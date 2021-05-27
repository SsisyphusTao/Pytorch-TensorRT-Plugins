import numpy as np

import tensorrt as trt
import torch

import common

import time

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
    if plugin_creator.name == 'Yolo_TRT':
        YoloCreator = plugin_creator
    if plugin_creator.name == 'BatchedNMS_TRT':
        nmsCreator = plugin_creator

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 288, 512)
    DTYPE = trt.float16
    NUM_CLS = 80

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Yolov5(object):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights
        self.engine = self.build_engine()

    def add_BatchNorm2d(self, input_tensor, parent):
        gamma = self.weights[parent + '.weight'].numpy()
        beta = self.weights[parent + '.bias'].numpy()
        mean = self.weights[parent + '.running_mean'].numpy()
        var = self.weights[parent + '.running_var'].numpy()
        eps = 1e-5

        scale = gamma / np.sqrt(var + eps)
        shift = beta - mean * gamma / np.sqrt(var + eps)
        power = np.ones_like(scale)

        return self.network.add_scale(input=input_tensor.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale, power=power)
    
    def add_conv(self, input_tensor, out_channels, kernel=1, stride=1, padding=None, group=1, act=True, parent=''):
        conv_w = self.weights[parent + '.conv.weight'].numpy()
        conv_b = self.weights[parent + '.conv.bias'].numpy()
        conv = self.network.add_convolution(input=input_tensor.get_output(0), num_output_maps=out_channels, kernel_shape=(kernel, kernel), kernel=conv_w, bias=conv_b)
        conv.stride = (stride, stride)
        conv.padding = (autopad(kernel, padding), autopad(kernel, padding))

        if act:
            ac = self.network.add_activation(input=conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
            ac.alpha = 0.1
            return ac
        else:
            return conv

    def add_focus(self, input_tensor, out_channels, kernel=1, stride=1, padding=None, group=1, act=True, parent=''):
        s1 = self.network.add_slice(input_tensor, (0,0,0), (ModelData.INPUT_SHAPE[0], ModelData.INPUT_SHAPE[1]//2, ModelData.INPUT_SHAPE[2]//2), (1,2,2))
        s2 = self.network.add_slice(input_tensor, (0,1,0), (ModelData.INPUT_SHAPE[0], ModelData.INPUT_SHAPE[1]//2, ModelData.INPUT_SHAPE[2]//2), (1,2,2))
        s3 = self.network.add_slice(input_tensor, (0,0,1), (ModelData.INPUT_SHAPE[0], ModelData.INPUT_SHAPE[1]//2, ModelData.INPUT_SHAPE[2]//2), (1,2,2))
        s4 = self.network.add_slice(input_tensor, (0,1,1), (ModelData.INPUT_SHAPE[0], ModelData.INPUT_SHAPE[1]//2, ModelData.INPUT_SHAPE[2]//2), (1,2,2))
        ct = self.network.add_concatenation([s1.get_output(0), s2.get_output(0), s3.get_output(0), s4.get_output(0)])
        return self.add_conv(ct, out_channels, 3, parent=parent+'.conv')

    def add_bottleneck(self, input_tensor, out_channels, shortcut=True, g=1, e=0.5, parent=''):
        c_ = int(out_channels * e)
        cv1 = self.add_conv(input_tensor, c_, parent=parent+'.cv1')
        cv2 = self.add_conv(cv1, out_channels, 3, parent=parent+'.cv2')
        if shortcut and input_tensor.get_output(0).shape[0] == out_channels:
            return self.network.add_elementwise(input_tensor.get_output(0), cv2.get_output(0), trt.ElementWiseOperation.SUM)
        else:
            return cv2

    def add_bottleneckcsp(self, input_tensor, out_channels, n=1, shortcut=True, g=1, e=0.5, parent=''):
        c_ = int(out_channels * e)
        m = self.add_conv(input_tensor, c_, parent=parent+'.cv1')
        for i in range(n):
            m = self.add_bottleneck(m, c_, shortcut, g, 1.0, parent=parent+'.m.%d'%i)
        conv2_w = self.weights[parent + '.cv2.weight'].numpy()
        conv2 = self.network.add_convolution(input=input_tensor.get_output(0), num_output_maps=c_, kernel_shape=(1, 1), kernel=conv2_w)
        conv3_w = self.weights[parent + '.cv3.weight'].numpy()
        conv3 = self.network.add_convolution(input=m.get_output(0), num_output_maps=c_, kernel_shape=(1, 1), kernel=conv3_w)
        ct = self.network.add_concatenation([conv3.get_output(0), conv2.get_output(0)])
        bn = self.add_BatchNorm2d(ct, parent=parent+'.bn')
        ac = self.network.add_activation(input=bn.get_output(0), type=trt.ActivationType.LEAKY_RELU)
        ac.alpha = 0.1
        cv4 = self.add_conv(ac, out_channels, parent=parent+'.cv4')
        return cv4
    
    def add_c3(self, input_tensor, out_channels, n=1, shortcut=True, g=1, e=0.5, parent=''):
        c_ = int(out_channels * e)
        m = self.add_conv(input_tensor, c_, parent='')
        cv2 = self.add_conv(input_tensor, c_, parent='')
        for i in range(n):
            m = self.add_bottleneck(m, c_, shortcut, g, 1.0, parent='')
        ct = self.network.add_concatenation([m.get_output(0), cv2.get_output(0)])
        cv3 = self.add_conv(ct, out_channels, parent='')
        return cv3

    def add_spp(self, input_tensor, out_channels, k=(5, 9, 13), parent=''):
        c_ = input_tensor.get_output(0).shape[0] // 2
        cv1 = self.add_conv(input_tensor, c_, parent=parent+'.cv1')
        m = [cv1.get_output(0)]
        for i in k:
            m_ = self.network.add_pooling(cv1.get_output(0), trt.PoolingType.MAX, (i, i))
            m_.stride = (1, 1)
            m_.padding = (i//2, i//2)
            m.append(m_.get_output(0))
        ct = self.network.add_concatenation(m)
        cv2 = self.add_conv(ct, out_channels, parent=parent+'.cv2')
        return cv2

    def add_detect(self, input_tensors, out_channels, parent=''):
        m = [self.network.add_convolution(input_tensors[i].get_output(0), out_channels, (1, 1), kernel=self.weights[parent + '.m.%d.weight'%i].numpy(), bias=self.weights[parent + '.m.%d.bias'%i].numpy()) for i in range(len(input_tensors))]
        return [self.network.add_activation(input=x.get_output(0), type=trt.ActivationType.SIGMOID) for x in m]

    def add_yoloHead(self, input_tensors):
        mh = ModelData.INPUT_SHAPE[1]/32
        mw = ModelData.INPUT_SHAPE[2]/32
        anchors = np.array([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], dtype=np.float32)
        num_cls = trt.PluginField("num_cls", np.array([3], dtype=np.int32), trt.PluginFieldType.INT32)
        max_det = trt.PluginField("max_det", np.array([3024], dtype=np.int32), trt.PluginFieldType.INT32)
        heights = trt.PluginField("heights", np.array([mh*4,mh*2,mh], dtype=np.int32), trt.PluginFieldType.INT32)
        widths  = trt.PluginField("widths", np.array([mw*4,mw*2,mw], dtype=np.int32), trt.PluginFieldType.INT32)
        strides = trt.PluginField("strides", np.array([8,16,32], dtype=np.int32), trt.PluginFieldType.INT32)
        anchors = trt.PluginField("anchors", anchors, trt.PluginFieldType.FLOAT32)
        field_collection = trt.PluginFieldCollection([num_cls, max_det, heights, widths, strides, anchors])
        yoloHead = YoloCreator.create_plugin(name='Yolo_TRT', field_collection=field_collection)

        return self.network.add_plugin_v2(inputs=[x.get_output(0) for x in input_tensors], plugin=yoloHead)

    def add_nms(self, input_tensors):
        shareLocation = trt.PluginField("shareLocation", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
        backgroundLabelId = trt.PluginField("backgroundLabelId", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32)
        numClasses  = trt.PluginField("numClasses", np.array([3], dtype=np.int32), trt.PluginFieldType.INT32)
        topK = trt.PluginField("topK", np.array([300], dtype=np.int32), trt.PluginFieldType.INT32)
        keepTopK = trt.PluginField("keepTopK", np.array([100], dtype=np.int32), trt.PluginFieldType.INT32)
        scoreThreshold = trt.PluginField("scoreThreshold", np.array([0.65], dtype=np.float32), trt.PluginFieldType.FLOAT32)
        iouThreshold = trt.PluginField("iouThreshold", np.array([0.5], dtype=np.float32), trt.PluginFieldType.FLOAT32)
        isNormalized = trt.PluginField("isNormalized", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
        clipBoxes = trt.PluginField("clipBoxes", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)

        field_collection = trt.PluginFieldCollection([shareLocation, backgroundLabelId, numClasses,
        topK, keepTopK, scoreThreshold, iouThreshold, isNormalized, clipBoxes])
        nms = nmsCreator.create_plugin(name='BatchedNMS_TRT', field_collection=field_collection)

        return self.network.add_plugin_v2(inputs=[input_tensors.get_output(x) for x in range(2)], plugin=nms)

    def populate_network(self):
        # Configure the network layers based on the self.weights provided.
        input_tensor = self.network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

        L0 = self.add_focus(input_tensor, 64, 3, parent='model.0')
        L1 = self.add_conv(L0, 128, 3, 2, parent='model.1')
        L2 = self.add_bottleneckcsp(L1, 128, 3, parent='model.2')
        L3 = self.add_conv(L2, 256, 3, 2, parent='model.3')
        L4 = self.add_bottleneckcsp(L3, 256, 9, parent='model.4')
        L5 = self.add_conv(L4, 512, 3, 2, parent='model.5')
        L6 = self.add_bottleneckcsp(L5, 512, 9, parent='model.6')
        L7 = self.add_conv(L6, 1024, 3, 2, parent='model.7')
        L8 = self.add_spp(L7, 1024, parent='model.8')
        L9 = self.add_bottleneckcsp(L8, 1024, 3, False, parent='model.9')

        L10 = self.add_conv(L9, 512, parent='model.10')
        L11 = self.network.add_resize(L10.get_output(0))
        L11.scales = (1., 2., 2.)
        L11.resize_mode = trt.ResizeMode.NEAREST
        L12 = self.network.add_concatenation([L11.get_output(0), L6.get_output(0)])
        L13 = self.add_bottleneckcsp(L12, 512, 3, False, parent='model.13')
        
        L14 = self.add_conv(L13, 256, parent='model.14')
        L15 = self.network.add_resize(L14.get_output(0))
        L15.scales = (1., 2., 2.)
        L15.resize_mode = trt.ResizeMode.NEAREST
        L16 = self.network.add_concatenation([L15.get_output(0), L4.get_output(0)])
        L17 = self.add_bottleneckcsp(L16, 256, 3, False, parent='model.17')

        L18 = self.add_conv(L17, 256, 3, 2, parent='model.18')
        L19 = self.network.add_concatenation([L18.get_output(0), L14.get_output(0)])
        L20 = self.add_bottleneckcsp(L19, 512, 3, False, parent='model.20')

        L21 = self.add_conv(L20, 512, 3, 2, parent='model.21')
        L22 = self.network.add_concatenation([L21.get_output(0), L10.get_output(0)])
        L23 = self.add_bottleneckcsp(L22, 1024, 3, False, parent='model.23')
        L24 = self.add_detect([L17, L20, L23], (ModelData.NUM_CLS+5)*3, parent='model.24')

        # self.network.mark_output(tensor=L24[0].get_output(0))
        # self.network.mark_output(tensor=L24[1].get_output(0))
        # self.network.mark_output(tensor=L24[2].get_output(0)) 
        yolo_out = self.add_yoloHead(L24)

        final_out = self.add_nms(yolo_out)
        self.network.mark_output(tensor=final_out.get_output(0))
        self.network.mark_output(tensor=final_out.get_output(1))
        self.network.mark_output(tensor=final_out.get_output(2))
        self.network.mark_output(tensor=final_out.get_output(3))
        final_out.get_output(0).name = 'kc'
        final_out.get_output(1).name = 'nb'
        final_out.get_output(2).name = 'ns'
        final_out.get_output(3).name = 'nc'

    def build_engine(self):
        # For more information on TRT basics, refer to the introductory samples.
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
            self.network = network
            builder.max_workspace_size = common.GiB(1)
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # Populate the network using self.weights from the PyTorch model.
            self.populate_network()
            # Build and return an engine.
            return builder.build_cuda_engine(self.network)

def load_random_test_case(pagelocked_buffer):
    # Select an image at random to be the test case.
    img = np.random.randn(1,*ModelData.INPUT_SHAPE).astype(np.float32)
    img_ = np.concatenate([img[..., ::2, ::2], img[..., 1::2, ::2], img[..., ::2, 1::2], img[..., 1::2, 1::2]], 1)
    print(img_.shape)
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img_.ravel())
    return img

def main():
    # # Get the PyTorch weights
    weights = torch.load('yolov5.pth')
    # Do inference with TensorRT.
    with Yolov5(weights).engine as engine:
        with open('yolov5.trt', "wb") as f:
            f.write(engine.serialize())
    with open('yolov5.trt', "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            load_random_test_case(pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [kc, nb, ns, nc] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
            print(kc.shape, nb.shape, ns.shape, nc.shape)

if __name__ == '__main__':
    main()