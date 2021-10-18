import numpy as np

import tensorrt as trt
import torch

import common
from mbv3_centernet_pytorch import get_pose_net

import time

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
    if plugin_creator.name == 'DCNv2_TRT':
        dcnCreator = plugin_creator

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "prob"
    DTYPE = trt.float16

class MobileNetv3(object):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights
        self.k = [3,3,3,5,5,5,3,3,3,3,3,3,5,5,5]
        self.isize = [16, 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160]
        self.esize = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 672, 960]
        self.osize = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 960]
        self.active = 6
        self.se = [3,4,5,10,11,12,13,14]
        self.s = [2,1,2,1,2,1,1,2,1,1,1,1,1,1,2,1,1]
        self.n = ['0.0','0.1','0.2','1.0','1.1','1.2','2.0','2.1','2.2','2.3','2.4','2.5','2.6','3.0','3.1']
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

    def add_hswish(self, input_tensor):
        hs = self.network.add_activation(input=input_tensor.get_output(0), type=trt.ActivationType.HARD_SIGMOID)
        hs.alpha = 1./6.
        hs.beta = 0.5
        return self.network.add_elementwise(input_tensor.get_output(0), hs.get_output(0), trt.ElementWiseOperation.PROD)
    def add_relu(self, input_tensor):
        return self.network.add_activation(input=input_tensor.get_output(0), type=trt.ActivationType.RELU)

    def add_deformConv(self, input_tensor, out_channels, parent, kernel=3, stride=1, padding=1, dilation=1, deformable_group=1):
        out_channels = trt.PluginField("out_channels", np.array([out_channels], dtype=np.int32), trt.PluginFieldType.INT32)
        kernel = trt.PluginField("kernel", np.array([kernel], dtype=np.int32), trt.PluginFieldType.INT32)
        deformable_group = trt.PluginField("deformable_group", np.array([deformable_group], dtype=np.int32), trt.PluginFieldType.INT32)
        dilation = trt.PluginField("dilation", np.array([dilation], dtype=np.int32), trt.PluginFieldType.INT32)
        padding = trt.PluginField("padding", np.array([padding], dtype=np.int32), trt.PluginFieldType.INT32)
        stride = trt.PluginField("stride", np.array([stride], dtype=np.int32), trt.PluginFieldType.INT32)
        weight = trt.PluginField("weight", self.weights[parent + '.conv.weight'].numpy(), trt.PluginFieldType.FLOAT32)
        bias = trt.PluginField("bias", self.weights[parent + '.conv.bias'].numpy(), trt.PluginFieldType.FLOAT32)
        field_collection = trt.PluginFieldCollection([out_channels, kernel, deformable_group, dilation, padding, stride, weight, bias])
        DCN = dcnCreator.create_plugin(name='DCNv2_TRT', field_collection=field_collection)

        conv_offset_mask_w = self.weights[parent + '.conv.conv_offset_mask.weight'].numpy()
        conv_offset_mask_b = self.weights[parent + '.conv.conv_offset_mask.bias'].numpy()
        conv_offset_mask = self.network.add_convolution(input=input_tensor.get_output(0),
                                                        num_output_maps=1*3*3*3,
                                                        kernel_shape=(3, 3),
                                                        kernel=conv_offset_mask_w,
                                                        bias=conv_offset_mask_b)
        conv_offset_mask.padding = (1,1)
        sigmoid_conv_offset_mask = self.network.add_activation(input=conv_offset_mask.get_output(0), type=trt.ActivationType.SIGMOID)

        dcn = self.network.add_plugin_v2(inputs=[input_tensor.get_output(0), conv_offset_mask.get_output(0), sigmoid_conv_offset_mask.get_output(0)], plugin=DCN)
        bn = self.add_BatchNorm2d(dcn, parent+'.actf.0')
        return self.add_relu(bn)

    def add_IDAUp(self, input_tensors, out_channels, up_f, parent):
        for i in range(1, len(up_f)):
            proj = self.add_deformConv(input_tensors[i], out_channels, parent+'.proj_%d'%i)
            f = up_f[i]
            up_w = self.weights[parent + '.up_%d.weight'%i].numpy()
            up = self.network.add_deconvolution(proj.get_output(0), out_channels, (f*2, f*2), up_w)
            up.stride = (f, f)
            up.padding = (f//2, f//2)
            up.num_groups = out_channels
            node = self.network.add_elementwise(input_tensors[i-1].get_output(0), up.get_output(0), trt.ElementWiseOperation.SUM)
            input_tensors[i] = self.add_deformConv(node, out_channels, parent+'.node_%d'%i)
        return input_tensors[-1]
    
    def add_head(self, input_tensor, out_channels, head):
        conv1_w = self.weights[head+'.0.weight'].numpy()
        conv1_b = self.weights[head+'.0.bias'].numpy()
        conv1 = self.network.add_convolution(input_tensor.get_output(0), 256, (3,3), conv1_w, conv1_b)
        conv1.padding = (1, 1)
        ac1 = self.add_relu(conv1)
        conv2_w = self.weights[head + '.2.weight'].numpy()
        conv2_b = self.weights[head+'.2.bias'].numpy()
        conv2 = self.network.add_convolution(ac1.get_output(0), out_channels, (1, 1), conv2_w, conv2_b)
        return conv2

    def SeModule(self, input_tensor, in_size, reduction=4, parent=''):
        wsize = input_tensor.get_output(0).shape[-1]
        avgpool = self.network.add_pooling(input=input_tensor.get_output(0), type=trt.PoolingType.AVERAGE, window_size=(wsize, wsize))
        conv1_w = self.weights[parent+'.1.weight'].numpy()
        conv1 = self.network.add_convolution(avgpool.get_output(0), in_size//reduction, (1,1), conv1_w)
        bn1 = self.add_BatchNorm2d(conv1, parent+'.2')
        ac1 = self.add_relu(bn1)
        conv2_w = self.weights[parent + '.4.weight'].numpy()
        conv2 = self.network.add_convolution(ac1.get_output(0), in_size, (1, 1), conv2_w)
        bn2 = self.add_BatchNorm2d(conv2, parent+'.5')
        hs = self.network.add_activation(input=bn2.get_output(0), type=trt.ActivationType.HARD_SIGMOID)
        hs.alpha = 1./6.
        hs.beta = 0.5
        return self.network.add_elementwise(input_tensor.get_output(0), hs.get_output(0), trt.ElementWiseOperation.PROD)

    def LinearBottleneck(self, input_tensor, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, parent=''):
        conv1_w = self.weights[parent + '.conv1.weight'].numpy()
        conv1 = self.network.add_convolution(input=input_tensor.get_output(0), num_output_maps=expand_size, kernel_shape=(1, 1), kernel=conv1_w)

        bn1 = self.add_BatchNorm2d(conv1, parent + '.bn1')
        nl1 = nolinear(bn1)

        conv2_w = self.weights[parent + '.conv2.weight'].numpy()
        conv2 = self.network.add_convolution(nl1.get_output(0), expand_size, (kernel_size, kernel_size), conv2_w)
        conv2.stride = (stride, stride)
        conv2.padding = (kernel_size // 2, kernel_size // 2)
        conv2.num_groups = expand_size
  
        bn2 = self.add_BatchNorm2d(conv2, parent + '.bn2')
        nl2 = nolinear(bn2)

        conv3_w = self.weights[parent + '.conv3.weight'].numpy()
        conv3 = self.network.add_convolution(input=nl2.get_output(0), num_output_maps=out_size, kernel_shape=(1, 1), kernel=conv3_w)

        bn3 = self.add_BatchNorm2d(conv3, parent + '.bn3')

        if semodule != False:
            se = self.SeModule(bn3, out_size, parent=parent+'.se.se')
        else:
            se = bn3

        if stride == 1 and in_size != out_size:
            conv_shortcut_w = self.weights[parent + '.shortcut.0.weight'].numpy()
            conv_shortcut = self.network.add_convolution(input=input_tensor.get_output(0), num_output_maps=out_size, kernel_shape=(1, 1), kernel=conv_shortcut_w)
            bn_shortcut = self.add_BatchNorm2d(conv_shortcut, parent+'.shortcut.1')
            out = self.network.add_elementwise(bn_shortcut.get_output(0), se.get_output(0), trt.ElementWiseOperation.SUM)
        elif stride == 1:
            out = self.network.add_elementwise(input_tensor.get_output(0), se.get_output(0), trt.ElementWiseOperation.SUM)
        else:
            out = se        
        return out

    def populate_network(self):
        # Configure the network layers based on the self.weights provided.
        input_tensor = self.network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

        conv1_w = self.weights['conv1.weight'].numpy()
        conv1 = self.network.add_convolution(input=input_tensor, num_output_maps=self.isize[0], kernel_shape=(3, 3), kernel=conv1_w)
        conv1.stride = (self.s[0], self.s[0])
        conv1.padding = (1, 1)

        bn1 = self.add_BatchNorm2d(conv1, 'bn1')
        hs1 = self.add_hswish(bn1)

        name = 'bneck'
        bneck0 = self.LinearBottleneck(hs1, self.k[0], self.isize[0], self.esize[0], self.osize[0], self.add_relu, False, self.s[1], name+self.n[0])
        for i in range(1, 15):
            if i < self.active:
                locals()[name+str(i)] = self.LinearBottleneck(locals()[name+str(i-1)], self.k[i], self.isize[i], self.esize[i], self.osize[i], self.add_relu, i in self.se, self.s[i+1], name+self.n[i])
            else:
                locals()[name+str(i)] = self.LinearBottleneck(locals()[name+str(i-1)], self.k[i], self.isize[i], self.esize[i], self.osize[i], self.add_hswish, i in self.se, self.s[i+1], name+self.n[i])
        out = [locals()['bneck2'], locals()['bneck5'], locals()['bneck12']]

        conv2_w = self.weights['conv2.weight'].numpy()
        conv2 = self.network.add_convolution(input=locals()['bneck14'].get_output(0), num_output_maps=self.osize[-1], kernel_shape=(1, 1), kernel=conv2_w)

        bn2 = self.add_BatchNorm2d(conv2, 'bn2')
        hs2 = self.add_hswish(bn2)
        out.append(hs2)

        ida_up = self.add_IDAUp(out, 24, [2 ** i for i in range(4)], 'ida_up')
        hm = self.add_head(ida_up, 2, 'hm')
        wh = self.add_head(ida_up, 2, 'wh')
        reg = self.add_head(ida_up, 2, 'reg')
        #hm_sig = self.network.add_activation(input=hm.get_output(0), type=trt.ActivationType.SIGMOID)
        hm_mask = self.network.add_pooling(input=hm.get_output(0), type=trt.PoolingType.MAX, window_size=(3, 3))
        hm_mask.stride = (1, 1)
        hm_mask.padding = (1, 1)

        hm.get_output(0).name = 'hm'
        wh.get_output(0).name = 'wh'
        reg.get_output(0).name = 'reg'
        hm_mask.get_output(0).name = 'hm_mask'

        self.network.mark_output(tensor=hm.get_output(0))
        self.network.mark_output(tensor=wh.get_output(0))
        self.network.mark_output(tensor=reg.get_output(0))
        self.network.mark_output(tensor=hm_mask.get_output(0))

    def build_engine(self):
        # For more information on TRT basics, refer to the introductory samples.
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
            self.network = network
            builder.max_workspace_size = 32
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # Populate the network using self.weights from the PyTorch model.
            self.populate_network()
            # Build and return an engine.
            return builder.build_cuda_engine(self.network)

def load_random_test_case(pagelocked_buffer):
    # Select an image at random to be the test case.
    img = np.random.randn(1,3,224,224).astype(np.float32)
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img.ravel())
    return img

def main():
    common.add_help(description="Yeah!")
    # Get the PyTorch weights
    mobilenetv3 = get_pose_net({'hm':2, 'wh':2, 'reg':2})
    mobilenetv3.init_params()
    mobilenetv3.eval()
    weights = mobilenetv3.state_dict()
    # Do inference with TensorRT.
    with MobileNetv3(weights).engine as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        with open('mobilenetv3-centernet.trt', "wb") as f:
            f.write(engine.serialize())

        with open('mobilenetv3-centernet.trt', "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            with engine.create_execution_context() as context:
                t = 0
                for _ in range(1):
                    img = load_random_test_case(pagelocked_buffer=inputs[0].host)
                    # For more information on performing inference, refer to the introductory samples.
                    # The common.do_inference function will return a list of outputs - we only have one in this case.
                    a = time.time()
                    [hm, wh, reg, _] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
                    t += time.time() - a

        with torch.no_grad():
            [baseline] = mobilenetv3.cuda()(torch.from_numpy(img).cuda())
            print('baseline: ', baseline['hm'].mean().cpu().numpy(), baseline['wh'].mean().cpu().numpy(), baseline['reg'].mean().cpu().numpy())
        print('output:   ', np.mean(hm), np.mean(wh), np.mean(reg))
    print('Time: ', t)

if __name__ == '__main__':
    main()
