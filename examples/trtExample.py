import tensorrt as trt
import numpy as np
import common

TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def get_trt_plugin(plugin_name):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                out_channels = trt.PluginField("out_channels", np.array([2], dtype=np.int32), trt.PluginFieldType.INT32)
                kernel = trt.PluginField("kernel", np.array([3], dtype=np.int32), trt.PluginFieldType.INT32)
                deformable_group = trt.PluginField("deformable_group", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                dilation = trt.PluginField("dilation", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                padding = trt.PluginField("padding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                stride = trt.PluginField("stride", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
                weight = trt.PluginField("weight", np.random.randn(2,2,3,3), trt.PluginFieldType.FLOAT32)
                bias = trt.PluginField("bias", np.random.randn(2), trt.PluginFieldType.FLOAT32)
                field_collection = trt.PluginFieldCollection([out_channels, kernel, deformable_group, dilation, padding, stride, weight, bias])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin

def add_yoloHead(network, input_tensors):
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == 'Yolo_TRT':
                YoloCreator = plugin_creator
        anchors = np.array([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], dtype=np.float32)
        num_cls = trt.PluginField("num_cls", np.array([3], dtype=np.int32), trt.PluginFieldType.INT32)
        max_det = trt.PluginField("max_det", np.array([3360], dtype=np.int32), trt.PluginFieldType.INT32)
        heights = trt.PluginField("heights", np.array([80,40,20], dtype=np.int32), trt.PluginFieldType.INT32)
        widths  = trt.PluginField("widths", np.array([128,64,32], dtype=np.int32), trt.PluginFieldType.INT32)
        strides = trt.PluginField("strides", np.array([8,16,32], dtype=np.int32), trt.PluginFieldType.INT32)
        anchors = trt.PluginField("anchors", anchors, trt.PluginFieldType.FLOAT32)
        field_collection = trt.PluginFieldCollection([num_cls, max_det, heights, widths, strides, anchors])
        yoloHead = YoloCreator.create_plugin(name='Yolo_TRT', field_collection=field_collection)

        return network.add_plugin_v2(inputs=[x.get_output(0) for x in input_tensors], plugin=yoloHead)

def main():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=common.EXPLICIT_BATCH) as network:
        builder.max_workspace_size = common.GiB(1)
        input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(1, 2, 4, 4))
        offset_mask_layer = network.add_input(name="offset_layer", dtype=trt.float32, shape=(27, 4, 4))
        offset_mask_sigmoid_layer = network.add_input(name="mask_layer", dtype=trt.float32, shape=(27, 4, 4))
        lrelu = network.add_plugin_v2(inputs=[input_layer, offset_mask_layer, offset_mask_sigmoid_layer], plugin=get_trt_plugin("DCNv2_TRT"))
        lrelu.get_output(0).name = "outputs"
        network.mark_output(lrelu.get_output(0))

        engine = builder.build_cuda_engine(network)
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            np.copyto(inputs[0].host, np.random.randn(1, 2, 4, 4).ravel())
            offset_mask = np.random.randn(27, 4, 4).ravel() #create by convolution layer in network
            np.copyto(inputs[1].host, offset_mask.ravel())
            np.copyto(inputs[2].host, sigmoid(offset_mask).ravel())
            output = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(output, output[0].shape)

if __name__ == '__main__':
    main()