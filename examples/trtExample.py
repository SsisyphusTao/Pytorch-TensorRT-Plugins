import tensorrt as trt
import numpy as np
import common

TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

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

def main():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=common.EXPLICIT_BATCH) as network:
        builder.max_workspace_size = common.GiB(1)
        input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(1, 2, 4, 4))
        offset_layer = network.add_input(name="offset_layer", dtype=trt.float32, shape=(18, 4, 4))
        mask_layer = network.add_input(name="mask_layer", dtype=trt.float32, shape=(9, 4, 4))
        lrelu = network.add_plugin_v2(inputs=[input_layer, offset_layer, mask_layer], plugin=get_trt_plugin("DCNv2_TRT"))
        lrelu.get_output(0).name = "outputs"
        network.mark_output(lrelu.get_output(0))

        engine = builder.build_cuda_engine(network)
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            np.copyto(inputs[0].host, np.random.randn(1, 2, 4, 4).ravel())
            np.copyto(inputs[1].host, np.random.randn(18, 4, 4).ravel())
            np.copyto(inputs[2].host, np.random.randn(9, 4, 4).ravel())
            output = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(output, output[0].shape)

if __name__ == '__main__':
    main()