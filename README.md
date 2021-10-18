# Plugins for Pytorch & TensorRT

It's a personal backup repository, but I will try my best to help you understand how to use them.

Now it contains:
|Plugins|Support|
|-|-|
|DCNv2 | pytorch1.5+/tensorrt7 |
|yolov5 detection | tensorrt7/tensorrt8 |

|Models|Support|
|-|-|
|mobilenetv3-centernet | pytorch1.5+/tensorrt7 |
|yolov5 (integrated with detection and nms) | tensorrt7/tensorrt8 |

## Installation

### Pytorch

>`cd pytorch`  
>`python setup.py install --user(optioanl)`  
>`cd ../examples`
>`python`  
>`from dcn_v2_wrapper import DeformableConv2DLayer as DCN`

### TensorRT

1. Copy plugin folders from `tensorrt` to [`NVIDIA/TensorRT/plugin`](https://github.com/NVIDIA/TensorRT/tree/master/plugin)

2. Add relative head file and initializePlugin() to [InferPlugin.cpp](https://github.com/NVIDIA/TensorRT/blob/master/plugin/InferPlugin.cpp) at proper place, for example

>`#include "dcnv2Plugin.h"`  
>`#include "yoloPlugin.h"`

>`initializePlugin<nvinfer1::plugin::DCNv2PluginCreator>(logger, libNamespace);`
>`initializePlugin<nvinfer1::plugin::YoloPluginCreator>(logger, libNamespace);`

3. Add name of plugin folder to `PLUGIN_LISTS` in [CMakeLists.txt](https://github.com/NVIDIA/TensorRT/blob/master/plugin/CMakeLists.txt)

4. Build and use `libnvinfer_plugin.so` following offical introduction.

## EXAMPLES

There are two pytorch2tensorrt transfer scripts in `examples` to show how these plugins work.  

### Evaluation

>`cd examples`  
>`python mbv3_centernet_trt7.py`  

The evaluation output is as follow which are mean values of `hm`, `wh`, `reg`  
![eval](https://user-images.githubusercontent.com/47047345/137699271-534c7a92-99d0-47f6-8628-3904b4041c61.png)
