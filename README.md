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

[InferPlugin.cpp](https://github.com/NVIDIA/TensorRT/blob/master/plugin/InferPlugin.cpp)
[CMakeLists.txt](https://github.com/NVIDIA/TensorRT/blob/master/plugin/CMakeLists.txt)

## EXAMPLES

Therea are two pytorch2tensorrt transfer scripts in `examples` to show how these plugins work.  

### Evaluation

>`cd examples`  
>`python mbv3_centernet_trt7.py`  

The evaluation output is as follow which are mean values of `hm`, `wh`, `reg`  
