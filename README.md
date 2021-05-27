# Plugins for Pytorch & TensorRT

It's a personal backup repository, but I will try my best to help you understand how to use them.

Now it contains:
|Plugins||
|-|-|
|DCNv2 | pytorch1.5+/tensorrt7 |
|yolov5 detection | tensorrt7|

|Models||
|-|-|
|mobilenetv3-centernet | pytorch1.5+/tensorrt7 |
|yolov5 (integrated with detection and nms) | tensorrt7 |

## HOW TO USE

### Pytorch

>`cd pytorchExtension`  
>`python ./setup.py install --user(optioanl)`  

Then, you can directly use `DeformableConv2DLayer` in `dcn_v2_wrapper.py` or referring to it to write your own layer.

### TensorRT

Copy all files in `tensorrtPlugin` folder to [TensorRT](https://github.com/NVIDIA/TensorRT)/plugin and build.


## EXAMPLE

I provide two models to show how these plugins work.  
I also provide a mobilenetv3-centernet model for evaluation, if all environment settled, `cd examples` and run `python mbv3_centernet_trt.py`
The evaluation output is as follow which are mean values of `hm`, `wh`, `reg`  
![eval](https://user-images.githubusercontent.com/47047345/119774547-e6e50b00-bef4-11eb-9c90-3afb97018265.png)
