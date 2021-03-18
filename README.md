# DCNv2 - Pytorch/TensorRT

Two plugin implementations of dcnv2 for Pytorch1.5+ and TensorRT7.

Easy use and easy understand.

## HOW TO USE

### pytorch

>`cd pytorchExtension`  
>`python ./setup.py install --user(optioanl)`  

Then, you can directly use `DeformableConv2DLayer` in `dcn_v2_wrapper.py` or referring to it to write your own layer.

### TensorRT

Copy all files in this folder to [TensorRT](https://github.com/NVIDIA/TensorRT)/plugin and build.
See `examples/trtExample.py` to find how to use it in python.

## EXAMPLE

I provide `examples/mobilenetv3.py` and `examples/mbv2ct.py` to show how to use dcnv2 in mobilenetv3-centernet in pytorch and how to transfer it into tensorrt.
I also provide a model for evaluation.
The evaluation output is like this:  
![eval.png](https://raw.githubusercontent.com/SsisyphusTao/DCNv2-Pytorch-TensorRT7/master/examples/eval.png)
