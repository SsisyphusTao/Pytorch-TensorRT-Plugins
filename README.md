# DCNv2 - Pytorch/TensorRT

Two plugin implementations of dcnv2 for pytorch and tensorRT.

Easy use and easy understand.

## HOW TO USE

### pytorch

>`cd pytorchExtension`  
>`python ./setup.py install --user(optioanl)`  

Then, you can directly use `DeformableConv2DLayer` in `dcn_v2_wrapper.py` or referring to it to write your own layer.

### TensorRT

Copy everything under folder `tensorrtPlugin` to `[TensorRT](https://github.com/NVIDIA/TensorRT)/plugin`.