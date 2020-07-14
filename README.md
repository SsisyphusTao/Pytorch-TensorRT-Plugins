# DCNv2 - Deformable Convolution Layer

This is the main operator in Deformable Convolutional Networks V2, which is also used in DLA network in [CenterNet](https://github.com/xingyizhou/CenterNet).

I reimplemented it with C++ API, and rewrote the building method, making it compatible with pytorch 1.5 and much more clear.

`Backward is not reentrant (minor)` error still exists, but doesn't have visible influence. For more details, you can see [the explanation](https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/DCNv2/README.md) from original project.

## HOW TO USE

>`python ./setup.py install --user(optioanl)`

Then, you can directly use `DeformableConv2DLayer` in `dcn_v2_wrapper.py` or referring to it to write your own layer.