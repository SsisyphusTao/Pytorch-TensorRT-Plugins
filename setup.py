from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
    name='dcn_op_v2',
    ext_modules=[
        CUDAExtension(
        'dcn_op_v2',
        sources=['src/dcn_v2_cuda.cc','src/dcn_v2_im2col_cuda.cu'],
    )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)