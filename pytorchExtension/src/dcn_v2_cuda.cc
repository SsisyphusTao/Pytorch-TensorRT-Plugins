#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "dcn_v2_im2col_cuda.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace torch;
using namespace c10::cuda;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

Tensor dcn_v2_cuda_forward(Tensor& input, Tensor& weight,
                         Tensor& bias,
                         Tensor& offset, Tensor& mask,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group)
{
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    // const int channels_out = weight.size(0);
    // const int channels_kernel = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    Tensor ones = input.new_full({1, height_out * width_out}, 1, {input.device()});
    Tensor columns = input.new_empty({channels * kernel_h * kernel_w, 1 * height_out * width_out}, {input.device()});

    vector<Tensor> outputs;

    for (int b = 0; b < batch; b++)
    {
        Tensor input_n = input.slice(0, b, b+1).squeeze();
        Tensor offset_n = offset.slice(0, b, b+1).squeeze();
        Tensor mask_n = mask.slice(0, b, b+1).squeeze();

        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)
        // long m_ = channels_out;
        // long n_ = height_out * width_out;
        // long k_ = 1;
        // THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
        //                  Tensor_data(state, ones), k_,
        //                  Tensor_data(state, bias), k_, 0.0f,
        //                  Tensor_data(state, output_n), n_);
        Tensor output_n = at::mm(bias.unsqueeze(1),ones);
        modulated_deformable_im2col_cuda(getCurrentCUDAStream(),
                                         input_n.data_ptr<float>(), offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group, columns.data_ptr<float>()); 

        //(k * m)  x  (m * n)
        // Y = WC
        // long m = channels_out;
        // long n = height_out * width_out;
        // long k = channels * kernel_h * kernel_w;
        // THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
        //                  Tensor_data(state, columns), n,
        //                  Tensor_data(state, weight), k, 1.0f,
        //                  Tensor_data(state, output_n), n);
        output_n = output_n.addmm(weight.flatten(1),columns);
        outputs.push_back(output_n.resize_({weight.size(0), height_out, width_out}));
    }
    return at::stack(at::TensorList(outputs));
}

vector<at::Tensor> dcn_v2_cuda_backward(Tensor& input, Tensor& weight,
                          Tensor& bias,
                          Tensor& offset, Tensor& mask,
                          Tensor& grad_output,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int deformable_group)
{
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    // const int channels_out = weight.size(0);
    // const int channels_kernel = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    Tensor ones = input.new_full({height_out * width_out}, 1, {input.device()});

    vector<Tensor> grad_input_list;
    Tensor grad_weight = weight.new_zeros(weight.sizes()).flatten(1);
    Tensor grad_bias = bias.new_zeros(bias.sizes());
    vector<Tensor> grad_offset_list;
    vector<Tensor> grad_mask_list;

    for (int b = 0; b < batch; b++)
    {
        Tensor input_n = input.slice(0, b, b+1).squeeze();
        Tensor offset_n = offset.slice(0, b, b+1).squeeze();
        Tensor mask_n = mask.slice(0, b, b+1).squeeze();
        Tensor grad_output_n = grad_output.slice(0, b, b+1).squeeze();
        Tensor grad_input_n = input_n.new_zeros(input_n.sizes());
        Tensor grad_offset_n = offset_n.new_zeros(offset_n.sizes());
        Tensor grad_mask_n = mask_n.new_zeros(mask_n.sizes());

        // long m = channels * kernel_h * kernel_w;
        // long n = height_out * width_out;
        // long k = channels_out;

        // THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
        //                  Tensor_data(state, grad_output_n), n,
        //                  Tensor_data(state, weight), m, 0.0f,
        //                  Tensor_data(state, columns), n);
        Tensor columns = at::mm(weight.flatten(1).t(), grad_output_n.flatten(1));

        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(getCurrentCUDAStream(),
                                               columns.data_ptr<float>(),
                                               input_n.data_ptr<float>(),
                                               offset_n.data_ptr<float>(),
                                               mask_n.data_ptr<float>(),
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n.data_ptr<float>(),
                                               grad_mask_n.data_ptr<float>());
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(getCurrentCUDAStream(),
                                         columns.data_ptr<float>(),
                                         offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n.data_ptr<float>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(getCurrentCUDAStream(),
                                         input_n.data_ptr<float>(),
                                         offset_n.data_ptr<float>(),
                                         mask_n.data_ptr<float>(),
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data_ptr<float>());
        // long m_ = channels_out;
        // long n_ = channels * kernel_h * kernel_w;
        // long k_ = height_out * width_out;

        // THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
        //                  Tensor_data(state, columns), k_,
        //                  Tensor_data(state, grad_output_n), k_, 1.0f,
        //                  Tensor_data(state, grad_weight), n_);

        grad_weight = grad_weight.addmm(grad_output_n.flatten(1), columns.flatten(1).t());
        grad_bias = grad_bias.addmv(grad_output_n.flatten(1), ones);

        // gradient w.r.t. bias
        // long m_ = channels_out;
        // long k__ = height_out * width_out;
    //     THCudaBlas_Sgemv(state,
    //                      't',
    //                      k_, m_, 1.0f,
    //                      Tensor_data(state, grad_output_n), k_,
    //                      Tensor_data(state, ones), 1, 1.0f,
    //                      Tensor_data(state, grad_bias), 1);

        grad_input_list.push_back(grad_input_n);
        grad_offset_list.push_back(grad_offset_n);
        grad_mask_list.push_back(grad_mask_n);
    }
    return {at::stack(at::TensorList(grad_input_list)).resize_(input.sizes()),
            grad_weight.resize_(weight.sizes()),
            grad_bias,
            at::stack(at::TensorList(grad_offset_list)).resize_(offset.sizes()),
            at::stack(at::TensorList(grad_mask_list)).resize_(mask.sizes())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcn_v2_cuda_forward, "DCN operator forward (CUDA)");
  m.def("backward", &dcn_v2_cuda_backward, "DCN operator backward (CUDA)");
}