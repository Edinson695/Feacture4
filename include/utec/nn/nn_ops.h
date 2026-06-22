
//
// Created by Ruben Rivas on 10/11/2020.
//

#ifndef PROG3_PF_EPIC1_FEATURE2_V2026_1_UTEC_NN_NN_OPS_H
#define PROG3_PF_EPIC1_FEATURE2_V2026_1_UTEC_NN_NN_OPS_H

#include "utec/algebra/tensor_backend.h"

namespace utec::tf::ops {

    template <typename T>
    Tensor<T> matmul(const Tensor<T>& lhs, const Tensor<T>& rhs) {
 
        int A = lhs.shape()[0];
        int B = lhs.shape()[1];
        int C = rhs.shape()[1];
 
        Tensor<T> result = Tensor<T>::zeros(Shape{ A, C });

        for (int i = 0; i < A; ++i) {
            for (int j = 0; j < C; ++j) {
                T sum = 0;
                for (int k = 0; k < B; ++k) {
                    sum += lhs.at({ i, k }) * rhs.at({ k, j });
                }
                result.at({ i, j }) = sum;
            }
        }

        return result;
    }

   
    template <typename T>
    Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel) {
        if (input.rank() != 4 || kernel.rank() != 4) {
            throw std::invalid_argument("Conv2D requiere tensores de rank 4 (NHWC)");
        }

        int B = input.shape()[0], H = input.shape()[1], W = input.shape()[2], C_in = input.shape()[3];
        int K_H = kernel.shape()[0], K_W = kernel.shape()[1], K_C_in = kernel.shape()[2], C_out = kernel.shape()[3];

        if (C_in != K_C_in) throw std::invalid_argument("Canales de entrada no coinciden con kernel");
        if (H < K_H || W < K_W) throw std::invalid_argument("Kernel es mas grande que la imagen");

        int H_out = H - K_H + 1;
        int W_out = W - K_W + 1;

        Tensor<T> out = Tensor<T>::zeros(Shape{ B, H_out, W_out, C_out });

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    for (int c_out = 0; c_out < C_out; ++c_out) {
                        T sum = 0;
                        for (int kh = 0; kh < K_H; ++kh) {
                            for (int kw = 0; kw < K_W; ++kw) {
                                for (int c_in = 0; c_in < C_in; ++c_in) {
                                    sum += input.at({ b, h + kh, w + kw, c_in }) * kernel.at({ kh, kw, c_in, c_out });
                                }
                            }
                        }
                        out.at({ b, h, w, c_out }) = sum;
                    }
                }
            }
        }
        return out;
    }

    template <typename T>
    Tensor<T> flatten_batch(const Tensor<T>& input) {
        if (input.rank() == 0) throw std::invalid_argument("No se puede aplanar un escalar");
        int batch = input.shape()[0];
        int rest = input.numel() / batch;

        Tensor<T> result = input;
        result.reshape(Shape{ batch, rest });
        return result;
    }

}

#endif //PROG3_PF_EPIC1_FEATURE2_V2026_1_UTEC_NN_NN_OPS_H