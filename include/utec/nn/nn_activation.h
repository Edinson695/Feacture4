#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_ACTIVATION_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_ACTIVATION_H

#include "utec/algebra/tensor_backend.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace utec::tf {

    enum class Activation {
        Linear,
        Relu,
        Softmax
    };

    template<typename T>
    Tensor<T> apply_activation(const Tensor<T>& x, Activation act) {
        if (act == Activation::Linear) return x;

        Tensor<T> result = x;

        if (act == Activation::Relu) {
            for (std::size_t i = 0; i < result.size(); ++i) {
                result[i] = std::max(T(0), result[i]);
            }
        }
        else if (act == Activation::Softmax) {
            if (result.rank() != 1 && result.rank() != 2) {
                throw std::invalid_argument("Softmax solo acepta tensores 1D o 2D");
            }

            std::size_t batch_size = (result.rank() == 2) ? result.shape()[0] : 1;
            std::size_t classes = (result.rank() == 2) ? result.shape()[1] : result.shape()[0];

            for (std::size_t b = 0; b < batch_size; ++b) {
                T max_val = result[b * classes];
                for (std::size_t c = 1; c < classes; ++c) {
                    max_val = std::max(max_val, result[b * classes + c]);
                }
                T sum_exp = 0;
                for (std::size_t c = 0; c < classes; ++c) {
                    result[b * classes + c] = std::exp(result[b * classes + c] - max_val);
                    sum_exp += result[b * classes + c];
                }
                for (std::size_t c = 0; c < classes; ++c) {
                    result[b * classes + c] /= sum_exp;
                }
            }
        }
        return result;
    }
}
#endif