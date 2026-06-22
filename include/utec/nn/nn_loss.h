#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_LOSS_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_LOSS_H

#include "utec/algebra/tensor_backend.h"
#include <cmath>
#include <stdexcept>

namespace utec::tf::losses {

    template<typename T = float>
    class CategoricalCrossentropy {
    public:
        Tensor<T> gradient(const Tensor<T>& y_true, const Tensor<T>& y_pred) const {
            Tensor<T> grad = y_pred - y_true;

            std::size_t batch_size = (y_true.rank() == 2) ? y_true.shape()[0] : 1;
        
            for (std::size_t i = 0; i < grad.numel(); ++i) {
                grad[i] = grad[i] / static_cast<T>(batch_size);
            }
            return grad;
        }

        T operator()(const Tensor<T>& y_true, const Tensor<T>& y_pred) const {
            if (y_true.rank() != y_pred.rank() || y_true.size() != y_pred.size()) {
                throw std::invalid_argument("Las formas de y_true y y_pred deben coincidir");
            }

            const T epsilon = 1e-7;
            T total_loss = 0;

            for (std::size_t i = 0; i < y_true.size(); ++i) {
                total_loss += y_true[i] * std::log(y_pred[i] + epsilon);
            }

            std::size_t batch_size = (y_true.rank() == 2) ? y_true.shape()[0] : 1;

            return -total_loss / static_cast<T>(batch_size);
        }
    };
}

using namespace utec::tf;

#endif // PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_LOSS_H