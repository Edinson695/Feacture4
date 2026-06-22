#ifndef PROG3_PF_EPIC1_FEATURE4_V2026_01_NN_OPTIMIZER_H
#define PROG3_PF_EPIC1_FEATURE4_V2026_01_NN_OPTIMIZER_H

#include "utec/algebra/tensor_backend.h"
#include "utec/algebra/shape.h"
#include <stdexcept>
#include <cstddef>

namespace utec::tf {
    namespace optimizers {

        template<typename T = float>
        class SGD {
        private:
            T learning_rate_; 

        public:
            explicit SGD(T learning_rate) : learning_rate_(learning_rate) {
                if (learning_rate <= 0) {
                    throw std::invalid_argument("El learning rate debe ser positivo");
                }
            }

       
            void update(Tensor<T>& parameter, const Tensor<T>& gradient) {
                if (!(parameter.shape() == gradient.shape())) {
                    throw std::invalid_argument("Las formas del parametro y el gradiente no coinciden");
                }

                for (std::size_t i = 0; i < parameter.numel(); ++i) {
                    
                    parameter[i] = parameter[i] - (learning_rate_ * gradient[i]);
                }
            }
        };

    }
}  

using namespace utec::tf;

#endif //PROG3_PF_EPIC1_FEATURE4_V2026_01_NN_OPTIMIZER_H