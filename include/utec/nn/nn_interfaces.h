#ifndef PROG3_PF_EPIC1_FEATURE4_V2026_1_NN_INTERFACES_H
#define PROG3_PF_EPIC1_FEATURE4_V2026_1_NN_INTERFACES_H

#include "utec/algebra/tensor_backend.h"
#include "utec/algebra/shape.h"
#include <memory>
#include <map>  
#include <string>

namespace utec::tf {
    template<typename T = float>
    class Layer {
    public:
        virtual ~Layer() noexcept(false) = default;

        virtual void build(const Shape& input_shape) = 0;
        virtual Tensor<T> forward(const Tensor<T>& x) = 0;

        virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;

        [[nodiscard]] virtual Shape output_shape() const = 0;

        [[nodiscard]] virtual std::map<std::string, Tensor<T>> parameters() const {
            return {};
        }

        [[nodiscard]] virtual std::map<std::string, Tensor<T>> gradients() const {
            return {};
        }

        [[nodiscard]] virtual std::unique_ptr<Layer> clone() const = 0;
    };

    namespace layers {
        class Input : public Layer<float> {
        private:
            Shape shape_;

        public:
            explicit Input(const Shape& shape) : shape_(shape) {
                if (shape.rank() == 0) {
                    throw std::invalid_argument("El Input shape no puede estar vacio");
                }
            }

            void build(const Shape& input_shape) override {}

            Tensor<float> forward(const Tensor<float>& x) override {
                return x;
            }

            Tensor<float> backward(const Tensor<float>& grad_output) override {
                return grad_output;
            }

            [[nodiscard]] Shape output_shape() const override {
                return shape_;
            }

            [[nodiscard]] std::unique_ptr<Layer<float>> clone() const override {
                return std::make_unique<Input>(*this);
            }
        };
    }
}
using namespace utec::tf;
#endif //PROG3_PF_EPIC1_FEATURE4_V2026_1_NN_INTERFACES_H