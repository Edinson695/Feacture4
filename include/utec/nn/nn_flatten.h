#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_FLATTEN_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_FLATTEN_H

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_ops.h"
#include <memory>

namespace utec::tf::layers {

    class Flatten : public Layer<float> {
    private:
        Shape input_shape_;
        Shape output_shape_;

        Tensor<float> cached_input_;

    public:
        void build(const Shape& input_shape) override {
            input_shape_ = input_shape;
            int total_elements = 1;
            for (std::size_t i = 0; i < input_shape.rank(); ++i) {
                total_elements *= input_shape[i];
            }
            output_shape_ = Shape{ total_elements };
        }

        Tensor<float> forward(const Tensor<float>& x) override {
            cached_input_ = x;
            return ops::flatten_batch(x);
        }

        [[nodiscard]] Shape output_shape() const override { return output_shape_; }

        [[nodiscard]] std::unique_ptr<Layer<float>> clone() const override {
            return std::make_unique<Flatten>(*this);
        }

        Tensor<float> backward(const Tensor<float>& grad_output) override {
            if (cached_input_.rank() == 0) {
                throw std::logic_error("No se puede hacer backward sin haber hecho forward primero");
            }

            return Tensor<float>::zeros(cached_input_.shape());
        }

        [[nodiscard]] std::map<std::string, Tensor<float>> gradients() const override {
            return {};
        }

    };
}
#endif