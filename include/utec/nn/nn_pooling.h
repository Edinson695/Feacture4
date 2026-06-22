#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_POOLING_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_POOLING_H

#include "utec/nn/nn_interfaces.h"
#include <stdexcept>
#include <limits>

namespace utec::tf::layers {

    class MaxPooling2D : public Layer<float> {
    private:
        Shape pool_size_;
        Shape input_shape_;
        Shape output_shape_;

        Tensor<float> cached_input_;

    public:
        explicit MaxPooling2D(const Shape& pool_size) : pool_size_(pool_size) {
            if (pool_size.rank() != 2 || pool_size[0] <= 0 || pool_size[1] <= 0) {
                throw std::invalid_argument("Pool size invalido");
            }
        }

        void build(const Shape& input_shape) override {
            input_shape_ = input_shape; // {H, W, C}
            int h_out = input_shape[0] / pool_size_[0];
            int w_out = input_shape[1] / pool_size_[1];
            output_shape_ = Shape{ h_out, w_out, input_shape[2] };
        }

        Tensor<float> forward(const Tensor<float>& x) override {
            cached_input_ = x;
            int B = x.shape()[0], H = x.shape()[1], W = x.shape()[2], C = x.shape()[3];
            int p_h = pool_size_[0], p_w = pool_size_[1];
            int H_out = H / p_h;
            int W_out = W / p_w;

            Tensor<float> out = Tensor<float>::zeros(Shape{ B, H_out, W_out, C });

            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        for (int c = 0; c < C; ++c) {
                            float max_val = std::numeric_limits<float>::lowest();
                            for (int ph = 0; ph < p_h; ++ph) {
                                for (int pw = 0; pw < p_w; ++pw) {
                                    float val = x.at({ b, h * p_h + ph, w * p_w + pw, c });
                                    if (val > max_val) max_val = val;
                                }
                            }
                            out.at({ b, h, w, c }) = max_val;
                        }
                    }
                }
            }
            return out;
        }

        [[nodiscard]] Shape output_shape() const override { return output_shape_; }

        [[nodiscard]] std::unique_ptr<Layer<float>> clone() const override {
            return std::make_unique<MaxPooling2D>(*this);
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