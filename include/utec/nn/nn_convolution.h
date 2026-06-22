#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_CONVOLUTION_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_01_NN_CONVOLUTION_H

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_ops.h"
#include <string>
#include <stdexcept>
#include <map>

namespace utec::tf::layers {

    class Conv2D : public Layer<float> {
    private:
        int filters_;
        Shape kernel_size_;
        Activation activation_;
        Shape input_shape_;
        Shape output_shape_;

        Tensor<float> weights_;
        Tensor<float> bias_;
        Tensor<float> grad_weights_;  
        Tensor<float> grad_bias_;     

        Tensor<float> cached_input_;
        inline static int conv_count = 0;
        std::string name_;

    public:
        Conv2D(int filters, const Shape& kernel_size, Activation activation = Activation::Linear)
            : filters_(filters), kernel_size_(kernel_size), activation_(activation) {
            if (filters <= 0) throw std::invalid_argument("Filtros invalidos");
            if (kernel_size.rank() != 2 || kernel_size[0] <= 0 || kernel_size[1] <= 0)
                throw std::invalid_argument("Kernel invalido");

            name_ = "conv2d_" + std::to_string(conv_count++);
        }

        void build(const Shape& input_shape) override {
            input_shape_ = input_shape;
            int h_out = input_shape[0] - kernel_size_[0] + 1;
            int w_out = input_shape[1] - kernel_size_[1] + 1;
            output_shape_ = Shape{ h_out, w_out, filters_ };

            if (weights_.rank() == 0) {
                weights_ = Tensor<float>::ones(Shape{ kernel_size_[0], kernel_size_[1], input_shape[2], filters_ });
            }
            if (bias_.rank() == 0) {
                bias_ = Tensor<float>::zeros(Shape{ filters_ });
            }

            grad_weights_ = Tensor<float>::zeros(weights_.shape());
            grad_bias_ = Tensor<float>::zeros(bias_.shape());
        }

        Tensor<float> forward(const Tensor<float>& x) override {
            cached_input_ = x;

            Tensor<float> out = ops::conv2d(x, weights_);
            int B = out.shape()[0], H = out.shape()[1], W = out.shape()[2];
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        for (int f = 0; f < filters_; ++f) {
                            out.at({ b, h, w, f }) += bias_[f];
                        }
                    }
                }
            }
            return apply_activation(out, activation_);
        }

        [[nodiscard]] Shape output_shape() const override { return output_shape_; }

        [[nodiscard]] std::map<std::string, Tensor<float>> parameters() const override {
            return {
                {"weights", weights_}, {"bias", bias_},
                {name_ + "/weights", weights_}, {name_ + "/bias", bias_}
            };
        }

        [[nodiscard]] std::map<std::string, Tensor<float>> gradients() const override {
            return {
                {"weights", grad_weights_}, {"bias", grad_bias_},
                {name_ + "/weights", grad_weights_}, {name_ + "/bias", grad_bias_}
            };
        }

        [[nodiscard]] std::unique_ptr<Layer<float>> clone() const override {
            return std::make_unique<Conv2D>(*this);
        }

        Tensor<float> backward(const Tensor<float>& grad_output) override {
            if (cached_input_.rank() == 0) {
                throw std::logic_error("No se puede hacer backward sin haber hecho forward primero");
            }

            int B = cached_input_.shape()[0];
            int H_in = cached_input_.shape()[1];
            int W_in = cached_input_.shape()[2];
            int C = cached_input_.shape()[3];

            int kH = kernel_size_[0];
            int kW = kernel_size_[1];
            int F = filters_;

            int H_out = H_in - kH + 1;
            int W_out = W_in - kW + 1;

            Tensor<float> Z = ops::conv2d(cached_input_, weights_);
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        for (int f = 0; f < F; ++f) {
                            Z.at({ b, h, w, f }) += bias_[f];
                        }
                    }
                }
            }

            Tensor<float> dZ = Tensor<float>::zeros(Z.shape());
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        for (int f = 0; f < F; ++f) {
                            float grad_val = grad_output.at({ b, h, w, f });
                            if (activation_ == Activation::Relu) {
                                dZ.at({ b, h, w, f }) = (Z.at({ b, h, w, f }) > 0.0f) ? grad_val : 0.0f;
                            }
                            else {
                                dZ.at({ b, h, w, f }) = grad_val;
                            }
                        }
                    }
                }
            }

            grad_weights_ = Tensor<float>::zeros(weights_.shape());
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                for (int c = 0; c < C; ++c) {
                                    for (int f = 0; f < F; ++f) {
                                        grad_weights_.at({ kh, kw, c, f }) +=
                                            cached_input_.at({ b, h + kh, w + kw, c }) * dZ.at({ b, h, w, f });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            grad_bias_ = Tensor<float>::zeros(bias_.shape());
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        for (int f = 0; f < F; ++f) {
                            grad_bias_.at({ f }) += dZ.at({ b, h, w, f });
                        }
                    }
                }
            }

            Tensor<float> dX = Tensor<float>::zeros(cached_input_.shape());
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                for (int c = 0; c < C; ++c) {
                                    for (int f = 0; f < F; ++f) {
                                        dX.at({ b, h + kh, w + kw, c }) +=
                                            dZ.at({ b, h, w, f }) * weights_.at({ kh, kw, c, f });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return dX;
        }

        void set_weights(const Tensor<float>& weights) {
            weights_ = weights;
        }

        void set_bias(const Tensor<float>& bias) {
            bias_ = bias;
        }
    };
}
#endif