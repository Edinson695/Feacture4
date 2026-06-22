#ifndef PROG3_PF_EPIC1_FEATURE4_V2026_01_NEURAL_NETWORK_H
#define PROG3_PF_EPIC1_FEATURE4_V2026_01_NEURAL_NETWORK_H

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_convolution.h"
#include "utec/nn/nn_pooling.h"
#include "utec/nn/nn_flatten.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <map>

namespace utec::tf {

    class Sequential {
    private:
        std::vector<std::unique_ptr<Layer<float>>> layers_;
        bool is_compiled_ = false;

    public:
        Sequential() = default;

        template <typename LayerT>
        void add(const LayerT& layer) {
            auto cloned = layer.clone();
            if (!layers_.empty()) {
                cloned->build(layers_.back()->output_shape());
            }
            layers_.push_back(std::move(cloned));
        }

        void compile(const optimizers::SGD<float>& optimizer, const losses::CategoricalCrossentropy<float>& loss) {
            is_compiled_ = true;
        }

        [[nodiscard]] bool compiled() const { return is_compiled_; }

        Tensor<float> predict(const Tensor<float>& batch) {
            if (layers_.empty()) throw std::invalid_argument("Modelo vacio");

            Shape expected_shape = layers_[0]->output_shape();
            if (batch.rank() != expected_shape.rank() + 1) {
                throw std::invalid_argument("Rank de batch invalido");
            }
            for (std::size_t i = 0; i < expected_shape.rank(); ++i) {
                if (batch.shape()[i + 1] != expected_shape[i]) {
                    throw std::invalid_argument("Dimensiones de batch incompatibles");
                }
            }

            Tensor<float> current = batch;
            for (const auto& layer : layers_) {
                current = layer->forward(current);
            }
            return current;
        }

        [[nodiscard]] std::map<std::string, Tensor<float>> parameters() const {
            std::map<std::string, Tensor<float>> all_params;
            for (const auto& layer : layers_) {
                auto p = layer->parameters();
                for (const auto& [key, tensor] : p) {
                    all_params[key] = tensor;
                }
            }
            return all_params;
        }
    };

}
#endif