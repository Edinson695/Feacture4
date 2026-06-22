#ifndef PROG3_PF_EPIC1_FEATURE4_V2026_01_TENSOR_BACKEND_H
#define PROG3_PF_EPIC1_FEATURE4_V2026_01_TENSOR_BACKEND_H

#include <Eigen/Dense> //Para gradescope
//#include "../../third_party/eigen/Eigen/Dense" //para mi
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>
#include "utec/algebra/shape.h"
#include <cmath>

namespace utec::tf {
    struct FitOptions {
        int epochs = 1;
        float learning_rate = 0.01f;
        int batch_size = 32;
    };

    template <typename T>
    class Tensor {
    public:
        Tensor() = default;
        explicit Tensor(const Shape& shape);
        Tensor(const Shape& shape, const std::vector<T>& values);
        static Tensor zeros(const Shape& shape);
        static Tensor ones(const Shape& shape);
        static Tensor from_data(const Shape& shape, const std::vector<T>& values);
        [[nodiscard]] const Shape& shape() const noexcept;
        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] std::size_t numel() const noexcept;
        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] Tensor reshaped(const Shape& new_shape) const;
        template <typename... Ix>
        T& operator()(Ix... indices);
        template <typename... Ix>
        const T& operator()(Ix... indices) const;
        void reshape(const Shape& new_shape);
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        T& at(std::initializer_list<int> indices);//a
        const T& at(std::initializer_list<int> indices) const;
        T& operator[](std::size_t index);
        const T& operator[](std::size_t index) const;
        Tensor(const Shape& shape, T constant_value);
    private:
        Shape shape_;
        Eigen::Array<T, Eigen::Dynamic, 1> data_;
        [[nodiscard]] std::size_t flat_index(std::span<const int> indices) const;
    };

    template<typename T>
    Tensor<T>::Tensor(const Shape& shape, T constant_value)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        data_.setConstant(constant_value);
    }
    template <typename T>
    Tensor<T>::Tensor(const Shape& shape)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        data_.setZero();
    }
    template<typename T>
    Tensor<T>::Tensor(const Shape& shape, const std::vector<T>& values)
        : shape_(shape),
        data_(static_cast<Eigen::Index>(shape.numel()))
    {
        if (values.size() != shape.numel())
            throw std::invalid_argument(
                "cantidad de datos incompatible con la forma");

        for (std::size_t i = 0; i < values.size(); ++i)
            data_(static_cast<Eigen::Index>(i)) = values[i];
    }
    template <typename T>
    std::size_t Tensor<T>::flat_index(std::span<const int> indices) const {
        if (indices.size() != shape_.rank()) {
            throw std::invalid_argument("numero de indices incompatible con el rank");
        }
        std::size_t flat = 0;
        std::size_t stride = 1;
        for (std::size_t i = shape_.rank(); i-- > 0;) {
            const int dim = shape_[i];
            const int idx = indices[i];
            if (idx < 0 || idx >= dim) {
                throw std::out_of_range("indice fuera de rango");
            }
            flat += static_cast<std::size_t>(idx) * stride;
            stride *= static_cast<std::size_t>(dim);
        }
        return flat;
    }
    template <typename T>
    Tensor<T> Tensor<T>::zeros(const Shape& shape) {
        Tensor<T> out(shape);
        out.data_.setZero();
        return out;
    }

    template<typename T>
    Tensor<T> Tensor<T>::ones(const Shape& shape) {
        Tensor<T> out(shape);
        out.data_.setOnes();
        return out;
    }

    template <typename T>
    template <typename... Ix>
    T& Tensor<T>::operator()(Ix... indices) {
        const int idx[] = { static_cast<int>(indices)... };
        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(idx, sizeof...(indices)))));
    }

    template <typename T>
    template <typename... Ix>
    const T& Tensor<T>::operator()(Ix... indices) const {
        const int idx[] = { static_cast<int>(indices)... };
        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(idx, sizeof...(indices)))));
    }


    template<typename T>
    const Shape& Tensor<T>::shape() const noexcept {
        return shape_;
    }

    template<typename T>
    std::size_t Tensor<T>::rank() const noexcept {
        return shape_.rank();
    }

    template<typename T>
    std::size_t Tensor<T>::numel() const noexcept {
        return shape_.numel();
    }


    template<typename T>
    Tensor<T> Tensor<T>::from_data(const Shape& shape, const std::vector<T>& values) {
        if (values.size() != shape.numel()) {
            throw std::invalid_argument("from_data incompatible");
        }
        return Tensor<T>(shape, values);
    }


    template<typename T>
    void Tensor<T>::reshape(const Shape& new_shape) {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("reshape incompatible: la cantidad de elementos no coincide");
        }
        shape_ = new_shape; // Solo actualizamos la metadata de la forma. Los datos quedan intactos.
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor& other) const {
        if (this->shape() != other.shape())
            throw std::invalid_argument("operator+ incompatible");
        Shape new_shape = this->shape();
        Tensor<T> out(new_shape);
        out.data_ = data_ + other.data_;
        return out;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor& other) const {
        if (!(shape_ == other.shape_)) {
            throw std::invalid_argument("las formas de los tensores no coinciden");
        }
        Tensor<T> newTensor(shape_);
        //-3 -1 1 3
        for (size_t i = 0; i < numel(); i++) {
            newTensor.data_[i] = data_[i] - other.data_[i];
        }
        return newTensor;
    }

    template <typename T>
    T& Tensor<T>::at(std::initializer_list<int> indices) {
        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(indices.begin(), indices.size()))));
    }

    template <typename T>
    const T& Tensor<T>::at(std::initializer_list<int> indices) const {
        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(indices.begin(), indices.size()))));
    }

    template <typename T>
    T& Tensor<T>::operator[](std::size_t index) {
        if (index >= numel()) {
            throw std::out_of_range("Indice plano fuera de rango");
        }
        return data_(static_cast<Eigen::Index>(index));
    }

    template <typename T>
    const T& Tensor<T>::operator[](std::size_t index) const {
        if (index >= numel()) {
            throw std::out_of_range("Indice plano fuera de rango");
        }
        return data_(static_cast<Eigen::Index>(index));
    }
    template<typename T>
    std::size_t Tensor<T>::size() const noexcept {
        return shape_.numel(); // O simplemente return numel();
    }

    template<typename T>
    Tensor<T> Tensor<T>::reshaped(const Shape& new_shape) const {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("reshaped: incomplatible: la cantidad de elementos no coincide");
        }

        Tensor<T> out = *this;

        out.shape_ = new_shape;
        return out;
    }

    template<typename T>
    bool allclose(const Tensor<T>& a, const Tensor<T>& b, T epsilon = 1e-5f) {
        if (!(a.shape() == b.shape())) {
            return false;
        }
        for (std::size_t i = 0; i < a.numel(); ++i) {
             if (std::abs(a[i] - b[i]) > epsilon) {
                return false;
            }
        }
        return true;
    }

}
using namespace utec::tf;

#endif 