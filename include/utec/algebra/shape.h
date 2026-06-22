#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <ostream>

namespace utec::tf {
    class Shape {
    public:
        Shape() = default;
        Shape(std::initializer_list<int> dims);
        Shape(std::initializer_list<std::size_t> dims);
        explicit Shape(std::vector<int> dims);
        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] std::size_t total_size() const noexcept;
        [[nodiscard]] std::size_t numel() const;
        [[nodiscard]] const std::vector<int>& dims() const noexcept;
        [[nodiscard]] int operator[](std::size_t i) const;
        friend bool operator==(const Shape&, const Shape&) = default;
        [[nodiscard]] std::size_t size() const noexcept;

    private:
        std::vector<int> dims_;
        void validate() const;
    };

    void Shape::validate() const {
        for (int dim : dims_) {
            if (dim <= 0) {
                throw std::invalid_argument("toda dimension debe ser positiva");
            }
        }
    }

    inline Shape::Shape(std::initializer_list<std::size_t> dims) {
        dims_.reserve(dims.size());
        for (std::size_t dim : dims) {
            dims_.push_back(static_cast<int>(dim));
        }
        validate();
    }
    Shape::Shape(std::initializer_list<int> dims) : dims_(dims) {
        validate();
    }

    Shape::Shape(std::vector<int> dims) : dims_(dims) {
        validate();
    }

    size_t Shape::rank() const noexcept {
        return dims_.size();
    }

    size_t Shape::numel() const {
        std::size_t total = 1;
        for (const int& d : dims_) {
            total *= d;
        }
        return total;
    }

    const std::vector<int>& Shape::dims() const noexcept {
        return dims_;
    }

    int Shape::operator[](std::size_t i) const {
        return dims_[i];
    }

    size_t Shape::total_size()const noexcept {
        size_t total = 1;
        for (const auto& v : dims_) {
            total *= v;
        }
        return total;
    }
    inline size_t Shape::size() const noexcept {
        return dims_.size();
    }

    inline std::ostream& operator<<(std::ostream& os, const Shape& s) {
        os << "Shape(";
        for (std::size_t i = 0; i < s.rank(); ++i) {
            os << s[i];
            if (i < s.rank() - 1) os << ", ";
        }
        os << ")";
        return os;
    }
}


#endif 