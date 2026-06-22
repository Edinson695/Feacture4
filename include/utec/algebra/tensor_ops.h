
#ifndef PROG3_PF_EPIC1_FEATURE4_V2026_1_UTEC_ALGEBRA_TENSOR_OPS_H
#define PROG3_PF_EPIC1_FEATURE4_V2026_1_UTEC_ALGEBRA_TENSOR_OPS_H

#include <stdexcept>
#include <iostream>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf::ops {
	
	template<typename T>
	bool allclose(const Tensor<T>& a, const Tensor<T> b, T epsilon = 1e-5) {

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


#endif 