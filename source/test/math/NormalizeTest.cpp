#include "math/Function.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

template<typename Scalar, size_t Dimension>
void Check() {
    Moer::Vector<Scalar, Dimension> input;
    for (size_t i = 0; i < Dimension; ++i)
        input[i] = Scalar(i + 1);
    const auto actual = Moer::Normalize(input);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(actual)>, Moer::Vectord<Dimension>>);
    double squared = 0;
    for (size_t i = 0; i < Dimension; ++i)
        squared += double(input[i]) * double(input[i]);
    for (size_t i = 0; i < Dimension; ++i) {
        const double expected = double(input[i]) / std::sqrt(squared);
        if (std::abs(actual[i] - expected) > 1e-12)
            throw std::runtime_error("Normalize changed direction or precision");
    }
    if (std::abs(Moer::Length(actual) - 1.0) > 1e-12)
        throw std::runtime_error("Normalize did not produce unit length");
}

int main() {
    try {
        Check<float, 2>();
        Check<float, 3>();
        Check<float, 4>();
        Check<double, 2>();
        Check<double, 3>();
        Check<double, 4>();
        Check<Moer::int32, 2>();
        Check<Moer::int32, 3>();
        Check<Moer::int32, 4>();
        const Moer::Vector<Moer::int32, 2> large(std::numeric_limits<Moer::int32>::max(), -std::numeric_limits<Moer::int32>::max());
        const auto                         normalized = Moer::Normalize(large);
        const double                       diagonal   = 1.0 / std::sqrt(2.0);
        if (std::abs(normalized.x - diagonal) > 1e-12 || std::abs(normalized.y + diagonal) > 1e-12)
            throw std::runtime_error("Normalize overflowed before conversion to double");
        std::cout << "Normalize float/double/integer vectors in 2/3/4 dimensions: success\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
