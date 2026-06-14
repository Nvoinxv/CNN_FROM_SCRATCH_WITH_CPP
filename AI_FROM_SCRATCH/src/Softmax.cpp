#include "../include/Softmax.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

Softmax::Softmax() : output_cache_({0}, {}) {}

BroadcastingTensor Softmax::forward(const BroadcastingTensor& input) {
    if (input.size() == 0) {
        throw std::invalid_argument("Softmax input cannot be empty");
    }

    std::vector<float> input_data = input.data();
    
    // Find maximum value to prevent overflow
    float max_val = -std::numeric_limits<float>::infinity();
    for (float val : input_data) {
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute exponentials and their sum
    std::vector<float> exp_data(input_data.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < input_data.size(); ++i) {
        exp_data[i] = std::exp(input_data[i] - max_val);
        sum_exp += exp_data[i];
    }

    // Compute probabilities
    std::vector<float> output_data(input_data.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = exp_data[i] / sum_exp;
    }

    output_cache_ = BroadcastingTensor(input.shape(), output_data);
    return output_cache_;
}

BroadcastingTensor Softmax::backward(const BroadcastingTensor& grad_output) {
    if (grad_output.shape() != output_cache_.shape()) {
        throw std::invalid_argument("Softmax backward shape mismatch");
    }

    const std::vector<float>& s = output_cache_.data();
    const std::vector<float>& dy = grad_output.data();
    size_t n = s.size();

    // Standalone Softmax gradient calculation:
    // dx_i = s_i * (dy_i - sum_j (dy_j * s_j))
    float dot_product = 0.0f;
    for (size_t j = 0; j < n; ++j) {
        dot_product += dy[j] * s[j];
    }

    std::vector<float> dx(n);
    for (size_t i = 0; i < n; ++i) {
        dx[i] = s[i] * (dy[i] - dot_product);
    }

    return BroadcastingTensor(output_cache_.shape(), dx);
}
