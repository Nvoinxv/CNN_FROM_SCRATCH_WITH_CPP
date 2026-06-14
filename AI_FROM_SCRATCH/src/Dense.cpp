#include "../include/Dense.h"
#include <random>
#include <cmath>
#include <stdexcept>

Dense::Dense(int in_features, int out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_({out_features, in_features}, std::vector<float>(out_features * in_features, 0.0f)),
      biases_({out_features}, std::vector<float>(out_features, 0.0f)),
      grad_weights_({out_features, in_features}, std::vector<float>(out_features * in_features, 0.0f)),
      grad_biases_({out_features}, std::vector<float>(out_features, 0.0f)),
      input_cache_({0}, {}) 
{
    initializeWeights();
}

void Dense::initializeWeights() {
    // He/Kaiming Normal Initialization
    float stddev = std::sqrt(2.0f / static_cast<float>(in_features_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    std::vector<float> weight_data(weights_.size());
    for (int i = 0; i < weights_.size(); ++i) {
        weight_data[i] = dist(gen);
    }
    weights_ = BroadcastingTensor(weights_.shape(), weight_data);

    // Initialize biases to 0.0f
    std::vector<float> bias_data(biases_.size(), 0.0f);
    biases_ = BroadcastingTensor(biases_.shape(), bias_data);
}

BroadcastingTensor Dense::forward(const BroadcastingTensor& input) {
    input_cache_ = input; // Cache input

    if (input.size() != in_features_) {
        throw std::invalid_argument("Dense input size mismatch: expected " + 
                                 std::to_string(in_features_) + ", got " + std::to_string(input.size()));
    }

    std::vector<float> output_data(out_features_, 0.0f);
    
    // Y = W * X + B
    for (int j = 0; j < out_features_; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < in_features_; ++i) {
            sum += input.data()[i] * weights_.data()[j * in_features_ + i];
        }
        output_data[j] = sum + biases_.data()[j];
    }

    return BroadcastingTensor({out_features_}, output_data);
}

BroadcastingTensor Dense::backward(const BroadcastingTensor& grad_output) {
    if (grad_output.size() != out_features_) {
        throw std::invalid_argument("Dense backward gradient size mismatch");
    }

    std::vector<float> d_weights(weights_.size(), 0.0f);
    std::vector<float> d_biases(biases_.size(), 0.0f);
    std::vector<float> d_input(in_features_, 0.0f);

    // 1. Calculate Bias Gradients: dL/db = dL/dy
    for (int j = 0; j < out_features_; ++j) {
        d_biases[j] = grad_output.data()[j];
    }
    grad_biases_ = BroadcastingTensor({out_features_}, d_biases);

    // 2. Calculate Weight Gradients: dL/dW = (dL/dy) * X^T
    for (int j = 0; j < out_features_; ++j) {
        float dy = grad_output.data()[j];
        for (int i = 0; i < in_features_; ++i) {
            d_weights[j * in_features_ + i] = dy * input_cache_.data()[i];
        }
    }
    grad_weights_ = BroadcastingTensor(weights_.shape(), d_weights);

    // 3. Calculate Input Gradients: dL/dX = W^T * (dL/dy)
    for (int i = 0; i < in_features_; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < out_features_; ++j) {
            sum += grad_output.data()[j] * weights_.data()[j * in_features_ + i];
        }
        d_input[i] = sum;
    }

    return BroadcastingTensor({in_features_}, d_input);
}
