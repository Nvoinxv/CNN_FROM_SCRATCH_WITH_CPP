#include "../include/Conv2D.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

Conv2D::Conv2D(
    int in_channels, 
    int out_channels, 
    int kernel_size, 
    int stride, 
    int padding
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    weights_({out_channels, in_channels, kernel_size, kernel_size}, std::vector<float>(out_channels * in_channels * kernel_size * kernel_size, 0.0f)),
    biases_({out_channels}, std::vector<float>(out_channels, 0.0f)),
    grad_weights_({out_channels, in_channels, kernel_size, kernel_size}, std::vector<float>(out_channels * in_channels * kernel_size * kernel_size, 0.0f)),
    grad_biases_({out_channels}, std::vector<float>(out_channels, 0.0f)),
    input_cache_({0}, {}) 
{
    initializeWeights();
}

void Conv2D::initializeWeights() {
    // He/Kaiming Normal Initialization
    // stddev = sqrt(2.0 / fan_in)
    float fan_in = static_cast<float>(in_channels_ * kernel_size_ * kernel_size_);
    float stddev = std::sqrt(2.0f / fan_in);

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

BroadcastingTensor Conv2D::forward(const BroadcastingTensor& input) {
    input_cache_ = input; // Cache input for backpropagation

    if (input.shape().size() < 3) {
        throw std::invalid_argument("Conv2D input tensor must have at least 3 dimensions [Channels, Height, Width]");
    }

    int in_c = input.shape()[0];
    int in_h = input.shape()[1];
    int in_w = input.shape()[2];

    if (in_c != in_channels_) {
        throw std::invalid_argument("Conv2D input channel mismatch");
    }

    int out_h = (in_h - kernel_size_ + 2 * padding_) / stride_ + 1;
    int out_w = (in_w - kernel_size_ + 2 * padding_) / stride_ + 1;

    std::vector<float> output_data(out_channels_ * out_h * out_w, 0.0f);

    auto get_padded_val = [&](int c, int y, int x) -> float {
        int original_y = y - padding_;
        int original_x = x - padding_;
        if (original_y >= 0 && original_y < in_h && original_x >= 0 && original_x < in_w) {
            return input.data()[c * in_h * in_w + original_y * in_w + original_x];
        }
        return 0.0f;
    };

    // Forward pass convolution operation
    for (int co = 0; co < out_channels_; ++co) {
        float bias_val = biases_.data()[co];
        for (int y = 0; y < out_h; ++y) {
            int in_y_start = y * stride_;
            for (int x = 0; x < out_w; ++x) {
                int in_x_start = x * stride_;
                
                float sum = 0.0f;
                for (int ci = 0; ci < in_channels_; ++ci) {
                    for (int ky = 0; ky < kernel_size_; ++ky) {
                        for (int kx = 0; kx < kernel_size_; ++kx) {
                            int weight_idx = co * (in_channels_ * kernel_size_ * kernel_size_) + 
                                             ci * (kernel_size_ * kernel_size_) + 
                                             ky * kernel_size_ + kx;
                            float w = weights_.data()[weight_idx];
                            float val = get_padded_val(ci, in_y_start + ky, in_x_start + kx);
                            sum += w * val;
                        }
                    }
                }
                
                output_data[co * out_h * out_w + y * out_w + x] = sum + bias_val;
            }
        }
    }

    return BroadcastingTensor({out_channels_, out_h, out_w}, output_data);
}

BroadcastingTensor Conv2D::backward(const BroadcastingTensor& grad_output) {
    int in_h = input_cache_.shape()[1];
    int in_w = input_cache_.shape()[2];
    int out_h = grad_output.shape()[1];
    int out_w = grad_output.shape()[2];

    std::vector<float> d_weights(weights_.size(), 0.0f);
    std::vector<float> d_biases(biases_.size(), 0.0f);
    std::vector<float> d_input(input_cache_.size(), 0.0f);

    auto get_padded_val = [&](int c, int y, int x) -> float {
        int original_y = y - padding_;
        int original_x = x - padding_;
        if (original_y >= 0 && original_y < in_h && original_x >= 0 && original_x < in_w) {
            return input_cache_.data()[c * in_h * in_w + original_y * in_w + original_x];
        }
        return 0.0f;
    };

    // 1. Calculate Bias Gradients
    for (int co = 0; co < out_channels_; ++co) {
        float sum_grad = 0.0f;
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                sum_grad += grad_output.data()[co * out_h * out_w + y * out_w + x];
            }
        }
        d_biases[co] = sum_grad;
    }
    grad_biases_ = BroadcastingTensor({out_channels_}, d_biases);

    // 2. Calculate Weight Gradients & Input Gradients
    for (int co = 0; co < out_channels_; ++co) {
        for (int y = 0; y < out_h; ++y) {
            int in_y_start = y * stride_;
            for (int x = 0; x < out_w; ++x) {
                int in_x_start = x * stride_;
                float grad_out_val = grad_output.data()[co * out_h * out_w + y * out_w + x];

                for (int ci = 0; ci < in_channels_; ++ci) {
                    for (int ky = 0; ky < kernel_size_; ++ky) {
                        for (int kx = 0; kx < kernel_size_; ++kx) {
                            int weight_idx = co * (in_channels_ * kernel_size_ * kernel_size_) + 
                                             ci * (kernel_size_ * kernel_size_) + 
                                             ky * kernel_size_ + kx;

                            // Update weight gradients
                            float input_val = get_padded_val(ci, in_y_start + ky, in_x_start + kx);
                            d_weights[weight_idx] += grad_out_val * input_val;

                            // Update input gradients (only for valid unpadded positions)
                            int original_y = in_y_start + ky - padding_;
                            int original_x = in_x_start + kx - padding_;
                            if (original_y >= 0 && original_y < in_h && original_x >= 0 && original_x < in_w) {
                                int input_idx = ci * in_h * in_w + original_y * in_w + original_x;
                                float w = weights_.data()[weight_idx];
                                d_input[input_idx] += grad_out_val * w;
                            }
                        }
                    }
                }
            }
        }
    }

    grad_weights_ = BroadcastingTensor(weights_.shape(), d_weights);
    return BroadcastingTensor(input_cache_.shape(), d_input);
}
