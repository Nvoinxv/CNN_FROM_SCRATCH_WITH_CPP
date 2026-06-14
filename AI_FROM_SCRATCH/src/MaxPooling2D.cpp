#include "../include/MaxPooling2D.h"
#include <stdexcept>
#include <algorithm>
#include <limits>

MaxPooling2D::MaxPooling2D(int pool_size, int stride)
    : pool_size_(pool_size), stride_(stride), in_channels_(0), in_height_(0), in_width_(0) {}

BroadcastingTensor MaxPooling2D::forward(const BroadcastingTensor& input) {
    if (input.shape().size() < 3) {
        throw std::invalid_argument("MaxPooling2D input tensor must have 3 dimensions [Channels, Height, Width]");
    }

    in_channels_ = input.shape()[0];
    in_height_ = input.shape()[1];
    in_width_ = input.shape()[2];

    int out_h = (in_height_ - pool_size_) / stride_ + 1;
    int out_w = (in_width_ - pool_size_) / stride_ + 1;

    std::vector<float> output_data(in_channels_ * out_h * out_w);
    max_indices_.resize(in_channels_ * out_h * out_w);

    for (int c = 0; c < in_channels_; ++c) {
        for (int y = 0; y < out_h; ++y) {
            int y_start = y * stride_;
            for (int x = 0; x < out_w; ++x) {
                int x_start = x * stride_;

                float max_val = -std::numeric_limits<float>::infinity();
                int max_idx = -1;

                for (int py = 0; py < pool_size_; ++py) {
                    for (int px = 0; px < pool_size_; ++px) {
                        int cur_y = y_start + py;
                        int cur_x = x_start + px;
                        if (cur_y < in_height_ && cur_x < in_width_) {
                            int input_idx = c * (in_height_ * in_width_) + cur_y * in_width_ + cur_x;
                            float val = input.data()[input_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = input_idx;
                            }
                        }
                    }
                }

                int out_idx = c * (out_h * out_w) + y * out_w + x;
                output_data[out_idx] = max_val;
                max_indices_[out_idx] = max_idx;
            }
        }
    }

    return BroadcastingTensor({in_channels_, out_h, out_w}, output_data);
}

BroadcastingTensor MaxPooling2D::backward(const BroadcastingTensor& grad_output) {
    int out_h = grad_output.shape()[1];
    int out_w = grad_output.shape()[2];

    std::vector<float> grad_input(in_channels_ * in_height_ * in_width_, 0.0f);

    for (int c = 0; c < in_channels_; ++c) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                int out_idx = c * (out_h * out_w) + y * out_w + x;
                int max_idx = max_indices_[out_idx];
                
                if (max_idx != -1) {
                    grad_input[max_idx] += grad_output.data()[out_idx];
                }
            }
        }
    }

    return BroadcastingTensor({in_channels_, in_height_, in_width_}, grad_input);
}
