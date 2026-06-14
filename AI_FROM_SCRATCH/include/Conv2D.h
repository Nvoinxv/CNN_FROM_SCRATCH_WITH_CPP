#ifndef CONV2D_H
#define CONV2D_H

#include "Layer.h"
#include <vector>

class Conv2D : public Layer {
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    BroadcastingTensor weights_;
    BroadcastingTensor biases_;

    // Gradients
    BroadcastingTensor grad_weights_;
    BroadcastingTensor grad_biases_;

    // Cached input for backward pass
    BroadcastingTensor input_cache_;

    // Initialize weights using He/Kaiming normal initialization
    void initializeWeights();

public:
    Conv2D(
        int in_channels, 
        int out_channels, 
        int kernel_size, 
        int stride = 1, 
        int padding = 0
    );

    BroadcastingTensor forward(const BroadcastingTensor& input) override;
    BroadcastingTensor backward(const BroadcastingTensor& grad_output) override;

    // Getters and Setters for Weights/Biases and their Gradients
    BroadcastingTensor& weights() { return weights_; }
    BroadcastingTensor& biases() { return biases_; }
    const BroadcastingTensor& grad_weights() const { return grad_weights_; }
    const BroadcastingTensor& grad_biases() const { return grad_biases_; }
};

#endif // CONV2D_H
