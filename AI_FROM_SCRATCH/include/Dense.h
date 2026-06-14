#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"
#include <vector>

class Dense : public Layer {
private:
    int in_features_;
    int out_features_;

    BroadcastingTensor weights_;
    BroadcastingTensor biases_;

    // Gradients
    BroadcastingTensor grad_weights_;
    BroadcastingTensor grad_biases_;

    // Cached input
    BroadcastingTensor input_cache_;

    // Initialize weights using He/Kaiming normal initialization
    void initializeWeights();

public:
    Dense(int in_features, int out_features);

    BroadcastingTensor forward(const BroadcastingTensor& input) override;
    BroadcastingTensor backward(const BroadcastingTensor& grad_output) override;

    // Getters and Setters
    BroadcastingTensor& weights() { return weights_; }
    BroadcastingTensor& biases() { return biases_; }
    const BroadcastingTensor& grad_weights() const { return grad_weights_; }
    const BroadcastingTensor& grad_biases() const { return grad_biases_; }
};

#endif // DENSE_H
