#ifndef LAYER_H
#define LAYER_H

#include "broadcasting.h"

class Layer {
public:
    virtual ~Layer() = default;

    /**
     * Perform the forward pass of this layer.
     * @param input Input tensor to the layer.
     * @return Output tensor resulting from the forward operation.
     */
    virtual BroadcastingTensor forward(const BroadcastingTensor& input) = 0;

    /**
     * Perform the backward pass of this layer.
     * Computes the gradient of the loss with respect to the input of this layer.
     * Also computes and stores gradients of parameters (weights/biases) if the layer has any.
     * @param grad_output Gradient of the loss with respect to the output of this layer.
     * @return Gradient of the loss with respect to the input of this layer.
     */
    virtual BroadcastingTensor backward(const BroadcastingTensor& grad_output) = 0;
};

#endif // LAYER_H
