#ifndef FLATTEN_H
#define FLATTEN_H

#include "Layer.h"
#include <vector>

class Flatten : public Layer {
private:
    std::vector<int> input_shape_;

public:
    Flatten() = default;

    BroadcastingTensor forward(const BroadcastingTensor& input) override;
    BroadcastingTensor backward(const BroadcastingTensor& grad_output) override;
};

#endif // FLATTEN_H
