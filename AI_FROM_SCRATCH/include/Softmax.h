#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Layer.h"
#include <vector>

class Softmax : public Layer {
private:
    BroadcastingTensor output_cache_;

public:
    Softmax();

    BroadcastingTensor forward(const BroadcastingTensor& input) override;
    BroadcastingTensor backward(const BroadcastingTensor& grad_output) override;
};

#endif // SOFTMAX_H
