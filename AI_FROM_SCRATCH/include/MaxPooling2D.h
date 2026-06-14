#ifndef MAXPOOLING2D_H
#define MAXPOOLING2D_H

#include "Layer.h"
#include <vector>

class MaxPooling2D : public Layer {
private:
    int pool_size_;
    int stride_;

    // Dimensions of cached input
    int in_channels_;
    int in_height_;
    int in_width_;

    // Cache the indices of the maximum values for backward pass
    std::vector<int> max_indices_;

public:
    MaxPooling2D(int pool_size = 2, int stride = 2);

    BroadcastingTensor forward(const BroadcastingTensor& input) override;
    BroadcastingTensor backward(const BroadcastingTensor& grad_output) override;
};

#endif // MAXPOOLING2D_H
