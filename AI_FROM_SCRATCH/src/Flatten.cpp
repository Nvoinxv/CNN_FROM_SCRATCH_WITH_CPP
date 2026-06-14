#include "../include/Flatten.h"
#include <numeric>

BroadcastingTensor Flatten::forward(const BroadcastingTensor& input) {
    input_shape_ = input.shape();
    
    // Output shape is simply a 1D shape with size = input.size()
    std::vector<int> out_shape = { input.size() };
    return BroadcastingTensor(out_shape, input.data());
}

BroadcastingTensor Flatten::backward(const BroadcastingTensor& grad_output) {
    // Reconstruct input shape
    return BroadcastingTensor(input_shape_, grad_output.data());
}
