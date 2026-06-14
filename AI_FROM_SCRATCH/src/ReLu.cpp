#include "../include/ReLu.h"

BroadcastingTensor ReLU::forward(
    const BroadcastingTensor& input
) {
    std::vector<float> result_data(input.data().size());

    for (size_t i = 0; i < input.data().size(); ++i) {
        result_data[i] = std::max(0.0f, input.data()[i]);
    }

    return BroadcastingTensor(input.shape(), result_data);
}

BroadcastingTensor ReLU::backward(
    const BroadcastingTensor& input,
    const BroadcastingTensor& grad_output
) {
    std::vector<float> grad_input_data(input.data().size());

    for (size_t i = 0; i < input.data().size(); ++i) {
        grad_input_data[i] = (input.data()[i] > 0.0f) ? grad_output.data()[i] : 0.0f;
    }

    return BroadcastingTensor(input.shape(), grad_input_data);
}
