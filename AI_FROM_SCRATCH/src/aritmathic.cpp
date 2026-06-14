#include "aritmathic.h"
#include <algorithm>
#include <stdexcept>

BroadcastingTensor AritmathicTensor::add(
    const BroadcastingTensor& tensor1,
    const BroadcastingTensor& tensor2
) {
    return tensor1 + tensor2;
}

BroadcastingTensor AritmathicTensor::subtract(
    const BroadcastingTensor& tensor1,
    const BroadcastingTensor& tensor2
) {
    return tensor1 - tensor2;
}

BroadcastingTensor AritmathicTensor::multiply(
    const BroadcastingTensor& tensor1,
    const BroadcastingTensor& tensor2
) {
    return tensor1 * tensor2;
}

BroadcastingTensor AritmathicTensor::divide(
    const BroadcastingTensor& tensor1,
    const BroadcastingTensor& tensor2
) {
    if (tensor2.data().empty() ||
        std::any_of(
            tensor2.data().begin(),
            tensor2.data().end(),
            [](float value) {
                return value == 0.0f;
            }
        ))
    {
        throw std::invalid_argument(
            "Pembagian dengan nol tidak diperbolehkan."
        );
    }

    return tensor1 / tensor2;
}