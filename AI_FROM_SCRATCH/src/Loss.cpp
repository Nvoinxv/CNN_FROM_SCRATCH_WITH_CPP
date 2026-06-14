#include "../include/Loss.h"

float CrossEntropyLoss::forward(
    const BroadcastingTensor& predictions,
    const BroadcastingTesnor& targets
) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument(
            "Bentuk prediksi dan target harus sama."
        );
    }

    float loss = 0.0f;
    for (size_t i = 0; i < predictions.data().size(); ++i) {
        loss -= targets.data()[i] * std::log(predictions.data()[i] + 1e-9f);
    }

    return loss / predictions.data().size();

}

float CrossEntropyLoss::backward(
    const BroadcastingTensor& predictions,
    const BroadcastingTensor& targets
) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument(
            "Bentuk prediksi dan target harus sama."
        );
    }

    std::vector<float> grad_data(predictions.data().size());
    for (size_t i = 0; i < predictions.data().size(); ++i) {
        grad_data[i] = -targets.data()[i] / (predictions.data()[i] + 1e-9f);
    }

    return BroadcastingTensor(predictions.shape(), grad_data);
}