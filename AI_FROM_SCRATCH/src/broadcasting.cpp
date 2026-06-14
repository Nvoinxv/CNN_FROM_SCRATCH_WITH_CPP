#include "../include/broadcasting.h"

BroadcastingTensor::BroadcastingTensor(
    const std::vector<int>& shape,
    const std::vector<float>& data
) : shape_(shape), data_(data) {
    if (size() != data.size()) {
        throw std::invalid_argument("Ukuran data tidak sesuai dengan bentuk.");
    }
}

const std::vector<int>& BroadcastingTensor::shape() const {
    return shape_;
}

const std::vector<float>& BroadcastingTensor::data() const {
    return data_;
}

int BroadcastingTensor::ndim() const {
    return shape_.size();
}

int BroadcastingTensor::size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
        std::multiplies<int>());
}

std::vector<int> BroadcastingTensor::broadcast_shape(
    const std::vector<int>& shape1,
    const std::vector<int>& shape2
) {
    std::vector<int> result_shape;

    int ndim1 = shape1.size();
    int ndim2 = shape2.size();
    int max_ndim = std::max(ndim1, ndim2);

    for (int i = 0; i < max_ndim; ++i) {
        int dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
        int dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::invalid_argument("Bentuk tidak dapat dibroadcast.");
        }

        result_shape.push_back(std::max(dim1, dim2));
    }

    std::reverse(result_shape.begin(), result_shape.end());

    return result_shape;
}

BroadcastingTensor BroadcastingTensor::add(
    const BroadcastingTensor& other
) const {
    std::vector<int> result_shape = broadcast_shape(shape_, other.shape());
    std::vector<float> result_data(size(), 0.0f);

    for (int i = 0; i < size(); ++i) {
        int idx1 = (i % shape_[0]) * (shape_[0] / result_shape[0]);
        int idx2 = (i % other.shape()[0]) * (other.shape()[0] / result_shape[0]);
        result_data[i] = data_[idx1] + other.data()[idx2];
    }
    return BroadcastingTensor(result_shape, result_data);
}