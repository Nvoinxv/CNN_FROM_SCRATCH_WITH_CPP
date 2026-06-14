#ifndef BROADCASTING_H
#define BROADCASTING_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <numeric>

class BroadcastingTensor {
private:
    std::vector<int> shape_;
    std::vector<float> data_;

public:
    // KONSTRUKTOR
    BroadcastingTensor(
        const std::vector<int>& shape,
        const std::vector<float>& data
    );

    // MENDAPATKAN INFORMASI
    const std::vector<int>& shape() const;
    const std::vector<float>& data() const;

    // UTILITAS
    int ndim() const;
    int size() const;

    // BROADCASTING
    static std::vector<int> broadcast_shape(
        const std::vector<int>& shape1,
        const std::vector<int>& shape2
    );

    // OPERASI ELEMENET-WISE
    BroadcastingTensor add(
        const BroadcastingTensor& other
    ) const;

    // OPERATOR OVERLOADING UNTUK ELEMENET-WISE MATH
    BroadcastingTensor operator+(const BroadcastingTensor& other) const;
    BroadcastingTensor operator-(const BroadcastingTensor& other) const;
    BroadcastingTensor operator*(const BroadcastingTensor& other) const;
    BroadcastingTensor operator/(const BroadcastingTensor& other) const;
};

#endif