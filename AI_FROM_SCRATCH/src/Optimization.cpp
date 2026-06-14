#include "../include/Optimization.h"
#include <cmath>
#include <stdexcept>

AdamOptimizer::AdamOptimizer(
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon
) : learning_rate_(learning_rate),
    beta1_(beta1),
    beta2_(beta2),
    epsilon_(epsilon),
    t_(0)
{
    // DIHILANGKAN KARENA HANYA BAGIAN KONSTRUKTOR SAJA
}

void AdamOptimizer::step(
    std::vector<BroadcastingTensor>& parameters,
    const std::vector<BroadcastingTensor>& gradients
) {
    if (parameters.size() != gradients.size()) {
        throw std::invalid_argument(
            "Jumlah parameters dan gradients tidak sama"
        );
    }

    if (m_.empty()) {
        m_.reserve(parameters.size());
        v_.reserve(parameters.size());
        for (const auto& p : parameters) {
            std::vector<float> zeros(p.size(), 0.0f);
            m_.emplace_back(p.shape(), zeros);
            v_.emplace_back(p.shape(), zeros);
        }
    }

    ++t_;

    const float bias_correction1 = 1.0f - std::pow(beta1_, t_);
    const float bias_correction2 = 1.0f - std::pow(beta2_, t_);

    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& param_shape = parameters[i].shape();

        if (gradients[i].shape() != param_shape) {
            throw std::invalid_argument(
                "AdamOptimizer::step - shape gradient tidak sesuai dengan parameter"
            );
        }

        const std::vector<float>& g  = gradients[i].data();
        const std::vector<float>& p  = parameters[i].data();
        const std::vector<float>& m_old = m_[i].data();
        const std::vector<float>& v_old = v_[i].data();

        const size_t n = p.size();
        std::vector<float> m_new(n);
        std::vector<float> v_new(n);
        std::vector<float> p_new(n);

        for (size_t j = 0; j < n; ++j) {
            // MELAKUKAN UPDATE PADA BIAS PERTAMA
            m_new[j] = beta1_ * m_old[j] + (1.0f - beta1_) * g[j];

            // MELAKUKAN UPDATE PADA BIAS KEDUA
            v_new[j] = beta2_ * v_old[j] + (1.0f - beta2_) * g[j] * g[j];

            // MELAKUKAN KOREKSI SETIAP LANGKAH PADA PERHITUNGAN UPDATE BIAS TERSEBUT
            float m_hat = m_new[j] / bias_correction1;
            float v_hat = v_new[j] / bias_correction2;

            // UPDATE PARAMETER
            p_new[j] = p[j] - learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }

        m_[i] = BroadcastingTensor(param_shape, m_new);
        v_[i] = BroadcastingTensor(param_shape, v_new);
        parameters[i] = BroadcastingTensor(param_shape, p_new);
    }
}