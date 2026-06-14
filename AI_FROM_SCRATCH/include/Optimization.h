#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include "broadcasting.h"

// OPTIMISASI DENGAN ALGORTIMA ADAM 
class AdamOptimizer {
private:
    // MENGINISIALISASI SETIAP PARAMETER 
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;

    std::vector<BroadcastingTensor> m_;
    std::vector<BroadcastingTensor> v_;

    int t_;

public:
    // KONSTRUKTOR 
    AdamOptimizer(
        float learning_rate,
        float beta1,
        float beta2,
        float epsilon
    );
    
    // LANGKAH PADA MELAKUKAN OPTIMISASI
    void step(
        std::vector<BroadcastingTensor>& parameters,
        const std::vector<BroadcastingTensor>& gradients
    );
};

#endif