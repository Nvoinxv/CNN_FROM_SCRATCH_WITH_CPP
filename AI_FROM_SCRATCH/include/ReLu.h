#ifndef RELU_H
#define RELU_H

#include "broadcasting.h"

// FUNGSI AKTIVITAS AI
class ReLU {
public:
    // OPERASI UNTUK MENGHITUNG INPUT (FORWARD)
    static BroadcastingTensor forward(
        const BroadcastingTensor& input
    );
    
    // OPERASI MENGHITUNG OUTPUT (BACKWARD)
    static BroadcastingTensor backward (
        const BroadcastingTensor& input,
        const BroadcastingTensor& grad_output
    );
};

#endif