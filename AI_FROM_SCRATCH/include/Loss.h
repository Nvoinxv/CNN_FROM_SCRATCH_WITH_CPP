#ifndef LOSS_H
#define LOSS_H

#include "broadcasting.h"
#include <math.h>

// FUNGSI LOSS UNTUK MENGHITUNG KESALAHAN MODEL
// DI SINI MENGGUNAKAN CROSS ENTROPY

class CrossEntropyLoss {
public:

    static float forward(
        const BroadcastingTensor& predictions,
        const BroadcastingTensor& targets
    );

    static BroadcastingTensor backward(
        const BroadcastingTensor& predictions,
        const BroadcastingTensor& targets
    );
};

#endif