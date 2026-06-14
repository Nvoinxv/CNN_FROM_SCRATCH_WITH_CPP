#ifndef ARITMATHIC_H
#define ARITMATHIC_H

#include "broadcasting.h"

class AritmathicTensor {
public:
    // LANGSUNG MAIN OPERASI TANPA KONSTRUKTOR
    // OPERASI PERTAMBAHAN
    static BroadcastingTensor add(
        const BroadcastingTensor& tensor1,
        const BroadcastingTensor& tensor2
    );
    
    // OPERASI PENGURANGAN
    static BroadcastingTensor subtract(
        const BroadcastingTensor& tensor1,
        const BroadcastingTensor& tensor2
    );
    
    // OPERASI PERKALIAN
    static BroadcastingTensor multiply(
        const BroadcastingTensor& tensor1,
        const BroadcastingTensor& tensor2
    );
    
    // OPERASI PEMBAGIAN
    static BroadcastingTensor divide(
        const BroadcastingTensor& tensor1,
        const BroadcastingTensor& tensor2
    );
};

#endif