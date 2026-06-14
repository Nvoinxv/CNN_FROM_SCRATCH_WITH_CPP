#include "include/broadcasting.h"
#include "include/aritmathic.h"
#include <iostream>

int main() {
    BroadcastingTensor tensor1({2, 3}, {1, 2, 3, 4, 5, 6});
    BroadcastingTensor tensor2({3}, {10, 20, 30});

    try {
        BroadcastingTensor result = tensor1.add(tensor2);
        for (float value : result.data()) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}