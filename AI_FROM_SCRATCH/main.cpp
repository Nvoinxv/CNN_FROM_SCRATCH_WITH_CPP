#include "include/broadcasting.h"
#include "include/aritmathic.h"
#include "include/ImageLoader.h"
#include "include/Layer.h"
#include "include/Conv2D.h"
#include "include/ReLu.h"
#include "include/MaxPooling2D.h"
#include "include/Flatten.h"
#include "include/Dense.h"
#include "include/Softmax.h"
#include "include/Loss.h"
#include "include/Optimization.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <random>
#include <numeric>

namespace fs = std::filesystem;

// Helper to load all images in a class folder
std::vector<BroadcastingTensor> loadImagesFromFolder(const fs::path& folder_path) {
    std::vector<BroadcastingTensor> tensors;
    if (!fs::exists(folder_path)) {
        std::cerr << "Directory not found: " << folder_path << std::endl;
        return tensors;
    }
    
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                try {
                    // Force load as grayscale (1 channel) for faster CPU training
                    BroadcastingTensor t = ImageLoader::loadImageAsTensor(entry.path().string(), 1, true);
                    tensors.push_back(t);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to load image " << entry.path() << " : " << e.what() << std::endl;
                }
            }
        }
    }
    return tensors;
}

void runMockTraining() {
    std::cout << "\n=== Running 2D CNN Mock Training from Scratch ===" << std::endl;
    std::cout << "Task: Classify 6x6 images into Vertical Lines (Class 0) or Horizontal Lines (Class 1)\n" << std::endl;

    // 1. Create Mock Dataset (6x6 pixels, 1 channel)
    std::vector<float> v1 = {
        0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0
    };
    std::vector<float> v2 = {
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0
    };
    std::vector<float> h1 = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    };
    std::vector<float> h2 = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    };

    std::vector<BroadcastingTensor> X = {
        BroadcastingTensor({1, 6, 6}, v1),
        BroadcastingTensor({1, 6, 6}, v2),
        BroadcastingTensor({1, 6, 6}, h1),
        BroadcastingTensor({1, 6, 6}, h2)
    };

    std::vector<BroadcastingTensor> y = {
        BroadcastingTensor({2}, {1.0f, 0.0f}),
        BroadcastingTensor({2}, {1.0f, 0.0f}),
        BroadcastingTensor({2}, {0.0f, 1.0f}),
        BroadcastingTensor({2}, {0.0f, 1.0f})
    };

    Conv2D conv(1, 2, 3, 1, 0); 
    MaxPooling2D pool(2, 2);
    Flatten flatten;
    Dense dense(8, 2);
    Softmax softmax;

    float learning_rate = 0.01f;
    AdamOptimizer optimizer(learning_rate, 0.9f, 0.999f, 1e-8f);

    const int epochs = 60;
    std::cout << "Starting Training for " << epochs << " epochs..." << std::endl;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f;
        for (size_t i = 0; i < X.size(); ++i) {
            BroadcastingTensor x_conv = conv.forward(X[i]);
            BroadcastingTensor x_relu = ReLU::forward(x_conv);
            BroadcastingTensor x_pool = pool.forward(x_relu);
            BroadcastingTensor x_flat = flatten.forward(x_pool);
            BroadcastingTensor logits = dense.forward(x_flat);
            BroadcastingTensor probs = softmax.forward(logits);

            float loss = CrossEntropyLoss::forward(probs, y[i]);
            epoch_loss += loss;

            BroadcastingTensor grad = CrossEntropyLoss::backward(probs, y[i]);
            grad = softmax.backward(grad);
            grad = dense.backward(grad);
            grad = flatten.backward(grad);
            grad = pool.backward(grad);
            grad = ReLU::backward(x_conv, grad);
            conv.backward(grad);

            std::vector<BroadcastingTensor> params = {
                conv.weights(), conv.biases(),
                dense.weights(), dense.biases()
            };
            std::vector<BroadcastingTensor> grads = {
                conv.grad_weights(), conv.grad_biases(),
                dense.grad_weights(), dense.grad_biases()
            };

            optimizer.step(params, grads);

            conv.weights() = params[0];
            conv.biases() = params[1];
            dense.weights() = params[2];
            dense.biases() = params[3];
        }
        if (epoch % 10 == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << " | Loss: " << (epoch_loss / X.size()) << std::endl;
        }
    }
}

void runRealTraining(const std::string& dataset_path) {
    std::cout << "\n=== Running 2D CNN Training with Real Dataset ===" << std::endl;
    std::cout << "Loading images from: " << dataset_path << std::endl;

    fs::path base_path(dataset_path);
    fs::path class0_dir = base_path / "class_0";
    fs::path class1_dir = base_path / "class_1";

    if (!fs::exists(class0_dir) || !fs::exists(class1_dir)) {
        std::cerr << "Error: Dataset folder must contain 'class_0' and 'class_1' subfolders." << std::endl;
        std::cerr << "Expected structure:\n"
                  << dataset_path << "/\n"
                  << "├── class_0/\n"
                  << "└── class_1/" << std::endl;
        return;
    }

    // Load Images
    std::vector<BroadcastingTensor> class0_imgs = loadImagesFromFolder(class0_dir);
    std::vector<BroadcastingTensor> class1_imgs = loadImagesFromFolder(class1_dir);

    std::cout << "Loaded Class 0 images: " << class0_imgs.size() << std::endl;
    std::cout << "Loaded Class 1 images: " << class1_imgs.size() << std::endl;

    size_t total_samples = class0_imgs.size() + class1_imgs.size();
    if (total_samples == 0) {
        std::cerr << "Error: No valid images found in the dataset directories." << std::endl;
        return;
    }

    // Combine into training vectors
    std::vector<BroadcastingTensor> X;
    std::vector<BroadcastingTensor> y;
    X.reserve(total_samples);
    y.reserve(total_samples);

    for (auto& img : class0_imgs) {
        X.push_back(img);
        y.push_back(BroadcastingTensor({2}, {1.0f, 0.0f})); // Class 0
    }
    for (auto& img : class1_imgs) {
        X.push_back(img);
        y.push_back(BroadcastingTensor({2}, {0.0f, 1.0f})); // Class 1
    }

    // Get input dimensions from the first image
    auto img_shape = X[0].shape();
    int C = img_shape[0];
    int H = img_shape[1];
    int W = img_shape[2];
    std::cout << "Detected Image Shape: [" << C << ", " << H << ", " << W << "]" << std::endl;

    // 2. Configure Dynamic CNN Model Layers
    // input: [C, H, W] -> conv -> [out_c, conv_h, conv_w]
    int out_channels = 4;
    int kernel_size = 3;
    int stride = 1;
    int padding = 0;
    
    int conv_h = (H - kernel_size + 2 * padding) / stride + 1;
    int conv_w = (W - kernel_size + 2 * padding) / stride + 1;
    
    // pooling -> [out_c, pool_h, pool_w]
    int pool_size = 2;
    int pool_stride = 2;
    int pool_h = (conv_h - pool_size) / pool_stride + 1;
    int pool_w = (conv_w - pool_size) / pool_stride + 1;

    int flat_features = out_channels * pool_h * pool_w;
    
    std::cout << "Configuring CNN Layers:" << std::endl;
    std::cout << "  Conv2D: [" << C << ", " << H << ", " << W << "] -> Output: [" 
              << out_channels << ", " << conv_h << ", " << conv_w << "]" << std::endl;
    std::cout << "  MaxPooling2D: pool=" << pool_size << " -> Output: [" 
              << out_channels << ", " << pool_h << ", " << pool_w << "]" << std::endl;
    std::cout << "  Flatten: Output: [" << flat_features << "]" << std::endl;
    std::cout << "  Dense: [" << flat_features << "] -> Output: [2]" << std::endl;

    if (conv_h <= 0 || pool_h <= 0) {
        std::cerr << "Error: Image resolution is too small for the CNN configuration." << std::endl;
        return;
    }

    Conv2D conv(C, out_channels, kernel_size, stride, padding); 
    MaxPooling2D pool(pool_size, pool_stride);
    Flatten flatten;
    Dense dense(flat_features, 2);
    Softmax softmax;

    // 3. Initialize Optimizer & Shuffler
    float learning_rate = 0.005f;
    AdamOptimizer optimizer(learning_rate, 0.9f, 0.999f, 1e-8f);

    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    // 4. Training Loop
    const int epochs = 25;
    std::cout << "\nStarting Training for " << epochs << " epochs..." << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    std::cout << std::left << std::setw(12) << "Epoch" << "Total Loss" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Shuffle dataset indices at the start of each epoch
        std::shuffle(indices.begin(), indices.end(), g);
        float epoch_loss = 0.0f;

        for (size_t idx : indices) {
            // Forward Pass
            BroadcastingTensor x_conv = conv.forward(X[idx]);
            BroadcastingTensor x_relu = ReLU::forward(x_conv);
            BroadcastingTensor x_pool = pool.forward(x_relu);
            BroadcastingTensor x_flat = flatten.forward(x_pool);
            BroadcastingTensor logits = dense.forward(x_flat);
            BroadcastingTensor probs = softmax.forward(logits);

            // Loss
            float loss = CrossEntropyLoss::forward(probs, y[idx]);
            epoch_loss += loss;

            // Backward Pass
            BroadcastingTensor grad = CrossEntropyLoss::backward(probs, y[idx]);
            grad = softmax.backward(grad);
            grad = dense.backward(grad);
            grad = flatten.backward(grad);
            grad = pool.backward(grad);
            grad = ReLU::backward(x_conv, grad);
            conv.backward(grad);

            // Update
            std::vector<BroadcastingTensor> params = {
                conv.weights(), conv.biases(),
                dense.weights(), dense.biases()
            };
            std::vector<BroadcastingTensor> grads = {
                conv.grad_weights(), conv.grad_biases(),
                dense.grad_weights(), dense.grad_biases()
            };

            optimizer.step(params, grads);

            conv.weights() = params[0];
            conv.biases() = params[1];
            dense.weights() = params[2];
            dense.biases() = params[3];
        }

        std::cout << std::left << std::setw(12) << epoch << (epoch_loss / total_samples) << std::endl;
    }
    std::cout << std::string(55, '-') << std::endl;
    std::cout << "Training complete." << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Broadcasting Tensor Test ===" << std::endl;
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

    if (argc >= 2) {
        // Run training on the specified folder path
        std::string dataset_path = argv[1];
        runRealTraining(dataset_path);
    } else {
        std::cout << "\nNote: To run on your real dataset, pass the path as an argument." << std::endl;
        std::cout << "Usage: " << argv[0] << " <path_to_dataset_folder>" << std::endl;
        std::cout << "Expected subfolders: '<dataset_folder>/class_0' and '<dataset_folder>/class_1'\n" << std::endl;
        
        runMockTraining();
    }

    return 0;
}