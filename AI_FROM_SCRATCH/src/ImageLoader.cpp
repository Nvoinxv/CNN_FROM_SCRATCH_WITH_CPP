#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/ImageLoader.h"
#include <stdexcept>
#include <iostream>

std::vector<float> ImageLoader::loadImage(
    const std::string& filepath,
    int& width,
    int& height,
    int& channels,
    int desired_channels,
    bool channels_first
) {
    // Load the image file using stb_image
    // stbi_load automatically detects and decodes PNG, JPG, BMP, etc.
    unsigned char* img_data = stbi_load(
        filepath.c_str(), 
        &width, 
        &height, 
        &channels, 
        desired_channels
    );

    if (img_data == nullptr) {
        throw std::runtime_error("ImageLoader Error: Failed to load image at path: " + filepath + 
                                 "\nReason: " + stbi_failure_reason());
    }

    // If desired_channels is specified, stb_image returns that number of channels.
    int num_channels = (desired_channels > 0) ? desired_channels : channels;
    channels = num_channels; // Update output reference

    int num_pixels = width * height;
    std::vector<float> normalized_data(num_pixels * num_channels);

    if (channels_first) {
        // CHW Layout: [Channels, Height, Width]
        // Group all red, all green, then all blue pixels together
        for (int c = 0; c < num_channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int src_idx = (y * width + x) * num_channels + c;
                    int dest_idx = c * num_pixels + (y * width + x);
                    normalized_data[dest_idx] = static_cast<float>(img_data[src_idx]) / 255.0f;
                }
            }
        }
    } else {
        // HWC Layout: [Height, Width, Channels]
        // Interleaved layout (R, G, B, R, G, B...)
        for (int i = 0; i < num_pixels * num_channels; ++i) {
            normalized_data[i] = static_cast<float>(img_data[i]) / 255.0f;
        }
    }

    // Free the memory allocated by stb_image
    stbi_image_free(img_data);

    return normalized_data;
}

BroadcastingTensor ImageLoader::loadImageAsTensor(
    const std::string& filepath,
    int desired_channels,
    bool channels_first
) {
    int width = 0;
    int height = 0;
    int channels = 0;

    std::vector<float> data = loadImage(
        filepath, 
        width, 
        height, 
        channels, 
        desired_channels, 
        channels_first
    );

    std::vector<int> shape;
    if (channels_first) {
        shape = {channels, height, width};
    } else {
        shape = {height, width, channels};
    }

    return BroadcastingTensor(shape, data);
}
