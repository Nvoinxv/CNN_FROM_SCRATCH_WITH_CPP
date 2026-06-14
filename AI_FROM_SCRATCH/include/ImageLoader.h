#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "broadcasting.h"
#include <string>
#include <vector>

class ImageLoader {
public:
    /**
     * Loads an image from the specified file path and returns the pixel data normalized to [0.0, 1.0].
     * Supports various formats like PNG, JPG, BMP, etc.
     * 
     * @param filepath Path to the image file.
     * @param width Output parameter to store the width of the image.
     * @param height Output parameter to store the height of the image.
     * @param channels Output parameter to store the number of channels (e.g., 1 for grayscale, 3 for RGB, 4 for RGBA).
     * @param desired_channels Number of channels you want to force-load (0 to use the file's natural channels, or 1/3/4).
     * @param channels_first If true, formats the output array as CHW (Channels, Height, Width).
     *                       If false, formats the output array as HWC (Height, Width, Channels).
     * @return A vector of floats representing the normalized pixel values in the range [0.0f, 1.0f].
     */
    static std::vector<float> loadImage(
        const std::string& filepath,
        int& width,
        int& height,
        int& channels,
        int desired_channels = 0,
        bool channels_first = true
    );

    /**
     * Loads an image directly into a BroadcastingTensor.
     * 
     * @param filepath Path to the image file.
     * @param desired_channels Number of channels to force-load (0 for auto, 1 for grayscale, 3 for RGB).
     * @param channels_first If true, returns shape [Channels, Height, Width].
     *                       If false, returns shape [Height, Width, Channels].
     * @return A BroadcastingTensor object initialized with the image data.
     */
    static BroadcastingTensor loadImageAsTensor(
        const std::string& filepath,
        int desired_channels = 0,
        bool channels_first = true
    );
};

#endif // IMAGE_LOADER_H
