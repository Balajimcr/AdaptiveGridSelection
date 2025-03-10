#ifndef FISHEYE_EFFECT_H
#define FISHEYE_EFFECT_H
#include <iostream>
#include <opencv2/opencv.hpp>

class FisheyeEffect {
public:
    FisheyeEffect(cv::Size imageSize) : imageSize(imageSize), bUseGeneratedMaps(false) {}

    void generateDistortionMaps(double distStrength, cv::Mat& mapX, cv::Mat& mapY, cv::Mat& normalized_magnitude) {
        mapX.create(imageSize, CV_32FC1);
        mapY.create(imageSize, CV_32FC1);

        cv::Point2f center(imageSize.width / 2.0f, imageSize.height / 2.0f);

        for (int y = 0; y < imageSize.height; y++) {
            for (int x = 0; x < imageSize.width; x++) {
                float deltaX = (x - center.x) / center.x;
                float deltaY = (y - center.y) / center.y;
                float distance = (sqrt(deltaX * deltaX + deltaY * deltaY)) / 2;
                float distortion = 1.0f + distance * distStrength;
                float newX = center.x + (deltaX * distortion * center.x);
                float newY = center.y + (deltaY * distortion * center.y);
                mapX.at<float>(y, x) = newX;
                mapY.at<float>(y, x) = newY;
            }
        }

        // Compute distortion magnitude and normalized magnitude
        normalized_magnitude = computeDistortionMagnitude(mapX, mapY);

        // Store the generated maps (optional, if you need class member storage)
        this->mapX = mapX;
        this->mapY = mapY;
        bUseGeneratedMaps = true;
    }

private:
    cv::Size imageSize;
    cv::Mat mapX, mapY;
    bool bUseGeneratedMaps;

    cv::Mat computeDistortionMagnitude(const cv::Mat& grid_x, const cv::Mat& grid_y) {
        // Validate input matrices
        if (grid_x.type() != CV_32F || grid_y.type() != CV_32F) {
            std::cerr << "Both grid_x and grid_y must be of type CV_32F" << std::endl;
            return cv::Mat();
        }
        if (grid_x.size() != grid_y.size()) {
            std::cerr << "grid_x and grid_y must have the same size" << std::endl;
            return cv::Mat();
        }

        // Compute gradients for both channels (grids)
        cv::Mat grad_x_dx, grad_y_dx, grad_x_dy, grad_y_dy;
        cv::Sobel(grid_x, grad_x_dx, CV_32F, 1, 0, 3);
        cv::Sobel(grid_x, grad_y_dx, CV_32F, 0, 1, 3);
        cv::Sobel(grid_y, grad_x_dy, CV_32F, 1, 0, 3);
        cv::Sobel(grid_y, grad_y_dy, CV_32F, 0, 1, 3);

        // Compute the magnitude of gradients
        cv::Mat magnitude_dx, magnitude_dy;
        cv::magnitude(grad_x_dx, grad_y_dx, magnitude_dx);
        cv::magnitude(grad_x_dy, grad_y_dy, magnitude_dy);

        // Combine the magnitudes to get the total magnitude of distortion
        cv::Mat total_magnitude = magnitude_dx + magnitude_dy; // Simple way to combine

        // Optionally, normalize the total magnitude for visualization
        cv::Mat normalized_magnitude;
        cv::normalize(total_magnitude, normalized_magnitude, 0, 1, cv::NORM_MINMAX);

        return normalized_magnitude;
    }
};

#endif // FISHEYE_EFFECT_H