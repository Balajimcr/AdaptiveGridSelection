#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

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
    
    // Simulate camera motion with rolling shutter effect
    void generateDistortionMapsCAMStab(cv::Mat& mapX, cv::Mat& mapY, cv::Mat& normalized_magnitude,
        float motionStrength = 1.0f,
        int smoothingWindowSize = 15) {

        mapX.create(imageSize, CV_32FC1);
        mapY.create(imageSize, CV_32FC1);
        // Generate base motion
        CameraMotion baseMotion = generateRandomMotion();

        // Generate per-row motions with smooth transitions to simulate rolling shutter
        std::vector<CameraMotion> rowMotions(imageSize.height);

        // Random motion components for interpolation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, motionStrength);

        std::vector<float> noiseX(imageSize.height), noiseY(imageSize.height);
        std::vector<float> noiseRotation(imageSize.height), noiseScale(imageSize.height);

        // Generate raw noise
        for (int y = 0; y < imageSize.height; y++) {
            noiseX[y] = noise(gen);
            noiseY[y] = noise(gen);
            noiseRotation[y] = noise(gen) * 0.1f; // Reduce rotation noise
            noiseScale[y] = noise(gen) * 0.01f;   // Reduce scale noise
        }

        // Apply smoothing to the noise
        smoothVector(noiseX, smoothingWindowSize);
        smoothVector(noiseY, smoothingWindowSize);
        smoothVector(noiseRotation, smoothingWindowSize);
        smoothVector(noiseScale, smoothingWindowSize);

        // Apply smoothed noise to base motion for each row
        for (int y = 0; y < imageSize.height; y++) {
            rowMotions[y].translationX = baseMotion.translationX + noiseX[y];
            rowMotions[y].translationY = baseMotion.translationY + noiseY[y];
            rowMotions[y].rotation = baseMotion.rotation + noiseRotation[y];
            rowMotions[y].scale = baseMotion.scale + noiseScale[y];
        }

        // Apply rolling shutter effect based on the row motions
        applyRollingShutterEffect(mapX, mapY, rowMotions);

        // Compute distortion magnitude and normalized magnitude
        normalized_magnitude = computeDistortionMagnitude(mapX, mapY);
    }
      

private:
    cv::Size imageSize;
    cv::Mat mapX, mapY;
    bool bUseGeneratedMaps;

    // New struct to hold camera motion parameters
    struct CameraMotion {
        float translationX, translationY;  // Translation in pixels
        float rotation;                    // Rotation in degrees
        float scale;                       // Scaling factor
    };

    // Generate random camera motion for simulation
    CameraMotion generateRandomMotion(float maxTranslation = 15.0f,
        float maxRotation = 2.0f,
        float maxScale = 0.05f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> transDist(-maxTranslation, maxTranslation);
        std::uniform_real_distribution<float> rotDist(-maxRotation, maxRotation);
        std::uniform_real_distribution<float> scaleDist(1.0f - maxScale, 1.0f + maxScale);

        CameraMotion motion;
        motion.translationX = transDist(gen);
        motion.translationY = transDist(gen);
        motion.rotation = rotDist(gen);
        motion.scale = scaleDist(gen);

        return motion;
    }

    // Apply rolling shutter effect to the distortion maps based on camera motion
    void applyRollingShutterEffect(cv::Mat& mapX, cv::Mat& mapY,
        const std::vector<CameraMotion>& rowMotions) {
        cv::Point2f center(imageSize.width / 2.0f, imageSize.height / 2.0f);

        for (int y = 0; y < imageSize.height; y++) {
            // Get the motion parameters for this row
            const CameraMotion& motion = rowMotions[y];

            // Create transformation matrix for this row
            cv::Mat M = cv::getRotationMatrix2D(center, motion.rotation, motion.scale);
            M.at<double>(0, 2) += motion.translationX;
            M.at<double>(1, 2) += motion.translationY;

            for (int x = 0; x < imageSize.width; x++) {
                // Get the current remapping coordinates
                cv::Point2f pt(mapX.at<float>(y, x), mapY.at<float>(y, x));

                // Apply the transformation to the point
                cv::Point2f transformedPt;
                transformedPt.x = M.at<double>(0, 0) * pt.x + M.at<double>(0, 1) * pt.y + M.at<double>(0, 2);
                transformedPt.y = M.at<double>(1, 0) * pt.x + M.at<double>(1, 1) * pt.y + M.at<double>(1, 2);

                // Update the map
                mapX.at<float>(y, x) = transformedPt.x;
                mapY.at<float>(y, x) = transformedPt.y;
            }
        }
    }

    // Helper method to smooth a vector using moving average
    void smoothVector(std::vector<float>& vec, int windowSize) {
        std::vector<float> smoothed = vec;
        for (int i = 0; i < vec.size(); i++) {
            float sum = 0.0f;
            int count = 0;
            for (int j = -windowSize / 2; j <= windowSize / 2; j++) {
                int idx = i + j;
                if (idx >= 0 && idx < vec.size()) {
                    sum += vec[idx];
                    count++;
                }
            }
            smoothed[i] = sum / count;
        }
        vec = smoothed;
    }

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