#ifndef GENERATE_GRID_H
#define GENERATE_GRID_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void Generate_FixedGrid(const cv::Mat& distortionMagnitude, std::vector<cv::Point>& fixedGridPoints, int gridX, int gridY) {
#ifdef DEBUG_GRID
    cv::Mat debugImage;
    debugImage = distortionMagnitude.clone();
    if (distortionMagnitude.type() == CV_32FC1) {
        debugImage.convertTo(debugImage, CV_8U, 255);
    }
    cv::cvtColor(debugImage, debugImage, cv::COLOR_GRAY2BGR);
#endif

    // Calculate cell dimensions once
    const float cellWidth = static_cast<float>(distortionMagnitude.cols) / (gridX - 1);
    const float cellHeight = static_cast<float>(distortionMagnitude.rows) / (gridY - 1);
    const int maxX = distortionMagnitude.cols - 1;
    const int maxY = distortionMagnitude.rows - 1;

    // Pre-allocate memory for efficiency
    fixedGridPoints.clear();
    fixedGridPoints.reserve(gridX * gridY);

    // Generate grid points (outer loop for j gives better cache locality)
    for (int j = 0; j < gridY; ++j) {
        const int y = std::min(static_cast<int>(j * cellHeight), maxY);

        for (int i = 0; i < gridX; ++i) {
            const int x = std::min(static_cast<int>(i * cellWidth), maxX);

            // Add point directly without additional clamping
            fixedGridPoints.emplace_back(x, y);

#ifdef DEBUG_GRID
            cv::circle(debugImage, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), 2);
#endif
        }
    }

#ifdef DEBUG_GRID
    cv::imshow("Fixed Grid Points", debugImage);
#endif
}

// Adaptive grid generation 
void GenerateAdaptiveGrid_HAG(const cv::Mat& magnitudeOfDistortion,
    std::vector<cv::Point>& adaptiveGridPoints,
    const int gridX, const int gridY,
    const float lowThreshold) {
    const int imageWidth = magnitudeOfDistortion.cols;
    const int imageHeight = magnitudeOfDistortion.rows;
    const int maxX = imageWidth - 1;
    const int maxY = imageHeight - 1;

    // Calculate cell dimensions 
    const float cellWidth = static_cast<float>(imageWidth) / (gridX - 1);
    const float cellHeight = static_cast<float>(imageHeight) / (gridY - 1);

#ifdef DEBUG_DRAW
    cv::Mat normalized;
    cvtColor(magnitudeOfDistortion, normalized, cv::COLOR_GRAY2BGR);
    const cv::Scalar blue(255, 0, 0), yellow(0, 255, 255), green(0, 255, 0), red(0, 0, 255);
#endif

    // Pre-allocate with estimated capacity (fixed grid + potential adaptive points)
    adaptiveGridPoints.clear();
    adaptiveGridPoints.reserve(gridX * gridY * 2);

    // First, generate the fixed grid points
    for (int j = 0; j < gridY; ++j) {
        const int y = std::min(static_cast<int>(j * cellHeight), maxY);

        for (int i = 0; i < gridX; ++i) {
            const int x = std::min(static_cast<int>(i * cellWidth), maxX);
            adaptiveGridPoints.emplace_back(x, y);

#ifdef DEBUG_DRAW
            cv::circle(normalized, cv::Point(x, y), 1, blue, 2);
#endif
        }
    }

    // Add adaptive points based on distortion
    for (int i = 0; i < gridX; ++i) {
        const int x = std::min(static_cast<int>(i * cellWidth), maxX);
        const float effectiveCellWidth = std::min(cellWidth, static_cast<float>(imageWidth - x));
        const int midX = std::min(static_cast<int>(x + (effectiveCellWidth / 2.0)), maxX);

        for (int j = 0; j < gridY; ++j) {
            const int y = std::min(static_cast<int>(j * cellHeight), maxY);

            // Calculate center point of the cell for distortion check
            const float effectiveCellHeight = std::min(cellHeight, static_cast<float>(imageHeight - y));
            const int midY = std::min(static_cast<int>(y + (effectiveCellHeight / 2.0)), maxY);

            // Get distortion value at center point
            const float distortionValue = magnitudeOfDistortion.at<float>(midY, midX);

            if (distortionValue >= lowThreshold) {
                // Add mid-point in the cell horizontally
                const cv::Point newPoint(midX, y);
                if (std::find(adaptiveGridPoints.begin(), adaptiveGridPoints.end(), newPoint) == adaptiveGridPoints.end()) {
                    adaptiveGridPoints.push_back(newPoint);

#ifdef DEBUG_DRAW
                    cv::circle(normalized, newPoint, 1,
                        (distortionValue > 0.9f) ? green : yellow, 2);
#endif
                }

                // For the last row, add additional point
                if (j == gridY - 2) {
                    const int lastRowY = std::min(static_cast<int>(y + cellHeight), maxY);
                    const cv::Point lastRowPoint(midX, lastRowY);

                    if (std::find(adaptiveGridPoints.begin(), adaptiveGridPoints.end(), lastRowPoint) == adaptiveGridPoints.end()) {
                        adaptiveGridPoints.push_back(lastRowPoint);

#ifdef DEBUG_DRAW
                        cv::circle(normalized, lastRowPoint, 2, red, 2);
#endif
                    }
                }
            }
        }
    }

#ifdef DEBUG_DRAW
    cv::imshow("Adaptive Grid Points", normalized);
    cv::waitKey(1);
#endif
}

#endif // FISHEYE_EFFECT_H