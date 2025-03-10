#ifndef GENERATE_GRIDV2_H
#define GENERATE_GRIDV2_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <random>
#include "GenerateGridv1.h"

using namespace cv;
using namespace std;


// Helper function for recursive grid generation with adaptive refinement
void recursivelyGenerateEnhancedGrid(
    const cv::Mat& distortionMagnitude,
    std::vector<cv::Point>& gridPoints,
    std::unordered_set<cv::Point, PointHash, PointEqual>& uniquePoints,
    int startI, int endI, int startJ, int endJ,
    float cellWidth, float cellHeight,
    int maxX, int maxY,
    float threshold
) {
    // Calculate the actual pixel coordinates for the corners
    const int topLeftX = std::min(static_cast<int>(startI * cellWidth), maxX);
    const int topLeftY = std::min(static_cast<int>(startJ * cellHeight), maxY);
    const int bottomRightX = std::min(static_cast<int>(endI * cellWidth), maxX);
    const int bottomRightY = std::min(static_cast<int>(endJ * cellHeight), maxY);

    // Add corner points if this is a leaf cell (single grid cell)
    if (startI == endI && startJ == endJ) {
        // Add the single corner point
        cv::Point corner(topLeftX, topLeftY);
        if (uniquePoints.insert(corner).second) {
            gridPoints.push_back(corner);
        }
        return;
    }

    // For larger sections, recursively divide
    if (startI < endI) {
        int midI = startI + (endI - startI) / 2;
        // Process left half
        recursivelyGenerateEnhancedGrid(distortionMagnitude, gridPoints, uniquePoints,
            startI, midI, startJ, endJ,
            cellWidth, cellHeight, maxX, maxY, threshold);
        // Process right half
        recursivelyGenerateEnhancedGrid(distortionMagnitude, gridPoints, uniquePoints,
            midI + 1, endI, startJ, endJ,
            cellWidth, cellHeight, maxX, maxY, threshold);
    }
    else if (startJ < endJ) {
        int midJ = startJ + (endJ - startJ) / 2;
        // Process top half
        recursivelyGenerateEnhancedGrid(distortionMagnitude, gridPoints, uniquePoints,
            startI, endI, startJ, midJ,
            cellWidth, cellHeight, maxX, maxY, threshold);
        // Process bottom half
        recursivelyGenerateEnhancedGrid(distortionMagnitude, gridPoints, uniquePoints,
            startI, endI, midJ + 1, endJ,
            cellWidth, cellHeight, maxX, maxY, threshold);
    }

    // If this is a single cell (grid square bounded by 4 corners), 
    // check if we need to add a center point
    if ((endI - startI == 1) && (endJ - startJ == 1)) {
        // Calculate center of the cell
        int centerX = (topLeftX + bottomRightX) / 2;
        int centerY = (topLeftY + bottomRightY) / 2;

        // Check if center value is above threshold
        float centerValue = 0.0f;
        if (centerY >= 0 && centerY < distortionMagnitude.rows &&
            centerX >= 0 && centerX < distortionMagnitude.cols) {
            centerValue = distortionMagnitude.at<float>(centerY, centerX);
        }

        // Add center point if gradient magnitude is above threshold
        if (centerValue > threshold) {
            cv::Point centerPoint(centerX, centerY);
            if (uniquePoints.insert(centerPoint).second) {
                gridPoints.push_back(centerPoint);
            }
        }
    }
}

void GenerateAdaptiveGrid_v2(
    const cv::Mat& distortionMagnitude,
    std::vector<cv::Point>& enhancedGridPoints,
    const int gridX,
    const int gridY,
    const float threshold
) {
    // First generate the base fixed grid
    enhancedGridPoints.clear();

    // Calculate cell dimensions once
    const float cellWidth = static_cast<float>(distortionMagnitude.cols) / (gridX - 1);
    const float cellHeight = static_cast<float>(distortionMagnitude.rows) / (gridY - 1);
    const int maxX = distortionMagnitude.cols - 1;
    const int maxY = distortionMagnitude.rows - 1;

    // Pre-allocate memory for efficiency (base grid + potential center points)
    enhancedGridPoints.reserve(gridX * gridY * 2);  // Approximate upper bound

    // Generate base grid points (corner points)
    for (int j = 0; j < gridY; ++j) {
        const int y = std::min(static_cast<int>(j * cellHeight), maxY);

        for (int i = 0; i < gridX; ++i) {
            const int x = std::min(static_cast<int>(i * cellWidth), maxX);
            enhancedGridPoints.emplace_back(x, y);
        }
    }

    // Track unique points for efficient insertion
    std::unordered_set<cv::Point, PointHash, PointEqual> uniquePoints;
    for (const auto& pt : enhancedGridPoints) {
        uniquePoints.insert(pt);
    }

    // Add center points for cells where value exceeds threshold
    for (int j = 0; j < gridY - 1; ++j) {
        for (int i = 0; i < gridX - 1; ++i) {
            // Calculate corners of this cell
            const int topLeftX = std::min(static_cast<int>(i * cellWidth), maxX);
            const int topLeftY = std::min(static_cast<int>(j * cellHeight), maxY);
            const int bottomRightX = std::min(static_cast<int>((i + 1) * cellWidth), maxX);
            const int bottomRightY = std::min(static_cast<int>((j + 1) * cellHeight), maxY);

            // Calculate center of the cell
            int centerX = (topLeftX + bottomRightX) / 2;
            int centerY = (topLeftY + bottomRightY) / 2;

            // Check if center value is above threshold
            float centerValue = 0.0f;
            if (centerY >= 0 && centerY < distortionMagnitude.rows &&
                centerX >= 0 && centerX < distortionMagnitude.cols) {
                centerValue = distortionMagnitude.at<float>(centerY, centerX);
            }

            // Add center point if gradient magnitude is above threshold
            if (centerValue > threshold) {
                cv::Point centerPoint(centerX, centerY);
                if (uniquePoints.insert(centerPoint).second) {
                    enhancedGridPoints.push_back(centerPoint);
                }
            }
        }
    }

    // Log results
    std::cout << "Enhanced grid generated: " << enhancedGridPoints.size() << " points" << std::endl;
    std::cout << "  - Base grid size: " << gridX << "x" << gridY << " = " << gridX * gridY << " points" << std::endl;
    std::cout << "  - Additional adaptive points: " << enhancedGridPoints.size() - gridX * gridY << std::endl;
}

#endif // FISHEYE_EFFECT_H