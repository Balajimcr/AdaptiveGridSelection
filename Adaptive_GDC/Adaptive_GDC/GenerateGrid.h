#pragma once
#include <iostream>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>

namespace HAG {
    // Hash structure for Point to use with unordered_set
    struct PointHash {
        size_t operator()(const cv::Point& p) const {
            return (size_t)(((uint64_t)p.x << 32) | (uint64_t)p.y);
        }
    };

    // Custom equality comparator for Point
    struct PointEqual {
        bool operator()(const cv::Point& lhs, const cv::Point& rhs) const {
            return lhs.x == rhs.x && lhs.y == rhs.y;
        }
    };

void Generate_FixedGrid(const cv::Mat& distortionMagnitude, std::vector<cv::Point>& fixedGridPoints, int gridX, int gridY) {
    
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
        }
    }
}

// Adaptive HAG grid generation with OpenMP parallelization
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

    // Pre-allocate with estimated capacity (fixed grid + potential adaptive points)
    const size_t estimatedSize = gridX * gridY * 2;
    adaptiveGridPoints.clear();
    adaptiveGridPoints.reserve(estimatedSize);

    // First, create a temporary vector for the fixed grid points
    std::vector<cv::Point> fixedGridPoints;
    fixedGridPoints.reserve(gridX * gridY);

    // Generate the fixed grid points in parallel
#pragma omp parallel
    {
        // Each thread creates its own local vector to avoid race conditions
        std::vector<cv::Point> localPoints;

#pragma omp for nowait schedule(dynamic, 4)
        for (int j = 0; j < gridY; ++j) {
            const int y = std::min(static_cast<int>(j * cellHeight), maxY);
            for (int i = 0; i < gridX; ++i) {
                const int x = std::min(static_cast<int>(i * cellWidth), maxX);
                localPoints.emplace_back(x, y);
            }
        }

        // Merge results safely
#pragma omp critical
        {
            fixedGridPoints.insert(fixedGridPoints.end(), localPoints.begin(), localPoints.end());
        }
    }

    // Create a vector for adaptive points
    std::vector<cv::Point> adaptivePoints;
    adaptivePoints.reserve(gridX * gridY);

    // Add adaptive points based on distortion in parallel
#pragma omp parallel
    {
        // Each thread creates its own local vector to avoid race conditions
        std::vector<cv::Point> localAdaptivePoints;

#pragma omp for nowait schedule(dynamic, 4)
        for (int i = 0; i < gridX; ++i) {
            const int x = std::min(static_cast<int>(i * cellWidth), maxX);
            const float effectiveCellWidth = std::min(cellWidth, static_cast<float>(imageWidth - x));

            for (int j = 0; j < gridY; ++j) {
                const int y = std::min(static_cast<int>(j * cellHeight), maxY);
                const float effectiveCellHeight = std::min(cellHeight, static_cast<float>(imageHeight - y));

                // Calculate midpoint of the cell (calculating once and reusing)
                const int midX = std::min(static_cast<int>(x + (effectiveCellWidth / 2.0)), maxX);
                const int midY = std::min(static_cast<int>(y + (effectiveCellHeight / 2.0)), maxY);

                // Get distortion value at center point
                const float distortionValue = magnitudeOfDistortion.at<float>(midY, midX);

                if (distortionValue >= lowThreshold) {
                    localAdaptivePoints.emplace_back(midX, midY);
                }
            }
        }

        // Merge results safely
#pragma omp critical
        {
            adaptivePoints.insert(adaptivePoints.end(),
                localAdaptivePoints.begin(),
                localAdaptivePoints.end());
        }
    }

    // Combine fixed and adaptive points (avoid multiple resizes)
    adaptiveGridPoints = std::move(fixedGridPoints);
    adaptiveGridPoints.insert(adaptiveGridPoints.end(),
        adaptivePoints.begin(),
        adaptivePoints.end());
}

/**
 * Recursively generates an adaptive grid of points based on distortion values.
 *
 * @param distortionMap       The map containing distortion magnitude values
 * @param resultPoints        Output vector where grid points will be stored
 * @param horizontalGridSize  Number of grid points along the horizontal axis
 * @param verticalGridSize    Number of grid points along the vertical axis
 * @param distortionThreshold Threshold value for adaptive point generation
 */
void GenerateAdaptiveGrid_HAG_v1(
    const cv::Mat& distortionMap,
    std::vector<cv::Point>& resultPoints,
    const int horizontalGridSize,
    const int verticalGridSize,
    const float distortionThreshold) {

    // Get image dimensions
    const int imageWidth = distortionMap.cols;
    const int imageHeight = distortionMap.rows;
    const int maxPixelX = imageWidth - 1;
    const int maxPixelY = imageHeight - 1;

    // Calculate the size of each grid cell in pixels
    const float cellPixelWidth = static_cast<float>(imageWidth) / (horizontalGridSize - 1);
    const float cellPixelHeight = static_cast<float>(imageHeight) / (verticalGridSize - 1);

    // Pre-allocate memory for the result points
    const size_t estimatedPointCount = horizontalGridSize * verticalGridSize * 2;
    resultPoints.clear();
    resultPoints.reserve(estimatedPointCount);

    // Set for preventing duplicate points
    std::set<std::pair<int, int>> uniquePoints;

    /**
     * Recursive function to process a grid region.
     *
     * @param startGridX Starting grid index on X-axis (horizontal)
     * @param endGridX   Ending grid index on X-axis (inclusive)
     * @param startGridY Starting grid index on Y-axis (vertical)
     * @param endGridY   Ending grid index on Y-axis (inclusive)
     */
    std::function<void(int, int, int, int)> processGridRegion =
        [&](int startGridX, int endGridX, int startGridY, int endGridY) {
        // Convert grid indices to pixel coordinates
        const int startPixelX = std::min(static_cast<int>(startGridX * cellPixelWidth), maxPixelX);
        const int startPixelY = std::min(static_cast<int>(startGridY * cellPixelHeight), maxPixelY);
        const int endPixelX = std::min(static_cast<int>(endGridX * cellPixelWidth), maxPixelX);
        const int endPixelY = std::min(static_cast<int>(endGridY * cellPixelHeight), maxPixelY);

        // Add the four corner points (part of the fixed grid)
        const std::vector<std::pair<int, int>> cornerPoints = {
            {startPixelX, startPixelY},  // Top-left corner
            {endPixelX, startPixelY},    // Top-right corner
            {startPixelX, endPixelY},    // Bottom-left corner
            {endPixelX, endPixelY}       // Bottom-right corner
        };

        // Add each corner point if not already added
        for (const auto& cornerPoint : cornerPoints) {
            // Insert returns a pair where second is true if the element was inserted
            if (uniquePoints.insert(cornerPoint).second) {
                resultPoints.emplace_back(cornerPoint.first, cornerPoint.second);
            }
        }

        // Base case: Single grid cell - check if we need an adaptive point
        if (startGridX == endGridX - 1 && startGridY == endGridY - 1) {
            // Calculate the midpoint of this cell in pixel coordinates
            const int midPixelX = std::min(static_cast<int>(startPixelX + (endPixelX - startPixelX) / 2.0), maxPixelX);
            const int midPixelY = std::min(static_cast<int>(startPixelY + (endPixelY - startPixelY) / 2.0), maxPixelY);

            // Get the distortion value at this midpoint
            const float distortionValue = distortionMap.at<float>(midPixelY, midPixelX);

            // If distortion exceeds threshold, add an adaptive point
            if (distortionValue >= distortionThreshold) {
                auto adaptivePoint = std::make_pair(midPixelX, midPixelY);
                if (uniquePoints.insert(adaptivePoint).second) {
                    resultPoints.emplace_back(midPixelX, midPixelY);
                }
            }
            return;  // Done processing this cell
        }

        // Recursive case: Calculate mid-grid indices for subdividing the region
        // Handle special cases where region can't be divided in one dimension
        int midGridX = (endGridX == startGridX) ? startGridX : startGridX + (endGridX - startGridX) / 2;
        int midGridY = (endGridY == startGridY) ? startGridY : startGridY + (endGridY - startGridY) / 2;

        // Safety check: Ensure we can make progress in at least one dimension
        if (midGridX == startGridX && midGridY == startGridY) {
            return;  // Can't divide further, prevent infinite recursion
        }

        // Process sub-regions (quadrants) if they have non-zero area
        // Top-left quadrant
        if (midGridX > startGridX && midGridY > startGridY) {
            processGridRegion(startGridX, midGridX, startGridY, midGridY);
        }

        // Top-right quadrant
        if (endGridX > midGridX && midGridY > startGridY) {
            processGridRegion(midGridX, endGridX, startGridY, midGridY);
        }

        // Bottom-left quadrant
        if (midGridX > startGridX && endGridY > midGridY) {
            processGridRegion(startGridX, midGridX, midGridY, endGridY);
        }

        // Bottom-right quadrant
        if (endGridX > midGridX && endGridY > midGridY) {
            processGridRegion(midGridX, endGridX, midGridY, endGridY);
        }
        };

    // Start the recursive processing from the entire grid
    // The grid indices are 0-based, so the end indices are (size-1)
    processGridRegion(0, horizontalGridSize - 1, 0, verticalGridSize - 1);
}
} // namespace HAG

