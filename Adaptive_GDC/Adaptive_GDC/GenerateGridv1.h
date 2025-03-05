#pragma once
#include <iostream>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <random>

using namespace cv;
using namespace std;

// Hash structure for Point to use with unordered_set
struct PointHash {
    size_t operator()(const Point& p) const {
        return (size_t)(((uint64_t)p.x << 32) | (uint64_t)p.y);
    }
};

// Custom equality comparator for Point
struct PointEqual {
    bool operator()(const Point& lhs, const Point& rhs) const {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }
};

// Calculate distance between two points
inline float pointDistance(const Point& p1, const Point& p2) {
    float dx = static_cast<float>(p1.x - p2.x);
    float dy = static_cast<float>(p1.y - p2.y);
    return std::sqrt(dx * dx + dy * dy);
}

// Optimized recursive version with minimum cell size check
void recursiveDivide(const Mat& image, int x, int y, int w, int h, int currentLevel,
    int maxLevel, float threshold, vector<Point>& corners,
    unordered_set<Point, PointHash, PointEqual>& uniqueCorners,
    int minCellWidth, int minCellHeight) {
    // Quick exit for invalid regions or cells smaller than minimum size
    if (w <= 0 || h <= 0 || w < minCellWidth || h < minCellHeight) return;

    // Early processing of leaf nodes or max depth reached
    if (currentLevel >= maxLevel) {
        // Add corners only if they don't already exist
        Point topLeft(x, y);
        Point topRight(x + w, y);
        Point bottomLeft(x, y + h);
        Point bottomRight(x + w, y + h);

        // Using the set for O(1) duplicate detection
        if (uniqueCorners.insert(topLeft).second) corners.push_back(topLeft);
        if (uniqueCorners.insert(topRight).second) corners.push_back(topRight);
        if (uniqueCorners.insert(bottomLeft).second) corners.push_back(bottomLeft);
        if (uniqueCorners.insert(bottomRight).second) corners.push_back(bottomRight);

        return;
    }

    // For non-leaf nodes, check intensity to decide whether to subdivide
    int centerX = std::min(x + w / 2, image.cols - 1);
    int centerY = std::min(y + h / 2, image.rows - 1);

    // Only sample pixel if needed - saves time in non-boundary cases
    if (currentLevel == maxLevel - 1) {
        float intensity = image.at<float>(centerY, centerX);
        if (intensity <= threshold) {
            // Low intensity, don't subdivide further, add corners now
            Point topLeft(x, y);
            Point topRight(x + w, y);
            Point bottomLeft(x, y + h);
            Point bottomRight(x + w, y + h);

            if (uniqueCorners.insert(topLeft).second) corners.push_back(topLeft);
            if (uniqueCorners.insert(topRight).second) corners.push_back(topRight);
            if (uniqueCorners.insert(bottomLeft).second) corners.push_back(bottomLeft);
            if (uniqueCorners.insert(bottomRight).second) corners.push_back(bottomRight);

            return;
        }
    }

    // Subdivide with minimal integer division (pre-compute half sizes once)
    int halfW = w / 2;
    int halfH = h / 2;
    int remW = w - halfW;  // Handle odd sizes correctly
    int remH = h - halfH;

    // Check if subdivided cells would be too small
    if (halfW < minCellWidth || halfH < minCellHeight) {
        // Cell too small to subdivide further, add corners now
        Point topLeft(x, y);
        Point topRight(x + w, y);
        Point bottomLeft(x, y + h);
        Point bottomRight(x + w, y + h);

        if (uniqueCorners.insert(topLeft).second) corners.push_back(topLeft);
        if (uniqueCorners.insert(topRight).second) corners.push_back(topRight);
        if (uniqueCorners.insert(bottomLeft).second) corners.push_back(bottomLeft);
        if (uniqueCorners.insert(bottomRight).second) corners.push_back(bottomRight);

        return;
    }

    // Use tail recursion optimization where possible
    recursiveDivide(image, x, y, halfW, halfH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
    recursiveDivide(image, x + halfW, y, remW, halfH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
    recursiveDivide(image, x, y + halfH, halfW, remH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
    recursiveDivide(image, x + halfW, y + halfH, remW, remH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
}

// Apply spatial filtering to reduce point density
vector<Point> filterPointsBySpacing(const vector<Point>& inputPoints, float minDistance) {
    if (minDistance <= 0) return inputPoints;

    vector<Point> filteredPoints;
    filteredPoints.reserve(inputPoints.size() / 2);  // Estimate conservatively

    // Use a grid-based approach for efficiency with large point sets
    const int gridSize = static_cast<int>(minDistance) + 1;

    // Find image bounds
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = INT_MIN, maxY = INT_MIN;

    for (const auto& pt : inputPoints) {
        minX = std::min(minX, pt.x);
        minY = std::min(minY, pt.y);
        maxX = std::max(maxX, pt.x);
        maxY = std::max(maxY, pt.y);
    }

    // Create occupancy grid
    int gridWidth = (maxX - minX) / gridSize + 1;
    int gridHeight = (maxY - minY) / gridSize + 1;

    // Use vector of vectors for the grid cells
    vector<vector<Point>> grid(gridWidth * gridHeight);

    // Place points into grid cells
    for (const auto& pt : inputPoints) {
        int gridX = (pt.x - minX) / gridSize;
        int gridY = (pt.y - minY) / gridSize;
        int idx = gridY * gridWidth + gridX;
        grid[idx].push_back(pt);
    }

    // Process each grid cell
    for (int i = 0; i < grid.size(); i++) {
        if (grid[i].empty()) continue;

        // Start with first point in cell
        filteredPoints.push_back(grid[i][0]);

        // Check remaining points in this cell and in neighborhood
        int gridX = i % gridWidth;
        int gridY = i / gridWidth;

        // Check neighboring cells (including this cell)
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = gridX + dx;
                int ny = gridY + dy;

                // Skip out-of-bounds cells
                if (nx < 0 || nx >= gridWidth || ny < 0 || ny >= gridHeight)
                    continue;

                int neighborIdx = ny * gridWidth + nx;

                // Skip empty cells
                if (grid[neighborIdx].empty())
                    continue;

                // Mark points that are too close to any point in filteredPoints
                for (int j = 0; j < grid[neighborIdx].size(); j++) {
                    const Point& candidate = grid[neighborIdx][j];
                    bool tooClose = false;

                    // Check against already accepted points
                    for (const Point& accepted : filteredPoints) {
                        if (pointDistance(candidate, accepted) < minDistance) {
                            tooClose = true;
                            break;
                        }
                    }

                    if (!tooClose) {
                        filteredPoints.push_back(candidate);
                    }
                }

                // Clear processed cell to avoid reprocessing
                grid[neighborIdx].clear();
            }
        }
    }

    return filteredPoints;
}

// Main function that generates the grid points
vector<Point> getGridPoints(const Mat& image, int maxLevel, float threshold,
    int minCellWidth, int minCellHeight, float pointSpacing = 0) {

    // Generate initial point set
    vector<Point> corners;
    corners.reserve((1 << (2 * maxLevel)) * 4);  // Conservative estimate

    unordered_set<Point, PointHash, PointEqual> uniqueCorners;
    recursiveDivide(image, 0, 0, image.cols, image.rows, 0, maxLevel, threshold,
        corners, uniqueCorners, minCellWidth, minCellHeight);

    // If no spacing control requested, return all points
    if (pointSpacing <= 0) {
        return corners;
    }

    // Apply spatial filtering to control point density
    return filterPointsBySpacing(corners, pointSpacing);
}

// Generate adaptive grid with controllable spacing
void GenerateAdaptiveGrid_v1(const Mat& magnitude_of_distortion,
    vector<Point>& GDC_Adaptive_Grid_Points,
    const float LowThreshold,
    const int Maxlevel = 6,
    int minCellWidth = 5,
    int minCellHeight = 5,
    float pointSpacing = 20.0) {  // Control minimum distance between points

    GDC_Adaptive_Grid_Points = getGridPoints(magnitude_of_distortion,
        Maxlevel,
        LowThreshold,
        minCellWidth,
        minCellHeight,
        pointSpacing);
}
