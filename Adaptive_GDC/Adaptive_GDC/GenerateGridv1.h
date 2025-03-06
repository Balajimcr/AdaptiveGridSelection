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

// Safe way to get a pixel value with bounds checking
inline float GetPixelValue(const Mat& image, int y, int x) {
    // Ensure coordinates are within bounds
    x = std::max(0, std::min(x, image.cols - 1));
    y = std::max(0, std::min(y, image.rows - 1));
    return image.at<float>(y, x);
}

// Helper to safely add a point ensuring it's within image bounds
inline void AddPoint(const Point& point, vector<Point>& corners,
    unordered_set<Point, PointHash, PointEqual>& uniqueCorners,
    int maxWidth, int maxHeight) {
    // Clamp point coordinates to image bounds
    Point boundedPoint(
        std::max(0, std::min(point.x, maxWidth)),
        std::max(0, std::min(point.y, maxHeight))
    );

    // Add only if it doesn't already exist
    if (uniqueCorners.insert(boundedPoint).second) {
        corners.push_back(boundedPoint);
    }
}

// Optimized recursive version with minimum cell size check and bounds safety
void recursiveDivide(const Mat& image, int x, int y, int w, int h, int currentLevel,
    int maxLevel, float threshold, vector<Point>& corners,
    unordered_set<Point, PointHash, PointEqual>& uniqueCorners,
    int minCellWidth, int minCellHeight) {

    // Ensure we don't go out of bounds
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::min(w, image.cols - x);
    h = std::min(h, image.rows - y);

    // Quick exit for invalid regions or cells smaller than minimum size
    if (w <= 0 || h <= 0 || w < minCellWidth || h < minCellHeight) return;

    // Early processing of leaf nodes or max depth reached
    if (currentLevel >= maxLevel) {
        // Add corners only if they don't already exist
        AddPoint(Point(x, y), corners, uniqueCorners, image.cols - 1, image.rows - 1);
        AddPoint(Point(x + w, y), corners, uniqueCorners, image.cols - 1, image.rows - 1);
        AddPoint(Point(x, y + h), corners, uniqueCorners, image.cols - 1, image.rows - 1);
        AddPoint(Point(x + w, y + h), corners, uniqueCorners, image.cols - 1, image.rows - 1);

        return;
    }

    // For non-leaf nodes, check intensity to decide whether to subdivide
    int centerX = x + w / 2;
    int centerY = y + h / 2;

    // Only sample pixel if needed - saves time in non-boundary cases
    if (currentLevel == maxLevel - 1) {
        float intensity = GetPixelValue(image, centerY, centerX);
        if (intensity <= threshold) {
            // Low intensity, don't subdivide further, add corners now
            AddPoint(Point(x, y), corners, uniqueCorners, image.cols - 1, image.rows - 1);
            AddPoint(Point(x + w, y), corners, uniqueCorners, image.cols - 1, image.rows - 1);
            AddPoint(Point(x, y + h), corners, uniqueCorners, image.cols - 1, image.rows - 1);
            AddPoint(Point(x + w, y + h), corners, uniqueCorners, image.cols - 1, image.rows - 1);

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
        AddPoint(Point(x, y), corners, uniqueCorners, image.cols - 1, image.rows - 1);
        AddPoint(Point(x + w, y), corners, uniqueCorners, image.cols - 1, image.rows - 1);
        AddPoint(Point(x, y + h), corners, uniqueCorners, image.cols - 1, image.rows - 1);
        AddPoint(Point(x + w, y + h), corners, uniqueCorners, image.cols - 1, image.rows - 1);

        return;
    }

    // Use tail recursion optimization where possible
    recursiveDivide(image, x, y, halfW, halfH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
    recursiveDivide(image, x + halfW, y, remW, halfH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
    recursiveDivide(image, x, y + halfH, halfW, remH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
    recursiveDivide(image, x + halfW, y + halfH, remW, remH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners, minCellWidth, minCellHeight);
}

// Apply spatial filtering to reduce point density with improved bounds checking
vector<Point> filterPointsBySpacing(const vector<Point>& inputPoints, float minDistance, int maxWidth, int maxHeight) {
    if (minDistance <= 0 || inputPoints.empty()) return inputPoints;

    vector<Point> filteredPoints;
    filteredPoints.reserve(inputPoints.size() / 2);  // Estimate conservatively

    // Use a grid-based approach for efficiency with large point sets
    const int gridSize = static_cast<int>(minDistance) + 1;

    // Find image bounds (clamped to valid range)
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = INT_MIN, maxY = INT_MIN;

    for (const auto& pt : inputPoints) {
        int x = std::max(0, std::min(pt.x, maxWidth));
        int y = std::max(0, std::min(pt.y, maxHeight));

        minX = std::min(minX, x);
        minY = std::min(minY, y);
        maxX = std::max(maxX, x);
        maxY = std::max(maxY, y);
    }

    // Handle empty range case
    if (minX > maxX || minY > maxY) {
        return filteredPoints;
    }

    // Create occupancy grid
    int gridWidth = (maxX - minX) / gridSize + 1;
    int gridHeight = (maxY - minY) / gridSize + 1;

    // Validate grid dimensions
    if (gridWidth <= 0 || gridHeight <= 0) {
        for (const auto& pt : inputPoints) {
            int x = std::max(0, std::min(pt.x, maxWidth));
            int y = std::max(0, std::min(pt.y, maxHeight));
            filteredPoints.push_back(Point(x, y));
        }
        return filteredPoints;
    }

    // Use vector of vectors for the grid cells
    vector<vector<Point>> grid(gridWidth * gridHeight);

    // Place points into grid cells
    for (const auto& pt : inputPoints) {
        int x = std::max(0, std::min(pt.x, maxWidth));
        int y = std::max(0, std::min(pt.y, maxHeight));

        int gridX = (x - minX) / gridSize;
        int gridY = (y - minY) / gridSize;

        // Ensure grid indices are valid
        gridX = std::max(0, std::min(gridX, gridWidth - 1));
        gridY = std::max(0, std::min(gridY, gridHeight - 1));

        int idx = gridY * gridWidth + gridX;
        if (idx >= 0 && idx < grid.size()) {
            grid[idx].push_back(Point(x, y));
        }
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

                // Ensure neighborIdx is within bounds
                if (neighborIdx < 0 || neighborIdx >= grid.size())
                    continue;

                // Skip empty cells
                if (grid[neighborIdx].empty())
                    continue;

                // Process the points in this cell
                vector<Point> acceptedPoints;

                for (const Point& candidate : grid[neighborIdx]) {
                    bool tooClose = false;

                    // Check against already accepted points
                    for (const Point& accepted : filteredPoints) {
                        if (pointDistance(candidate, accepted) < minDistance) {
                            tooClose = true;
                            break;
                        }
                    }

                    // Also check against points accepted within this cell
                    for (const Point& accepted : acceptedPoints) {
                        if (pointDistance(candidate, accepted) < minDistance) {
                            tooClose = true;
                            break;
                        }
                    }

                    if (!tooClose) {
                        acceptedPoints.push_back(candidate);
                    }
                }

                // Add accepted points from this cell
                filteredPoints.insert(filteredPoints.end(), acceptedPoints.begin(), acceptedPoints.end());

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

    // Input validation
    if (image.empty()) {
        cerr << "Warning: Input image is empty!" << endl;
        return vector<Point>();
    }

    if (image.type() != CV_32F) {
        cerr << "Warning: Input image should be CV_32F type!" << endl;
    }

    // Adjust parameters to be valid
    maxLevel = std::max(1, maxLevel);
    minCellWidth = std::max(1, minCellWidth);
    minCellHeight = std::max(1, minCellHeight);
    pointSpacing = std::max(0.0f, pointSpacing);

    // Generate initial point set
    vector<Point> corners;
    corners.reserve((1 << (2 * std::min(maxLevel, 10))) * 4);  // Conservative estimate, capped to avoid overflow

    unordered_set<Point, PointHash, PointEqual> uniqueCorners;
    recursiveDivide(image, 0, 0, image.cols, image.rows, 0, maxLevel, threshold,
        corners, uniqueCorners, minCellWidth, minCellHeight);

    // If no spacing control requested, return all points
    if (pointSpacing <= 0) {
        return corners;
    }

    // Apply spatial filtering to control point density
    return filterPointsBySpacing(corners, pointSpacing, image.cols - 1, image.rows - 1);
}

// Generate adaptive grid with controllable spacing
void GenerateAdaptiveGrid_v1(const Mat& magnitude_of_distortion,
    vector<Point>& GDC_Adaptive_Grid_Points,
    const float LowThreshold,
    const int Maxlevel = 6,
    int minCellWidth = 10,
    int minCellHeight = 10,
    float pointSpacing = 20.0) {  // Control minimum distance between points

    // Input validation
    if (magnitude_of_distortion.empty()) {
        cerr << "Error: Input distortion magnitude map is empty!" << endl;
        GDC_Adaptive_Grid_Points.clear();
        return;
    }

    // Make sure we have a float image (or convert if needed)
    Mat floatMagnitude;
    if (magnitude_of_distortion.type() != CV_32F) {
        magnitude_of_distortion.convertTo(floatMagnitude, CV_32F);
    }
    else {
        floatMagnitude = magnitude_of_distortion;
    }

    // Generate grid points with validated parameters
    GDC_Adaptive_Grid_Points = getGridPoints(floatMagnitude,
        Maxlevel,
        LowThreshold,
        minCellWidth,
        minCellHeight,
        pointSpacing);

    cout << "Generated " << GDC_Adaptive_Grid_Points.size() << " adaptive grid points." << endl;
}