#pragma once
#include <iostream>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Forward declaration for a non-recursive implementation
void iterativeDivide(const Mat& image, int maxLevel, float threshold, vector<Point>& corners);

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

// Optimized recursive version (if needed for compatibility)
void recursiveDivide(const Mat& image, int x, int y, int w, int h, int currentLevel,
    int maxLevel, float threshold, vector<Point>& corners,
    unordered_set<Point, PointHash, PointEqual>& uniqueCorners) {
    // Quick exit for invalid regions
    if (w <= 0 || h <= 0) return;

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

    // Use tail recursion optimization where possible
    recursiveDivide(image, x, y, halfW, halfH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners);
    recursiveDivide(image, x + halfW, y, remW, halfH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners);
    recursiveDivide(image, x, y + halfH, halfW, remH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners);
    recursiveDivide(image, x + halfW, y + halfH, remW, remH, currentLevel + 1, maxLevel, threshold, corners, uniqueCorners);
}

// Non-recursive implementation for better performance
void iterativeDivide(const Mat& image, int maxLevel, float threshold, vector<Point>& corners) {
    // Using a queue for breadth-first processing
    struct QuadNode {
        int x, y, w, h, level;
        QuadNode(int _x, int _y, int _w, int _h, int _l) :
            x(_x), y(_y), w(_w), h(_h), level(_l) {
        }
    };

    // Using a flat vector as queue for better cache locality
    vector<QuadNode> queue;
    queue.reserve(1024);  // Reserve space to avoid frequent reallocations

    // Start with the whole image
    queue.emplace_back(0, 0, image.cols, image.rows, 0);

    // Use a hash set for O(1) duplicate checks
    unordered_set<Point, PointHash, PointEqual> uniqueCorners;
    uniqueCorners.reserve(image.cols * image.rows / 16);  // Pre-allocate reasonable size

    const int maxX = image.cols - 1;
    const int maxY = image.rows - 1;

    while (!queue.empty()) {
        // Process front node and remove it
        QuadNode node = queue.back();
        queue.pop_back();  // Using as a stack (depth-first) for better cache behavior

        if (node.w <= 0 || node.h <= 0) continue;  // Skip invalid regions

        if (node.level >= maxLevel) {
            // Leaf node, add corners
            Point topLeft(node.x, node.y);
            Point topRight(node.x + node.w, node.y);
            Point bottomLeft(node.x, node.y + node.h);
            Point bottomRight(node.x + node.w, node.y + node.h);

            if (uniqueCorners.insert(topLeft).second) corners.push_back(topLeft);
            if (uniqueCorners.insert(topRight).second) corners.push_back(topRight);
            if (uniqueCorners.insert(bottomLeft).second) corners.push_back(bottomLeft);
            if (uniqueCorners.insert(bottomRight).second) corners.push_back(bottomRight);

            continue;
        }

        // Check intensity at center to decide if we should subdivide
        int centerX = std::min(node.x + node.w / 2, maxX);
        int centerY = std::min(node.y + node.h / 2, maxY);

        if (node.level == maxLevel - 1) {
            float intensity = image.at<float>(centerY, centerX);

            if (intensity <= threshold) {
                // Don't subdivide further
                Point topLeft(node.x, node.y);
                Point topRight(node.x + node.w, node.y);
                Point bottomLeft(node.x, node.y + node.h);
                Point bottomRight(node.x + node.w, node.y + node.h);

                if (uniqueCorners.insert(topLeft).second) corners.push_back(topLeft);
                if (uniqueCorners.insert(topRight).second) corners.push_back(topRight);
                if (uniqueCorners.insert(bottomLeft).second) corners.push_back(bottomLeft);
                if (uniqueCorners.insert(bottomRight).second) corners.push_back(bottomRight);

                continue;
            }
        }

        // Subdivide (push in reverse order for better locality)
        int halfW = node.w / 2;
        int halfH = node.h / 2;
        int remW = node.w - halfW;
        int remH = node.h - halfH;

        // Bottom-right
        queue.emplace_back(node.x + halfW, node.y + halfH, remW, remH, node.level + 1);
        // Bottom-left
        queue.emplace_back(node.x, node.y + halfH, halfW, remH, node.level + 1);
        // Top-right
        queue.emplace_back(node.x + halfW, node.y, remW, halfH, node.level + 1);
        // Top-left
        queue.emplace_back(node.x, node.y, halfW, halfH, node.level + 1);
    }
}

// Main function that chooses the best implementation
vector<Point> getGridPoints(const Mat& image, int maxLevel, float threshold) {
    vector<Point> corners;
    // Pre-allocate space - based on expected size
    corners.reserve((1 << (2 * maxLevel)) * 4);  // Conservative estimate

    // Use the iterative version for better performance
    //iterativeDivide(image, maxLevel, threshold, corners);

    // Alternative: recursive version with explicit deduplication
    
    unordered_set<Point, PointHash, PointEqual> uniqueCorners;
    recursiveDivide(image, 0, 0, image.cols, image.rows, 0, maxLevel, threshold, corners, uniqueCorners);
    

    return corners;
}

void GenerateAdaptiveGrid_v1(const Mat& magnitude_of_distortion,
    vector<Point>& GDC_Adaptive_Grid_Points,
    const int Maxlevel,
    const float LowThreshold) {
    GDC_Adaptive_Grid_Points = getGridPoints(magnitude_of_distortion, Maxlevel, LowThreshold);
}