#pragma once
#include <iostream>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include "GenerateGrid.hpp"

namespace HAG {
    /**
    * Applies gradient-enhanced Lloyd's relaxation to a set of points based on a distortion map.
    *
    * @param distortionMap The map containing distortion magnitude values.
    * @param points The vector of points to be relaxed.
    * @param relaxationIterations Number of Lloyd's relaxation iterations to perform (default: 3).
    * @param relaxationWeight Weight factor for centroid movement (default: 5.5).
    * @param gradientWeight Weight factor for gradient influence (default: 0.3).
    * @param distortionThreshold Threshold value for adaptive point generation (default: 1.0).
    */
    void ApplyGradientEnhancedRelaxation(
        const cv::Mat& distortionMap,
        std::vector<cv::Point>& points,
        int relaxationIterations = 3,
        float relaxationWeight = 5.5,
        float gradientWeight = 0.3,
        float distortionThreshold = 1.0) {

        // Early exit if no points or no iterations
        if (points.empty() || relaxationIterations <= 0) {
            return;
        }

        // Get image dimensions
        const int imageWidth = distortionMap.cols;
        const int imageHeight = distortionMap.rows;
        const int maxX = imageWidth - 1;
        const int maxY = imageHeight - 1;

        // DEBUG: Print initial state
        std::cout << "\n=== Initial Relaxation State ===\n"
            << "Points: " << points.size() << "\n"
            << "Iterations: " << relaxationIterations << "\n"
            << "Relaxation weight: " << relaxationWeight << "\n"
            << "Gradient weight: " << gradientWeight << "\n"
            << "Distortion threshold: " << distortionThreshold << "\n";

        // Pre-compute gradient of the distortion map
        cv::Mat gradientX, gradientY, gradientMagnitude, gradientDirection;

        // Calculate the gradient using Sobel operator
        cv::Sobel(distortionMap, gradientX, CV_32F, 1, 0, 3);
        cv::Sobel(distortionMap, gradientY, CV_32F, 0, 1, 3);

        // Compute gradient magnitude and direction
        cv::cartToPolar(gradientX, gradientY, gradientMagnitude, gradientDirection);

        // DEBUG: Print gradient stats
        double minGrad, maxGrad;
        cv::minMaxLoc(gradientMagnitude, &minGrad, &maxGrad);
        // std::cout << "Gradient magnitude range: [" << minGrad << ", " << maxGrad << "]\n";

        // Normalize gradient magnitude to [0,1] range
        double minVal, maxVal;
        cv::minMaxLoc(gradientMagnitude, &minVal, &maxVal);
        if (maxVal > minVal) {
            gradientMagnitude = (gradientMagnitude - minVal) / (maxVal - minVal);
        }

        // DEBUG: Print normalized gradient stats
        cv::minMaxLoc(gradientMagnitude, &minGrad, &maxGrad);
        // std::cout << "Normalized gradient range: [" << minGrad << ", " << maxGrad << "]\n";

        // Convert points to floating-point for more precise movement
        std::vector<cv::Point2f> floatPoints;
        floatPoints.reserve(points.size());
        for (const auto& pt : points) {
            floatPoints.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
        }

        // Perform Lloyd's Relaxation iterations
        for (int iter = 0; iter < relaxationIterations; ++iter) {
            // DEBUG: Print iteration header
            std::cout << "\n--- Iteration " << (iter + 1) << "/" << relaxationIterations << " ---\n";

            // Create a Voronoi diagram for the current points
            cv::Rect bounds(0, 0, imageWidth, imageHeight);
            cv::Mat voronoiDiagram = cv::Mat::zeros(imageHeight, imageWidth, CV_32SC1);
            std::vector<cv::Point2f> centroids(floatPoints.size(), cv::Point2f(0, 0));
            std::vector<int> pointCounts(floatPoints.size(), 0);

            // Convert to integer points for Voronoi
            std::vector<cv::Point> intPoints;
            intPoints.reserve(floatPoints.size());
            for (const auto& pt : floatPoints) {
                intPoints.emplace_back(
                    std::min(std::max(0, static_cast<int>(std::round(pt.x))), maxX),
                    std::min(std::max(0, static_cast<int>(std::round(pt.y))), maxY)
                );
            }

            // Build Voronoi diagram using OpenCV's approach
            // Each pixel in voronoiDiagram will be labeled with the index of the nearest point + 1
            // (0 is reserved for pixels outside all regions)
            for (int y = 0; y < imageHeight; ++y) {
                for (int x = 0; x < imageWidth; ++x) {
                    int nearestIdx = -1;
                    float minDist = std::numeric_limits<float>::max();

                    for (size_t i = 0; i < intPoints.size(); ++i) {
                        float dx = x - intPoints[i].x;
                        float dy = y - intPoints[i].y;
                        float dist = dx * dx + dy * dy;  // Squared distance is sufficient for comparison

                        if (dist < minDist) {
                            minDist = dist;
                            nearestIdx = static_cast<int>(i);
                        }
                    }

                    if (nearestIdx >= 0) {
                        // Label this pixel with the index of the nearest point (adding 1 to avoid 0)
                        voronoiDiagram.at<int>(y, x) = nearestIdx + 1;

                        // Add this pixel to the centroid calculation
                        centroids[nearestIdx].x += x;
                        centroids[nearestIdx].y += y;
                        pointCounts[nearestIdx]++;
                    }
                }
            }

            // Calculate final centroids by averaging
            for (size_t i = 0; i < centroids.size(); ++i) {
                if (pointCounts[i] > 0) {
                    centroids[i].x /= pointCounts[i];
                    centroids[i].y /= pointCounts[i];

                    // Ensure centroids are within bounds
                    centroids[i].x = std::min(std::max(0.0f, centroids[i].x), static_cast<float>(maxX));
                    centroids[i].y = std::min(std::max(0.0f, centroids[i].y), static_cast<float>(maxY));
                }
                else {
                    // If no pixels assigned to this point, keep it where it is
                    centroids[i] = floatPoints[i];
                }
            }

            // DEBUG: Print centroid stats
            // std::cout << "Computed centroids: " << centroids.size() << "\n";

            // Prepare new positions
            std::vector<cv::Point2f> newPositions(floatPoints.size());

            // Calculate new positions for each point
#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(floatPoints.size()); ++i) {
                const cv::Point2f& currentPoint = floatPoints[i];
                const cv::Point2f& centroid = centroids[i];

                // Standard Lloyd's movement vector (towards centroid)
                cv::Point2f centroidMovement = (centroid - currentPoint) * relaxationWeight;

                // Sample the gradient at current point
                int px = std::min(std::max(0, static_cast<int>(currentPoint.x)), maxX);
                int py = std::min(std::max(0, static_cast<int>(currentPoint.y)), maxY);

                // Get gradient properties at this location
                float gradMag = gradientMagnitude.at<float>(py, px);
                float gradDir = gradientDirection.at<float>(py, px);

                // Convert gradient direction from polar to cartesian
                cv::Point2f gradientVector(
                    gradMag * std::cos(gradDir),
                    gradMag * std::sin(gradDir)
                );

                // Scale by gradient weight
                gradientVector *= gradientWeight;

                // Get local distortion value for adaptive weighting
                float distortionValue = distortionMap.at<float>(py, px);
                float distortionFactor = std::min(distortionValue / distortionThreshold, 1.0f);

                // Combine movements: more gradient influence in high distortion areas
                cv::Point2f combinedMovement = centroidMovement + (gradientVector * distortionFactor);

                // Apply movement to get new position
                cv::Point2f newPosition = currentPoint + combinedMovement;

                // Ensure new position is within image boundaries
                newPosition.x = std::max(0.0f, std::min(newPosition.x, static_cast<float>(maxX)));
                newPosition.y = std::max(0.0f, std::min(newPosition.y, static_cast<float>(maxY)));

                // Store new position
                newPositions[i] = newPosition;
            }

            // DEBUG: Movement statistics
            float maxMovement = 0;
            for (size_t i = 0; i < floatPoints.size(); ++i) {
                float movement = cv::norm(floatPoints[i] - newPositions[i]);
                if (movement > maxMovement) maxMovement = movement;
            }
            // std::cout << "Max point movement: " << maxMovement << " pixels\n";

            // Update all points with their new positions
            floatPoints = std::move(newPositions);
        }

        // Convert floating-point points back to integer points
        points.clear();
        points.reserve(floatPoints.size());

        // Use a set to eliminate duplicates
        std::set<std::pair<int, int>> uniquePoints;

        for (const auto& pt : floatPoints) {
            int x = static_cast<int>(std::round(pt.x));
            int y = static_cast<int>(std::round(pt.y));

            // Only add if it's a new point
            auto result = uniquePoints.insert({ x, y });
            if (result.second) {
                points.emplace_back(x, y);
            }
        }
    }
    /**
     * Recursively generates an adaptive grid of points based on distortion values,
     * then applies Lloyd's Relaxation for more uniform point distribution.
     *
     * @param distortionMap       The map containing distortion magnitude values
     * @param resultPoints        Output vector where grid points will be stored
     * @param horizontalGridSize  Number of grid points along the horizontal axis
     * @param verticalGridSize    Number of grid points along the vertical axis
     * @param distortionThreshold Threshold value for adaptive point generation
     * @param relaxationIterations Number of Lloyd's relaxation iterations to perform (default: 3)
     * @param relaxationWeight    Weight factor for centroid movement (0.0-1.0, default: 0.5)
     */
    void GenerateAdaptiveGrid_HAG_v2(
        const cv::Mat& distortionMap,
        std::vector<cv::Point>& resultPoints,
        const int horizontalGridSize = 30,
        const int verticalGridSize = 30,
        const float distortionThreshold = 0.85f,
        const int relaxationIterations = 6,
        const float relaxationWeight = 2.0) {

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

        float gradientWeight = 0.3;

        ApplyGradientEnhancedRelaxation(distortionMap, resultPoints, relaxationIterations, relaxationWeight, gradientWeight, distortionThreshold);
        
    }
} // namespace HAG

