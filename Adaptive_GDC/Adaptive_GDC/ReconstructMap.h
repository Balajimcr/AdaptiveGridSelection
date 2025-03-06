#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <libInterpolate/Interpolators/_2D/ThinPlateSplineInterpolator.hpp>

namespace _2D {
    class ThinPlateSplineInter : public ThinPlateSplineInterpolator<double>
    {
    public:
        VectorType getX() { return *(this->X); }
        VectorType getY() { return *(this->Y); }
        MatrixType getZ() { return *(this->Z); }
    };
}

/**
 * Comparison operator for cv::Point to use in std::map
 * Needed to properly sort and compare Points as keys
 */
struct PointCompare {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        if (a.y == b.y) {
            return a.x < b.x;
        }
        return a.y < b.y;
    }
};

/**
 * Error metrics structure to store reconstruction evaluation results
 */
struct ReconstructionErrorMetrics {
    double meanError;      // Average Euclidean distance error
    double maxError;       // Maximum error value
    double rmse;           // Root Mean Square Error
    double psnr;           // Peak Signal-to-Noise Ratio
    double executionTimeMs; // Reconstruction time in milliseconds
    cv::Mat errorMap;      // Visualization-ready error map (color-coded)
    int errorPixelCount;   // Number of pixels where interpolation failed
};

/**
 * Extract distortion values at specified grid points from the distortion maps
 *
 * @param mapX The X-coordinate distortion map (CV_32F)
 * @param mapY The Y-coordinate distortion map (CV_32F)
 * @param gridPoints Vector of grid point locations
 * @return A map from grid point coordinates to distortion values (x,y)
 */
std::map<cv::Point, cv::Point2f, PointCompare> ExtractGridDistortionValues(
    const cv::Mat& mapX,
    const cv::Mat& mapY,
    const std::vector<cv::Point>& gridPoints
) {
    // Verify input maps are of correct type
    CV_Assert(mapX.type() == CV_32F && mapY.type() == CV_32F);
    CV_Assert(mapX.size() == mapY.size());

    // Create output map
    std::map<cv::Point, cv::Point2f, PointCompare> gridDistortionMap;

    // Extract distortion values at each grid point
    for (const auto& point : gridPoints) {
        // Ensure the point is within bounds
        if (point.x >= 0 && point.x < mapX.cols &&
            point.y >= 0 && point.y < mapX.rows) {
            // Get distortion values at this grid point
            float x_distortion = mapX.at<float>(point.y, point.x);
            float y_distortion = mapY.at<float>(point.y, point.x);

            // Store the values in the map
            gridDistortionMap[point] = cv::Point2f(x_distortion, y_distortion);
        }
        else {
            // Handle out-of-bounds points (should not happen with proper grid generation)
            std::cerr << "Warning: Grid point (" << point.x << "," << point.y
                << ") is out of bounds!" << std::endl;

            // Add default values or handle as needed
            gridDistortionMap[point] = cv::Point2f(0.0f, 0.0f);
        }
    }

    return gridDistortionMap;
}

/**
 * Reconstruct full distortion maps from grid points using Thin Plate Spline interpolation
 *
 * @param gridDistortionMap Map of grid points to their distortion values
 * @param imageSize Size of the output distortion maps
 * @param reconstructedMapX Output X distortion map
 * @param reconstructedMapY Output Y distortion map
 * @return True if reconstruction was successful, false otherwise
 */
bool ReconstructDistortionMaps(
    const std::map<cv::Point, cv::Point2f, PointCompare>& gridDistortionMap,
    const cv::Size& imageSize,
    cv::Mat& reconstructedMapX,
    cv::Mat& reconstructedMapY
) {
    // Check if we have enough grid points for interpolation
    if (gridDistortionMap.size() < 3) {
        std::cerr << "Error: Thin Plate Spline requires at least 3 grid points!" << std::endl;
        return false;
    }

    try {
        // Initialize output maps
        reconstructedMapX = cv::Mat(imageSize, CV_32F);
        reconstructedMapY = cv::Mat(imageSize, CV_32F);

        // Get the size of the grid distortion map
        int gridSize = gridDistortionMap.size();

        // Create interpolators for X and Y distortion values
        _2D::ThinPlateSplineInterpolator<double> interpolateX;
        _2D::ThinPlateSplineInterpolator<double> interpolateY;

        // Create vectors to store grid data
        _2D::ThinPlateSplineInterpolator<double>::VectorType
            gridPointsX(gridSize),
            gridPointsY(gridSize),
            distortionValuesX(gridSize),
            distortionValuesY(gridSize);

        // Fill the vectors with grid points and their distortion values
        int i = 0;
        for (const auto& entry : gridDistortionMap) {
            // Extract grid point coordinates
            const cv::Point& gridPoint = entry.first;

            // Extract distortion values at this grid point
            const cv::Point2f& distortionValue = entry.second;

            // Store grid point coordinates
            gridPointsX[i] = static_cast<double>(gridPoint.x);
            gridPointsY[i] = static_cast<double>(gridPoint.y);

            // Store distortion values
            distortionValuesX[i] = static_cast<double>(distortionValue.x);
            distortionValuesY[i] = static_cast<double>(distortionValue.y);

            i++;
        }

        // Configure the interpolators
        //interpolateX.setSmoothing(0.1); // Adjust smoothing if needed
        //0interpolateY.setSmoothing(0.1); // Adjust smoothing if needed

        // Set the data for interpolation
        interpolateX.setData(gridPointsX, gridPointsY, distortionValuesX);
        interpolateY.setData(gridPointsX, gridPointsY, distortionValuesY);

        // Create a timer to measure reconstruction time
        cv::TickMeter tm;
        tm.start();

        // Track interpolation errors
        int errorCount = 0;

        // Interpolate to fill the complete distortion maps
#pragma omp parallel for collapse(2) reduction(+:errorCount)
        for (int y = 0; y < imageSize.height; ++y) {
            for (int x = 0; x < imageSize.width; ++x) {
                try {
                    // Interpolate distortion values at this pixel
                    double interpolatedX = interpolateX(static_cast<double>(x), static_cast<double>(y));
                    double interpolatedY = interpolateY(static_cast<double>(x), static_cast<double>(y));

                    // Store interpolated values in output maps
                    reconstructedMapX.at<float>(y, x) = static_cast<float>(interpolatedX);
                    reconstructedMapY.at<float>(y, x) = static_cast<float>(interpolatedY);
                }
                catch (const std::exception& e) {
                    // Handle interpolation errors at specific points
                    errorCount++;

                    // Find the nearest grid point as a fallback
                    double minDist = std::numeric_limits<double>::max();
                    cv::Point2f nearestValue(0.0f, 0.0f);

                    for (const auto& entry : gridDistortionMap) {
                        double dist = std::sqrt(std::pow(entry.first.x - x, 2) +
                            std::pow(entry.first.y - y, 2));
                        if (dist < minDist) {
                            minDist = dist;
                            nearestValue = entry.second;
                        }
                    }

                    // Use nearest neighbor value
                    reconstructedMapX.at<float>(y, x) = nearestValue.x;
                    reconstructedMapY.at<float>(y, x) = nearestValue.y;

                    // Only print errors in debug builds to avoid flooding output
#ifdef _DEBUG
                    if (errorCount < 10) { // Limit the number of error messages
                        std::cout << "Error at (" << x << "," << y << "): " << e.what() << std::endl;
                    }
#endif
                }
            }
        }

        tm.stop();

        // Log completion
        std::cout << "Reconstruction complete using Thin Plate Spline interpolation" << std::endl;
        std::cout << "  - Grid points used: " << gridSize << std::endl;
        std::cout << "  - Output map size: " << imageSize.width << "x" << imageSize.height << std::endl;
        std::cout << "  - Reconstruction time: " << tm.getTimeMilli() << " ms" << std::endl;

        if (errorCount > 0) {
            std::cout << "  - Interpolation errors: " << errorCount << " pixels ("
                << (errorCount * 100.0 / (imageSize.width * imageSize.height)) << "%)" << std::endl;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during TPS reconstruction: " << e.what() << std::endl;
        return false;
    }
}



/**
 * Reconstruct distortion maps from grid points and evaluate against ground truth
 *
 * @param mapX_GT Ground truth X distortion map
 * @param mapY_GT Ground truth Y distortion map
 * @param gridPoints Vector of grid point locations
 * @param reconstructedMapX Output reconstructed X distortion map
 * @param reconstructedMapY Output reconstructed Y distortion map
 * @return Error metrics structure with detailed evaluation results
 */
ReconstructionErrorMetrics ReconstructMap(
    const cv::Mat& mapX_GT,
    const cv::Mat& mapY_GT,
    const std::vector<cv::Point>& gridPoints,
    cv::Mat& reconstructedMapX,
    cv::Mat& reconstructedMapY
) {
    ReconstructionErrorMetrics metrics;
    metrics.errorPixelCount = 0;
    cv::TickMeter tm;

    // Verify input maps
    CV_Assert(mapX_GT.type() == CV_32F && mapY_GT.type() == CV_32F);
    CV_Assert(mapX_GT.size() == mapY_GT.size());
    CV_Assert(!gridPoints.empty());

    // Extract distortion values at grid points
    std::map<cv::Point, cv::Point2f, PointCompare> gridDistortionMap =
        ExtractGridDistortionValues(mapX_GT, mapY_GT, gridPoints);

    // Start timing
    tm.start();

    // Perform reconstruction using existing function
    bool success = ReconstructDistortionMaps(
        gridDistortionMap,
        mapX_GT.size(),
        reconstructedMapX,
        reconstructedMapY
    );

    tm.stop();
    metrics.executionTimeMs = tm.getTimeMilli();

    // Handle reconstruction failure
    if (!success) {
        std::cerr << "Reconstruction failed for grid with " << gridPoints.size() << " points" << std::endl;
        metrics.meanError = -1.0;
        metrics.maxError = -1.0;
        metrics.rmse = -1.0;
        metrics.psnr = -1.0;
        metrics.errorMap = cv::Mat();
        return metrics;
    }

    // Calculate error metrics
    cv::Mat errorX, errorY, errorMagnitude;

    // Compute absolute differences between ground truth and reconstruction
    cv::absdiff(mapX_GT, reconstructedMapX, errorX);
    cv::absdiff(mapY_GT, reconstructedMapY, errorY);

    // Compute error magnitude (Euclidean distance in distortion space)
    cv::sqrt(errorX.mul(errorX) + errorY.mul(errorY), errorMagnitude);

    // Calculate mean error
    cv::Scalar meanErrorScalar = cv::mean(errorMagnitude);
    metrics.meanError = meanErrorScalar[0];

    // Calculate max error
    double minError, maxError;
    cv::minMaxLoc(errorMagnitude, &minError, &maxError);
    metrics.maxError = maxError;

    // Calculate RMSE
    cv::Scalar mseScalar = cv::mean(errorMagnitude.mul(errorMagnitude));
    metrics.rmse = std::sqrt(mseScalar[0]);

    // Calculate PSNR
    double maxVal = 0.0;
    cv::minMaxLoc(mapX_GT, nullptr, &maxVal);
    double maxY;
    cv::minMaxLoc(mapY_GT, nullptr, &maxY);
    maxVal = std::max(maxVal, maxY);

    if (metrics.rmse > 0.0) {
        metrics.psnr = 20.0 * std::log10(maxVal / metrics.rmse);
    }
    else {
        metrics.psnr = std::numeric_limits<double>::infinity();
    }

    // Create visualization-ready error map
    cv::Mat errorMapColorized;
    cv::Mat errorMapNormalized;

    // Normalize error to 0-255 range
    cv::normalize(errorMagnitude, errorMapNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply colormap for better visualization
    cv::applyColorMap(errorMapNormalized, errorMapColorized, cv::COLORMAP_JET);

    // Add error value text overlay
    std::stringstream ss;
    ss << "RMSE: " << std::fixed << std::setprecision(5) << metrics.rmse;
    cv::putText(errorMapColorized, ss.str(), cv::Point(20, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    ss.str("");
    ss << "Mean: " << std::fixed << std::setprecision(5) << metrics.meanError;
    cv::putText(errorMapColorized, ss.str(), cv::Point(20, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    ss.str("");
    ss << "Max: " << std::fixed << std::setprecision(5) << metrics.maxError;
    cv::putText(errorMapColorized, ss.str(), cv::Point(20, 90),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Add grid point count
    ss.str("");
    ss << "Points: " << gridPoints.size();
    cv::putText(errorMapColorized, ss.str(), cv::Point(20, 120),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Store the visualization in metrics
    metrics.errorMap = errorMapColorized;

    return metrics;
}

#pragma once
