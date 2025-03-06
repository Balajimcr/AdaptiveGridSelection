// Adaptive_GDC.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FishEyeEffect.h"
#include "GenerateGrid.h"
#include "utils.h"
#include "GenerateGridv1.h"
#include "ReconstructMap.h"

void TestFishEyeEffect() {
    cv::Size imageSize(1280, 720);
    FisheyeEffect fisheye(imageSize);
    double distortionStrength = 2.5;

    cv::Mat mapX, mapY, normalizedMagnitude;
    fisheye.generateDistortionMaps(distortionStrength, mapX, mapY, normalizedMagnitude);

    // Use mapX, mapY, and normalizedMagnitude as needed
    cv::imshow("Normalized Magnitude", normalizedMagnitude);
    cv::waitKey(0);
}

void TestAdaptiveGridGeneration() {
    const cv::Size imageSize(1280, 720);
    const int gridX = 30, gridY = 30;
    const int gridX_FG = 33, gridY_FG = 33;
    const float GradientLowThreshold = 0.75;
    const float DistorstionStrength = 2.75;
    int MAX_LEVEL = 6;

    // Initialize fisheye distortion
    FisheyeEffect distorter(imageSize);

    // Compute distortion magnitude
    cv::Mat mapX, mapY;
    cv::Mat distortionMagnitude;

    distorter.generateDistortionMaps(DistorstionStrength, mapX, mapY, distortionMagnitude);

    // Convert to 8-bit for visualization
    cv::Mat distortionMagnitude_8U;
    convertScaleAbs(distortionMagnitude, distortionMagnitude_8U, 255);

    cv::TickMeter tm_FM, tm_HAG, tm_v1;

    // Generate grid points
    std::vector<cv::Point> fixedGridPoints, adaptiveGridPoints, adaptiveGridPoints_v1;
    tm_FM.start();
    Generate_FixedGrid(distortionMagnitude, fixedGridPoints, gridX_FG, gridY_FG);
    tm_FM.stop();
    tm_HAG.start();
    GenerateAdaptiveGrid_HAG(distortionMagnitude, adaptiveGridPoints, gridX, gridY, GradientLowThreshold);
    tm_HAG.stop();

    tm_v1.start();
    GenerateAdaptiveGrid_v1(distortionMagnitude, adaptiveGridPoints_v1, GradientLowThreshold);
    tm_v1.stop();
    
    std::cout << "[Timing] Time Taken for Fixed Grid: " << tm_FM.getTimeMilli() << " ms, Total Points: " << fixedGridPoints.size() << std::endl;
    std::cout << "[Timing] Time Taken for HAG   Grid: " << tm_HAG.getTimeMilli()<< " ms, Total Points: " << adaptiveGridPoints.size() << std::endl;
    std::cout << "[Timing] Time Taken for RD(V1)Grid: " << tm_v1.getTimeMilli() << " ms, Total Points: " << adaptiveGridPoints_v1.size() << std::endl;

    // Visualize and draw grids
    cv::Mat fixedGridImage, adaptiveGridImage, adaptiveGridImage_v1;
    createGridVisualization(distortionMagnitude_8U, fixedGridPoints, fixedGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints, adaptiveGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints_v1, adaptiveGridImage_v1);

    // Display and save results
    displayAndSaveImage(fixedGridImage, "Fixed Grid Map");
    displayAndSaveImage(adaptiveGridImage, "Adaptive Grid Map");
    displayAndSaveImage(adaptiveGridImage_v1, "Adaptive Grid Map v1");

    cv::waitKey();
}

void TestAdaptiveGridRemapping() {
    const cv::Size imageSize(1280, 720);
    const int gridX = 30, gridY = 30;
    const int gridX_FG = 33, gridY_FG = 33;
    const float GradientLowThreshold = 0.95;
    const float DistorstionStrength = 2.75;
    int MAX_LEVEL = 6;

    // Initialize fisheye distortion
    FisheyeEffect distorter(imageSize);

    // Compute distortion magnitude
    cv::Mat mapX, mapY;
    cv::Mat distortionMagnitude;

    distorter.generateDistortionMaps(DistorstionStrength, mapX, mapY, distortionMagnitude);

    // Convert to 8-bit for visualization
    cv::Mat distortionMagnitude_8U;
    convertScaleAbs(distortionMagnitude, distortionMagnitude_8U, 255);

    cv::TickMeter tm_FM, tm_HAG, tm_v1;

    // Generate grid points
    std::vector<cv::Point> fixedGridPoints, adaptiveGridPoints, adaptiveGridPoints_v1;
    tm_FM.start();
    Generate_FixedGrid(distortionMagnitude, fixedGridPoints, gridX_FG, gridY_FG);
    tm_FM.stop();
    tm_HAG.start();
    GenerateAdaptiveGrid_HAG(distortionMagnitude, adaptiveGridPoints, gridX, gridY, GradientLowThreshold);
    tm_HAG.stop();

    tm_v1.start();
    GenerateAdaptiveGrid_v1(distortionMagnitude, adaptiveGridPoints_v1, GradientLowThreshold);
    tm_v1.stop();

    std::cout << "[Timing] Time Taken for Fixed Grid: " << tm_FM.getTimeMilli() << " ms, Total Points: " << fixedGridPoints.size() << std::endl;
    std::cout << "[Timing] Time Taken for HAG   Grid: " << tm_HAG.getTimeMilli() << " ms, Total Points: " << adaptiveGridPoints.size() << std::endl;
    std::cout << "[Timing] Time Taken for RD(V1)Grid: " << tm_v1.getTimeMilli() << " ms, Total Points: " << adaptiveGridPoints_v1.size() << std::endl;

    // Visualize and draw grids
    cv::Mat fixedGridImage, adaptiveGridImage, adaptiveGridImage_v1;
    createGridVisualization(distortionMagnitude_8U, fixedGridPoints, fixedGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints, adaptiveGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints_v1, adaptiveGridImage_v1);

    // Display and save results
    displayAndSaveImage(fixedGridImage, "Fixed Grid Map");
    displayAndSaveImage(adaptiveGridImage, "Adaptive Grid Map");
    displayAndSaveImage(adaptiveGridImage_v1, "Adaptive Grid Map v1");

    // Image Remapping - Reconstruction and evaluation
    std::cout << "\n=== Distortion Map Reconstruction Evaluation ===\n" << std::endl;

    // Reconstruct and evaluate each grid type
    cv::Mat fixedReconstructedX, fixedReconstructedY;
    cv::Mat adaptiveReconstructedX, adaptiveReconstructedY;
    cv::Mat adaptiveV1ReconstructedX, adaptiveV1ReconstructedY;

    // Fixed Grid
    std::cout << "Evaluating Fixed Grid (" << fixedGridPoints.size() << " points):" << std::endl;
    ReconstructionErrorMetrics fixedMetrics = ReconstructMap(
        mapX, mapY, fixedGridPoints, fixedReconstructedX, fixedReconstructedY);

    // HAG Grid
    std::cout << "\nEvaluating HAG Grid (" << adaptiveGridPoints.size() << " points):" << std::endl;
    ReconstructionErrorMetrics hagMetrics = ReconstructMap(
        mapX, mapY, adaptiveGridPoints, adaptiveReconstructedX, adaptiveReconstructedY);

    // RD(V1) Grid
    std::cout << "\nEvaluating RD(V1) Grid (" << adaptiveGridPoints_v1.size() << " points):" << std::endl;
    ReconstructionErrorMetrics v1Metrics = ReconstructMap(
        mapX, mapY, adaptiveGridPoints_v1, adaptiveV1ReconstructedX, adaptiveV1ReconstructedY);

    // Display error maps
    displayAndSaveImage(fixedMetrics.errorMap, "Fixed Grid Error Map");
    displayAndSaveImage(hagMetrics.errorMap, "HAG Grid Error Map");
    displayAndSaveImage(v1Metrics.errorMap, "RD(V1) Grid Error Map");

    // Create comparison table
    std::cout << "\n=== Reconstruction Results Summary ===\n" << std::endl;
    std::cout << "Grid Type   | Points | RMSE      | Mean Error | Max Error  | PSNR (dB) | Time (ms)" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    printf("%-12s| %-7zu| %-10.5f| %-11.5f| %-11.5f| %-10.2f| %-9.2f\n",
        "Fixed Grid", fixedGridPoints.size(), fixedMetrics.rmse,
        fixedMetrics.meanError, fixedMetrics.maxError,
        fixedMetrics.psnr, fixedMetrics.executionTimeMs);

    printf("%-12s| %-7zu| %-10.5f| %-11.5f| %-11.5f| %-10.2f| %-9.2f\n",
        "HAG Grid", adaptiveGridPoints.size(), hagMetrics.rmse,
        hagMetrics.meanError, hagMetrics.maxError,
        hagMetrics.psnr, hagMetrics.executionTimeMs);

    printf("%-12s| %-7zu| %-10.5f| %-11.5f| %-11.5f| %-10.2f| %-9.2f\n",
        "RD(V1) Grid", adaptiveGridPoints_v1.size(), v1Metrics.rmse,
        v1Metrics.meanError, v1Metrics.maxError,
        v1Metrics.psnr, v1Metrics.executionTimeMs);

    // Create side-by-side comparison of error maps
    cv::Mat comparisonImage(imageSize.height, imageSize.width * 3, CV_8UC3);
    fixedMetrics.errorMap.copyTo(comparisonImage(cv::Rect(0, 0, imageSize.width, imageSize.height)));
    hagMetrics.errorMap.copyTo(comparisonImage(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height)));
    v1Metrics.errorMap.copyTo(comparisonImage(cv::Rect(2 * imageSize.width, 0, imageSize.width, imageSize.height)));

    // Add grid type labels to the top of each error map in the comparison
    cv::putText(comparisonImage, "Fixed Grid",
        cv::Point(imageSize.width / 2 - 60, 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(255, 255, 255), 2);

    cv::putText(comparisonImage, "HAG Grid",
        cv::Point(imageSize.width + imageSize.width / 2 - 60, 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(255, 255, 255), 2);

    cv::putText(comparisonImage, "RD(V1) Grid",
        cv::Point(2 * imageSize.width + imageSize.width / 2 - 60, 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(255, 255, 255), 2);

    displayAndSaveImage(comparisonImage, "Error Map Comparison");

    // Optional: Demonstrate the reconstructed distortion on a test image
    cv::Mat testImage = cv::imread("test_image.jpg");
    if (!testImage.empty()) {
        cv::resize(testImage, testImage, imageSize);

        // Apply original distortion
        cv::Mat originalDistorted;
        cv::remap(testImage, originalDistorted, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        // Apply reconstructed distortions
        cv::Mat fixedDistorted, hagDistorted, v1Distorted;
        cv::remap(testImage, fixedDistorted, fixedReconstructedX, fixedReconstructedY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::remap(testImage, hagDistorted, adaptiveReconstructedX, adaptiveReconstructedY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::remap(testImage, v1Distorted, adaptiveV1ReconstructedX, adaptiveV1ReconstructedY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        // Display results
        displayAndSaveImage(originalDistorted, "Original Distortion");
        displayAndSaveImage(fixedDistorted, "Fixed Grid Distortion");
        displayAndSaveImage(hagDistorted, "HAG Grid Distortion");
        displayAndSaveImage(v1Distorted, "RD(V1) Grid Distortion");

        // Create side-by-side comparison
        cv::Mat distortionComparison(imageSize.height, imageSize.width * 4, CV_8UC3);
        originalDistorted.copyTo(distortionComparison(cv::Rect(0, 0, imageSize.width, imageSize.height)));
        fixedDistorted.copyTo(distortionComparison(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height)));
        hagDistorted.copyTo(distortionComparison(cv::Rect(2 * imageSize.width, 0, imageSize.width, imageSize.height)));
        v1Distorted.copyTo(distortionComparison(cv::Rect(3 * imageSize.width, 0, imageSize.width, imageSize.height)));

        // Add labels
        cv::putText(distortionComparison, "Original",
            cv::Point(imageSize.width / 2 - 50, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(255, 255, 255), 2);

        cv::putText(distortionComparison, "Fixed Grid",
            cv::Point(imageSize.width + imageSize.width / 2 - 60, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(255, 255, 255), 2);

        cv::putText(distortionComparison, "HAG Grid",
            cv::Point(2 * imageSize.width + imageSize.width / 2 - 60, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(255, 255, 255), 2);

        cv::putText(distortionComparison, "RD(V1) Grid",
            cv::Point(3 * imageSize.width + imageSize.width / 2 - 60, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(255, 255, 255), 2);

        displayAndSaveImage(distortionComparison, "Distortion Comparison");
    }

    // Calculate compression ratios
    double originalSize = 2 * imageSize.width * imageSize.height; // X and Y maps
    double fixedGridSize = fixedGridPoints.size();
    double hagGridSize = adaptiveGridPoints.size();
    double v1GridSize = adaptiveGridPoints_v1.size();

    std::cout << "\n=== Compression Analysis ===" << std::endl;
    std::cout << "Original Map Size: " << originalSize / 1024.0 << " KB" << std::endl;
    std::cout << "Fixed Grid Size: " << fixedGridSize / 1024.0 << " KB (Ratio: "
        << originalSize / fixedGridSize << ":1)" << std::endl;
    std::cout << "HAG Grid Size: " << hagGridSize / 1024.0 << " KB (Ratio: "
        << originalSize / hagGridSize << ":1)" << std::endl;
    std::cout << "RD(V1) Grid Size: " << v1GridSize / 1024.0 << " KB (Ratio: "
        << originalSize / v1GridSize << ":1)" << std::endl;

    // Detailed compression ratio table
    std::cout << "\n=== Detailed Compression Ratio ===" << std::endl;
    std::cout << "Grid Type   | Points | Original Size (KB) | Grid Size (KB) | Compression Ratio" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    printf("%-12s| %-7zu| %-19.2f| %-14.2f| %-17.2f\n",
        "Fixed Grid", fixedGridPoints.size(), originalSize / 1024.0,
        fixedGridSize / 1024.0, originalSize / fixedGridSize);

    printf("%-12s| %-7zu| %-19.2f| %-14.2f| %-17.2f\n",
        "HAG Grid", adaptiveGridPoints.size(), originalSize / 1024.0,
        hagGridSize / 1024.0, originalSize / hagGridSize);

    printf("%-12s| %-7zu| %-19.2f| %-14.2f| %-17.2f\n",
        "RD(V1) Grid", adaptiveGridPoints_v1.size(), originalSize / 1024.0,
        v1GridSize / 1024.0, originalSize / v1GridSize);

    cv::waitKey();
}


int main() {
    //TestAdaptiveGridGeneration();
	TestAdaptiveGridRemapping();

    return 0;
}