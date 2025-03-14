// Adaptive_GDC.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <filesystem>
#include <fstream>
#include "FishEyeEffect.h"
#include "GenerateGrid.h"
#include "utils.h"
#include "GenerateGridv1.h"
#include "GenerateGridv2.h"
#include "ReconstructMap.h"
#include "TestImageReconstruct.hpp"

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
    const cv::Size imageSize(3840,2160);
    const int gridX = 30, gridY = 30;
    const int gridX_FG = 33, gridY_FG = 33;
    const float GradientLowThreshold = 0.85;
    const float DistorstionStrength = 2.75;

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
    HAG::Generate_FixedGrid(distortionMagnitude, fixedGridPoints, gridX_FG, gridY_FG);
    tm_FM.stop();
    tm_HAG.start();
    HAG::GenerateAdaptiveGrid_HAG(distortionMagnitude, adaptiveGridPoints, gridX, gridY, GradientLowThreshold);
    tm_HAG.stop();

    tm_v1.start();
    HAG::GenerateAdaptiveGrid_HAG2(distortionMagnitude, adaptiveGridPoints_v1, gridX,gridY,GradientLowThreshold);
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
    displayAndSaveImage(adaptiveGridImage_v1, "Adaptive Grid Map HAG2");

    cv::waitKey();
}

void TestAdaptiveGridRemapping1() {
    const cv::Size imageSize(1280, 720);
    const int gridX = 30, gridY = 30;
    const int gridX_FG = 33, gridY_FG = 33;
    const float GradientLowThreshold = 0.95f;
    const float DistorstionStrength = 2.75;

    // Create Outputs directory if it doesn't exist
    create_directory("Outputs");

    // Create a single CSV file for all outputs
    std::ofstream outputCSV("Outputs/grid_analysis_results.csv", std::ios::out);

    // Initialize fisheye distortion
    FisheyeEffect distorter(imageSize);

    // Compute distortion magnitude
    cv::Mat mapX, mapY;
    cv::Mat distortionMagnitude;

    distorter.generateDistortionMaps(DistorstionStrength, mapX, mapY, distortionMagnitude);

    // Convert to 8-bit for visualization
    cv::Mat distortionMagnitude_8U;
    convertScaleAbs(distortionMagnitude, distortionMagnitude_8U, 255);

    cv::TickMeter tm_FM, tm_HAG, tm_v1, tm_v2;

    // Generate grid points
    std::vector<cv::Point> fixedGridPoints, adaptiveGridPoints, adaptiveGridPoints_v1, adaptiveGridPoints_v2;
    tm_FM.start();
    HAG::Generate_FixedGrid(distortionMagnitude, fixedGridPoints, gridX_FG, gridY_FG);
    tm_FM.stop();
    tm_HAG.start();
    HAG::GenerateAdaptiveGrid_HAG(distortionMagnitude, adaptiveGridPoints, gridX, gridY, GradientLowThreshold);
    tm_HAG.stop();

    tm_v1.start();
    HAG::GenerateAdaptiveGrid_HAG2(distortionMagnitude, adaptiveGridPoints_v1, gridX, gridY, GradientLowThreshold);
    tm_v1.stop();

    tm_v2.start();
    GenerateAdaptiveGrid_v1(distortionMagnitude, adaptiveGridPoints_v2, GradientLowThreshold);
    tm_v2.stop();

    std::cout << "[Timing] Time Taken for Fixed Grid: " << tm_FM.getTimeMilli() << " ms, Total Points: " << fixedGridPoints.size() << std::endl;
    std::cout << "[Timing] Time Taken for HAG   Grid: " << tm_HAG.getTimeMilli() << " ms, Total Points: " << adaptiveGridPoints.size() << std::endl;
    std::cout << "[Timing] Time Taken for HAG2  Grid: " << tm_v1.getTimeMilli() << " ms, Total Points: " << adaptiveGridPoints_v1.size() << std::endl;
    std::cout << "[Timing] Time Taken for HAG_V2Grid: " << tm_v2.getTimeMilli() << " ms, Total Points: " << adaptiveGridPoints_v2.size() << std::endl;

    // Visualize and draw grids
    cv::Mat fixedGridImage, adaptiveGridImage, adaptiveGridImage_v1, adaptiveGridImage_v2;
    createGridVisualization(distortionMagnitude_8U, fixedGridPoints, fixedGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints, adaptiveGridImage);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints_v1, adaptiveGridImage_v1);
    createGridVisualization(distortionMagnitude_8U, adaptiveGridPoints_v2, adaptiveGridImage_v2);

    // Display and save results
    displayAndSaveImage(fixedGridImage, "Grid_Fixed Grid Map");
    displayAndSaveImage(adaptiveGridImage, "Grid_Adaptive Grid Map");
    displayAndSaveImage(adaptiveGridImage_v1, "Grid_Adaptive Grid Map 2");
    displayAndSaveImage(adaptiveGridImage_v2, "Grid_Adaptive Grid Map v2");

    // Image Remapping - Reconstruction and evaluation
    std::cout << "\n=== Distortion Map Reconstruction Evaluation ===\n" << std::endl;

    // Reconstruct and evaluate each grid type
    cv::Mat fixedReconstructedX, fixedReconstructedY;
    cv::Mat adaptiveReconstructedX, adaptiveReconstructedY;
    cv::Mat adaptiveV1ReconstructedX, adaptiveV1ReconstructedY;
    cv::Mat adaptiveV2ReconstructedX, adaptiveV2ReconstructedY;

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
    
    // RD(V1) Grid
    std::cout << "\nEvaluating RD(V2) Grid (" << adaptiveGridPoints_v2.size() << " points):" << std::endl;
    ReconstructionErrorMetrics v2Metrics = ReconstructMap(
        mapX, mapY, adaptiveGridPoints_v2, adaptiveV2ReconstructedX, adaptiveV2ReconstructedY);

    // Display error maps
    displayAndSaveImage(fixedMetrics.errorMap, "ReConError_Fixed Grid Error Map");
    displayAndSaveImage(hagMetrics.errorMap, "ReConError_HAG Grid Error Map");
    displayAndSaveImage(v1Metrics.errorMap, "ReConError_RD(V1) Grid Error Map");
    displayAndSaveImage(v2Metrics.errorMap, "ReConError_RD(V2) Grid Error Map");

    {
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
        printf("%-12s| %-7zu| %-10.5f| %-11.5f| %-11.5f| %-10.2f| %-9.2f\n",
            "RD(V2) Grid", adaptiveGridPoints_v1.size(), v2Metrics.rmse,
            v2Metrics.meanError, v2Metrics.maxError,
            v2Metrics.psnr, v2Metrics.executionTimeMs);
    }

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

    displayAndSaveImage(comparisonImage, "ReCon_Error Map Comparison");

    // Optional: Demonstrate the reconstructed distortion on a test image
    cv::Mat testImage(imageSize,CV_8UC3);
    DrawGrid(testImage);
    if (!testImage.empty()) {
        cv::resize(testImage, testImage, imageSize);

        // Apply original distortion
        cv::Mat originalDistorted;
        cv::remap(testImage, originalDistorted, mapX, mapY, cv::INTER_CUBIC, cv::BORDER_REFLECT);

        // Apply reconstructed distortions
        cv::Mat fixedDistorted, hagDistorted, v1Distorted;
        cv::remap(testImage, fixedDistorted, fixedReconstructedX, fixedReconstructedY, cv::INTER_CUBIC, cv::BORDER_REFLECT);
        cv::remap(testImage, hagDistorted, adaptiveReconstructedX, adaptiveReconstructedY, cv::INTER_CUBIC, cv::BORDER_REFLECT);
        cv::remap(testImage, v1Distorted, adaptiveV1ReconstructedX, adaptiveV1ReconstructedY, cv::INTER_CUBIC, cv::BORDER_REFLECT);

        // Display results
        displayAndSaveImage(originalDistorted, "ReCon_Original Distortion");
        displayAndSaveImage(fixedDistorted, "ReCon_Fixed Grid Distortion");
        displayAndSaveImage(hagDistorted, "ReCon_HAG Grid Distortion");
        displayAndSaveImage(v1Distorted, "ReCon_RD(V1) Grid Distortion");

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

        displayAndSaveImage(distortionComparison, "ReCon_Distortion Comparison");
    }

    // Calculate compression ratios
    double originalSize = 2 * imageSize.width * imageSize.height; // X and Y maps
    double fixedGridSize = fixedGridPoints.size();
    double hagGridSize = adaptiveGridPoints.size();
    double v1GridSize = adaptiveGridPoints_v1.size();

    {
        // Detailed compression ratio table
        std::cout << "\n=== Compression Ratio ===" << std::endl;
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
    }

    if (outputCSV.is_open()) {
        // Write timing section to CSV
        outputCSV << "TIMING RESULTS\n";
        outputCSV << "Grid Type,Time (ms),Total Points\n";
        outputCSV << "Fixed Grid," << tm_FM.getTimeMilli() << "," << fixedGridPoints.size() << "\n";
        outputCSV << "HAG Grid," << tm_HAG.getTimeMilli() << "," << adaptiveGridPoints.size() << "\n";
        outputCSV << "RD(V1) Grid," << tm_v1.getTimeMilli() << "," << adaptiveGridPoints_v1.size() << "\n\n";

        // Write reconstruction results section to CSV
        outputCSV << "RECONSTRUCTION RESULTS\n";
        outputCSV << "Grid Type,Points,RMSE,Mean Error,Max Error,PSNR (dB),Time (ms)\n";
        outputCSV << "Fixed Grid," << fixedGridPoints.size() << ","
            << fixedMetrics.rmse << "," << fixedMetrics.meanError << ","
            << fixedMetrics.maxError << "," << fixedMetrics.psnr << ","
            << fixedMetrics.executionTimeMs << "\n";
        outputCSV << "HAG Grid," << adaptiveGridPoints.size() << ","
            << hagMetrics.rmse << "," << hagMetrics.meanError << ","
            << hagMetrics.maxError << "," << hagMetrics.psnr << ","
            << hagMetrics.executionTimeMs << "\n";
        outputCSV << "RD(V1) Grid," << adaptiveGridPoints_v1.size() << ","
            << v1Metrics.rmse << "," << v1Metrics.meanError << ","
            << v1Metrics.maxError << "," << v1Metrics.psnr << ","
            << v1Metrics.executionTimeMs << "\n\n";

        // Write compression results section to CSV
        outputCSV << "COMPRESSION RESULTS\n";
        outputCSV << "Grid Type,Points,Original Size (KB),Grid Size (KB),Compression Ratio\n";
        outputCSV << "Fixed Grid," << fixedGridPoints.size() << ","
            << originalSize / 1024.0 << "," << fixedGridSize / 1024.0 << ","
            << originalSize / fixedGridSize << "\n";
        outputCSV << "HAG Grid," << adaptiveGridPoints.size() << ","
            << originalSize / 1024.0 << "," << hagGridSize / 1024.0 << ","
            << originalSize / hagGridSize << "\n";
        outputCSV << "RD(V1) Grid," << adaptiveGridPoints_v1.size() << ","
            << originalSize / 1024.0 << "," << v1GridSize / 1024.0 << ","
            << originalSize / v1GridSize << "\n";

        // Close the CSV file
        outputCSV.close();
    }

    cv::waitKey();
}


void TestAdaptiveGridRemapping() {
    const cv::Size imageSize(1280, 720);
    const int gridX = 30, gridY = 30;
    const int gridX_FG = 33, gridY_FG = 33;
    const float GradientLowThreshold = 0.95f;
    const float DistortionStrength = 2.75;

    // Create evaluator
    GridEvaluator evaluator(imageSize, DistortionStrength);

    // Setup distortion maps
    evaluator.setupDistortionMaps();

    // Register different grid generation algorithms to test

    // Fixed Grid (takes Mat, vector<Point>, int, int)
    evaluator.registerGridAlgorithm(
        "Fixed Grid",
        &HAG::Generate_FixedGrid,  // Use function pointer style
        gridX_FG, gridY_FG
    );

    // HAG Grid (takes Mat, vector<Point>, int, int, float)
    evaluator.registerGridAlgorithm(
        "HAG Grid",
        &HAG::GenerateAdaptiveGrid_HAG,
        gridX, gridY,
        GradientLowThreshold
    );

    //// RD(V1) Grid (takes Mat, vector<Point>, float, int)
    //evaluator.registerGridAlgorithm(
    //    "RD(V1) Grid",
    //    &GenerateAdaptiveGrid_v1,
    //    GradientLowThreshold,
    //    6  // Max level
    //);

    // V2 Grid (takes Mat, vector<Point>, int, int, float)
    evaluator.registerGridAlgorithm(
        "RD(V2) Grid",
        &HAG::GenerateAdaptiveGrid_HAG2,
        gridX, gridY,
        GradientLowThreshold
    );

    // Evaluate all registered algorithms
    evaluator.evaluateAllGrids();

    // Wait for user input
    cv::waitKey();
}


int main() {
    TestAdaptiveGridGeneration();
	//TestAdaptiveGridRemapping();

    return 0;
}