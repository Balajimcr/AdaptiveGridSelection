// Adaptive_GDC.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <filesystem>
#include <fstream>
#include "FishEyeEffect.h"
#include "GenerateGrid.h"
#include "GenerateGridv2.h"
#include "utils.h"
#include "ReconstructMap.h"
#include "TestImageReconstruct.hpp"

void TestFishEyeEffect() {
    cv::Size imageSize(1280, 720);
    FisheyeEffect fisheye(imageSize);
    float motionStrength = 1.0f;
    int smoothingWindowSize = 15;

    cv::Mat mapX, mapY, normalizedMagnitude;
    fisheye.generateDistortionMapsCAMStab(mapX, mapY, normalizedMagnitude,motionStrength,smoothingWindowSize);

    // Create color image from grayscale magnitude map
    cv::Mat distortionMagnitude_Color;
    cv::convertScaleAbs(normalizedMagnitude, normalizedMagnitude, 255);
    cv::applyColorMap(normalizedMagnitude, distortionMagnitude_Color, cv::COLORMAP_JET);

    // Use mapX, mapY, and normalizedMagnitude as needed
    cv::imshow("Normalized Magnitude", distortionMagnitude_Color);
    cv::waitKey();
}

void TestAdaptiveGridGeneration() {
    // Test parameters
    const cv::Size imageSize(1280, 720);
    const int gridX = 30, gridY = 30;
    const int fixedGridX = 33, fixedGridY = 33;
    const float distortionThreshold = 0.85f;
    const float distortionStrength = 2.75f;
    const int numIterations = 20;

    std::cout << "=== Testing Adaptive Grid Generation ===" << std::endl;
    std::cout << "Image Size: " << imageSize.width << "x" << imageSize.height << std::endl;
    std::cout << "Distortion Strength: " << distortionStrength << std::endl;
    std::cout << "Threshold: " << distortionThreshold << std::endl << std::endl;

    // Initialize fisheye distortion
    FisheyeEffect distorter(imageSize);

    // Compute distortion maps and magnitude
    cv::Mat mapX, mapY, distortionMagnitude;
    distorter.generateDistortionMaps(distortionStrength, mapX, mapY, distortionMagnitude);

    // Convert to 8-bit for visualization
    cv::Mat distortionMagnitude_8U;
    cv::convertScaleAbs(distortionMagnitude, distortionMagnitude_8U, 255);

    // Apply colormap for better visualization
    cv::Mat distortionHeatmap;
    cv::applyColorMap(distortionMagnitude_8U, distortionHeatmap, cv::COLORMAP_JET);
    displayAndSaveImage(distortionHeatmap, "Distortion Magnitude Map");

    // Timing and point storage
    cv::TickMeter tm_FixedGrid, tm_HAG, tm_HAG_v1;
    std::vector<cv::Point> fixedGridPoints, HAG_Points, HAG_v1_Points;

    // Accumulate total time
    double totalTime_FixedGrid = 0, totalTime_HAG = 0, totalTime_HAG_v1 = 0;

    // Pre-allocate vectors
    fixedGridPoints.reserve(fixedGridX * fixedGridY);
    HAG_Points.reserve(gridX * gridY * 2);
    HAG_v1_Points.reserve(gridX * gridY * 2); // Estimate maximum points

    // Run tests
    for (int i = 0; i < numIterations; ++i) {
        fixedGridPoints.clear();
        HAG_Points.clear();

        tm_FixedGrid.start();
        HAG::Generate_FixedGrid(distortionMagnitude, fixedGridPoints, fixedGridX, fixedGridY);
        tm_FixedGrid.stop();
        totalTime_FixedGrid += tm_FixedGrid.getTimeMilli();
        tm_FixedGrid.reset();

        tm_HAG.start();
        HAG::GenerateAdaptiveGrid_HAG(distortionMagnitude, HAG_Points, gridX, gridY, distortionThreshold);
        tm_HAG.stop();
        totalTime_HAG += tm_HAG.getTimeMilli();
        tm_HAG.reset();

        tm_HAG_v1.start();
        HAG::GenerateAdaptiveGrid_HAG_v1(distortionMagnitude, HAG_v1_Points, gridX, gridY, distortionThreshold);
        tm_HAG_v1.stop();
        totalTime_HAG_v1 += tm_HAG_v1.getTimeMilli();
        tm_HAG_v1.reset();
    }

    // Calculate average times
    double avgTime_FixedGrid = totalTime_FixedGrid / numIterations;
    double avgTime_HAG = totalTime_HAG / numIterations;
    double avgTime_HAG_v1 = totalTime_HAG_v1 / numIterations;
    
    // Count adaptive points
    int adaptivePointsHAG = static_cast<int>(HAG_Points.size()) - gridX * gridY;
    int adaptivePointsHAG_v1 = static_cast<int>(HAG_v1_Points.size()); // All points are adaptive in v1

    // Calculate efficiency metrics - points per millisecond
    double fixedGridEfficiency = fixedGridPoints.size() / std::max(0.1, avgTime_FixedGrid);
    double HAGEfficiency = HAG_Points.size() / std::max(0.1, avgTime_HAG);
    double HAG_v1Efficiency = HAG_v1_Points.size() / std::max(0.1, avgTime_HAG_v1);

    // Print comparison table
    std::cout << "=== Grid Generation Comparison (over " << numIterations << " runs) ===" << std::endl;
    std::cout << std::left << std::setw(15) << "Method" << std::setw(15) << "Avg Time (ms)" << std::setw(15) << "Total Points" << std::setw(15) << "Adaptive Pts" << std::setw(15) << "Points/ms" << std::endl;
    std::cout << std::left << std::setw(15) << "Fixed Grid" << std::setw(15) << std::fixed << std::setprecision(3) << avgTime_FixedGrid << std::setw(15) << fixedGridPoints.size() << std::setw(15) << "N/A" << std::setw(15) << std::fixed << std::setprecision(2) << fixedGridEfficiency << std::endl;
    std::cout << std::left << std::setw(15) << "HAG" << std::setw(15) << std::fixed << std::setprecision(3) << avgTime_HAG << std::setw(15) << HAG_Points.size() << std::setw(15) << adaptivePointsHAG << std::setw(15) << std::fixed << std::setprecision(2) << HAGEfficiency << std::endl;
    std::cout << std::left << std::setw(15) << "HAG_v1" << std::setw(15) << std::fixed << std::setprecision(3) << avgTime_HAG_v1 << std::setw(15) << HAG_v1_Points.size() << std::setw(15) << adaptivePointsHAG_v1 << std::setw(15) << std::fixed << std::setprecision(2) << HAG_v1Efficiency << std::endl;

    // Helper function to create grid visualization with colored points
    auto createEnhancedGridVisualization = [&distortionMagnitude_8U](const std::vector<cv::Point>& points, const std::string& title, int baseGridSize) {
        cv::Mat output;
        cv::cvtColor(distortionMagnitude_8U, output, cv::COLOR_GRAY2BGR);
        cv::applyColorMap(output, output, cv::COLORMAP_JET);

        for (size_t i = 0; i < points.size(); i++) {
            cv::Scalar color = (i < baseGridSize) ? cv::Scalar(255, 0, 0) :
                cv::Scalar(0, 255, 255 * (1.0f - std::min(1.0f, static_cast<float>(i - baseGridSize) / std::max(1.0f, static_cast<float>(points.size() - baseGridSize)))));
            cv::circle(output, points[i], 1, color, 2, cv::LINE_AA);
        }

        cv::putText(output, title, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        cv::putText(output, "Points: " + std::to_string(points.size()) + ((points.size() > baseGridSize) ? " (" + std::to_string(points.size() - baseGridSize) + " adaptive)" : ""),
            cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        return output;
        };

    // Generate visualizations
    cv::Mat fixedGridImage = createEnhancedGridVisualization(fixedGridPoints, "Fixed Grid", fixedGridX * fixedGridY);
    cv::Mat HAGImage = createEnhancedGridVisualization(HAG_Points, "HAG", gridX * gridY);
    cv::Mat HAGv1Image = createEnhancedGridVisualization(HAG_v1_Points, "HAG v1", gridX * gridY);

    // Display and save results
    displayAndSaveImage(fixedGridImage, "Fixed Grid Map");
    displayAndSaveImage(HAGImage, "HAG Grid Map");
    displayAndSaveImage(HAGv1Image, "HAG Grid v1 Map");
}

void TestAdaptiveGridRemapping() {
    // Test parameters
    const cv::Size imageSize(1280, 720);
    const int gridX = 30, gridY = 30;
    const int fixedGridX = 33, fixedGridY = 33;
    const float GradientLowThreshold = 0.85f;
    const float DistortionStrength = 2.75f;
    const int numIterations = 50;

    // Create evaluator
    GridEvaluator evaluator(imageSize, DistortionStrength);

    // Setup distortion maps
    evaluator.setupDistortionMaps();

    // Register different grid generation algorithms to test

    // Fixed Grid (takes Mat, vector<Point>, int, int)
    evaluator.registerGridAlgorithm(
        "Fixed Grid",
        &HAG::Generate_FixedGrid,  // Use function pointer style
        fixedGridX, fixedGridY
    );

    // HAG Grid (takes Mat, vector<Point>, int, int, float)
    evaluator.registerGridAlgorithm(
        "HAG Grid",
        &HAG::GenerateAdaptiveGrid_HAG,
        gridX, gridY,
        GradientLowThreshold
    );

    // HAG Grid (takes Mat, vector<Point>, int, int, float)
    evaluator.registerGridAlgorithm(
        "HAGv1Grid",
        &HAG::GenerateAdaptiveGrid_HAG_v1,
        gridX, gridY,
        GradientLowThreshold
    );

    //// V2 Grid (takes Mat, vector<Point>, int, int, float)
    //evaluator.registerGridAlgorithm(
    //    "RD(V2) Grid",
    //    &HAG::GenerateAdaptiveGrid_HAG_v2,
    //    gridX, gridY,
    //    GradientLowThreshold
    //);

    // Evaluate all registered algorithms
    evaluator.evaluateAllGrids();

    // Wait for user input
    cv::waitKey();
}


int main() {
    TestAdaptiveGridGeneration();
	//TestAdaptiveGridRemapping();
    //TestFishEyeEffect();

    // Wait for user input
    cv::waitKey();
    return 0;
}