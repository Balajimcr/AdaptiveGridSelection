// Adaptive_GDC.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FishEyeEffect.h"
#include "GenerateGrid.h"
#include "utils.h"
#include "GenerateGridv1.h"

void logGridStatistics(int fixedPoints, int adaptivePoints, int TimeTakenms_FG, int TimeTakenms_AG, std::string AdaptiveGridPointsMethod) {
    int pointsDiff = fixedPoints - adaptivePoints;
    int TimeDiff = TimeTakenms_FG - TimeTakenms_AG;
    double savedPercentage = static_cast<double>(pointsDiff) / fixedPoints * 100;
    double savedPercentage_Time = static_cast<double>(TimeDiff) / TimeTakenms_FG * 100;

    std::cout << "Fixed Grid Points: " << fixedPoints
        << AdaptiveGridPointsMethod << " : " << adaptivePoints
        << ", Points Saved: " << pointsDiff
        << " (" << savedPercentage << "%)"
        << " ( Time Taken : " << TimeTakenms_AG << "ms)"
        << " ( Time Saved : " << savedPercentage_Time << "%)"
        << std::endl;
}

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
    GenerateAdaptiveGrid_v1(distortionMagnitude, adaptiveGridPoints_v1, MAX_LEVEL, GradientLowThreshold);
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


int main() {
    TestAdaptiveGridGeneration();

    return 0;
}