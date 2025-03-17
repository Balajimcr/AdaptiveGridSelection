#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <functional>
#include <filesystem>

#include "FishEyeEffect.hpp"
#include "GenerateGrid.hpp"
#include "utils.hpp"
#include "ReconstructMap.hpp"

// Structure to store grid generation results
struct GridGenerationResult {
    std::string name;                  // Name of the grid generation method
    std::vector<cv::Point> gridPoints; // Generated grid points
    double generationTimeMs;           // Time taken to generate the grid
    cv::Mat visualizationImage;        // Visualization of the grid
};

// Structure to store reconstruction results
struct ReconstructionResult {
    std::string gridName;              // Name of the grid used
    size_t pointCount;                 // Number of points in the grid
    ReconstructionErrorMetrics metrics; // Error metrics from reconstruction
    cv::Mat reconstructedMapX;          // Reconstructed X map
    cv::Mat reconstructedMapY;          // Reconstructed Y map
    cv::Mat distortedImage;             // Example distorted image
};

/**
 * Class for evaluating different grid generation algorithms
 */
class GridEvaluator {
private:
    cv::Size imageSize;                                  // Size of the distortion maps
    float distortionStrength;                           // Strength of distortion for testing
    cv::Mat mapX;                                       // Original X distortion map
    cv::Mat mapY;                                       // Original Y distortion map
    cv::Mat distortionMagnitude;                        // Magnitude of distortion
    cv::Mat distortionMagnitude_8U;                     // 8-bit version for visualization
    std::vector<GridGenerationResult> gridResults;      // Results from different grid algorithms
    std::vector<ReconstructionResult> reconstructionResults; // Results from reconstruction
    cv::Mat testImage;                                  // Test image for visualization

    // Directory for output files
    std::string outputDirectory = "Outputs";

public:
    /**
     * Constructor
     * @param size Image size for testing
     * @param strength Distortion strength
     */
    GridEvaluator(const cv::Size& size = cv::Size(1280, 720), float strength = 2.75)
        : imageSize(size), distortionStrength(strength) {

        // Create output directory
        create_directory(outputDirectory.c_str());

        // Initialize test image
        initializeTestImage();
    }

    /**
     * Setup distortion maps for testing
     */
    void setupDistortionMaps() {
        // Initialize fisheye distortion
        FisheyeEffect distorter(imageSize);

        // Generate distortion maps
        distorter.generateDistortionMaps(distortionStrength, mapX, mapY, distortionMagnitude);

        // Convert magnitude to 8-bit for visualization
        convertScaleAbs(distortionMagnitude, distortionMagnitude_8U, 255);
    }

    /**
     * Initialize a test image
     */
    void initializeTestImage() {
        testImage = cv::Mat(imageSize, CV_8UC3);
        DrawGrid(testImage);
    }

    /**
     * Register a grid generation algorithm for testing - Fixed Grid version
     */
    void registerGridAlgorithm(
        const std::string& name,
        void (*generator)(const cv::Mat&, std::vector<cv::Point>&, int, int),
        int param1, int param2
    ) {
        // Create result container
        GridGenerationResult result;
        result.name = name;

        // Time the grid generation
        cv::TickMeter tm;
        tm.start();

        // Call the generator function with parameters
        generator(distortionMagnitude, result.gridPoints, param1, param2);

        tm.stop();
        result.generationTimeMs = tm.getTimeMilli();

        // Create visualization
        createGridVisualization(distortionMagnitude_8U, result.gridPoints, result.visualizationImage);

        // Add to results
        gridResults.push_back(result);

        // Log
        std::cout << "[Timing] " << name << ": " << result.generationTimeMs
            << " ms, Total Points: " << result.gridPoints.size() << std::endl;
    }

    /**
     * Register a grid generation algorithm for testing - HAG Grid version
     */
    void registerGridAlgorithm(
        const std::string& name,
        void (*generator)(const cv::Mat&, std::vector<cv::Point>&, int, int, float),
        int param1, int param2, float param3
    ) {
        // Create result container
        GridGenerationResult result;
        result.name = name;

        // Time the grid generation
        cv::TickMeter tm;
        tm.start();

        // Call the generator function with parameters
        generator(distortionMagnitude, result.gridPoints, param1, param2, param3);

        tm.stop();
        result.generationTimeMs = tm.getTimeMilli();

        // Create visualization
        createGridVisualization(distortionMagnitude_8U, result.gridPoints, result.visualizationImage);

        // Add to results
        gridResults.push_back(result);

        // Log
        std::cout << "[Timing] " << name << ": " << result.generationTimeMs
            << " ms, Total Points: " << result.gridPoints.size() << std::endl;
    }

    /**
     * Register a grid generation algorithm for testing - RDV1 Grid version
     */
    void registerGridAlgorithm(
        const std::string& name,
        void (*generator)(const cv::Mat&, std::vector<cv::Point>&, float, int),
        float param1, int param2
    ) {
        // Create result container
        GridGenerationResult result;
        result.name = name;

        // Time the grid generation
        cv::TickMeter tm;
        tm.start();

        // Call the generator function with parameters
        generator(distortionMagnitude, result.gridPoints, param1, param2);

        tm.stop();
        result.generationTimeMs = tm.getTimeMilli();

        // Create visualization
        createGridVisualization(distortionMagnitude_8U, result.gridPoints, result.visualizationImage);

        // Add to results
        gridResults.push_back(result);

        // Log
        std::cout << "[Timing] " << name << ": " << result.generationTimeMs
            << " ms, Total Points: " << result.gridPoints.size() << std::endl;
    }

    /**
     * Run reconstruction and evaluation for all registered grid algorithms
     */
    void evaluateAllGrids() {
        std::cout << "\n=== Distortion Map Reconstruction Evaluation ===\n" << std::endl;

        // Evaluate each grid
        for (const auto& gridResult : gridResults) {
            ReconstructionResult result;
            result.gridName = gridResult.name;
            result.pointCount = gridResult.gridPoints.size();

            std::cout << "\nEvaluating " << gridResult.name << " ("
                << gridResult.gridPoints.size() << " points):" << std::endl;

            // Perform reconstruction
            result.metrics = ReconstructMap(
                mapX, mapY, gridResult.gridPoints,
                result.reconstructedMapX, result.reconstructedMapY);

            // Apply reconstruction to test image
            cv::remap(testImage, result.distortedImage,
                result.reconstructedMapX, result.reconstructedMapY,
                cv::INTER_CUBIC, cv::BORDER_REFLECT);

            // Save to results
            reconstructionResults.push_back(result);

            // Display and save error map
            displayAndSaveImage(result.metrics.errorMap,
                "ReConError_" + gridResult.name + " Error Map");
        }

        // Print summary table
        printReconstructionSummary();

        // Create comparison visualizations
        createErrorMapComparison();
        createDistortionComparison();

        // Compute and print compression ratios
        printCompressionRatios();

        // Save results to CSV
        saveResultsToCSV();
    }

private:
    /**
     * Print summary of reconstruction results
     */
    void printReconstructionSummary() {
        std::cout << "\n=== Reconstruction Results Summary ===\n" << std::endl;
        std::cout << "Grid Type   | Points | RMSE      | Mean Error | Max Error  | PSNR (dB) | Time (ms)" << std::endl;
        std::cout << "----------------------------------------------------------------------------" << std::endl;

        for (const auto& result : reconstructionResults) {
            printf("%-12s| %-7zu| %-10.5f| %-11.5f| %-11.5f| %-10.2f| %-9.2f\n",
                result.gridName.c_str(), result.pointCount, result.metrics.rmse,
                result.metrics.meanError, result.metrics.maxError,
                result.metrics.psnr, result.metrics.executionTimeMs);
        }
    }

    /**
     * Create side-by-side comparison of error maps
     */
    void createErrorMapComparison() {
        // Determine how many grids to compare (max 4)
        int numGrids = std::min(4, static_cast<int>(reconstructionResults.size()));

        if (numGrids <= 1) return;  // Need at least 2 for comparison

        // Create composite image
        cv::Mat comparisonImage(imageSize.height, imageSize.width * numGrids, CV_8UC3);

        // Copy error maps and add labels
        for (int i = 0; i < numGrids; i++) {
            reconstructionResults[i].metrics.errorMap.copyTo(
                comparisonImage(cv::Rect(i * imageSize.width, 0, imageSize.width, imageSize.height)));

            // Add grid type label
            cv::putText(comparisonImage, reconstructionResults[i].gridName,
                cv::Point(i * imageSize.width + imageSize.width / 2 - 60, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
        }

        // Display and save
        displayAndSaveImage(comparisonImage, "ReCon_Error Map Comparison");
    }

    /**
     * Create side-by-side comparison of distorted images
     */
    void createDistortionComparison() {
        // Determine how many grids to compare
        int numGrids = reconstructionResults.size();

        if (numGrids == 0) return;

        // Create original distorted image
        cv::Mat originalDistorted;
        cv::remap(testImage, originalDistorted, mapX, mapY, cv::INTER_CUBIC, cv::BORDER_REFLECT);

        // Create composite image (original + reconstructed)
        cv::Mat comparisonImage(imageSize.height, imageSize.width * (numGrids + 1), CV_8UC3);

        // Copy original image
        originalDistorted.copyTo(comparisonImage(cv::Rect(0, 0, imageSize.width, imageSize.height)));
        cv::putText(comparisonImage, "Original",
            cv::Point(imageSize.width / 2 - 50, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(255, 255, 255), 2);

        // Copy reconstructed images
        for (int i = 0; i < numGrids; i++) {
            reconstructionResults[i].distortedImage.copyTo(
                comparisonImage(cv::Rect((i + 1) * imageSize.width, 0, imageSize.width, imageSize.height)));

            // Add grid type label
            cv::putText(comparisonImage, reconstructionResults[i].gridName,
                cv::Point((i + 1) * imageSize.width + imageSize.width / 2 - 60, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
        }

        // Display and save
        displayAndSaveImage(comparisonImage, "ReCon_Distortion Comparison");
    }

    /**
     * Calculate and print compression ratios
     */
    void printCompressionRatios() {
        // Calculate original size in bytes (one float per pixel per map)
        double originalSize = 2 * imageSize.width * imageSize.height * sizeof(float);

        std::cout << "\n=== Compression Ratio ===" << std::endl;
        std::cout << "Grid Type   | Points | Original Size (KB) | Grid Size (KB) | Compression Ratio" << std::endl;
        std::cout << "----------------------------------------------------------------------------" << std::endl;

        for (const auto& result : reconstructionResults) {
            // Grid size is number of points * (sizeof(Point) + sizeof(Point2f)) for coordinate and value
            double gridSize = result.pointCount * (sizeof(cv::Point) + sizeof(cv::Point2f));

            printf("%-12s| %-7zu| %-19.2f| %-14.2f| %-17.2fx\n",
                result.gridName.c_str(), result.pointCount, originalSize / 1024.0,
                gridSize / 1024.0, originalSize / gridSize);
        }
    }

    /**
     * Save results to CSV file
     */
    void saveResultsToCSV() {
        std::string filename = outputDirectory + "/grid_analysis_results.csv";
        std::ofstream outputCSV(filename, std::ios::out);

        if (!outputCSV.is_open()) {
            std::cerr << "Failed to open CSV file for writing: " << filename << std::endl;
            return;
        }

        // Write timing section to CSV
        outputCSV << "TIMING RESULTS\n";
        outputCSV << "Grid Type,Time (ms),Total Points\n";

        for (const auto& gridResult : gridResults) {
            outputCSV << gridResult.name << ","
                << gridResult.generationTimeMs << ","
                << gridResult.gridPoints.size() << "\n";
        }
        outputCSV << "\n";

        // Write reconstruction results section to CSV
        outputCSV << "RECONSTRUCTION RESULTS\n";
        outputCSV << "Grid Type,Points,RMSE,Mean Error,Max Error,PSNR (dB),Time (ms)\n";

        for (const auto& result : reconstructionResults) {
            outputCSV << result.gridName << ","
                << result.pointCount << ","
                << result.metrics.rmse << ","
                << result.metrics.meanError << ","
                << result.metrics.maxError << ","
                << result.metrics.psnr << ","
                << result.metrics.executionTimeMs << "\n";
        }
        outputCSV << "\n";

        // Write compression results section to CSV
        double originalSize = 2 * imageSize.width * imageSize.height * sizeof(float);

        outputCSV << "COMPRESSION RESULTS\n";
        outputCSV << "Grid Type,Points,Original Size (KB),Grid Size (KB),Compression Ratio\n";

        for (const auto& result : reconstructionResults) {
            double gridSize = result.pointCount * (sizeof(cv::Point) + sizeof(cv::Point2f));

            outputCSV << result.gridName << ","
                << result.pointCount << ","
                << originalSize / 1024.0 << ","
                << gridSize / 1024.0 << ","
                << originalSize / gridSize << "\n";
        }

        outputCSV.close();
        std::cout << "Results saved to " << filename << std::endl;
    }
};

