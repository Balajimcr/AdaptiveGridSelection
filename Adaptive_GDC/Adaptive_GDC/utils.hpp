#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Platform-specific directory creation
#ifdef _WIN32
#include <direct.h>
#define create_directory(dir) _mkdir(dir)
#else
#include <filesystem>
#define create_directory(dir) std::filesystem::create_directory(dir)
#endif

/**
 * @brief Draws a grid on a given OpenCV Mat image.
 *
 * This function draws a grid on the input image with specified grid dimensions.
 * The grid lines are drawn in blue color.
 *
 * @param mSrc The input OpenCV Mat image to draw the grid on.
 * @param Grid_X The number of grid cells along the horizontal axis (default: 35).
 * @param Grid_Y The number of grid cells along the vertical axis (default: 35).
 */
void DrawGrid(cv::Mat mSrc, const int Grid_X = 35, const int Grid_Y = 35) {
    int width = mSrc.size().width;
    int height = mSrc.size().height;

    // Calculate cell dimensions
    const int cellwidth = width / Grid_X;
    const int cellheight = width / Grid_Y; // Corrected to use Grid_Y

    // Set the image to white before drawing the grid
    mSrc.setTo(cv::Scalar::all(255));

    // Draw horizontal grid lines
    for (int i = 0; i < height; i += cellheight) { // Corrected to use cellheight
        cv::line(mSrc, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 0, 0), 2);
    }

    // Draw vertical grid lines
    for (int i = 0; i < width; i += cellwidth) { // Corrected to use cellwidth
        cv::line(mSrc, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 0, 0), 2);
    }
}

/**
 * @brief Draws points on an OpenCV Mat image.
 *
 * This function draws circles at the specified points on the input image.
 *
 * @param image The input OpenCV Mat image to draw points on.
 * @param points The vector of cv::Point objects representing the points to be drawn.
 * @param color The color of the points.
 * @param radius The radius of the points (default: 2).
 * @param thickness The thickness of the point borders (default: -1, filled circle).
 */
void DrawPoints(cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color, int radius = 2, int thickness = -1) {
    for (const auto& point : points) {
        cv::circle(image, point, radius, color, thickness);
    }
}

/**
 * @brief Creates a grid visualization by drawing points on a base image.
 *
 * This function converts a grayscale base image to BGR and draws grid points on it.
 *
 * @param baseImage The grayscale base image.
 * @param gridPoints The vector of grid points to be drawn.
 * @param outputImage The output BGR image with grid points drawn.
 */
void createGridVisualization(const cv::Mat& baseImage, const std::vector<cv::Point>& gridPoints, cv::Mat& outputImage) {
    cv::cvtColor(baseImage, outputImage, cv::COLOR_GRAY2BGR);
    DrawPoints(outputImage, gridPoints, cv::Scalar(255, 0, 0), 1, 2);
}

/**
 * @brief Displays and saves an image, resizing it if necessary.
 *
 * This function displays an image in a window and saves it to a file.
 * If the image resolution exceeds a maximum size, it is resized before display.
 *
 * @param image The input OpenCV Mat image to be displayed and saved.
 * @param windowName The name of the window and the base name of the saved file.
 */
void displayAndSaveImage(const cv::Mat& image, const std::string& windowName) {
    cv::Mat displayImage = image;
    const int maxWidth = 1280;
    const int maxHeight = 720;

    // Check if the image resolution exceeds the maximum display size
    if (image.cols > maxWidth || image.rows > maxHeight) {
        float aspectRatio = static_cast<float>(image.cols) / image.rows;
        int newWidth = maxWidth;
        int newHeight = maxHeight;

        // Adjust dimensions to maintain aspect ratio
        if (aspectRatio > 1) {
            newHeight = static_cast<int>(maxWidth / aspectRatio);
        }
        else {
            newWidth = static_cast<int>(maxHeight * aspectRatio);
        }

        cv::resize(image, displayImage, cv::Size(newWidth, newHeight));
    }

    // Display the image
    imshow(windowName, displayImage);
    cv::waitKey(1);

    // Create the "Outputs" directory if it doesn't exist
    const std::string foldername = "Outputs/";
    create_directory(foldername.c_str());

    // Construct the filename and save the image
    std::string filename = foldername + windowName + ".png";
    imwrite(filename, image);
}

/**
 * @brief Checks if a point exists in a vector of points.
 *
 * This inline function checks if a given point is present in a vector of points.
 *
 * @param points The vector of cv::Point objects to search in.
 * @param pt The cv::Point object to search for.
 * @return True if the point exists in the vector, false otherwise.
 */
inline bool pointExists(const std::vector<cv::Point>& points, const cv::Point& pt) {
    return std::find(points.begin(), points.end(), pt) != points.end();
}