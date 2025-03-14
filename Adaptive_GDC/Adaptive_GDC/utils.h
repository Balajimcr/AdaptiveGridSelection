#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <direct.h>
#define create_directory(dir) _mkdir(dir)
#else
#include <filesystem>
#define create_directory(dir) std::filesystem::create_directory(dir)
#endif

// Function for drawing a grid 
void DrawGrid(cv::Mat mSrc, const int Grid_X = 35, const int Grid_Y = 35) {
    int width = mSrc.size().width;
    int height = mSrc.size().height;

    const int cellwidth = width / Grid_X;
    const int cellheight = width / Grid_X;

	mSrc.setTo(cv::Scalar::all(255));

    for (int i = 0; i < height; i += cellwidth)
        cv::line(mSrc, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 0, 0), 2);

    for (int i = 0; i < width; i += cellheight)
        cv::line(mSrc, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 0, 0), 2);
}

void DrawPoints(cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color, int radius = 2, int thickness = -1) {
    for (const auto& point : points) {
        cv::circle(image, point, radius, color, thickness);
    }
}

void createGridVisualization(const cv::Mat& baseImage, const std::vector<cv::Point>& gridPoints, cv::Mat& outputImage) {
    cv::cvtColor(baseImage, outputImage, cv::COLOR_GRAY2BGR);
    DrawPoints(outputImage, gridPoints, cv::Scalar(255, 0, 0), 1, 2);
}

// Function to display and save an image
void displayAndSaveImage(const cv::Mat& image, const std::string& windowName) {
    cv::Mat displayImage = image;
    const int maxWidth = 1280;
    const int maxHeight = 720;

    // Check if the image resolution is more than 1280x720
    if (image.cols > maxWidth || image.rows > maxHeight) {
        float aspectRatio = static_cast<float>(image.cols) / image.rows;
        int newWidth = maxWidth;
        int newHeight = maxHeight;

        if (aspectRatio > 1) {
            newHeight = static_cast<int>(maxWidth / aspectRatio);
        } else {
            newWidth = static_cast<int>(maxHeight * aspectRatio);
        }

        cv::resize(image, displayImage, cv::Size(newWidth, newHeight));
    }

    imshow(windowName, displayImage);
    cv::waitKey(1);

    // Construct the filename using the window name and ".png" extension
    const std::string foldername = "Outputs/";
    std::string filename = foldername + windowName + ".png";
    imwrite(filename, image);
}

// Helper function for point comparison
inline bool pointExists(const std::vector<cv::Point>& points, const cv::Point& pt) {
    return std::find(points.begin(), points.end(), pt) != points.end();
}

#endif //