#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

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
    imshow(windowName, image);
    cv::waitKey(1);

    // Construct the filename using the window name and ".png" extension
    std::string filename = windowName + ".png";
    imwrite(filename, image);
}

// Helper function for point comparison
inline bool pointExists(const std::vector<cv::Point>& points, const cv::Point& pt) {
    return std::find(points.begin(), points.end(), pt) != points.end();
}