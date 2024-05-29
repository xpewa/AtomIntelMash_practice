#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include "calibrate.h"

#include <opencv2/opencv.hpp>

std::string PATH = "../img/";


int main(int argc, char const* argv[]) {

    std::vector<cv::Mat> images;
    for (const auto & file : fs::directory_iterator(PATH)) {
        cv::Mat img = cv::imread(file.path(), cv::IMREAD_GRAYSCALE);
        images.push_back(img);
    }

    Calibrate calibrate;

    std::vector<Point> points = calibrate.find_points(images[1]);
    calibrate.draw_points(images[1], points);

    return 0;
}


