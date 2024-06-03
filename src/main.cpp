#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include "calibrate.h"

#include <opencv2/opencv.hpp>

std::string PATH = "../img/";

int DESK_SIZE_X = 4;
int DESK_SIZE_Y = 2;
int SIZE_SQUARE = 50;


int main(int argc, char const* argv[]) {

    std::vector<cv::Mat> images;
    for (const auto & file : fs::directory_iterator(PATH)) {
        cv::Mat img = cv::imread(file.path(), cv::IMREAD_GRAYSCALE);
        images.push_back(img);
    }

    Calibrate calibrate;

    std::vector<Point> imagePoints = calibrate.find_points(images[2]);
//    calibrate.draw_points(images[2], imagePoints);

    std::vector<Point> objectPoints = calibrate.getObjectPoints(DESK_SIZE_X, DESK_SIZE_Y, SIZE_SQUARE);
//    int count_obj_points = 0;
//    for (int i = 0; i < objectPoints.size(); i += 1) {
//        std::cout << "Object " << objectPoints[i] << std::endl;
//        ++count_obj_points;
//    }
//    std::cout << "Count Object point: " << count_obj_points << std::endl;

    Point p1(10, 20);
    Point p2(50, 60);
    std::vector<Point> u = {p1};
    std::vector<Point> v = {p2};

//    K intrinsics_camera_param = calibrate.getK(u, v);
    K intrinsics_camera_param = calibrate.getK(imagePoints, objectPoints);

    return 0;
}


