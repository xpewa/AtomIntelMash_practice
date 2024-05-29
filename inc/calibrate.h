#ifndef ATOMINTELMASH_PRACTICE_CALIBRATE_H
#define ATOMINTELMASH_PRACTICE_CALIBRATE_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct Point {
    Point (int x, int y) : x(x), y(y) {}
    Point () {}
    int x;
    int y;

    friend std::ostream& operator<<(std::ostream& out, const Point& p){
        return out << "Point: " << p.x << " " << p.y;
    }
};


struct Calibrate {
private:
    cv::Mat __GaussianBlur(cv::Mat const & img);
    std::vector<Point> __PrevittOperator(cv::Mat const & img);
    std::vector<Point> __delete_similar_points(std::vector<Point> const & points);
public:
    std::vector<Point> find_points(cv::Mat const & src);
    void draw_points(cv::Mat const & img, std::vector<Point> const & points);
};

#endif //ATOMINTELMASH_PRACTICE_CALIBRATE_H
