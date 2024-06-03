#ifndef ATOMINTELMASH_PRACTICE_CALIBRATE_H
#define ATOMINTELMASH_PRACTICE_CALIBRATE_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <armadillo>

//extern "C" {
//#include <lapacke.h>
//#include <cblas.h>
//}

struct Point {
    Point (int x, int y) : x(x), y(y) {}
    Point () {}
    int x;
    int y;

    friend std::ostream& operator<<(std::ostream& out, const Point& p){
        return out << "Point: " << p.x << " " << p.y;
    }
    static bool comp(Point p1, Point p2) {
        int epsilon = 10;
        return p1.x - p2.x > epsilon ? p1.x < p2.x : p1.y < p2.y;
    }
    static void sortPoint(std::vector<Point> & points) {
        std::sort(points.begin(), points.end(), Point::comp);
    }
};

struct K {
    int fx, fy, cx, cy;
};


struct Calibrate {
private:
    cv::Mat __GaussianBlur(cv::Mat const & img);
    std::vector<Point> __PrevittOperator(cv::Mat const & img);
    std::vector<Point> __delete_similar_points(std::vector<Point> const & points);
    arma::mat __findHomography(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints);
    arma::mat __v(arma::mat const & H, int p, int q);
    arma::mat __V(std::vector<arma::mat> const & matH);
    arma::mat __B(arma::mat const & V);
public:
    std::vector<Point> find_points(cv::Mat const & src);
    void draw_points(cv::Mat const & img, std::vector<Point> const & points);
    std::vector<Point> getObjectPoints(int deskSizeX, int deskSizeY, int sizeSquare);
    K getK(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints);
};

#endif //ATOMINTELMASH_PRACTICE_CALIBRATE_H
