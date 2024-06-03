#include "calibrate.h"

cv::Mat Calibrate::__GaussianBlur(cv::Mat const & img) {
    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            float k1 = 0.0625;
            float k2 = 0.125;
            float k3 = 0.0625;
            float k4 = 0.125;
            float k5 = 0.25;
            float k6 = 0.125;
            float k7 = 0.0625;
            float k8 = 0.125;
            float k9 = 0.0625;

            int p1 = img.at<uchar>(y - 1, x - 1);
            int p2 = img.at<uchar>(y - 1, x);
            int p3 = img.at<uchar>(y - 1, x + 1);
            int p4 = img.at<uchar>(y, x - 1);
            int p5 = img.at<uchar>(y, x);
            int p6 = img.at<uchar>(y, x + 1);
            int p7 = img.at<uchar>(y + 1, x - 1);
            int p8 = img.at<uchar>(y + 1, x);
            int p9 = img.at<uchar>(y + 1, x + 1);

            res.at<uchar>(y, x) = k1*p1 + k2*p2 + k3*p3 + k4*p4 + k5*p5 + k6*p6 + k7*p7 + k8*p8 + k9*p9;
        }
    }
    return res;
}

std::vector<Point> Calibrate::__PrevittOperator(cv::Mat const & img) {
    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            res.at<uchar>(y, x) = img.at<uchar>(y, x);
        }
    }

    std::vector<std::vector<int>> Gx(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Gy(img.cols, std::vector<int>(img.rows, 0));
    std::vector<std::vector<int>> Hp(img.cols, std::vector<int>(img.rows, 0));
    std::vector<Point> points;

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            int z1 = img.at<uchar>(y - 1, x - 1);
            int z2 = img.at<uchar>(y - 1, x);
            int z3 = img.at<uchar>(y - 1, x + 1);
            int z4 = img.at<uchar>(y, x - 1);
            int z6 = img.at<uchar>(y, x + 1);
            int z7 = img.at<uchar>(y + 1, x - 1);
            int z8 = img.at<uchar>(y + 1, x);
            int z9 = img.at<uchar>(y + 1, x + 1);

            int gx = (z7 + z8 + z9) - (z1 + z2 + z3);
            int gy = (z3 + z6 + z9) - (z1 + z4 + z7);
            Gx[x][y] = gx;
            Gy[x][y] = gy;
        }
    }

    for (int x = 10; x < Gx.size() - 10; ++x) {
        for (int y = 10; y < Gx[0].size() - 10; ++y) {
            float k = 0.2; // 0.2
            int gp1 = 0;
            int gp2 = 0;
            int gp3 = 0;

            for (int i = x - 1; i < x + 2; ++i) {
                for (int j = y - 1; j < y + 2; ++j) {
                    int gx = Gx[i][j];
                    int gy = Gy[i][j];
                    gp1 += gx*gx;
                    gp2 += gx*gy;
                    gp3 += gy*gy;
                }
            }

            int hp = (gp1 * gp3 - gp2*gp2) - k*(gp1 + gp3)*(gp1 + gp3);
            Hp[x][y] = hp;

            if (hp > 1000000) {
                Point p = Point(x, y);
                points.push_back(p);
            }
        }
    }

    return points;
}

std::vector<Point> Calibrate::__delete_similar_points(std::vector<Point> const & points) {
    uint accuracy = 10;
    std::vector<Point> res;

    for (int i = 0; i < points.size(); ++i) {
        bool is_similar = false;
        for (int j = 0; j < i; ++j) {
            if (std::abs(points[i].x - points[j].x) < accuracy && std::abs(points[i].y - points[j].y) < accuracy) {
                is_similar = true;
            }
        }
        if ( ! is_similar ) {
            res.push_back(points[i]);
        }
    }
    return res;
}

std::vector<Point> Calibrate::find_points(cv::Mat const & src) {
    cv::Mat img;

    img = __GaussianBlur(src);
    for (int i = 0; i < 10; ++i) { // 10
        img = __GaussianBlur(img);
    }

    std::vector<Point> points = __PrevittOperator(img);

    points = __delete_similar_points(points);

    Point::sortPoint(points);

    return points;
}

void Calibrate::draw_points(cv::Mat const & img, std::vector<Point> const & points) {
    uint count_points = 0;
    for (int i = 0; i < points.size(); ++i) {
        std::cout << points[i] << std::endl;
        ++count_points;
    }
    std::cout << "Count points: " << count_points << std::endl;

    cv::Mat res(cv::Size(img.cols, img.rows), CV_8UC1, 255);
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            res.at<uchar>(y, x) = img.at<uchar>(y, x);
        }
    }

    for (int i = 0; i < points.size(); ++ i) {
        Point p = points[i];
        cv::Point centerCircle(p.x, p.y);
        cv::Scalar colorCircle(0);
        cv::circle(res, centerCircle, 10, colorCircle, cv::FILLED);
    }

    cv::imshow("draw_points", res);
    cv::waitKey();
}


std::vector<Point> Calibrate::getObjectPoints(int deskSizeX, int deskSizeY, int sizeSquare) {
    std::vector<Point> res;
    for (int x = 0; x < deskSizeX; x += 1) {
        for (int y = 0; y < deskSizeY; y += 1) {
            res.push_back(Point(x * sizeSquare, y * sizeSquare));
        }
    }
    return res;
}

arma::mat Calibrate::__findHomography(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints) {
    if (objectPoints.size() != imagePoints.size()) {
        arma::mat H;
        return H;
    }

    std::vector<int> P, u;
//    for (int i = 0; i < objectPoints.size(); ++i) {
//        P.push_back(objectPoints[i].x);
//        P.push_back(objectPoints[i].y);
//        P.push_back(0);
//        P.push_back(1);
//    }
//    for (int i = 0; i < imagePoints.size(); ++i) {
//        u.push_back(imagePoints[i].x);
//        u.push_back(imagePoints[i].y);
//        u.push_back(1);
//    }
//
//    std::vector<std::vector<int>> Q;
//    std::vector<int> Q1, Q2;
//    for (int i = 0; i < P.size(); ++i) {
//        Q1.push_back(P[i]);
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q1.push_back(0);
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q1.push_back(-u[i % 3]*P[i]);
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q2.push_back(0);
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q2.push_back(P[i]);
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q2.push_back(-u[(i % 3) + 1]*P[i]);
//    }
//
//    Q.push_back(Q1);
//    Q.push_back(Q2);

//    arma::vec P(objectPoints.size() * 3);
//    for (int i = 0; i < objectPoints.size(); ++i) {
//        Point point = objectPoints[i];
//        P[4*i] = point.x;
//        P[4*i + 1] = point.y;
//        P[4*i + 2] = 1;
//    }
//    arma::vec u(imagePoints.size() * 3);
//    for (int i = 0; i < imagePoints.size(); ++i) {
//        Point point = imagePoints[i];
//        u[3*i] = point.x;
//        u[3*i + 1] = point.y;
//        u[3*i + 2] = 1;
//    }
//
//    arma::mat Q(2, 3*P.size());
//    for (int i = 0; i < P.size(); ++i) {
//        Q.row(0).col(i) = P[i];
//    }
//    for (int i = P.size(); i < 2 * P.size(); ++i) {
//        Q.row(0).col(i) = 0;
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q.row(0).col(2 * P.size() + i) = -u[(i % 3)]*P[i];
//    }
//
//    for (int i = 0; i < P.size(); ++i) {
//        Q.row(1).col(i) = 0;
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q.row(1).col(P.size() + i) = P[i];
//    }
//    for (int i = 0; i < P.size(); ++i) {
//        Q.row(1).col(2 * P.size() + i) = -u[(i % 3) + 1]*P[i];
//    }

//    std::cout << Q << std::endl;


    arma::mat Q(objectPoints.size() * 2, 9);

    for (int i = 0; i < objectPoints.size(); ++i) {
        Q.col(0).row(2 * i) = - objectPoints[i].x;
        Q.col(1).row(2 * i) = 0;
        Q.col(2).row(2 * i) = objectPoints[i].x * imagePoints[i].x;
        Q.col(3).row(2 * i) = -objectPoints[i].y;
        Q.col(4).row(2 * i) = 0;
        Q.col(5).row(2 * i) = objectPoints[i].y * imagePoints[i].x;
        Q.col(6).row(2 * i) = -1;
        Q.col(7).row(2 * i) = 0;
        Q.col(8).row(2 * i) = imagePoints[i].x;
        Q.col(0).row(2 * i + 1) = 0;
        Q.col(1).row(2 * i + 1) = -objectPoints[i].x;
        Q.col(2).row(2 * i + 1) = objectPoints[i].x * imagePoints[i].y;
        Q.col(3).row(2 * i + 1) = 0;
        Q.col(4).row(2 * i + 1) = - objectPoints[i].y;
        Q.col(5).row(2 * i + 1) = objectPoints[i].y * imagePoints[i].y;
        Q.col(6).row(2 * i + 1) = 0;
        Q.col(7).row(2 * i + 1) = -1;
        Q.col(8).row(2 * i + 1) = imagePoints[i].y;
    }

    arma::mat U;
    arma::vec S;
    arma::mat V;

//    std::cout << Q << std::endl;

    arma::svd(U, S, V, Q);

    arma::mat H = V.col(8);

    H = reshape(H, 3, 3);
    return H;
}


arma::mat Calibrate::__v(arma::mat const & H, int p, int q) {
    arma::mat v(1, 6);
    double H_1p = H.at(0, p);
    double H_1q = H.at(0, q);
    double H_2p = H.at(1, p);
    double H_2q = H.at(1, q);
    double H_3p = H.at(2, p);
    double H_3q = H.at(2, q);
    v.row(0).col(0) = H_1p * H_1q;
    v.row(0).col(1) = H_1p * H_3q + H_1q * H_3p;
    v.row(0).col(2) = H_2p * H_3q + H_2q * H_3p;
    v.row(0).col(3) = H_3p * H_3q;
    v.row(0).col(4) = H_2p * H_2q;
    v.row(0).col(5) = H_1p * H_2q + H_1q * H_2p;

    return v;
}

arma::mat Calibrate::__V(std::vector<arma::mat> const & matH) {
    arma::mat V(matH.size() * 2, 6);
    for (int i = 0; i < matH.size(); ++i) {
        arma::mat v12 = this->__v(matH[i], 1, 2);
        arma::mat v11 = this->__v(matH[i], 1, 1);
        arma::mat v22 = this->__v(matH[i], 2, 2);

        for (int j = 0; j < 6; ++j) {
            V.at(2 * i, j) = v12.at(0, j);

            V.at(2 * i + 1, j) = v11.at(0, j) - v22.at(0, j);
        }
    }

    return V;
}


arma::mat Calibrate::__B(arma::mat const & V) {
    arma::mat U;
    arma::vec S;
    arma::mat svdB;

    arma::svd(U, S, svdB, V);

    svdB = svdB.col(svdB.n_cols - 1);

    arma::mat B(3, 3);
    B.at(0, 0) = svdB.at(0, 0);
    B.at(0, 1) = svdB.at(0, 5);
    B.at(0, 2) = svdB.at(0, 1);
    B.at(1, 0) = svdB.at(0, 5);
    B.at(1, 1) = svdB.at(0, 4);
    B.at(1, 2) = svdB.at(0, 2);
    B.at(2, 0) = svdB.at(0, 1);
    B.at(2,1) = svdB.at(0, 2);
    B.at(2, 2) = svdB.at(0, 3);

    return B;
}


K Calibrate::getK(std::vector<Point> const & imagePoints, std::vector<Point> const & objectPoints) {

    arma::mat H = this->__findHomography(imagePoints, objectPoints);
//    std::cout << H << std::endl;


    std::vector<arma::mat> matH;
    matH.push_back(H);
    arma::mat V = this->__V(matH);
//    std::cout << V << std::endl;

    arma::mat B = this->__B(V);
    std::cout << B << std::endl;
    std::cout << arma::inv(B) << std::endl;

//    arma::mat matK;
//    arma::chol(matK, arma::inv(B));
//    std::cout << matK << std::endl;

    arma::mat matK;
    arma::chol(matK, B);
    std::cout << matK << std::endl;



    K res;
    return res;
}

