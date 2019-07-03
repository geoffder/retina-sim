#ifndef RETINA_UTILS_H
#define RETINA_UTILS_H

#include <iostream>
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include "type_defs.h"

std::tuple<Eigen::VectorXd, Eigen::VectorXd> gridVecs(int nrows, int ncols);
Eigen::MatrixXi circleMask(Eigen::VectorXd xgrid, Eigen::VectorXd ygrid, Eigen::VectorXd xOnes, Eigen::VectorXd yOnes, std::array<double, 2> origin, double radius);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> rotateGrids(std::array<double, 2> origin, Eigen::VectorXd xgrid, Eigen::VectorXd xOnes, Eigen::VectorXd yOnes, Eigen::VectorXd ygrid, double degrees);
Eigen::MatrixXi rectMask(Eigen::VectorXd *xgrid, Eigen::VectorXd *ygrid, Eigen::VectorXd *xOnes, Eigen::VectorXd *yOnes, std::array<double, 2> origin, double orient, double width, double height);
Eigen::MatrixXi ellipseMask(Eigen::VectorXd xgrid, Eigen::VectorXd ygrid, Eigen::VectorXd xOnes, Eigen::VectorXd yOnes, std::array<double, 2> origin, double theta, double axis0, double axis1);
double deg2rad(double degrees);
void MatrixXiToCSV(std::string fname, Eigen::MatrixXi mat);
void MatrixXdToCSV(std::string fname, Eigen::MatrixXd mat);

#endif //RETINA_UTILS_H
