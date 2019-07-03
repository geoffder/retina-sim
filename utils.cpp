#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "type_defs.h"
#include "utils.h"

using namespace Eigen;


std::tuple<VectorXd, VectorXd> gridVecs(int nrows, int ncols) {
    // range from 0 to dim_size-1
    VectorXd xvec = VectorXd::LinSpaced(ncols, 0, ncols-1);
    VectorXd yvec = VectorXd::LinSpaced(nrows, 0, nrows-1);

    // package for output
    std::tuple<VectorXd, VectorXd> out = std::make_tuple(xvec, yvec);
    return out;
}

/* Take X and Y grid matrices and use them to calculate the distance from the
 * origin to every element of the matrix. Return a boolean mask identifying
 * all elements that are within a maximum radius (a circle shape).
 */
MatrixXi circleMask(Eigen::VectorXd xgrid, Eigen::VectorXd ygrid, Eigen::VectorXd xOnes,
                    Eigen::VectorXd yOnes, std::array<double, 2> origin, double radius) {
    // squared euclidean distance (not taking sqrt, square the radius instead)
    MatrixXd rgrid = (
            (xgrid.array() - origin[0]).square().matrix()*yOnes.transpose()
            + xOnes*(ygrid.array() - origin[1]).square().matrix().transpose()
    );
    // convert to boolean based on distance from origin vs radius of desired circle
    MatrixXi mask = (rgrid.array() <= pow(radius, 2)).cast<int>();
    return mask;
}

// Rotate X and Y grid matrices clockwise by given degrees, around an origin.
std::tuple<MatrixXd, MatrixXd> rotateGrids(std::array<double, 2> origin, VectorXd xgrid, VectorXd ygrid, VectorXd xOnes,
                                           VectorXd yOnes, double degrees) {
    // convert orientation angle to radians
    double theta = deg2rad(degrees);

    // rotate x and y grids around origin
    xgrid = xgrid.array() - origin[0];
    ygrid = ygrid.array() - origin[1];

    // looks stupid, but it's faster to add origin back in before blowing up vector to matrix
    MatrixXd x_rot = ((cos(theta)*xgrid).array() + origin[0]).matrix()*yOnes.transpose() - xOnes*sin(theta)*ygrid.transpose();
    MatrixXd y_rot = sin(theta)*xgrid*yOnes.transpose() + xOnes*((cos(theta)*ygrid).array() + origin[1]).matrix().transpose();

    std::tuple<MatrixXd, MatrixXd> out = std::make_tuple(x_rot, y_rot);
    return out;
}

/* Take X and Y grid matrices and draw a rectangular boolean mask. Rectangle is defined by WxH dims
 * centred on an origin, and oriented in an arbitrary angle.
 */
MatrixXi rectMask(VectorXd *xgrid, VectorXd *ygrid, VectorXd *xOnes, VectorXd *yOnes, std::array<double, 2> origin,
                  double orient, double width, double height) {
    MatrixXi mask; // integer

    // rotate coordinates according to orientation of desired rectangle mask
    auto [xgrid_rot, ygrid_rot] = rotateGrids(origin, *xgrid, *ygrid, *xOnes, *yOnes, orient); // returns MatrixXd
    // convert to boolean based on distances from origin on rotated x and y planes
    mask = (
                ((xgrid_rot.array() - origin[0]).abs() <= width/2).cast<int>()
                * ((ygrid_rot.array() - origin[1]).abs() <= height/2).cast<int>()
            );
    return mask;
}

// axis0 and axis1 are the full length of the minor and major axes of the ellipse (like diam, not rad)
Eigen::MatrixXi ellipseMask(Eigen::VectorXd xgrid, Eigen::VectorXd ygrid, Eigen::VectorXd xOnes,
                            Eigen::VectorXd yOnes, std::array<double, 2> origin, double theta, double axis0, double axis1) {
    Eigen::MatrixXd x, y;  // double
    Eigen::MatrixXi mask;   // integer

    // squared euclidean distance (not taking sqrt, square the radius instead)
    xgrid = xgrid.array() - origin[0];
    ygrid = ygrid.array() - origin[1];
    x = xgrid*cos(theta) + ygrid*sin(theta);
    y = xgrid*sin(theta) + ygrid*cos(theta);

    // convert to boolean based on distance from origin vs radius of desired circle
    mask = (
            (
                    (x.array()/axis0).square().matrix()* yOnes.transpose()
                    + xOnes * (y.array()/axis1).square().matrix().transpose()
            ).array() <= 1
    ).cast<int>();
    return mask;
}

void MatrixXiToCSV(std::string fname, MatrixXi mat) {
    // CSV format described in eigen_types.h
    std::ofstream file(fname.c_str());
    file << mat.format(CSVFormat);
    file.close();
}

void MatrixXdToCSV(std::string fname, MatrixXd mat) {
    // CSV format described in eigen_types.h
    std::ofstream file(fname.c_str());
    file << mat.format(CSVFormat);
    file.close();
}

double deg2rad(double degrees) {
    return degrees * 3.14159265359/180;
}

