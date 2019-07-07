#ifndef RETINA_STIM_H
#define RETINA_STIM_H

#include <iostream>
#include <tuple>
#include <vector>
#include <chrono>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "utils.h"
#include "type_defs.h"

class Stim {
private:
    std::array<int, 3> dims;                  // network dimensions {X, Y, space_redux}
    int reduX, reduY;
    Eigen::VectorXd * net_xvec;             // pointer to network X range grid used for generation of masks
    Eigen::VectorXd * net_yvec;             // pointer to network Y range grid used for generation of masks
    Eigen::VectorXd * net_xOnes;
    Eigen::VectorXd * net_yOnes;
    int t;                                // track net time (move is called on every time step)
    int dt;                               // timestep of network model
    std::array<double, 2> pos;                           // centre coordinates
    double tOn;                                 // time stimulus appears
    double tOff;                                // time stimulus turns off
    bool alive;                       // false after tOff
    double vel;                              // velocity
    double theta, theta_rad;                 // angle of movement
    double orient, orient_rad;               // angle of orientation
    double amp;                              // intensity of stimulus
    double dAmp;                             // rate and direction of change in intensity of stimulus
    std::string type;                        // type of stimulus tag/label (e.g. circle, bar)
    double radius = 0;                           // stimulus radius (circle type parameter)
    double length;                           // length (bar type parameter)
    double width;                            // width (bar type parameter)
    Eigen::MatrixXi mask;                    // mask defining this stimulus
    Eigen::MatrixXi delta;                   // mask defining the change in stimulus (timestep delta)
    Eigen::SparseMatrix<int> mask_sparse;    // sparse representation of the stimulus mask (fast computation)
    Eigen::SparseMatrix<int> delta_sparse;   // sparse representation of the delta mask (fast computation)
    std::vector<double> xPosRec;             // stored position from each timestep
    std::vector<double> yPosRec;             // stored position from each timestep
    std::vector<double> ampRec;              // stored amplitude from each timestep
    std::vector<double> orientRec;           // stored angle of orientation from each timestep

public:
    Stim(const std::array<int, 3> net_dims, Eigen::VectorXd &xgrid, Eigen::VectorXd &ygrid,
            Eigen::VectorXd &xOnes, Eigen::VectorXd &yOnes, const int net_dt, const std::array<double, 2> start_pos,
            const double time_on, const double time_off, const double velocity, const double direction,
            const double orientation, const double amplitude, const double change) {
        // network properties
        dims = net_dims;
        reduX = dims[0]/dims[2], reduY = dims[1]/dims[2];
        net_xvec = &xgrid;
        net_yvec = &ygrid;
        net_xOnes = &xOnes;
        net_yOnes = &yOnes;
        t = 0;
        dt = net_dt;
        // general stim properties
        alive = true;
        pos = start_pos;
        tOn = time_on;
        tOff = time_off;
        // movement
        vel = velocity;
        theta = direction;
        theta_rad = theta*3.14159265359/180;
        // spatial orientation (only matters for radially asymmetric objects)
        orient = orientation;
        orient_rad = orient*3.14159265359/180;
        // "contrast"
        amp = amplitude;
        dAmp = change;
        // initialize
        mask = Eigen::MatrixXi::Zero(reduX, reduY);
        delta = Eigen::MatrixXi::Zero(reduX, reduY);
        mask_sparse = mask.sparseView();
        delta_sparse = delta.sparseView();
    }

    void setCircle(const double rad) {
        type = "circle";
        radius = rad;
    }

    void setBar(const double wid, const double len) {
        type = "bar";
        width = wid;
        length = len;
    }

    void setEllipse(const double axis0, const double axis1) {
        type = "ellipse";
        width = axis0;
        length = axis1;
    }

    void setOrientation(const double angle) {
        orient = angle;
    }

    void setVelocity(const double velocity) {
       vel = velocity;
    }

    Eigen::MatrixXi getMask() const {
        return mask;
    }

    Eigen::SparseMatrix<int> getSparseMask() const {
        return mask_sparse;
    }

    Eigen::SparseMatrix<int> getSparseDelta() const {
        return delta_sparse;
    }

    double getTheta() const {
        return theta;
    }

    double getAmp() const {
        return amp;
    }

    void drawMask() {
        Eigen::MatrixXi old_mask = mask;
        if ((t >= tOff) && alive) {
            alive = false;
            mask = mask.array() * 0;
        } else {
            if (type == "bar") {
                mask = rectMask(net_xvec, net_yvec, net_xOnes, net_yOnes, pos, orient, width, length);
            } else if (type == "circle") {
                mask = circleMask(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, radius);
            } else if (type == "ellipse") {
                mask = ellipseMask(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, orient, width, length);
            }
            if (amp < 0) {
                mask = mask.array() * -1;
            }
        }
        delta = mask - old_mask;
    }

    // update centre coordinate of stimulus (if moving), then redraw mask
    void move() {
        // TODO: right now, this wil not work with a motionless stimulus. Fix it after initial testing
        if ((t >= tOn) && alive) {
            if (vel != 0) {
                pos[0] += vel * dt * cos(theta_rad);
                pos[1] += vel * dt * sin(theta_rad);
                drawMask();
                mask_sparse = mask.sparseView();
                delta_sparse = delta.sparseView();
            }
            amp += dAmp;
        } else if (!alive && (delta.nonZeros() > 0)) {
            // zero out delta so it doesn't remain forever after stim turns off
            delta = delta.array() * 0;
            delta_sparse = delta.sparseView();
        }

        // record stimulus characteristics that are subject to change for movie reconstruction
        xPosRec.push_back(pos[0]);
        yPosRec.push_back(pos[1]);
        ampRec.push_back(amp);
        orientRec.push_back(orient);

//        if (t % 50 == 0) {
//            std::cout << mask << "\n\n";
//        }

        t += dt;  // increment time to keep in sync with network
    }

    Eigen::MatrixXd getRecTable() {
        // Position, amplitude and orientation of stimulus at each time-step for this stimulus.
        // Use these to draw stimulus movies to be used by the decoder.
        int recLen = xPosRec.size();
        Eigen::MatrixXd all_recs = Eigen::MatrixXd::Ones(recLen, 4);

        for(std::size_t i = 0; i < recLen; ++i){
            all_recs(i, 0) = xPosRec[i];
            all_recs(i, 1) = yPosRec[i];
            all_recs(i, 2) = ampRec[i];
            all_recs(i, 3) = orientRec[i];
        }
        return all_recs;
    }

    void saveParams(const std::string &filepath) {
        std::ofstream paramFile;
        paramFile.open(filepath);
        // JSON formatting using raw string literals
        paramFile << R"({"type": ")" << type << R"(", "radius": )" << radius << R"(, "width": )" << width;
        paramFile << R"(, "length": )" << length << R"(, "tOn": )" << tOn << R"(, "tOff": )" << tOff << "}";
        paramFile.close();
    }
};


#endif //RETINA_STIM_H
