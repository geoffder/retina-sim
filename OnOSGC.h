//
// Created by geoff on 2019-05-10.
//

#ifndef RETINA_ONOSGC_H
#define RETINA_ONOSGC_H

#include <array>
#include <math.h>
#include <random>

#include "Cell.h"
#include "utils.h"
#include "type_defs.h"


class OnOSGC : public Cell {
protected:
    double theta;      // preferred orientation
    double axis0;
    double axis1;
    std::array<double, 3> cardinals = {0, 90};  // constant

public:
    OnOSGC(Eigen::VectorXd &xgrid, Eigen::VectorXd &ygrid, Eigen::VectorXd &xOnes, Eigen::VectorXd &yOnes,
            const double net_dt, const std::array<double, 2> cell_pos, std::mt19937 &gen)
            :Cell(xgrid, ygrid, xOnes, yOnes, net_dt, cell_pos) {
        type = "OnOSGC";
        // spatial properties
        diam = 8;  // of soma (old 15)
        somaMask = circleMask(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, diam/2);
        // Orientation-selective properties
        axis0 = 25;
        axis1 = 100;
        surround_rad = 100;
        theta = rollPreferred(gen);  // choose a cardinal direction preference for this cell
        std::tie(rfCentre, rfSurround) = buildRF(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, axis0, axis1, surround_rad, theta);
        rfCentre_sparse = rfCentre.sparseView();  // convert from dense matrix to sparse
        rfSurround_sparse = rfSurround.sparseView();  // convert from dense matrix to sparse
        // active / synaptic properties
        sustained = true;
        onoff = false;
        dtau = 200;
    }

    double rollPreferred(std::mt19937 &gen) {
        // sample one cardinal direction from the array
        std::vector<double> choice;  // iterator to receive sample output
        std::sample(cardinals.begin(), cardinals.end(), std::back_inserter(choice), 1, gen);
        return choice[0];
    }

    // (check this) axis0 and axis1 are the full length of the minor and major axes of the ellipse (like diam, not rad)
    // does not need to override base class method, this has an additional parameter
    rfPair buildRF(Eigen::VectorXd xgrid, Eigen::VectorXd ygrid, Eigen::VectorXd xOnes, Eigen::VectorXd yOnes,
                   std::array<double, 2> origin, double axis0, double axis1, double surradius, double theta) {
        Eigen::MatrixXd rgrid, x, y;  // double
        Eigen::MatrixXi centre, surround;   // integer

        // squared euclidean distance (not taking sqrt, square the radius instead)
        // for elliptical centre
        xgrid = xgrid.array() - origin[0];
        ygrid = ygrid.array() - origin[1];
        x = xgrid*cos(theta) + ygrid*sin(theta);
        y = xgrid*sin(theta) + ygrid*cos(theta);
        // for circular surround
        rgrid = (
                xgrid.array().square().matrix() * yOnes.transpose()
                + xOnes * ygrid.array().square().matrix().transpose()
        );

        // convert to boolean based on distance from origin vs radius of desired circle
        centre = (
                    (
                            (x.array()/axis0).square().matrix()* yOnes.transpose()
                             + xOnes * (y.array()/axis1).square().matrix().transpose()
                    ).array() <= 1
                ).cast<int>();
        surround = (pow(axis0, 2) <= rgrid.array()).cast<int>()
                   * (rgrid.array() <= pow(axis0+surradius, 2)).cast<int>();
        return std::make_tuple(centre, surround);
    }

    // override base method to add in theta
    std::string getParamStr() override {
        std::stringstream stream;
        // JSON formatting using raw string literals
        stream << R"({"type": ")" << type << R"(", "theta": )" << theta << R"(, "diam": )" << diam;
        stream << R"(, "rf_ax0": )" << axis0 << R"(, "rf_ax1": )" << axis1 << R"(, "surround_rad": )" << surround_rad;
        stream << R"(, "dtau": )" << dtau << "}";
        std::string params = stream.str();
        return params;
    }

    void stimulate(const Stim &stim) override {
        // use stimulus itself if sustained, and delta of stimulus if transient
        Eigen::SparseMatrix<int> stim_mask = sustained ? stim.getSparseMask() : stim.getSparseDelta();

        Eigen::SparseMatrix<int> centre_overlap, surround_overlap;
        centre_overlap = stim_mask.cwiseProduct(rfCentre_sparse);
        surround_overlap = stim_mask.cwiseProduct(rfSurround_sparse);

        double strength = (centre_overlap.sum() - std::max(0, surround_overlap.sum())*.2) * abs(stim.getAmp());

        Vm += strength*.025;
    }

};


#endif //RETINA_ONOSGC_H
