//
// Created by geoff on 2019-05-10.
//

#ifndef RETINA_ONALPHA_H
#define RETINA_ONALPHA_H

#include <random>

#include "Cell.h"
#include "utils.h"
#include "type_defs.h"

class OnAlpha : public Cell {
public:
    OnAlpha(Eigen::VectorXd &xgrid, Eigen::VectorXd &ygrid, Eigen::VectorXd &xOnes, Eigen::VectorXd &yOnes,
            const double net_dt, const std::array<double, 2> cell_pos, std::mt19937 &gen)
            :Cell(xgrid, ygrid, xOnes, yOnes, net_dt, cell_pos) {
        type = "OnAlpha";
        // spatial properties
        diam = 8;  // 15
        centre_rad = 100;  // 200
        surround_rad = 200;
        somaMask = circleMask(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, diam/2);
        std::tie(rfCentre, rfSurround) = buildRF(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, centre_rad, surround_rad);
        rfCentre_sparse = rfCentre.sparseView();  // convert from dense matrix to sparse
        rfSurround_sparse = rfSurround.sparseView();  // convert from dense matrix to sparse
        // active / synaptic properties
        onoff = false;
        sustained = !std::uniform_int_distribution<> (0, 1)(gen);  // randomly set cell to sustained or transient
        dtau = sustained ? 200 : 100;  // decay tau depends on sustained/transient status (true : false)
    }

    void stimulate(const Stim &stim) override {
        // use stimulus itself if sustained, and delta of stimulus if transient
        Eigen::SparseMatrix<int> stim_mask = sustained ? stim.getSparseMask() : stim.getSparseDelta();

        Eigen::SparseMatrix<int> centre_overlap, surround_overlap;
        centre_overlap = stim_mask.cwiseProduct(rfCentre_sparse);
        surround_overlap = stim_mask.cwiseProduct(rfSurround_sparse);

        double strength = (centre_overlap.sum() - std::max(0, surround_overlap.sum())*.1) * abs(stim.getAmp());

        Vm += strength*(sustained ? .025 : .5);
    }

    // override base method to add in sustained/transient identifier
    std::string getParamStr() override {
        std::stringstream stream;
        // JSON formatting using raw string literals
        stream << R"({"type": ")" << type << R"(", "sustained": )" << sustained << R"(, "diam": )" << diam;
        stream << R"(, "centre_rad": )" << centre_rad << R"(, "surround_rad": )" << surround_rad;
        stream << R"(, "dtau": )" << dtau << "}";
        std::string params = stream.str();
        return params;
    }
};


#endif //RETINA_ONALPHA_H
