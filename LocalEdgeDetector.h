//
// Created by geoff on 2019-05-10.
//

#ifndef RETINA_LOCALEDGEDETECTOR_H
#define RETINA_LOCALEDGEDETECTOR_H

#include "Cell.h"
#include "utils.h"
#include "type_defs.h"


class LocalEdgeDetector : public Cell {
public:
    LocalEdgeDetector(Eigen::VectorXd &xgrid, Eigen::VectorXd &ygrid, Eigen::VectorXd &xOnes, Eigen::VectorXd &yOnes,
                      const double net_dt, const std::array<double, 2> cell_pos)
                      :Cell(xgrid, ygrid, xOnes, yOnes, net_dt, cell_pos) {
        type = "LocalEdgeDetector";
        // spatial properties
        diam = 8;  // 15
        centre_rad = 23;  // 45
        surround_rad = 50;
        somaMask = circleMask(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, diam/2);
        std::tie(rfCentre, rfSurround) = buildRF(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, centre_rad, surround_rad);
        rfCentre_sparse = rfCentre.sparseView();  // convert from dense matrix to sparse
        rfSurround_sparse = rfSurround.sparseView();  // convert from dense matrix to sparse
        // active / synaptic properties
        sustained = false;
        onoff = true;
        dtau = 100;
    }

    void stimulate(const Stim &stim) override {
        // use stimulus itself if sustained, and delta of stimulus if transient
        Eigen::SparseMatrix<int> stim_mask = sustained ? stim.getSparseMask() : stim.getSparseDelta();

        Eigen::SparseMatrix<int> centre_overlap, surround_overlap;
        centre_overlap = stim_mask.cwiseProduct(rfCentre_sparse);
        surround_overlap = stim_mask.cwiseProduct(rfSurround_sparse);

        // take absolute value of overlap since this is an ON-OFF cell
        centre_overlap = centre_overlap.cwiseProduct(centre_overlap).cwiseSqrt();
        surround_overlap = surround_overlap.cwiseProduct(surround_overlap).cwiseSqrt();

        double strength = (centre_overlap.sum() - surround_overlap.sum()*.1) * abs(stim.getAmp());

        Vm += strength*6;
    }
};


#endif //RETINA_LOCALEDGEDETECTOR_H
