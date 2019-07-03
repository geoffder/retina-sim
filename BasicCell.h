//
// Created by geoff on 2019-05-03.
//

#ifndef RETINA_BASICCELL_H
#define RETINA_BASICCELL_H

#include "Cell.h"
#include "utils.h"
#include "type_defs.h"

class BasicCell : public Cell {
public:
    BasicCell(Eigen::VectorXd &xgrid, Eigen::VectorXd &ygrid, Eigen::VectorXd &xOnes, Eigen::VectorXd &yOnes,
              const double net_dt, const std::array<double, 2> cell_pos)
              :Cell(xgrid, ygrid, xOnes, yOnes, net_dt, cell_pos) {
        type = "Basic";
        // spatial properties
        diam = 8;  // 15
        centre_rad = 100;
        surround_rad = 200;
        somaMask = circleMask(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, diam/2);
        std::tie(rfCentre, rfSurround) = buildRF(*net_xvec, *net_yvec, *net_xOnes, *net_yOnes, pos, centre_rad, surround_rad);
        rfCentre_sparse = rfCentre.sparseView();  // convert from dense matrix to sparse
        rfSurround_sparse = rfSurround.sparseView();  // convert from dense matrix to sparse
        // active / synaptic properties
        sustained = true;
        onoff = false;
        dtau = 10;
    }

};


#endif //RETINA_BASICCELL_H
