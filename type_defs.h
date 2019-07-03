//
// Created by geoff on 2019-03-20.
//

#ifndef RETINA_TYPES_H
#define RETINA_TYPES_H

#include <Eigen/Dense>
#include <chrono>

namespace Eigen{
    // boolean dynamic-size matrix (these turn out to be slower than int matrices, discontinued use.)
    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
    // Using Eigen::IOFormat to describe CSV printing
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
}

// not Eigen, but a type def useful to have for the whole project
typedef std::chrono::high_resolution_clock Clock;

// pair of centre and surround receptive fields (dynamic integer Eigen matrices)
typedef std::tuple<Eigen::MatrixXi, Eigen::MatrixXi> rfPair;

#endif //RETINA_TYPES_H
