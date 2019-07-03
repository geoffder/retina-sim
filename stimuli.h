#ifndef RETINA_STIMULI_H
#define RETINA_STIMULI_H

#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "type_defs.h"
#include "utils.h"
#include "NetworkModel.h"

void runExperiment_1(NetworkModel &net, std::string &netFolder);

#endif //RETINA_STIMULI_H
