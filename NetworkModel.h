#ifndef RETINA_NETWORKMODEL_H
#define RETINA_NETWORKMODEL_H

#ifdef _WIN32
    #include <windows.h>
#endif

#include <iostream>
#include <tuple>
#include <vector>
#include <chrono>
#include <random>

#include <Eigen/Dense>

#include "Cell.h"
#include "OnDSGC.h"
#include "OnOffDSGC.h"
#include "BasicCell.h"
#include "LocalEdgeDetector.h"
#include "OnAlpha.h"
#include "OffAlpha.h"
#include "OnOSGC.h"

#include "Stim.h"
#include "utils.h"
#include "type_defs.h"


class NetworkModel {
private:
    std::array<int, 2> dims;
    std::array<double, 2> origin;
    Eigen::VectorXd xvec;                  // coordinate range (along rows) grid used to calculate masks
    Eigen::VectorXd yvec;                  // coordinate range (down columns) grid used to calculate masks
    Eigen::VectorXd xOnes;                  // coordinate range (along rows) grid used to calculate masks
    Eigen::VectorXd yOnes;                  // coordinate range (down columns) grid used to calculate masks
    int margin;                             // cell free margin (allow stimulus to move in from outside)
    int tstop;                              // end time of a simulation run
    double dt;                              // timestep of network
    double t;                               // current time
    int runs;                               // number of runs completed in this Network simulation

    std::vector<Cell*> cells;                // Pointers to all Cell objects residing in this Network
    std::vector<double> cell_Xs;            // Centre X coordinates of all cells in the network
    std::vector<double> cell_Ys;            // Centre Y coordinates of all cells in the network
    std::vector<Stim> stims;                // All Stimuli objects added to the Network

    std::random_device rd;  // Obtain a seed for the random number engine
    std::mt19937 gen;  // mersenne_twister_engine, to be seeded with rd

public:
    NetworkModel(const std::array<int, 2> net_dims, const int cell_margin, const int time_stop, const double delta) {
        // spatial
        dims = net_dims;
        origin[0] = dims[0]/2.0, origin[1] = dims[1]/2.0;
        std::tie(xvec, yvec) = gridVecs(dims[0], dims[1]);
        xOnes = Eigen::VectorXd::Ones(dims[0]);
        yOnes = Eigen::VectorXd::Ones(dims[1]);
        margin = cell_margin;
        // temporal
        tstop = time_stop;
        dt = delta;
        t = 0;
        runs = 0;
        // seed random number generation
        gen = std::mt19937 (rd());
    }

    std::tuple<double, double> getOrigin() {
        return std::make_tuple(origin[0], origin[1]);
    }

    void populate(const int spacing, double jitter) {
        double theta, radius;
        std::array<double, 2> pos = {0, 0};

        // Regularly spaced axes giving the default layout of cells in the network.
        Eigen::VectorXd xgridvec = Eigen::VectorXd::LinSpaced((dims[1]-margin*2)/spacing, margin, dims[1]-margin);
        Eigen::VectorXd ygridvec = Eigen::VectorXd::LinSpaced((dims[1]-margin*2)/spacing, margin, dims[1]-margin);

        // uniform distribution for angle cell is offset from grid-point
        std::uniform_real_distribution<> Uniform(0, 1.0);
        // Normal distribution for distance cell is offset from grid-point
        std::normal_distribution<> Gaussian{0, 1};

        for (int i = 0; i < xgridvec.size(); ++i) {
            for (int j = 0; j < ygridvec.size(); ++j) {

                theta = 2 * 3.14159265359 * Uniform(gen);
                radius = Gaussian(gen) * jitter;

                pos[0] = xgridvec[i] + radius * cos(theta);
                pos[1] = ygridvec[j] + radius * sin(theta);

                // build cell of random type and add pointer to cells vector
                cells.push_back(buildRandomCell(pos));
                cell_Xs.push_back(pos[0]), cell_Ys.push_back(pos[1]);
            }
        }
    }

    Cell* buildRandomCell (const std::array<double, 2> cell_pos) {
        std::uniform_int_distribution<> IntDist(0, 5); // distribution in range (inclusive)
        int r = IntDist(gen);
        switch (r) {
            case 0:
                return new OnDSGC(xvec, yvec, xOnes, yOnes, dt, cell_pos, gen);
            case 1:
                return new OnOffDSGC(xvec, yvec, xOnes, yOnes, dt, cell_pos, gen);
            case 2:
                return new LocalEdgeDetector(xvec, yvec, xOnes, yOnes, dt, cell_pos);
            case 3:
                return new OnAlpha(xvec, yvec, xOnes, yOnes, dt, cell_pos, gen);
            case 4:
                return new OffAlpha(xvec, yvec, xOnes, yOnes, dt, cell_pos, gen);
            case 5:
                return new OnOSGC(xvec, yvec, xOnes, yOnes, dt, cell_pos, gen);
            default:
                return nullptr; // should never come here...
        }
    }

    std::vector<Cell*> getCells() {
        return cells;
    }

    // Cell Coordinates organized into a Nx2 matrix (for output to file)
    Eigen::Matrix<double, Eigen::Dynamic, 2> getCellXYs() {
        Eigen::MatrixXd cell_XYs = Eigen::MatrixXd::Ones(cell_Xs.size(), 2);
        for(std::size_t i = 0; i < cell_Xs.size(); ++i){
            cell_XYs(i, 0) = cell_Xs[i];
            cell_XYs(i, 1) = cell_Ys[i];
        }
        return cell_XYs;
    }

    // Construct table with columns of all cell recordings (for output to file)
    Eigen::MatrixXd getRecTable() {
        int recLen = cells[0] -> getRec().size();
        Eigen::MatrixXd all_recs = Eigen::MatrixXd::Ones(recLen, cells.size());
        for(std::size_t c = 0; c < cells.size(); ++c){
            auto rec = cells[c] -> getRec();
            for(int i = 0; i < recLen; ++i) {
                all_recs(i, c) = rec[i];
            }
            cells[c] -> clearRec();
        }
        return all_recs;
    }

    void clearCells() {
        // de-allocate memory of each of the Cell class pointers
        for(auto& cell : cells){
            delete cell;
        }
        // clear the vector the pointers are stored in.
        cells.clear();
        // clear cell coordinate vectors
        cell_Xs.clear();
        cell_Ys.clear();
    }

    // Generating new Stimuli within NetworkModel for ease of access to it's variables.
    void newStim(const std::array<double, 2> start_pos, const int time_on, const int time_off, const double velocity,
                 const double direction, const double orientation, const double amplitude, const double change,
                 const std::string &type, const double radius=50, const double width=50, const double length=100) {

        stims.emplace_back(dims, xvec, yvec, xOnes, yOnes, dt, start_pos, time_on, time_off, velocity,
                           direction, orientation, amplitude, change);

        if (type == "bar"){
            stims[stims.size()-1].setBar(width, length);
        } else if (type == "circle"){
            stims[stims.size()-1].setCircle(radius);
        } else if (type == "ellipse"){
            stims[stims.size()-1].setEllipse(width, length);
        }
    }

    std::vector<Stim> getStims() {
        return stims;
    }

    void clearStims() {
        stims.clear();
    }

    void step() {
        for(auto& stim : stims){
            stim.move();

            for(auto& cell : cells){
                cell -> stimulate(stim);
            }
        }
        //std::cout << "\n";

        for(auto& cell : cells){
            cell -> decay();
        }
        t += dt;
    }

    void run(const std::string &folder, const std::string &label) {
        auto run_start = Clock::now();
        t = 0;
        for(int i = 0; i*dt < tstop; ++i){
            step();
        }
        for(auto& cell : cells){
            cell -> setVm(0);
        }
        ++runs;
        runToFile(folder, label);
        auto run_time = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - run_start).count();
        std::cout << run_time << " milliseconds run time\n";
    }

    void runToFile(const std::string &folder, const std::string &label) {
        #ifdef _WIN32
            CreateDirectory((folder+label).c_str(), nullptr);
        #else
            auto err = system(("mkdir -p '" + folder+label + "'").c_str());
        #endif

        MatrixXdToCSV(folder + label + "/cellRecs.csv", getRecTable());
        for (std::size_t i = 0; i < stims.size(); ++i) {
            MatrixXdToCSV(folder + label + "/stimRecs" + std::to_string(i) + ".csv", stims[i].getRecTable());
            stims[i].saveParams(folder + label + "/stimParams" + std::to_string(i) + ".txt");
        }
    }

    void netToFile(const std::string &folder) {
        #ifdef _WIN32
            CreateDirectory((folder).c_str(), nullptr);
        #else
            auto err = system(("mkdir -p '" + folder + "'").c_str());
        #endif


        MatrixXdToCSV(folder + "/cellMat.csv", cellMatrix());
        MatrixXdToCSV(folder + "/cellCoords.csv", getCellXYs());

        // global parameters of the network
        std::ofstream netParamFile;
        netParamFile.open(folder + "/netParams.txt");
        netParamFile << R"({"xdim": )" << dims[0] << R"(, "ydim": )" << dims[1] << R"(, "margin": )" << margin;
        netParamFile << R"(, "tstop": )" << tstop << R"(, "dt": )" << dt << "}";
        netParamFile.close();
        // parameters for each cell in the network
        std::ofstream cellParamFile;
        cellParamFile.open(folder + "/cellParams.txt");
        for(auto& cell : cells){
            cellParamFile << cell -> getParamStr() << "\n";
        }
        cellParamFile.close();
    }

    // Add up spatial masks of all cells and their receptive fields for display (for output to file)
    Eigen::MatrixXd cellMatrix() {
        Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(dims[0], dims[1]);
        for(auto& cell : cells){
            mat += cell -> getSoma().cast<double>();
            mat += cell -> getRFCentre().cast<double>()*.2;
        }
        return mat;
    }
};

#endif //RETINA_NETWORKMODEL_H
