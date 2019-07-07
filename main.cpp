#ifdef _WIN32
    #include <windows.h>
#endif



#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
//#include <eigen3-hdf5.hpp>

#include "type_defs.h"
#include "utils.h"
#include "stimuli.h"

#include "NetworkModel.h"
#include "Cell.h"
#include "Stim.h"


using namespace Eigen;

int main() {
    // Eigen::setNbThreads(4);
    std::cout << "threads used by Eigen: " << Eigen::nbThreads( ) << std::endl;
    #ifdef _WIN32
        std::string baseFolder = "D://retina-sim-data/";
    #else
        std::string baseFolder = "/media/geoff/Data/retina-sim-data/";
    #endif
    std::string netFolder;
    std::cout << "Use base folder " << baseFolder << "? [Y/N]" << std::endl;
    std::string answer;
    std::cin >> answer;
    if (answer.find('N') != std::string::npos || answer.find('n') != std::string::npos){
        std::cout << "Enter new base folder path:" << std::endl;
        std::cin >> baseFolder;
    }


    std::array<int, 3> net_dims = {700, 700, 4};  // third element is spatial downsampling factor
    NetworkModel net(net_dims, 200, 3000, 5.0);  // margin is subject to downsampling

    for(int i = 0; i < 20; ++i) {
        // fill the empty network object with cells
        std::cout << "Constructing net" << i << "..." << std::endl;
        net.populate(20, 10.0);  // subject to spatial downsampling
        std::cout << "Number of cells: " << net.getCells().size() << std::endl;

        // create network directory
        netFolder = baseFolder+"net"+std::to_string(i)+"/";

        #ifdef _WIN32
            CreateDirectory((netFolder).c_str(), nullptr);
        #else
            auto err = system(("mkdir -p '" + netFolder + "'").c_str());
        #endif

        // run series of stimuli, saving results along the way
        runExperiment_1(net, netFolder);

        std::cout << "saving network information...\n\n";
        net.netToFile(netFolder);
        net.clearCells();
    }

    // wait for ENTER to close terminal
    std::string wait;
    std::cout << "Type anything and hit ENTER to terminate..." << std::endl;
    std::cin >> wait;

    return 0;
}