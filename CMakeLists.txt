cmake_minimum_required(VERSION 3.13)
project(retina-sim)
set(CMAKE_CXX_STANDARD 17)

if (WIN32)
 include_directories(C:/eigen3)
endif (WIN32)

if (UNIX)
    include_directories(~/eigen3)
endif (UNIX)

#include_directories(C:/eigen3-hdf5-master)
#include_directories(C:/HDF_Group/HDF5/1.10.5/include)

add_executable(retina-sim
        main.cpp Cell.cpp Cell.h NetworkModel.cpp NetworkModel.h Stim.cpp Stim.h utils.cpp utils.h type_defs.h OnOffDSGC.cpp OnOffDSGC.h BasicCell.cpp BasicCell.h LocalEdgeDetector.cpp LocalEdgeDetector.h OnAlpha.cpp OnAlpha.h OffAlpha.cpp OffAlpha.h OnDSGC.cpp OnDSGC.h OnOSGC.cpp OnOSGC.h stimuli.cpp stimuli.h)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp -march=native")
#SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -march=native)

# linking libraries statically to make sure that .exe has all the MinGW stuff it needs
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
# linking libgcc and libstdc++ alone still gave an error about missing libgomp-1.dll when running executable
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++ -static")