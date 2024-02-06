cmake_minimum_required(VERSION 3.27)
project(neural-network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(neural-network main.cpp
        NormalRandom.cpp
        MSE.cpp
        ReLu.cpp
        Sigmoid.cpp
        RegularNode.cpp
        ActivatedNode.cpp
        Layer.cpp)
target_link_libraries(neural-network Eigen3::Eigen)
