cmake_minimum_required(VERSION 3.27)
project(neural-network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH third-party-library/eigen mnist-fashion)
find_package(Eigen3 3.4 NO_MODULE)
if (NOT Eigen3_FOUND)
    include_directories(${CMAKE_SOURCE_DIR}/third-party-library/eigen/)
endif ()
include_directories(third-party-library/EigenRand)

find_package(MNIST)
include_directories(${MNIST_INCLUDE_DIR})
add_definitions(-DMNIST_DATA_DIR="${MNIST_DATA_DIR}" -DMNIST_FASHION_DATA_DIR="${MNIST_FASHION_DATA_DIR}")

add_executable(neural-network main.cpp
        ActivationFunction.cpp
        LossFunction.cpp
        Layer.cpp
        Network.cpp
        View.cpp
        Data.cpp
        Except.cpp
        MnistTest.cpp)

if (Eigen3_FOUND)
    target_link_libraries(neural-network Eigen3::Eigen)
endif ()
