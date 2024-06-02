cmake_minimum_required(VERSION 3.22)
project(neural-network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH third-party-library/eigen mnist-fashion)
include_directories(${CMAKE_SOURCE_DIR}/third-party-library/eigen/)
include_directories(third-party-library/EigenRand)

find_package(MNIST)
include_directories(${MNIST_INCLUDE_DIR})
add_definitions(-DMNIST_DATA_DIR="${MNIST_DATA_DIR}" -DMNIST_FASHION_DATA_DIR="${MNIST_FASHION_DATA_DIR}")

set(NEURAL_NETWORK ${CMAKE_SOURCE_DIR}/ActivationFunction.cpp
        ${CMAKE_SOURCE_DIR}/LossFunction.cpp
        ${CMAKE_SOURCE_DIR}/Layer.cpp
        ${CMAKE_SOURCE_DIR}/MiniBatchGD.cpp
        ${CMAKE_SOURCE_DIR}/AdamWOptimizer.cpp
        ${CMAKE_SOURCE_DIR}/MomentumOptimizer.cpp
        ${CMAKE_SOURCE_DIR}/SchedulerUtils.cpp
        ${CMAKE_SOURCE_DIR}/ExponentialDecay.cpp
        ${CMAKE_SOURCE_DIR}/PolynomialDecay.cpp
        ${CMAKE_SOURCE_DIR}/StepDecay.cpp
        ${CMAKE_SOURCE_DIR}/Network.cpp
        ${CMAKE_SOURCE_DIR}/View.cpp
        ${CMAKE_SOURCE_DIR}/Data.cpp
        ${CMAKE_SOURCE_DIR}/Except.cpp
        ${CMAKE_SOURCE_DIR}/MnistTest.cpp)

add_library(net-lib ${NEURAL_NETWORK})
add_executable(neural-network main.cpp)
target_link_libraries(neural-network net-lib)

add_subdirectory(tests)
