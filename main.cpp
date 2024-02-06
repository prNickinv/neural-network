#include <iostream>
#include <random>
#include <utility>

#include <Eigen/Dense>

#include "MSE.h"
#include "Network.h"
#include "Sigmoid.h"

int main() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution distrib(0.0, 1.0);
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> example_data;
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> val_data;

  for (int i = 0; i != 500; ++i) {
    Eigen::VectorXd input = Eigen::VectorXd::NullaryExpr(
        2, [&]() { return std::round((distrib(generator))); });
    // if the values of the input vector are equal, the target is 1, otherwise 0;
    Eigen::VectorXd target{{input(0) == input(1) ? 1.0 : 0.0}};
    example_data.emplace_back(input, target);
  }

  for (int i = 0; i != 500; ++i) {
    Eigen::VectorXd input = Eigen::VectorXd::NullaryExpr(
        2, [&]() { return std::round((distrib(generator))); });
    Eigen::VectorXd target{{input(0) == input(1) ? 1.0 : 0.0}};
    val_data.emplace_back(input, target);
  }

  std::vector<int> layers_data{2, 10, 5, 1};
  NeuralNetwork::Network<NeuralNetwork::Sigmoid, NeuralNetwork::MSE>
      example_net(layers_data, 2.0 / 100, 1, 50);

  example_net.TrainNetwork(example_data, val_data);

  return 0;
}
