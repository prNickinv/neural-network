#ifndef GLOBALUSINGS_H
#define GLOBALUSINGS_H

#include <vector>

#include <Eigen/Dense>

namespace NeuralNetwork {

using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;
using Vectors = std::vector<Vector>;

} // namespace NeuralNetwork

#endif //GLOBALUSINGS_H
