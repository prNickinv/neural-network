#ifndef VIEW_H
#define VIEW_H

#include <random>
#include <vector>

#include <Eigen/Dense>

namespace NeuralNetwork {

class View {
 public:
  static std::vector<int> GenerateViewVector(int);
  static void ShuffleViewVector(std::vector<int>&);

 private:
  static constexpr int rand_seed_{42};
  static std::mt19937_64 gen_;
};

} // namespace NeuralNetwork

#endif //VIEW_H
