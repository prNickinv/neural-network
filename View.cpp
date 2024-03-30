#include "View.h"

#include <algorithm>
#include <numeric>

namespace NeuralNetwork {

std::vector<int> View::GenerateViewVector(int data_size) {
  std::vector<int> view_vector(data_size);
  std::iota(view_vector.begin(), view_vector.end(), 0);
  return view_vector;
}

void View::ShuffleViewVector(std::vector<int>& view_vector) {
  std::shuffle(view_vector.begin(), view_vector.end(), gen_);
}

std::mt19937_64 View::gen_{rand_seed_};

} // namespace NeuralNetwork
