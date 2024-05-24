#ifndef ADAMWOPTIMIZER_H
#define ADAMWOPTIMIZER_H

#include <iostream>

#include <Eigen/Dense>

#include "GlobalUsings.h"

namespace NeuralNetwork {

struct AdamWMoments {
  Matrix m_w; // First moment for weights
  Matrix v_w; // Second moment for weights
  Vector m_b; // First moment for bias
  Vector v_b; // Second moment for bias
};

class AdamWOptimizer {
 public:
  AdamWOptimizer() = default;
  AdamWOptimizer(Index, Index);
  explicit AdamWOptimizer(std::istream&);

  void UpdateMoments(const Matrix&, const Vector&);
  AdamWMoments ComputeCorrectedMoments() const;
  double GetEpsilon() const;

  friend std::ostream& operator<<(std::ostream&, const AdamWOptimizer&);
  friend std::istream& operator>>(std::istream&, AdamWOptimizer&);

 private:
  double beta1_{0.9};
  double beta2_{0.999};
  double epsilon_{1e-8};
  int time_{0};
  AdamWMoments adam_;
};

} // namespace NeuralNetwork

#endif //ADAMWOPTIMIZER_H
