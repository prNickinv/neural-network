#ifndef SCHEDULERUTILS_H
#define SCHEDULERUTILS_H

#include <variant>

#include "ExponentialDecay.h"
#include "PolynomialDecay.h"
#include "StepDecay.h"

namespace NeuralNetwork {

using Scheduler =
    std::variant<double, ExponentialDecay, PolynomialDecay, StepDecay>;

namespace SchedulerUtils {

// Overload pattern & deduction guide
template<class... Ts>
struct Overload : Ts... {
  using Ts::operator()...;
};
template<class... Ts>
Overload(Ts...) -> Overload<Ts...>;

Scheduler GetScheduler(std::istream&);

} // namespace SchedulerUtils

} // namespace NeuralNetwork

#endif //SCHEDULERUTILS_H
