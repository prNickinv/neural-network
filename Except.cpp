#include "Except.h"

#include <exception>
#include <iostream>

namespace Except {

void React() {
  try {
    throw;
  } catch (std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception caught" << std::endl;
  }
}

} // namespace Except
