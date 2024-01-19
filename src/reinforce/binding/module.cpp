#ifndef REINFORCE_PYTHON_MODULE_BINDING_HPP
#define REINFORCE_PYTHON_MODULE_BINDING_HPP

#include <pybind11/pybind11.h>

#include "declarations.hpp"
#include "reinforce/reinforce.hpp"

namespace py = pybind11;

using namespace force;

PYBIND11_MODULE(_reinforce, m) {
   pyforce::init::env_gridworld(m);
}

#endif  // REINFORCE_PYTHON_MODULE_BINDING_HPP
