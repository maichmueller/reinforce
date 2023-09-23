
#ifndef REINFORCE_PYBINDING_DECLARATIONS_HPP
#define REINFORCE_PYBINDING_DECLARATIONS_HPP

#include <pybind11/pybind11.h>
#include "reinforce/reinforce.hpp"

namespace py = pybind11;

namespace pyforce {

template <size_t dim>
void init_env_gridworld(py::module_& m);

}

#include "declarations.tcc"

#endif  // REINFORCE_PYBINDING_DECLARATIONS_HPP
