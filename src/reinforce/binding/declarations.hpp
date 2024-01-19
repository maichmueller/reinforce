
#ifndef REINFORCE_PYBINDING_DECLARATIONS_HPP
#define REINFORCE_PYBINDING_DECLARATIONS_HPP

#include <pybind11/pybind11.h>

#include "reinforce/reinforce.hpp"

namespace py = pybind11;

namespace pyforce {

struct init {

   template < size_t dim >
   static void env_gridworld(py::module_& m);

};
}  // namespace pyforce
#include "declarations.tcc"

#endif  // REINFORCE_PYBINDING_DECLARATIONS_HPP
