

#include "sandbox.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <range/v3/all.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "pybind11/embed.h"

struct M {
   M() { std::cout << "CTOR\n"; }
   ~M() { std::cout << "DTOR\n"; }
   M(const M&) { std::cout << "COPY\n"; }
   M(M&&) { std::cout << "MOVE\n"; }
   M& operator=(const M&)
   {
      std::cout << "COPY assignment\n";
      return *this;
   }
   M& operator=(M&&)
   {
      std::cout << "MOVE assignment\n";
      return *this;
   }

};


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

inline py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
   py::buffer_info buf1 = input1.request(), buf2 = input2.request();

   if (buf1.ndim != 1 || buf2.ndim != 1)
      throw std::runtime_error("Number of dimensions must be one");

   if (buf1.size != buf2.size)
      throw std::runtime_error("Input shapes must match");

   /* No pointer is passed, so NumPy will allocate the buffer */
   auto result = py::array_t<double>(buf1.size);

   py::buffer_info buf3 = result.request();

   double *ptr1 = static_cast<double *>(buf1.ptr);
   double *ptr2 = static_cast<double *>(buf2.ptr);
   double *ptr3 = static_cast<double *>(buf3.ptr);

   for (size_t idx = 0; idx < buf1.shape[0]; idx++)
      ptr3[idx] = ptr1[idx] + ptr2[idx];

   return result;
}

int main()
{
   py::scoped_interpreter g{};
   py::exec("import sys; print(sys.executable)");
   py::module_ sys = py::module_::import("sys");
   py::print(sys.attr("path"));
   py::exec("import numpy");
//   py::module_ m = py::module_::import("numpy.core.multiarray");


//   py::array_t<double> npArray = py::array_t<double>(6);
//   py::array_t<double> npArray2 = py::array_t<double>(7);
//
//   auto rest = add_arrays(npArray, npArray2);



   return 0;
}
