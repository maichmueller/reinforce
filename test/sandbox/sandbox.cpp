

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
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "xtensor/xarray.hpp"

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

// #include <array>
// #include <random>
// #include <iostream>
//
// #include <range/v3/all.hpp>
//
// #include "fmt/format.h"
// #include "fmt/core.h"
// #include "fmt/std.h"
#define FORCE_IMPORT_ARRAY

int main()
{
   //   pybind11::scoped_interpreter g{};
   //   xt::import_numpy();
   //
   //   spdlog::set_level(spdlog::level::debug);
   //   xt::random::seed(0);
   //   constexpr auto n_samples = 1000;
   //   auto samples = sample_broken(n_samples);
   ////   auto samples = sample_working(n_samples);
   //   SPDLOG_DEBUG(samples);
   //

   return 0;
}
