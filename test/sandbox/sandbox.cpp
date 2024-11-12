

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

// #include "pybind11/embed.h"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "xtensor/xarray.hpp"

struct M {
   M(int i_) : i(i_) { std::cout << "CTOR[i=" << i << "]\n"; }
   ~M() { std::cout << "DTOR[i=" << i << "]\n"; }
   M(const M&) { std::cout << "COPY[i=" << i << "]\n"; }
   M(M&&) { std::cout << "MOVE[i=" << i << "]\n"; }
   M& operator=(const M&)
   {
      std::cout << "COPY[i=" << i << "] assignment\n";
      return *this;
   }
   M& operator=(M&&)
   {
      std::cout << "MOVE=[i=" << i << "] assignment\n";
      return *this;
   }

   auto operator+(const M& other) { return i + other.i; }

   int i;
};

std::ostream& operator<<(std::ostream& os, const M& m)
{
   return os << "M[i=" << m.i << "]";
}

template <>
struct fmt::formatter< M >: public fmt::ostream_formatter {};

#include "reinforce/utils/hof.hpp"
#include "reinforce/utils/utils.hpp"

using namespace force;
using namespace hof;

int main()
{
   auto k = std::make_unique< std::tuple< M, M > >(1, 2);
   auto p = std::make_unique< std::tuple< M, M > >(3, 4);

   fmt::println(
      "Values & Addresses:\n\t{} - {}\n\t{} - {}", *k, fmt::ptr(k.get()), *p, fmt::ptr(p.get())
   );

   std::cout << "End of manual construction." << std::endl;

   fmt::println("--------------------------------------");

   auto f = compose<  //
      hof::fork< 2 >,  //
      construct< std::tuple >,  //
      map< force::detail::dereffer > >{};

   auto fv = f(k, p);

   fmt::println(
      "Function:\n"
      "\t compose<\n"
      "\t\t hof::fork<2>,\n"
      "\t\t constructor<std::tuple>,\n"
      "\t\t map<dereffer>\n"
      "\t >:\n"
   );

   fmt::println("Values:\n\t{}", fv);

   fmt::println(
      "Values & Addresses:\n\t{} - {}\n\t{} - {}\n\t{} - {}\n\t{} - {}\n\t{} - {}\n\t{} - {}",
      std::get< 0 >(std::get< 0 >(fv)),
      fmt::ptr(&std::get< 0 >(std::get< 0 >(fv))),
      std::get< 1 >(std::get< 0 >(fv)),
      fmt::ptr(&std::get< 1 >(std::get< 0 >(fv))),
      std::get< 0 >(std::get< 1 >(fv)),
      fmt::ptr(&std::get< 0 >(std::get< 1 >(fv))),
      std::get< 1 >(std::get< 1 >(fv)),
      fmt::ptr(&std::get< 1 >(std::get< 1 >(fv))),
      std::get< 0 >(fv),
      fmt::ptr(&std::get< 0 >(fv)),
      std::get< 1 >(fv),
      fmt::ptr(&std::get< 1 >(fv))
   );

   fmt::println("--------------------------------------");

   auto g = compose<  //
      zip_map< project< 0 >, project< 1 > >,  //
      hof::fork< 2 >,  //
      construct< std::tuple >,  //
      map< force::detail::dereffer > >{};

   auto value = 3;
   auto gv = g(k, p);

   fmt::println(
      "Function:\n"
      "\t compose<\n"
      "\t\t zip_map<project<0>, project<1>>,\n"
      "\t\t hof::fork<2>,\n"
      "\t\t constructor<std::tuple>,\n"
      "\t\t map<dereffer>\n"
      "\t >"
   );

   fmt::println("Values:\n\t{}", gv);

   fmt::println("--------------------------------------");

   auto h = compose<  //
      std::plus<>,  //
      zip_map< project< 1 >, project< 0 > >,  //
      zip_map< project< 0 >, project< 1 > >,  //
      hof::fork< 2 >,  //
      construct< std::tuple >,  //
      map< force::detail::dereffer > >{};

   auto hv = h(k, p);

   fmt::println(
      "Function:\n"
      "\t compose<\n"
      "\t\t std::plus<>,\n"
      "\t\t zip_map<project<1>, project<0>>,\n"
      "\t\t zip_map<project<0>, project<1>>,\n"
      "\t\t hof::fork<2>,\n"
      "\t\t constructor<std::tuple>,\n"
      "\t\t map<dereffer>\n"
      "\t >"
   );

   fmt::println("Values:\n\t{}", hv);

   fmt::println("--------------------------------------");

   std::cout << "The end." << std::endl;
};