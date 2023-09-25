

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

#include <xtensor-python/pyarray.hpp>

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

template < typename T >
class MyC {
   template < typename U >
   MyC(std::variant< U, std::vector< U > > v) : m_v(v)
   {
   }

  private:
   std::variant< T, std::vector< T > > m_v;
};

template < typename U >
MyC(std::variant< U, std::vector< U > > v) -> MyC< U >;

int main()
{
   pybind11::scoped_interpreter g{};
   xt::import_numpy();

   //   MyC{3};
   //   //   auto arr = xt::pyarray<double>::from_shape(xt::svector{2,3});
   //      auto arr = xt::pyarray< double >{{0, 1, 2}, {3, 4, 5}};
   xt::xarray< double > arr2{
      {std::numeric_limits< double >::infinity(), -std::numeric_limits< double >::infinity(), 0}};
   //      xt::xarray< double > arr2 = xt::adapt(
   //      std::move(arr.ptr()), 6, xt::acquire_ownership{}, xt::svector{2, 3}
   //   );
   //   arr.release();
   std::cout << "is inf: " << std::isinf(arr2(0)) << std::endl;
   std::cout << "is inf: " << std::isinf(arr2(1)) << std::endl;
   std::cout << "is inf: " << std::isinf(arr2(2)) << std::endl;
      std::cout << arr2 << std::endl;
   //   std::cout << arr2 << std::endl;
   //   arr2 = std::move(arr);
   //   std::cout << arr << std::endl;
   //   std::cout << arr2 << std::endl;
}
