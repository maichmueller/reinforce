

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

#include <span>
#include <xtensor-python/pyarray.hpp>

#include "xtensor/xadapt.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include <xtensor/xstrided_view.hpp>
#include "reinforce/utils/xarray_formatter.hpp"

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
   //   pybind11::scoped_interpreter g{};
   //   xt::import_numpy();
   using T = double;
   xt::xarray< float > arr1 = xt::arange(0, 90);
//   arr1 = arr1.reshape(xt::svector{90});
//   auto coordinates = xt::unravel_index(0, xt::svector{3,3});
   // add all entries of the samples spot
//   xt::xstrided_slice_vector index_stride(coordinates.begin(), coordinates.end());
   xt::xstrided_slice_vector index_stride{};
   index_stride.emplace_back(xt::all());
   auto entry_view = xt::strided_view(arr1, index_stride);
   auto draw_shape = xt::svector<size_t>{90};
   auto draw = xt::random::randn< T >(draw_shape, T(0), T(1.));
   entry_view = draw;
   std::cout << entry_view;
   //   MyC{3};
   //   //   auto arr = xt::pyarray<double>::from_shape(xt::svector{2,3});
   //      auto arr = xt::pyarray< double >{{0, 1, 2}, {3, 4, 5}};
   //   xt::xarray< double > arr2{
   //      {std::numeric_limits< double >::infinity(), -std::numeric_limits< double >::infinity(),
   //      0}};
   //   //      xt::xarray< double > arr2 = xt::adapt(
   //   //      std::move(arr.ptr()), 6, xt::acquire_ownership{}, xt::svector{2, 3}
   //   //   );
   //   //   arr.release();
   //   std::cout << "is inf: " << std::isinf(arr2(0)) << std::endl;
   //   std::cout << "is inf: " << std::isinf(arr2(1)) << std::endl;
   //   std::cout << "is inf: " << std::isinf(arr2(2)) << std::endl;
   //   std::cout << arr2 << std::endl;
   //   arr2 = std::move(arr);
   //   std::cout << arr << std::endl;
   //   std::cout << arr2 << std::endl;
}
