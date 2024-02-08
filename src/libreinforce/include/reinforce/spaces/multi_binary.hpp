
#ifndef REINFORCE_MULTI_BINARY_HPP
#define REINFORCE_MULTI_BINARY_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <iterator>
#include <optional>
#include <random>
#include <range/v3/all.hpp>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xstorage.hpp>

#include "mono_space.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

class TypedMultiBinarySpace: public TypedMonoSpace< int8_t, TypedMultiBinarySpace > {
  public:
   using value_type = int8_t;
   friend class TypedMonoSpace;
   using base = TypedMonoSpace;
   using base::shape;
   using base::rng;

   template < std::convertible_to< int > T >
   TypedMultiBinarySpace(const xarray< T >& shape, std::optional< size_t > seed = std::nullopt)
       : base(xt::svector< int >(shape.begin(), shape.end()), seed)
   {
      if(shape.dimension() > 1) {
         throw std::invalid_argument(fmt::format(
            "Shape of given array has to be one-dimensional (flat). Passed: {}", shape.dimension()
         ));
      }
   }

   template < std::convertible_to< int > T >
   TypedMultiBinarySpace(const xt::svector< T >& shape, std::optional< size_t > seed = std::nullopt)
       : base(std::move(shape), seed)
   {
   }

   template < std::convertible_to< int > T >
   TypedMultiBinarySpace(
      std::initializer_list< T > shape,
      std::optional< size_t > seed = std::nullopt
   )
       : base(std::move(shape), seed)
   {
   }

   template < typename Rng >
      requires std::convertible_to< ranges::value_type_t< detail::raw_t< Rng > >, int >
   TypedMultiBinarySpace(const Rng& shape, std::optional< size_t > seed = std::nullopt)
       : base(xt::svector< int >(shape.begin(), shape.end()), seed)
   {
   }

   bool operator==(const TypedMultiBinarySpace& rhs) const
   {
      return ranges::equal(shape(), rhs.shape());
   }

   std::string repr() { return fmt::format("MultiBinary({})", shape()); }

  private:
   xarray< value_type >
   _sample(size_t nr_samples, const std::optional< xarray< value_type > >& mask = {});

   xarray< value_type > _sample(const std::optional< xarray< value_type > >& mask = {})
   {
      return _sample(1, mask);
   }

   bool _contains(const value_type& value) const { return value <= 1; }
};

}  // namespace force

#endif  // REINFORCE_MULTI_DISCRETE_HPP
