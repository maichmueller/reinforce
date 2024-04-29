
#ifndef REINFORCE_SPACES_MULTI_BINARY_HPP
#define REINFORCE_SPACES_MULTI_BINARY_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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
#include <xtensor/xoperation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xstorage.hpp>

#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"
#include "space.hpp"

namespace force {

class MultiBinarySpace: public Space< xarray< int8_t >, MultiBinarySpace > {
  public:
   friend class Space;
   using base = Space;
   using data_type = int8_t;
   using typename base::value_type;
   using typename base::multi_value_type;
   using base::shape;
   using base::rng;

   template < std::convertible_to< int > T >
   explicit MultiBinarySpace(const xarray< T >& shape, std::optional< size_t > seed = std::nullopt)
       : base(xt::svector< int >(shape.begin(), shape.end()), seed)
   {
      if(shape.dimension() > 1) {
         throw std::invalid_argument(fmt::format(
            "Shape of given array has to be one-dimensional (flat). Passed: {}", shape.dimension()
         ));
      }
   }

   template < std::convertible_to< int > T >
   explicit MultiBinarySpace(
      const xt::svector< T >& shape,
      std::optional< size_t > seed = std::nullopt
   )
       : base(std::move(shape), seed)
   {
   }

   template < std::convertible_to< int > T >
   MultiBinarySpace(std::initializer_list< T > shape, std::optional< size_t > seed = std::nullopt)
       : base(std::move(shape), seed)
   {
   }

   template < typename Rng >
      requires std::convertible_to< ranges::value_type_t< detail::raw_t< Rng > >, int >
   explicit MultiBinarySpace(const Rng& shape, std::optional< size_t > seed = std::nullopt)
       : base(xt::svector< int >(shape.begin(), shape.end()), seed)
   {
   }

   bool operator==(const MultiBinarySpace& rhs) const
   {
      return ranges::equal(shape(), rhs.shape());
   }

   std::string repr() { return fmt::format("MultiBinary({})", shape()); }

  private:
   [[nodiscard]] xt::svector< int > samples_shape(size_t nr_samples) const;

   [[nodiscard]] multi_value_type _sample(size_t nr_samples, std::nullopt_t = std::nullopt) const;

   [[nodiscard]] multi_value_type _sample(size_t nr_samples, const value_type& mask) const;

   [[nodiscard]] multi_value_type _sample(std::nullopt_t = std::nullopt) const
   {
      return _sample(1);
   }
   [[nodiscard]] multi_value_type _sample(const value_type& mask) const { return _sample(1, mask); }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      const auto& incoming_shape = value.shape();
      const auto& incoming_dim = incoming_shape.size();
      const auto space_dim = shape().size();

      if(incoming_dim < space_dim or space_dim + 1 < incoming_dim) {
         return false;
      }
      return ranges::all_of(
                ranges::views::zip(
                   shape(),
                   incoming_shape | ranges::views::drop(size_t{space_dim + 1 == incoming_dim})
                ),
                [](auto pair) { return std::cmp_equal(std::get< 0 >(pair), std::get< 1 >(pair)); }
             )
             and xt::all(xt::less_equal(value, 1) and xt::greater_equal(value, 0));
   }
};

}  // namespace force

#endif  // REINFORCE_SPACES_MULTI_BINARY_HPP
