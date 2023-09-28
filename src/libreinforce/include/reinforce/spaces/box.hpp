#ifndef REINFORCE_BOX_HPP
#define REINFORCE_BOX_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <range/v3/all.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

template < typename T >
   requires std::is_integral_v< T > || std::is_floating_point_v< T >
class TypedBox: public TypedSpace< T > {
  public:
   using base = TypedSpace< T >;
   using base::shape;
   using base::rng;

   template < typename... Args >
   TypedBox(const T& low, const T& high, Args&&... args)
       : TypedBox(xarray< T >{low}, xarray< T >{high}, std::forward< Args >(args)...)
   {
   }
   template < template < typename... > class Array, typename... Args >
      requires detail::is_any_v< Array< T >, pyarray< T >, std::vector< T > >
   TypedBox(const Array< T >& low, const Array< T >& high, Args&&... args)
       : TypedBox(xarray< T >(low), xarray< T >(high), std::forward< Args >(args)...)
   {
   }

   template < typename... Args >
   TypedBox(const xarray< T >& low, const xarray< T >& high, Args&&... args)
       : base(std::forward< Args >(args)...),
         m_low(low),
         m_high(high),
         m_bounded_below(not xt::isinf(low)),
         m_bounded_above(not xt::isinf(high))
   {
      auto low_shape = low.shape();
      auto high_shape = high.shape();

      SPDLOG_DEBUG(fmt::format(
         "Low shape {}, high shape: {}, specified shape: {}", low_shape, high_shape, shape()
      ));
      if(not ranges::equal(high_shape, low_shape)) {
         throw std::invalid_argument(fmt::format(
            "'Low' and 'High' bound arrays need to have the same shape. Given:\n{}\nand\n{}",
            low_shape,
            high_shape
         ));
      }
      if(shape().size() > 0) {
         if(not ranges::equal(shape(), low_shape)) {
            throw std::invalid_argument(fmt::format(
               "Given shape and shape of 'Low' and 'High' bound arrays have to be the. "
               "Given:\n{}\nand\n{}\nand\n{}",
               shape(),
               low_shape,
               high_shape
            ));
         }
      } else {
         shape() = ranges::to<xt::svector<int>>(low_shape | detail::cast);
      }
   }

   bool is_bounded(std::string_view manner = "")
   {
      constexpr auto below = [&] { return xt::all(m_bounded_below); };
      constexpr auto above = [&] { return xt::all(m_bounded_above); };

      if(manner == "below") {
         return below();
      }
      if(manner == "above") {
         return above();
      }
      assert(manner.empty());
      return below() and above();
   }

   xarray< T > sample(const std::optional< xarray< bool > >& /*unused*/ = std::nullopt) override
   {
      xarray< T > samples = xt::empty< T >(shape());
      SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples.shape()));
      for(auto&& [i, bounds] :
          ranges::views::enumerate(ranges::views::zip(m_bounded_below, m_bounded_above))) {
         auto&& [lower_bounded, upper_bounded] = bounds;
         auto& entry = samples.data_element(i);
         switch(auto choice = (lower_bounded ? 1 : -1) + int(upper_bounded)) {
            case -1: {
               // (-infinity, infinity)
               entry = std::normal_distribution< T >{}(rng());
               break;
            }
            case 0: {
               // (-infinity, B]
               entry = m_high.data_element(i) - std::exponential_distribution< T >{1}(rng());
               break;
            }
            case 1: {
               // [A, infinity)
               entry = m_low.data_element(i) + std::exponential_distribution< T >{1}(rng());
               break;
            }
            case 2: {
               // [A, B]
               using distribution = std::conditional_t<
                  std::is_integral_v< T >,
                  std::uniform_int_distribution< T >,
                  std::uniform_real_distribution< T > >;
               entry = distribution{m_low.data_element(i), m_high.data_element(i)}(rng());
               break;
            }
            default: {
               throw std::logic_error(
                  fmt::format("Case generation created unexpected case {}.", choice)
               );
            }
         }
      }
      return samples;
   }

   bool contains(const T& t) const override
   {
      return ranges::any_of(ranges::views::zip(m_low, m_high), [&](const auto& low_high) {
         auto&& [low, high] = low_high;
         return low <= t and high >= t;
      });
   }

   // Checks whether this space can be flattened to a Box
   bool is_flattenable() const override { return true; }

   std::string repr() { return fmt::format("Box({}, {}, {})", m_low, m_high, shape()); }

   bool operator==(const TypedBox< T >& other)
   {
      return xt::equal(shape(), other.shape()) and xt::equal(m_low, other.m_low)
             and xt::equal(m_high, other.m_high);
   }

   // Add other methods and properties as needed

  private:
   xarray< T > m_low;
   xarray< T > m_high;
   xarray< bool > m_bounded_below;
   xarray< bool > m_bounded_above;
};

}  // namespace force

#endif  // REINFORCE_BOX_HPP
