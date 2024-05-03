#ifndef REINFORCE_BOX_HPP
#define REINFORCE_BOX_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <range/v3/all.hpp>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>
#include <xtensor/xrandom.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xview.hpp>

#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_extension.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

template < typename T >
   requires std::is_integral_v< T > || std::is_floating_point_v< T >
class BoxSpace: public Space< xarray< T >, BoxSpace< T > > {
  public:
   friend class Space< xarray< T >, BoxSpace >;
   using base = Space< xarray< T >, BoxSpace >;
   using data_type = T;
   using typename base::value_type;
   using base::shape;
   using base::rng;

   template < typename U, typename V, typename Range >
      requires(std::is_integral_v< U > || std::is_floating_point_v< U >)
                 and (std::is_integral_v< V > || std::is_floating_point_v< V >)
   BoxSpace(
      const U& low,
      const V& high,
      Range&& shape_,
      std::optional< size_t > seed = std::nullopt
   )
       : base(FWD(shape_), seed),
         m_low(xt::full< T >(shape(), static_cast< T >(low))),
         m_high(xt::full< T >(shape(), static_cast< T >(high))),
         m_bounded_below(xt::full(shape(), std::isinf(low))),
         m_bounded_above(xt::full(shape(), std::isinf(high)))
   {
   }
   template < template < typename... > class Array, typename... Args >
      requires detail::is_any_v< Array< T >, pyarray< T >, std::vector< T > >
   BoxSpace(const Array< T >& low, const Array< T >& high, Args&&... args)
       : BoxSpace(xarray< T >(low), xarray< T >(high), std::forward< Args >(args)...)
   {
   }

   template < std::ranges::range Range = xt::svector< int > >
   BoxSpace(
      xarray< T > low,
      xarray< T > high,
      const Range& shape_ = {},
      std::optional< size_t > seed = std::nullopt
   );

   BoxSpace(xarray< T > low, xarray< T > high, std::optional< size_t > seed)
       : BoxSpace(std::move(low), std::move(high), xt::svector< int >{}, seed)
   {
   }

   bool is_bounded(std::string_view manner = "");

   template < ranges::range Range >
      requires std::forward_iterator< ranges::iterator_t< Range > >
   std::pair< T, T > bounds(const Range& mdindex) const
   {
      return std::pair{
         m_low.element(mdindex.begin(), mdindex.end()),
         m_high.element(mdindex.begin(), mdindex.end())
      };
   }
   std::pair< T, T > bounds(std::initializer_list< T > mdindex) const
   {
      return std::pair{
         m_low.element(mdindex.begin(), mdindex.end()),
         m_high.element(mdindex.begin(), mdindex.end())
      };
   }

   bool operator==(const BoxSpace& rhs) const
   {
      // we can safely use static-cast here, because the base checks for type-identity first and
      // only calls equals if the types of two compared objects are the same (hence
      // TypedDiscrete<T>)
      return xt::all(xt::equal(m_low, rhs.m_low))  //
             and xt::all(xt::equal(m_high, rhs.m_high))
             and xt::all(xt::equal(m_bounded_below, rhs.m_bounded_below))
             and xt::all(xt::equal(m_bounded_above, rhs.m_bounded_above));
   }

   // Checks whether this space can be flattened to a Box
   [[nodiscard]] bool is_flattenable() const { return true; }

   std::string repr() { return fmt::format("Box({}, {}, {})", m_low, m_high, shape()); }

  private:
   xarray< T > m_low;
   xarray< T > m_high;
   xarray< bool > m_bounded_below;
   xarray< bool > m_bounded_above;

   [[nodiscard]] value_type _sample(
      const std::optional< xarray< bool > >& /*unused*/ = std::nullopt
   ) const;

   [[nodiscard]] value_type _sample(
      size_t nr_samples,
      const std::optional< xarray< bool > >& /*unused*/ = std::nullopt
   ) const;

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return base::_isin_shape_and_bounds(value, m_low, m_high);
   }
};

/// Deduction guides

template < typename U, typename V, typename Range >
   requires(std::is_integral_v< U > || std::is_floating_point_v< U >)
              and (std::is_integral_v< V > || std::is_floating_point_v< V >)
BoxSpace(const U& low, const V& high, Range&& shape_, std::optional< size_t > seed = std::nullopt)
   -> BoxSpace< std::common_type_t< U, V > >;

/// Definitions
///
///

template < typename T >
   requires std::is_integral_v< T > || std::is_floating_point_v< T >
template < std::ranges::range Range >
BoxSpace< T >::BoxSpace(
   xarray< T > low,
   xarray< T > high,
   const Range& shape_,
   std::optional< size_t > seed
)
    : base(
         std::ranges::empty(shape_)
            ? xt::svector< int >(std::ranges::begin(low.shape()), std::ranges::end(low.shape()))
            : xt::svector< int >(std::ranges::begin(shape_), std::ranges::end(shape_)),
         seed
      ),
      m_low(std::move(low)),
      m_high(std::move(high)),
      m_bounded_below(not xt::isinf(m_low)),
      m_bounded_above(not xt::isinf(m_high))
{
   using namespace fmt::literals;

   auto low_shape = m_low.shape();
   auto high_shape = m_high.shape();

   SPDLOG_DEBUG(fmt::format(
      "Low shape {}, high shape: {}, specified shape: {}", low_shape, high_shape, shape()
   ));
   if(not ranges::equal(high_shape, low_shape) or not ranges::equal(high_shape, shape())) {
      throw std::invalid_argument(fmt::format(
         "Shape of 'Low' and 'High' bound arrays, as well as the explicit shape need to match. "
         "Given {}, {}, and {} respectively.",
         low_shape,
         high_shape,
         shape()
      ));
   }
   SPDLOG_DEBUG(fmt::format(
      "Bounds:\n{}", std::invoke([&] {
         xarray< std::string > bounds = xt::empty< std::string >(shape());
         for(auto [i, bound_string] : ranges::views::enumerate(
                ranges::views::zip(m_low, m_high) | ranges::views::transform([](auto pair) {
                   return fmt::format(
                      "{bracket_open}{lower},{upper}{bracket_close}",
                      "bracket_open"_a = std::isinf(pair.first) ? "(" : "[",
                      "bracket_close"_a = std::isinf(pair.second) ? ")" : "]",
                      "lower"_a = pair.first,
                      "upper"_a = pair.second
                   );
                })
             )) {
            bounds.flat(i) = bound_string;
         };
         return bounds;
      })
   ));
   if(ranges::any_of(ranges::views::zip(m_high, m_low), [](auto&& pair) {
         return std::get< 0 >(pair) < std::get< 1 >(pair);
      })) {
      throw std::invalid_argument(
         "Some value-positions in 'low' are greater than their corresponding 'high' values."
      );
   }
}

template < typename T >
   requires std::is_integral_v< T > || std::is_floating_point_v< T >
auto BoxSpace< T >::_sample(const std::optional< xarray< bool > >&) const -> value_type
{
   xarray< T > samples = xt::empty< T >(shape());
   SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples.shape()));
   for(auto&& [i, bounds] :
       ranges::views::enumerate(ranges::views::zip(m_bounded_below, m_bounded_above))) {
      auto&& [lower_bounded, upper_bounded] = bounds;
      // for single samples we can use flat indexing
      auto& entry = samples.data_element(i);
      switch(auto choice = (lower_bounded ? 1 : -1) + int(upper_bounded)) {
         case -1: {
            // (-infinity, infinity)
            entry = static_cast< T >(std::normal_distribution< double >{}(rng()));
            break;
         }
         case 0: {
            // (-infinity, B]
            entry = m_high.data_element(i)
                    - static_cast< T >(std::exponential_distribution< double >{1}(rng()));
            break;
         }
         case 1: {
            // [A, infinity)
            entry = m_low.data_element(i)
                    + static_cast< T >(std::exponential_distribution< double >{1}(rng()));
            break;
         }
         case 2: {
            // [A, B]
            using distribution = std::conditional_t<
               std::is_integral_v< T >,
               std::uniform_int_distribution< T >,
               std::uniform_real_distribution< T > >;
            entry = static_cast< T >(
               distribution{m_low.data_element(i), m_high.data_element(i)}(rng())
            );
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

template < typename T >
   requires std::is_integral_v< T > || std::is_floating_point_v< T >
auto BoxSpace< T >::_sample(
   size_t nr_samples,
   const std::optional< xarray< bool > >& /*unused*/
) const -> value_type
{
   xarray< T > samples = xt::empty< T >(prepend(shape(), static_cast< int >(nr_samples)));
   SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples.shape()));

   for(auto&& [i, bounds] :
       ranges::views::enumerate(ranges::views::zip(m_bounded_below, m_bounded_above))) {
      auto&& [lower_bounded, upper_bounded] = bounds;
      // convert the flat index i to an indexing list for the given shape
      auto coordinates = xt::unravel_index(static_cast< int >(i), shape());
      // select the sampling indices (as if e.g. samples[x,y,:] on a numpy array) to emplace
      auto slice_vec = prepend(
         xt::xstrided_slice_vector(coordinates.begin(), coordinates.end()), xt::all()
      );
      SPDLOG_DEBUG(fmt::format("Slice: {}", slice_vec));
      auto entry_view = xt::strided_view(samples, slice_vec);
      auto draw_shape = xt::svector{nr_samples};
      switch(auto choice = (lower_bounded ? 1 : -1) + int(upper_bounded)) {
            // we use the `double` versions of all the sampling functions of xtensor even if `T`
            // were to be integral. xtensor casts the sampled double floating points to `T`
            // implicitly, so we do not need to handle this manually.
         case -1: {
            // (-infinity, infinity)
            entry_view = xt::random::randn(draw_shape, 0., 1., rng());
            break;
         }
         case 0: {
            // (-infinity, B]
            entry_view = m_high.data_element(i) - xt::random::exponential(draw_shape, 1., rng());
            break;
         }
         case 1: {
            // [A, infinity)
            entry_view = m_low.data_element(i) + xt::random::exponential(draw_shape, 1., rng());
            break;
         }
         case 2: {
            // [A, B]
            using distribution = std::conditional_t<
               std::is_integral_v< T >,
               decltype(AS_LAMBDA(xt::random::randint)),
               decltype(AS_LAMBDA(xt::random::rand< double >)) >;

            entry_view = distribution{
            }(draw_shape, m_low.data_element(i), m_high.data_element(i), rng());
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

template < typename T >
   requires std::is_integral_v< T > || std::is_floating_point_v< T >
bool BoxSpace< T >::is_bounded(const std::string_view manner)
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

}  // namespace force

#endif  // REINFORCE_BOX_HPP
