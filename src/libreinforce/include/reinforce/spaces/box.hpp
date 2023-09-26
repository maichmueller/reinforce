#ifndef REINFORCE_BOX_HPP
#define REINFORCE_BOX_HPP

#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <variant>
#include <vector>

#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

template < typename T >
   requires std::is_integral_v< T > || std::is_floating_point_v< T >
class TypedBox: public TypedSpace< xarray< T > > {
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

   TypedBox(
      const xarray< T >& low,
      const xarray< T >& high,
      const std::optional< std::vector< int > >& shape_ = std::nullopt,
      std::optional< size_t > seed = std::nullopt
   )
       : base(shape_, seed),
         low(low),
         high(high),
         low_repr(_short_repr(low)),
         high_repr(_short_repr(high)),
         bounded_below(not xt::isinf(low)),
         bounded_above(not xt::isinf(high))
   {
   }

   bool is_bounded(std::string_view manner = "")
   {
      const auto below = [&] { return xt::all(bounded_below); };
      const auto above = [&] { return xt::all(bounded_above); };

      if(manner == "below") {
         return below();
      }
      if(manner == "above") {
         return above();
      }
      return below() and above();
   }

   T sample(const std::optional< xarray< int8_t > >& /*unused*/ = std::nullopt)
   {
      xarray< T > sample = xt::empty< T >(shape());
      size_t shape_size = ranges::accumulate(shape(), size_t(0), std::multiplies{});
      // Masking arrays which classify the coordinates according to interval type
      std::vector< bool > unbounded(shape_size, false);
      std::vector< bool > upp_bounded(shape_size, false);
      std::vector< bool > low_bounded(shape_size, false);
      std::vector< bool > bounded(shape_size, false);

      for(const auto idx : ranges::views::iota(0UL, shape_size)) {
         unbounded[idx] = not bounded_below[idx] and not bounded_above[idx];
         upp_bounded[idx] = not bounded_below[idx] and bounded_above[idx];
         low_bounded[idx] = bounded_below[idx] and not bounded_above[idx];
         bounded[idx] = bounded_below[idx] and bounded_above[idx];
      }

      for(auto&& [i, bounds] :
          ranges::views::enumerate(ranges::views::zip(bounded_below, bounded_above))) {
         auto&& [low_bound, upp_bound] = bounds;
         switch((low_bound ? 1 : -1) + int(upp_bound)) {
            case -1: {
               // (-infinity, infinity)
               sample += static_cast< T >(std::normal_distribution< double >()(rng()));
            }
            case 0: {
               // (-infinity, B]
               sample -= static_cast< T >(std::exponential_distribution< double >()(rng()))
                         + high[i];
            }
            case 1: {
               // [A, infinity)
               sample = low[i] + xt::random::exponential(xt::xshape< 1 >{}, 1., rng());
            }
            case 2: {
               // [A, B]
               sample += xt::random::rand(xt::xshape< 1 >{}, low[i], high[i], rng());
            }
         }
         if(not low_bound and not upp_bound) {
            sample += static_cast< T >(std::normal_distribution< double >()(rng()));
         } else if(low_bound and not upp_bound) {
            sample += static_cast< T >(std::exponential_distribution< double >()(rng())) + low[i];
         } else if(not low_bound and upp_bound) {
            sample -= static_cast< T >(std::exponential_distribution< double >()(rng())) + high[i];
         } else if(low_bound and upp_bound) {
            sample += static_cast< T >(
               std::uniform_real_distribution< double >(low[i], high[i])(rng())
            );
         }
      }

      if constexpr(std::is_integral_v< T >) {
         sample = static_cast< T >(std::floor(sample));
      }

      return sample;
   }

   template < typename U >
      requires std::is_convertible_v< U, T >
   bool contains(U&& x)
   {
      T t(x);
      return low <= t and high >= t;
   }

   std::string repr()
   {
      return fmt::format(
         "Box({}, {}, {})",
         low_repr,
         high_repr,
         ranges::accumulate(shape(), size_t(0), std::multiplies{})
      );
   }

   bool operator==(const TypedBox< T >& other)
   {
      return xt::equal(shape(), other.shape()) and xt::equal(low, other.low)
             and xt::equal(high, other.high);
   }

   // Add other methods and properties as needed

  private:
   xarray< T > low;
   xarray< T > high;
   std::string low_repr;
   std::string high_repr;
   xarray< bool > bounded_below;
   xarray< bool > bounded_above;
};

}  // namespace force

#endif  // REINFORCE_BOX_HPP
