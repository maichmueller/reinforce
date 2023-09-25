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
class Box: public Space< xarray< T > > {
  public:
   using base = Space< T >;
   using base::shape;
   using base::rng;

   template < typename... Args >
   Box(const T& low, const T& high, Args&&... args)
       : Box(xarray< T >{low}, xarray< T >{high}, std::forward< Args >(args)...)
   {
   }
   template < template < typename... > class Array, typename... Args >
      requires detail::is_any_v< Array< T >, pyarray< T >, std::vector< T > >
   Box(const Array< T >& low, const Array< T >& high, Args&&... args)
       : Box(xarray< T >{low}, xarray< T >{high}, std::forward< Args >(args)...)
   {
   }

   Box(
      const xarray< T >& low,
      const xarray< T >& high,
      const std::optional< std::vector< int > >& shape_ = std::nullopt,
      std::optional< size_t > seed = std::nullopt
   )
       : base(shape_, seed),
         low(broadcast_value(low, shape(), '-')),
         high(broadcast_value(high, shape(), '+'))
   {
      low = std::vector< T >(shape_size, low);
      high = std::vector< T >(shape_size, high);

      low_repr = _short_repr(low);
      high_repr = _short_repr(high);
   }

   bool is_bounded(std::string_view manner = "both")
   {
      bool below = std::all_of(bounded_below.begin(), bounded_below.end(), [](bool b) {
         return b;
      });
      bool above = std::all_of(bounded_above.begin(), bounded_above.end(), [](bool b) {
         return b;
      });

      if(manner == "both") {
         return below && above;
      } else if(manner == "below") {
         return below;
      } else if(manner == "above") {
         return above;
      } else {
         throw std::invalid_argument("manner is not in {'below', 'above', 'both'}");
      }
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
          ranges::views::enumerate(ranges::views::zip(bounded_below, bounded_above)
          )) {
         auto&& [low_bound, upp_bound] = bounds;
         switch((-1) % low_bound + 1 % upp_bound) {
            case -1: {
               // [A, infinity)
               sample = low[i] + xt::random::exponential(xt::xshape<1>{}, 1., rng());
            }
            case 0 : {
               // [A, B]
               sample += xt::random::rand(xt::xshape<1>{}, low[i], high[i], rng());
            }
            case 1: {
               // (-infinity, B]
               sample -= static_cast< T >(std::exponential_distribution< double >()(rng())) + high[i];
            }
            case 2: {
               // (-infinity, infinity)
               sample += static_cast< T >(std::normal_distribution< double >()(rng()));
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

   bool operator==(const Box< T >& other)
   {
      return xt::equal(shape(), other.shape()) and xt::equal(low, other.low)
             and xt::equal(high, other.high);
   }

   // Add other methods and properties as needed

  private:
   std::vector< bool > bounded_below;
   std::vector< bool > bounded_above;
   xarray< T > low;
   xarray< T > high;
   std::string low_repr;
   std::string high_repr;


   std::vector< T >
   broadcast_value(const xarray< T >& value, const std::vector< int >& shape, char inf_sign)
   {
      if(xt::isinf(value)) {
         if(inf_sign == '+') {
            return std::numeric_limits< T >::infinity();
         }
         if(inf_sign == '-') {
            return -std::numeric_limits< T >::infinity();
         }
      } else {
         return std::vector< T >(shape().size(), value);
      }
      if(xt::any(xt::isinf(value))) {
         xarray< T > inf_value = broadcast_value(static_cast< T >(0), 1, inf_sign);
         xarray< T > inf_value return broadcasted;
      }
      return value;
   }
};

}  // namespace force

#endif  // REINFORCE_BOX_HPP
