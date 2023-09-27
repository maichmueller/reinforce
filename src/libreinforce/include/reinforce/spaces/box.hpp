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
         m_low(low),
         m_high(high),
         m_bounded_below(not xt::isinf(low)),
         m_bounded_above(not xt::isinf(high))
   {
      auto low_shape = low.shape();
      auto high_shape = high.shape();
      if(low_shape != high_shape) {
         throw std::invalid_argument(fmt::format(
            "'Low' and 'High' bound arrays need to have the same shape. Given:\n{}\nand\n{}",
            low_shape,
            high_shape
         ));
      }
      if(shape_.has_value() and *shape_ != low_shape) {
         throw std::invalid_argument(fmt::format(
            "Given shape and shape of 'Low' and 'High' bound arrays have to be the. "
            "Given:\n{}\nand\n{}\nand\n{}",
            shape(),
            low_shape,
            high_shape
         ));
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
      ASSERT(manner.empty());
      return below() and above();
   }

   T sample(const std::optional< xarray< int8_t > >& /*unused*/ = std::nullopt)
   {
      xarray< T > samples = xt::empty< T >(shape());

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

   template < typename U >
      requires std::is_convertible_v< U, T >
   bool contains(U&& x)
   {
      T t(std::forward< U >(x));
      return m_low <= t and m_high >= t;
   }

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
