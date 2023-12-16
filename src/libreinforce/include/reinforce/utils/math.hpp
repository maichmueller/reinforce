#ifndef REINFORCE_MATH_HPP
#define REINFORCE_MATH_HPP

#include <cstdlib>
#include <tuple>

namespace force {

template < typename T = double >
constexpr auto infinity = std::numeric_limits< T >::infinity();

namespace detail {

template < typename T >
struct modulo_result {
   T quot;
   T rem;
};
}  // namespace detail

template < typename result_type = long, std::integral T1, std::integral T2 >
constexpr detail::modulo_result< result_type > modulo(T1 number, T2 dividend)
{
   if(std::is_constant_evaluated()) {
      auto number_ = result_type(number);
      auto dividend_ = result_type(dividend);
      return detail::modulo_result< result_type >{
         number_ / dividend_, number_ - (number_ / dividend_) * dividend_
      };
   }
   auto [quotient, remainder] = std::div(
      static_cast< long >(number), static_cast< long >(dividend)
   );
   return detail::modulo_result< result_type >{quotient, remainder};
}

}  // namespace force

#endif  // REINFORCE_MATH_HPP
