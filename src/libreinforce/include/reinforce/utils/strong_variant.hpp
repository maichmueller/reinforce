

#ifndef REINFORCE_STRONG_VARIANT_HPP
#define REINFORCE_STRONG_VARIANT_HPP

#include <type_traits>
#include <variant>

namespace force {
template < typename... Ts >
class strong_variant: public std::variant< Ts... > {
  public:
   template < typename T >
      requires std::disjunction_v< std::is_same< T, Ts >... >
   strong_variant(T&& t) : std::variant< Ts... >(std::forward< T >(t))
   {
   }

   strong_variant() : std::variant< Ts... >() {}
};

}  // namespace force

#endif  // REINFORCE_STRONG_VARIANT_HPP
