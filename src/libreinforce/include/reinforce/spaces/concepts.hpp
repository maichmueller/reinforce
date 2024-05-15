
#ifndef REINFORCE_SPACES_CONCEPTS_HPP
#define REINFORCE_SPACES_CONCEPTS_HPP

#include <concepts>

#include "reinforce/utils/type_traits.hpp"

namespace force {

template < typename T >
concept discrete_reqs = std::integral< T >;

template < typename T >
concept box_reqs = std::is_integral_v< T > or std::is_floating_point_v< T >;

template < typename T >
concept multidiscrete_reqs = std::integral< T >;

}  // namespace force

#endif  // REINFORCE_SPACES_CONCEPTS_HPP
