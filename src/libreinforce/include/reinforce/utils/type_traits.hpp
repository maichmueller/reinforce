
#ifndef REINFORCE_TYPE_TRAITS_HPP
#define REINFORCE_TYPE_TRAITS_HPP

#include <type_traits>

#include "xtensor_typedefs.hpp"

namespace force::detail {

/// is_specialization checks whether T is a specialized template class of 'Template'
/// This has the limitation of
/// usage:
///     constexpr bool is_vector = is_specialization< std::vector< int >, std::vector>;
///
/// Note that this trait has 2 limitations:
///  1) Does not work with non-type parameters.
///     (i.e. templates such as std::array will not be compatible with this type trait)
///  2) Generally, templated typedefs do not get captured as the underlying template but as the
///     typedef template. As such the following scenarios will return:
///          specialization<uptr<int>, uptr> << std::endl;            -> false
///          specialization<std::unique_ptr<int>, uptr>;              -> false
///          specialization<std::unique_ptr<int>, std::unique_ptr>;   -> true
///          specialization<uptr<int>, std::unique_ptr>;              -> true
///     for a typedef template template <typename T> using uptr = std::unique_ptr< T >;
template < class T, template < class... > class Template >
struct is_specialization: std::false_type {};

template < template < class... > class Template, class... Args >
struct is_specialization< Template< Args... >, Template >: std::true_type {};

template < class T, template < class... > class Template >
constexpr bool is_specialization_v = is_specialization< T, Template >::value;

/// logical XOR of the conditions (using fold expressions and bitwise xor)
template < typename... Conditions >
struct logical_xor: std::integral_constant< bool, (Conditions::value ^ ...) > {};
/// helper variable to get the contained value of the trait
template < typename... Conditions >
constexpr bool logical_xor_v = logical_xor< Conditions... >::value;

/// logical AND of the conditions (merely aliased)
template < typename... Conditions >
using logical_and = std::conjunction< Conditions... >;
/// helper variable to get the contained value of the trait
template < typename... Conditions >
constexpr bool logical_and_v = logical_and< Conditions... >::value;

/// logical OR of the conditions (merely aliased)
template < typename... Conditions >
using logical_or = std::disjunction< Conditions... >;
/// helper variable to get the contained value of the trait
template < typename... Conditions >
constexpr bool logical_or_v = logical_or< Conditions... >::value;
/// check if type T matches any of the given types in Ts...

/// logical NEGATION of the conditions (specialized for booleans)
template < bool... conditions >
constexpr bool none_of = logical_and_v< std::integral_constant< bool, not conditions >... >;

/// logical ANY of the conditions (specialized for booleans)
template < bool... conditions >
constexpr bool any_of = logical_or_v< std::integral_constant< bool, conditions >... >;

/// logical AND of the conditions (specialized for booleans)
template < bool... conditions >
constexpr bool all_of = logical_and_v< std::integral_constant< bool, conditions >... >;

template < class T, class... Ts >
struct is_any: ::std::disjunction< ::std::is_same< T, Ts >... > {};
template < class T, class... Ts >
inline constexpr bool is_any_v = is_any< T, Ts... >::value;

template < class T, class... Ts >
struct is_none: ::std::negation< is_any< T, Ts... > > {};
template < class T, class... Ts >
inline constexpr bool is_none_v = is_none< T, Ts... >::value;

template < class T, class... Ts >
struct all_same: ::std::conjunction< ::std::is_same< T, Ts >... > {};
template < class T, class... Ts >
inline constexpr bool all_same_v = all_same< T, Ts... >::value;

template < typename T >
using raw = std::remove_cvref< T >;

template < typename T >
using raw_t = typename raw< T >::type;

template < typename T >
using value_t = typename T::value_type;

template < typename T >
using multi_value_t = typename T::multi_value_type;

template < typename T >
using data_t = typename T::data_type;

template < typename T, typename U >
consteval bool same_as(const T&, const U&)
{
   return std::same_as< T, U >;
}

template < typename T >
concept has_value_type = requires(T t) { typename T::value_type; };

template < class Array >
concept is_xarray = has_value_type< Array >
                    and detail::is_any_v<
                       Array,
                       pyarray< typename Array::value_type >,
                       xarray< typename Array::value_type > >;

template < class Array, typename T >
concept is_xarray_of = detail::is_any_v< Array, pyarray< T >, xarray< T > >;

template < class Array >
concept is_xarray_ref = std::is_lvalue_reference_v< Array > and is_xarray< detail::raw_t< Array > >;

template < class Array, typename T = typename Array::value_type >
concept is_xarray_ref_of = std::is_lvalue_reference_v< Array >
                           and is_xarray_of< detail::raw_t< Array >, T >;

}  // namespace force::detail

#endif  // REINFORCE_TYPE_TRAITS_HPP
