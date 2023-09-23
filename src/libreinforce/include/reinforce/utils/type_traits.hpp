
#ifndef REINFORCE_TYPE_TRAITS_HPP
#define REINFORCE_TYPE_TRAITS_HPP

#include <type_traits>

namespace force {

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


}

#endif  // REINFORCE_TYPE_TRAITS_HPP
