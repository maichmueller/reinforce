#ifndef REINFORCE_FORMAT_HPP
#define REINFORCE_FORMAT_HPP

#include <sstream>
#include <string>
#include <type_traits>

#include "fmt/format.h"
#include "fmt/std.h"

namespace force::detail {

template < typename T >
inline std::string to_string(const T& value) = delete;

template <>
inline std::string to_string(const ::std::nullopt_t&)
{
   return "None";
}

template < typename To >
To from_string(std::string_view str);

// template < typename T >
// struct printable;

template < typename T >
struct printable: std::false_type {};

template < typename T >
   requires(requires(T t) { ::force::detail::to_string(t); })
struct printable< T >: std::true_type {};

template < typename T >
constexpr bool printable_v = printable< T >::value;

template <>
struct printable< std::monostate >: std::true_type {};

}  // namespace force::detail

namespace force {
template < typename T >
   requires(::force::detail::printable_v< T >)
inline auto& operator<<(std::ostream& os, const T& value)
{
   return os << ::force::detail::to_string(value);
}

template < typename T >
   requires(::force::detail::printable_v< T >)
inline auto& operator<<(std::stringstream& os, const T& value)
{
   return os << ::force::detail::to_string(value);
}
}  // namespace force
template < typename T >
   requires(::force::detail::printable_v< T >)
struct fmt::formatter< T >: fmt::ostream_formatter {};

#endif  // REINFORCE_FORMAT_HPP
