
#ifndef REINFORCE_UTILS_HPP
#define REINFORCE_UTILS_HPP

#include <utility>

namespace force::utils {

template < typename... Ts >
struct overload: Ts... {
   using Ts::operator()...;
};
template < typename... Ts >
overload(Ts...) -> overload< Ts... >;

constexpr auto identity_pr = [](auto&& obj) -> decltype(auto) {
   return std::forward< decltype(obj) >(obj);
};

constexpr auto identity = [](auto&& obj) { return obj; };

inline void hash_combine([[maybe_unused]] std::size_t& /*seed*/) {}

template < typename T, typename... Rest >
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest)
{
   std::hash< T > hasher;
   seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
   hash_combine(seed, rest...);
}

// taken from the proposal https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0870r4.html
template < class From, class To >
inline constexpr bool is_convertible_without_narrowing_v = false;

template < class T, class U >
concept construct_without_narrowing = requires(U&& x) {
   {
      std::type_identity_t< T[] >{std::forward< U >(x)}
   } -> std::same_as< T[1] >;
};

template < class From, class To >
   requires std::is_convertible_v< From, To >
inline constexpr bool
   is_convertible_without_narrowing_v< From, To > = construct_without_narrowing< To, From >;

}  // namespace force::utils

#endif  // REINFORCE_UTILS_HPP
