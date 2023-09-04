
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

constexpr auto identity_pr = [](auto&& obj) -> decltype(auto)
{
   return std::forward< decltype(obj) >(obj);
};

constexpr auto identity = [](auto&& obj)
{
   return obj;
};

inline void hash_combine([[maybe_unused]] std::size_t& /*seed*/) {}

template < typename T, typename... Rest >
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest)
{
   std::hash< T > hasher;
   seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
   hash_combine(seed, rest...);
}

}  // namespace force::utils

#endif  // REINFORCE_UTILS_HPP
