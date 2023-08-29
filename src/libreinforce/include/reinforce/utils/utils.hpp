
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

}  // namespace force::utils

#endif  // REINFORCE_UTILS_HPP
