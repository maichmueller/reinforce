
#ifndef REINFORCE_TUPLE_UTILS_HPP
#define REINFORCE_TUPLE_UTILS_HPP

#include <cstddef>
#include <stdexcept>
#include <tuple>

#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"

namespace force::detail {

template < long left, long right, typename Ret = void >
struct visit_impl_binary_search {
   static_assert(
      left >= 0 && right >= 0,
      "left and right binary search boundaries have to be positive."
   );
   static_assert(
      right >= left,
      "right boundary of binary search cannot be less than left boundary."
   );
   static constexpr long mid = left + (right - left) / 2;

   template < typename Tuple, typename F, typename... Args >
   static constexpr auto visit(Tuple const& tuple, std::size_t idx, F fun, Args&&... args)
      -> decltype(fun(std::get< mid >(tuple), FWD(args)...))
   {
      using R = decltype(fun(std::get< mid >(tuple), FWD(args)...));
      return visit_impl_binary_search< left, right, R >::template visit< R >(
         tuple, idx, fun, FWD(args)...
      );
   }

   template < typename R, typename Tuple, typename F, typename... Args >
   static constexpr R visit(Tuple const& tuple, std::size_t idx, F&& fun, Args&&... args)
   {
      if(idx == mid) {
         return fun(std::get< mid >(tuple), FWD(args)...);
      }
      if(idx < mid) {
         return visit_impl_binary_search< left, mid, R >::template visit< R >(
            tuple, idx, FWD(fun), FWD(args)...
         );
      }
      // idx > mid
      return visit_impl_binary_search< mid, right, R >::template visit< R >(
         tuple, idx, FWD(fun), FWD(args)...
      );
   }
};

/// Specialization for cases which are never reached
/// This specialization needs to be defined anyway,
/// since these cases are considered during compile
/// time of the binary search, even if the runtime
/// idx could never reach here with an unsigned
/// index type
template < long left, long right, typename Ret >
   requires(left < 0 || right < 0)
struct visit_impl_binary_search< left, right, Ret > {
   template < typename Tuple, typename F, typename... Args >
   static constexpr auto
   visit(Tuple const& /*tuple*/, std::size_t /*idx*/, F /*fun*/, Args&&... /*args*/)
   {
      throw std::logic_error("Calling visit< <0, <0, Ret > > specialization is a logic error");
      return Ret{};
   }

   template < typename R, typename Tuple, typename F, typename... Args >
   static constexpr R
   visit(Tuple const& /*tuple*/, std::size_t /*idx*/, F /*fun*/, Args&&... /*args*/)
   {
      throw std::logic_error("Calling visit< <0, <0, Ret > > specialization is a logic error");
      return Ret{};
   }
};
template < typename Tuple >
void _check_idx(size_t idx)
{
   if(idx >= std::tuple_size< Tuple >::value) {
      throw std::out_of_range("Index out of range");
   }
   SPDLOG_DEBUG(
      fmt::format("Check idx: Passed! {} >= {} is FALSE", idx, std::tuple_size< Tuple >::value)
   );
}

template < std::size_t N, typename... Tuples >
auto zip_tuples_at(Tuples&&... tuples)
{
   return std::forward_as_tuple(std::get< N >(std::forward< Tuples >(tuples))...);
}

template < std::size_t... Ns, typename... Tuples >
constexpr auto zip_tuples_impl(std::index_sequence< Ns... >, Tuples&&... tuples)
{
   return std::tuple{zip_tuples_at< Ns >(FWD(tuples)...)...};
}

template < size_t N, typename T >
using T_ = T;

template < typename T, std::size_t... Is >
constexpr auto create_tuple(T value, std::index_sequence< Is... >)
{
   return std::tuple< T_< Is, T >... >{(static_cast< void >(Is), value)...};
}

}  // namespace force::detail

namespace force {

template < typename R, typename Tuple, typename F >
   requires(std::tuple_size< Tuple >::value > 0)
constexpr R visit_at_unchecked(Tuple const& tuple, std::size_t idx, F fun, auto&&... args) noexcept(
   noexcept(detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value >::template visit<
            R >(tuple, idx, fun, FWD(args)...))
)
{
   return detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value >::template visit<
      R >(tuple, idx, fun, FWD(args)...);
}

template < typename Tuple, typename F >
   requires(std::tuple_size< Tuple >::value > 0)
constexpr decltype(auto) visit_at_unchecked(
   Tuple const& tuple,
   std::size_t idx,
   F fun,
   auto&&... args
) noexcept(noexcept(detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value >::
                       visit(tuple, idx, fun, FWD(args)...)))
{
   return detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value >::visit(
      tuple, idx, fun, FWD(args)...
   );
}

template < typename R, typename Tuple, typename F >
   requires(std::tuple_size< Tuple >::value > 0)
constexpr R visit_at(Tuple const& tuple, std::size_t idx, F fun, auto&&... args)
{
   detail::_check_idx< Tuple >(idx);
   return visit_at_unchecked< R >(tuple, idx, fun, FWD(args)...);
}

template < typename Tuple, typename F >
   requires(std::tuple_size< Tuple >::value > 0)
constexpr decltype(auto) visit_at(Tuple const& tuple, std::size_t idx, F fun, auto&&... args)
{
   detail::_check_idx< Tuple >(idx);
   return visit_at_unchecked(tuple, idx, fun, FWD(args)...);
}

template < typename Tuple1, typename... TuplesTail >
   requires(
      (sizeof...(TuplesTail) > 0)
      and ((std::tuple_size_v< detail::raw_t< Tuple1 > > == std::tuple_size_v< detail::raw_t< TuplesTail > >) && ...)
   )
constexpr auto zip_tuples(Tuple1&& tuple, TuplesTail&&... tuple_tail)
{
   constexpr size_t tuple_len = std::tuple_size_v< detail::raw_t< Tuple1 > >;
   return zip_tuples_impl(std::make_index_sequence< tuple_len >(), FWD(tuple), FWD(tuple_tail)...);
}

template < size_t N, class T >
auto create_tuple(T value = {})
{
   return detail::create_tuple< T >(value, std::make_index_sequence< N >{});
}

}  // namespace force

#endif  // REINFORCE_TUPLE_UTILS_HPP
