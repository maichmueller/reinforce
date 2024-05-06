
#ifndef REINFORCE_TUPLE_UTILS_HPP
#define REINFORCE_TUPLE_UTILS_HPP

#include <cstddef>
#include <stdexcept>
#include <tuple>

#include "reinforce/utils/macro.hpp"

namespace force::detail {

template < std::size_t left, std::size_t right >
struct visit_impl_binary_search {
   static constexpr size_t mid = left + (right - left) / 2;

   template < typename Tuple, typename F, typename... Args >
   static constexpr auto visit(Tuple const& tuple, std::size_t idx, F fun, Args&&... args) noexcept(
      noexcept(fun(std::get< right - 1U >(tuple), FWD(args)...))
      && noexcept(visit_impl_binary_search< left, right >::visit(tuple, idx, fun, FWD(args)...))
   )
   {
      using R = decltype(fun(std::get< mid >(tuple), FWD(args)...));
      return visit_impl_binary_search< left, right >::visit< R >(tuple, idx, fun, FWD(args)...);
   }

   template < typename R, typename Tuple, typename F, typename... Args >
   static constexpr R visit(Tuple const& tuple, std::size_t idx, F&& fun, Args&&... args) noexcept(
      noexcept(fun(std::get< mid >(tuple), FWD(args)...))
      && noexcept(visit_impl_binary_search< left, mid - 1 >::template visit<
                  R >(tuple, idx, FWD(fun), FWD(args)...))
      && noexcept(visit_impl_binary_search< mid, right >::template visit<
                  R >(tuple, idx, FWD(fun), FWD(args)...))
   )
   {
      if(idx == mid) {
         return fun(std::get< mid >(tuple), FWD(args)...);
      }
      if(idx < mid) {
         return visit_impl_binary_search< left, mid - 1 >::template visit< R >(
            tuple, idx, FWD(fun), FWD(args)...
         );
      }
      // idx > mid
      return visit_impl_binary_search< mid, right >::template visit< R >(
         tuple, idx, FWD(fun), FWD(args)...
      );
   }
};
template < typename Tuple >
void _check_idx(size_t idx)
{
   if(idx >= std::tuple_size< Tuple >::value) {
      throw std::out_of_range("Index out of range");
   }
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

}  // namespace force::detail

namespace force {

template < typename R, typename Tuple, typename F >
   requires(std::tuple_size< Tuple >::value > 0)
constexpr R visit_at_unchecked(Tuple const& tuple, std::size_t idx, F fun, auto&&... args) noexcept(
   noexcept(detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value - 1 >::
               template visit< R >(tuple, idx, fun, FWD(args)...))
)
{
   return detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value - 1 >::
      template visit< R >(tuple, idx, fun, FWD(args)...);
}

template < typename Tuple, typename F >
   requires(std::tuple_size< Tuple >::value > 0)
constexpr void visit_at_unchecked(
   Tuple const& tuple,
   std::size_t idx,
   F fun,
   auto&&... args
) noexcept(noexcept(detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value - 1 >::
                       visit(tuple, idx, fun, FWD(args)...)))
{
   detail::visit_impl_binary_search< 0, std::tuple_size< Tuple >::value - 1 >::visit(
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
constexpr void visit_at(Tuple const& tuple, std::size_t idx, F fun, auto&&... args)
{
   detail::_check_idx< Tuple >(idx);
   return visit_at_unchecked(tuple, idx, fun, FWD(args)...);
}

template < typename Tuple1, typename... TuplesTail >
   requires(
      (sizeof...(TuplesTail) > 0)
      and ((std::tuple_size_v< Tuple1 > == std::tuple_size_v< TuplesTail >) && ...)
   )
constexpr auto zip_tuples(TuplesTail&&... tuples)
{
   constexpr size_t tuple_len = std::tuple_size_v< detail::raw_t< Tuple1 > >;
   return zip_tuples_impl(
      std::make_index_sequence< tuple_len >(), std::forward< TuplesTail >(tuples)...
   );
}

}  // namespace force

#endif  // REINFORCE_TUPLE_UTILS_HPP
