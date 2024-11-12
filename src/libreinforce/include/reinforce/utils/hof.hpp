#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"

// for comparison
// https://godbolt.org/z/n4WdqGWcv

namespace force::hof {

template < typename T >
struct unpack: std::true_type {
   using type = T;
};

template < typename T >
constexpr bool unpack_v = unpack< T >::value;

/// we consider a type that can be unpacked with std::get with at least 1 entry unpackable
template < typename T >
concept unpackable = requires(T t) { std::get< 0 >(t); };

template < typename F, typename... Fs >
   requires std::is_default_constructible_v< F > and (std::is_default_constructible_v< Fs > and ...)
struct compose {
   static_assert(
      sizeof...(Fs) > 0,
      "Logic error, case of 0 variadic args should be caught by specialization."
   );
   template < typename... Args >
   constexpr decltype(auto) operator()(Args&&... args) const
   {
      using result_t = detail::raw_t< std::invoke_result_t< compose< Fs... >, Args... > >;
      if constexpr(unpackable< result_t > and unpack_v< F >) {
         return std::apply(F{}, compose< Fs... >{}(FWD(args)...));
      } else {
         return F{}(compose< Fs... >{}(FWD(args)...));
      }
   }
};

template < typename F >
   requires std::is_default_constructible_v< F >
struct compose< F > {
   template < typename... Args >
   constexpr decltype(auto) operator()(Args&&... args) const
   {
      return F{}(FWD(args)...);
   }
};

AS_STRUCT(fwd, std::forward_as_tuple);

AS_STRUCT(gather, std::forward_as_tuple);
template <>
struct unpack< gather >: std::false_type {};

template < size_t N >
struct project {
   template < typename T >
   constexpr decltype(auto) operator()(T&& t) const noexcept(noexcept(std::get< N >(FWD(t))))
   {
      return std::get< N >(FWD(t));
   }
};

template < template < typename... > class T, typename = void >
struct construct {
   template < typename... Args >
   constexpr auto operator()(Args&&... args) const noexcept(noexcept(T< Args... >{FWD(args)...}))
   {
      return T< Args... >{FWD(args)...};
   }
};

namespace detail {
template < typename T >
struct adhoc {
   using type = T;
};
}  // namespace detail

template < typename T >
struct construct< detail::adhoc, T > {
   constexpr auto operator()(auto&&... args) const noexcept(noexcept(T{FWD(args)...}))
   {
      return T{FWD(args)...};
   }
};

template < template < typename... > class T, typename U >
struct unpack< construct< T, U > >: public std::false_type {};

template < typename T >
struct unpack< construct< detail::adhoc, T > >: public std::false_type {};

template < size_t N >
struct fork {
   template < class T >
   constexpr auto operator()(T&& ref) const noexcept
   {
      return as_tuple(FWD(ref), std::make_index_sequence< N >{});
   }

  private:
   template < typename T, std::size_t... Is >
   constexpr auto as_tuple(T value, std::index_sequence< Is... >) const
   {
      // no forwarding here or dangling references!
      return std::tuple((static_cast< void >(Is), value)...);
   }
};

template < typename... Fs >
   requires(sizeof...(Fs) > 0)
struct map {
   template < typename... Args >
   constexpr auto operator()(Args&&... args) const
   {
      constexpr auto composed_func = compose< Fs... >{};
      return hof::fwd{}(composed_func(FWD(args))...);
   }
};

template < typename... Fs >
   requires(sizeof...(Fs) > 0)
struct zip_map {
   template < typename... Args >
      requires(sizeof...(Args) == sizeof...(Fs))
   constexpr auto operator()(Args&&... args) const
   {
      return hof::fwd{}(Fs{}(args)...);
   }
};

template < size_t... Is >
struct slice {
   template < typename... Args >
   constexpr auto operator()(Args&&... args) const noexcept
   {
      // create a tuple by forwarding exactly those arguments at indices Is...
      constexpr auto fwd = hof::fwd{};
      return fwd(project< Is >{}(fwd(args...))...);
   };
};

}  // namespace force::hof