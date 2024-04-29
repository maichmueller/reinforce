#ifndef NOR_SANDBOX_HPP
#define NOR_SANDBOX_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <reinforce/utils/type_traits.hpp>
#include <utility>

namespace py = pybind11;

struct Messenger {
   explicit Messenger(std::string name_ = "DefaultObject") : name(std::move(name_))
   {
      std::cout << "CTOR - " << name << "\n";
   }
   ~Messenger() { std::cout << "DTOR - " << (name.empty() ? "moved away..." : name) << "\n"; }
   Messenger(const Messenger& other) : name(other.name)
   {
      std::cout << "COPY - " << name << "\n";
      if(not name.empty()) {
         try {
            auto subs = name.substr(name.size() - 1, 1);
            auto order = std::stoi(subs);
            name = name.substr(0, name.size() - 1) + std::to_string(order + 1);
         } catch(std::invalid_argument) {
            name += std::to_string(2);
         }
      }
   }
   Messenger(Messenger&& other) noexcept : name(std::move(other.name))
   {
      std::cout << "MOVE - " << name << "\n";
   }
   Messenger& operator=(const Messenger& other)
   {
      if(this == &other) {
         return *this;
      }
      name = other.name;
      std::cout << "COPY assignment - " << name << "\n";
      return *this;
   }
   Messenger& operator=(Messenger&& other) noexcept
   {
      if(this == &other) {
         return *this;
      }
      name = std::move(other.name);
      std::cout << "MOVE assignment - " << name << "\n";
      return *this;
   }

   std::string name;
};

template < class U, class T, bool can_move >
struct wrapper {
   T* ptr;
   wrapper(T& t) : ptr(std::addressof(t)) {}

   using unwrapped_type = std::conditional_t<
      can_move,
      std::conditional_t< std::is_lvalue_reference< U >{}, T&, T&& >,
      std::conditional_t< std::is_rvalue_reference< U >{}, T&&, T& > >;
   using tuple_element_type = U;

   unwrapped_type unwrap() const { return std::forward< unwrapped_type >(*ptr); }
};

template < class... Wrappers, std::size_t... Is >
auto unwrap_tuple(const std::tuple< Wrappers... >& t, std::index_sequence< Is... >)
{
   return std::tuple< typename Wrappers::tuple_element_type... >(std::get< Is >(t).unwrap()...);
}

template < class... Wrappers >
auto unwrap_tuple(const std::tuple< Wrappers... >& t)
{
   return unwrap_tuple(t, std::index_sequence_for< Wrappers... >());
}

template < bool can_move, class V, class T >
auto wrap_and_flatten(T& t, char)
{
   return std::tuple{wrapper< V, T, can_move >(t)};
}
template < class T >
struct is_tuple: std::false_type {};
template < class... Ts >
struct is_tuple< std::tuple< Ts... > >: std::true_type {};
template < class T >
struct is_tuple< const T >: is_tuple< T > {};
template < class T >
struct is_tuple< volatile T >: is_tuple< T > {};

template <
   bool can_move,
   class,
   class Tuple,
   // requires force::detail::is_specialization_v<std::decay_t< Tuple >, std::tuple >
   class = std::enable_if_t< is_tuple< std::decay_t< Tuple > >::value > >
auto wrap_and_flatten(Tuple& t, int);

template < bool can_move, class Tuple, std::size_t... Is >
auto wrap_and_flatten(Tuple& t, std::index_sequence< Is... >)
{
   return std::tuple_cat(
      wrap_and_flatten< can_move, std::tuple_element_t< Is, std::remove_cv_t< Tuple > > >(
         std::get< Is >(t), 0
      )...
   );
}

template < bool can_move, class V, class Tuple, class >
auto wrap_and_flatten(Tuple& t, int)
{
   using seq_type = std::make_index_sequence< std::tuple_size< Tuple >{} >;
   return wrap_and_flatten< can_move >(t, seq_type());
}

template < class Tuple >
auto wrap_and_flatten_tuple(Tuple&& t)
{
   constexpr bool can_move = ! std::is_lvalue_reference< Tuple >{};
   using seq_type = std::make_index_sequence< std::tuple_size< std::decay_t< Tuple > >{} >;
   return wrap_and_flatten< can_move >(t, seq_type());
}

template < typename T >
auto merge_tuple(T&& t)
{
   return unwrap_tuple(wrap_and_flatten_tuple(std::forward< T >(t)));
}

#endif  // NOR_SANDBOX_HPP
