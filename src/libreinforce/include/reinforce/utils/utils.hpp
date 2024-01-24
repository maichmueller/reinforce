
#ifndef REINFORCE_UTILS_HPP
#define REINFORCE_UTILS_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <iterator>
#include <optional>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>
#include <random>
#include <range/v3/all.hpp>
#include <span>
#include <utility>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

#include "macro.hpp"
#include "reinforce/utils/type_traits.hpp"

namespace force::detail {

template < typename T, typename U = T >
constexpr std::pair< std::unique_ptr< T[] >, size_t > c_array(const size_t size, U&& value)
{
   auto data = std::make_unique< T[] >(size);
   auto data_span = std::span{new T[size], size};
   std::ranges::fill(data, std::forward< U >(value));
   return {std::move(data), size};
}

template < typename T, std::ranges::range Rng >
constexpr std::pair< std::unique_ptr< T[] >, size_t > c_array(const size_t size, Rng&& range)
{
   auto data = std::make_unique< T[] >(size);
   std::ranges::move(std::forward< Rng >(range), data.get());
   return {std::move(data), size};
}

template < typename T, std::ranges::forward_range Rng >
constexpr std::pair< std::unique_ptr< T[] >, size_t > c_array(Rng&& range)
{
   auto len = std::ranges::distance(range);
   return c_array< T >(static_cast< size_t >(len), FWD(range));
}

// Seed a PRNG
inline auto create_rng(std::optional< size_t > seed)
{
   if(seed.has_value()) {
      return pcg64{*seed};
   }
   return pcg64{pcg_extras::seed_seq_from< std::random_device >{}};
}

class rng_mixin {
  public:
   explicit rng_mixin(std::optional< size_t > seed = std::nullopt) : m_rng(create_rng(seed)) {}
   // Seed the PRNG of this space
   void seed(size_t seed) { m_rng = create_rng(seed); }

   /// const rng reference for external rng state inspection
   [[nodiscard]] auto& rng() const { return m_rng; }
   /// mutable rng reference for derived classes to forward random state
   auto& rng() { return m_rng; }

  private:
   pcg64 m_rng;
};

template < typename To >
constexpr auto static_to = [](const auto& t) { return static_cast< To >(t); };

template < typename Tuple, std::size_t... Ints >
constexpr auto tuple_slice(Tuple&& tuple, std::index_sequence< Ints... >)
{
   return std::tuple< std::tuple_element_t< Ints, Tuple >... >(
      std::get< Ints >(std::forward< Tuple >(tuple))...
   );
}

template < typename T >
consteval bool always_false(T)
{
   return false;
}

template < typename Iterator, typename Sentinel >
   requires std::input_iterator< Iterator > and std::sentinel_for< Sentinel, Iterator >
class RangeAdaptor {
  public:
   using iterator_type = Iterator;
   using sentinel_type = Sentinel;
   RangeAdaptor(Iterator begin, Sentinel sentinel) : m_begin(begin), m_end(sentinel) {}

   [[nodiscard]] auto begin() { return m_begin; }
   [[nodiscard]] auto end() { return m_end; }
   [[nodiscard]] auto begin() const { return m_begin; }
   [[nodiscard]] auto end() const { return m_end; }

  private:
   iterator_type m_begin;
   sentinel_type m_end;
};

template < typename Iterator, typename Sentinel >
class SizedRangeAdaptor: public RangeAdaptor< Iterator, Sentinel > {
  public:
   SizedRangeAdaptor(Iterator begin, Sentinel sentinel, size_t size)
       : RangeAdaptor< Iterator, Sentinel >(begin, sentinel), m_size(size)
   {
   }
   [[nodiscard]] auto size() const { return m_size; }

  private:
   size_t m_size;
};

template < typename ReturnArray, typename T, size_t N >
ReturnArray
adapt_stdarray(std::array< T, N > arr, xt::layout_type layout = xt::layout_type::row_major)
{
   std::vector< T > data(std::move_iterator(arr.begin()), std::move_iterator(arr.end()));
   return ReturnArray(xt::adapt(data, std::vector{N}, layout));
}

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

template < typename T >
decltype(auto) deref(T&& t)
{
   return std::forward< T >(t);
}

template < typename T >
   requires std::is_pointer_v< std::remove_cvref_t< T > >
            or is_specialization_v< std::remove_cvref_t< T >, std::reference_wrapper >
decltype(auto) deref(T&& t)
{
   if constexpr(is_specialization_v< std::remove_cvref_t< T >, std::reference_wrapper >) {
      return std::forward< T >(t).get();
   } else {
      return *std::forward< T >(t);
   }
}

template < typename T >
// clang-format off
   requires(
      is_specialization_v< std::remove_cvref_t< T >, std::shared_ptr >
      or is_specialization_v< std::remove_cvref_t< T >, std::unique_ptr >
   )
decltype(auto)  // clang-format on
deref(T&& t)
{
   return *std::forward< T >(t);
}

template < ranges::range Range >
class deref_view: public ranges::view_base {
  public:
   struct iterator;
   deref_view() = default;
   deref_view(ranges::range auto&& base) : m_base(base) {}

   iterator begin() { return ranges::begin(m_base); }
   iterator end() { return ranges::end(m_base); }

  private:
   Range m_base;
};

template < ranges::range Range >
struct deref_view< Range >::iterator {
   using base = ranges::iterator_t< Range >;
   using value_type = std::remove_cvref_t< decltype(deref(*(std::declval< Range >().begin()))) >;
   using difference_type = ranges::range_difference_t< Range >;

   iterator() = default;

   iterator(const base& b) : m_base{b} {}

   iterator operator++(int)
   {
      auto tmp = *this;
      ++*this;
      return tmp;
   }

   iterator& operator++()
   {
      ++m_base;
      return *this;
   }

   decltype(auto) operator*() const { return deref(*m_base); }

   bool operator==(iterator const& rhs) const { return m_base == rhs.m_base; }

  private:
   base m_base;
};

template < ranges::range Range >
deref_view(Range&&) -> deref_view< ranges::cpp20::views::all_t< Range > >;

struct deref_fn {
   template < typename Rng >
   auto operator()(Rng&& rng) const
   {
      return deref_view{ranges::views::all(std::forward< Rng >(rng))};
   }

   template < typename Rng >
   friend auto operator|(Rng&& rng, deref_fn const&)
   {
      return deref_view{ranges::views::all(std::forward< Rng >(rng))};
   }
};

template < typename ExpectedType, typename Range >
concept expected_value_type = requires(Range rng) {
   {
      *(rng.begin())
   } -> std::convertible_to< ExpectedType >;
};

struct CoordinateHasher {
   using is_transparent = std::true_type;

   template < ranges::range Range >
   size_t operator()(const Range& coords) const noexcept
      requires expected_value_type< size_t, Range >
   {
      return std::hash< std::string >{}(fmt::join(coords, ","));
   }
};

}  // namespace force::detail

namespace ranges::views {

constexpr ::force::detail::deref_fn deref{};

}  // namespace ranges::views

#endif  // REINFORCE_UTILS_HPP
