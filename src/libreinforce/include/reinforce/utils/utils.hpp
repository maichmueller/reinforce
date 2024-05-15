
#ifndef REINFORCE_UTILS_HPP
#define REINFORCE_UTILS_HPP

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <pcg_extras.hpp>
#include <pcg_random.hpp>
#include <random>
#include <range/v3/all.hpp>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

#include "macro.hpp"
#include "reinforce/utils/type_traits.hpp"

namespace force {

auto copy(auto item)
{
   return item;
}

template < typename T, typename U >
decltype(auto) extend(xt::svector< T > base, xt::svector< U >& extension)
{
   for(const auto& elem : extension) {
      base.push_back(static_cast< T >(elem));
   }
   return base;
}

template < typename T, typename U >
decltype(auto) extend(xt::svector< T > base, xt::svector< U >&& extension)
{
   for(auto&& elem : std::move(extension)) {
      base.push_back(static_cast< T >(FWD(elem)));
   }
   return base;
}

template < typename Container, typename T >
decltype(auto) append(Container&& container, T&& elem)
{
   container.push_back(FWD(elem));
   return FWD(container);
}

template < typename Container, typename T >
   requires std::is_const_v< std::remove_reference_t< Container > >
auto append(Container&& container, T&& elem)
{
   auto tmp = container;
   return append(tmp, FWD(elem));
}

template < typename Container, typename T >
decltype(auto) prepend(Container&& container, T&& elem)
{
   container.insert(
      container.begin(), static_cast< detail::value_t< detail::raw_t< Container > > >(FWD(elem))
   );
   return FWD(container);
}

template < typename Container, typename T >
   requires std::is_const_v< std::remove_reference_t< Container > >
auto prepend(Container&& container, T&& elem)
{
   auto tmp = container;
   return prepend(tmp, FWD(elem));
}

}  // namespace force

namespace force::detail {

template < typename KeyT = size_t, typename KeyCompare = std::less< KeyT > >
class Counter {
  public:
   using map_type = std::map< KeyT, size_t, KeyCompare >;
   using value_type = typename map_type::value_type;
   using mapped_type = typename map_type::mapped_type;
   using key_type = typename map_type::key_type;
   using key_compare = typename map_type::key_compare;
   using allocator_type = typename map_type::allocator_type;

   template < std::ranges::range Rng >
   explicit Counter(Rng&& rng)
   {
      static_assert(
         std::convertible_to< std::ranges::range_value_t< std::remove_cvref_t< Rng > >, KeyT >,
         "Range needs to hold value types that are convertible to the key type."
      );
      for(const auto& elem : rng) {
         ++m_map[static_cast< KeyT >(elem)];
      }
   }

   [[nodiscard]] const auto& map() const { return m_map; }

   [[nodiscard]] auto begin() { return m_map.begin(); }
   [[nodiscard]] auto end() { return m_map.end(); }
   [[nodiscard]] auto begin() const { return m_map.begin(); }
   [[nodiscard]] auto end() const { return m_map.end(); }

  private:
   map_type m_map = {};
};

template < size_t N >
struct StringLiteral {
   constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }
   char value[N];
};

template < class T >
constexpr std::string_view type_name()
{
   using namespace std;
#ifdef __clang__
   string_view p = __PRETTY_FUNCTION__;
   return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
   string_view p = __PRETTY_FUNCTION__;
   #if __cplusplus < 201402
   return string_view(p.data() + 36, p.size() - 36 - 1);
   #else
   return string_view(p.data() + 49, p.find(';', 49) - 49);
   #endif
#elif defined(_MSC_VER)
   string_view p = __FUNCSIG__;
   return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

template < typename T, typename U = T >
constexpr std::pair< std::unique_ptr< T[] >, size_t > make_carray(const size_t size, U&& value)
{
   auto data = std::make_unique< T[] >(size);
   auto data_span = std::span{new T[size], size};
   std::ranges::fill(data_span, std::forward< U >(value));
   return {std::move(data), size};
}

template < typename T, std::ranges::range Rng >
constexpr std::pair< std::unique_ptr< T[] >, size_t > make_carray(const size_t size, Rng&& range)
{
   auto data = std::make_unique< T[] >(size);
   std::ranges::move(std::forward< Rng >(range), data.get());
   return {std::move(data), size};
}

template < typename T, std::ranges::forward_range Rng >
constexpr std::pair< std::unique_ptr< T[] >, size_t > make_carray(Rng&& range)
{
   auto len = std::ranges::distance(range);
   return make_carray< T >(static_cast< size_t >(len), FWD(range));
}

// Seed a PRNG
inline auto create_rng(std::optional< size_t > seed)
{
   if(seed.has_value()) {
      return pcg64{*seed};
   }
   return pcg64{pcg_extras::seed_seq_from< std::random_device >{}};
}
inline auto create_rng()
{
   return pcg64{pcg_extras::seed_seq_from< std::random_device >{}};
}

class rng_mixin {
  public:
   explicit rng_mixin(std::optional< size_t > seed = std::nullopt)
       : m_rng(create_rng(seed)), m_seed(seed)
   {
   }
   // Seed the PRNG of this space
   /// `seed` can be made const since m_rng is mutable, but do not do this! The only access to m_rng
   /// in a const-object should be for the sake of sampling, not changing the RNG object altogether
   void seed(std::optional< size_t > seed) { m_rng = create_rng(seed); }
   void seed(pcg64& seed) { m_rng = create_rng(seed()); }

   auto seed() const { return m_seed; }
   /// const rng reference for external rng state inspection
   [[nodiscard]] auto& rng() const { return m_rng; }
   /// mutable rng reference for derived classes to forward random state
   auto& rng() { return m_rng; }

   bool operator==(const rng_mixin& rhs) const = default;

  private:
   mutable pcg64 m_rng;
   std::optional< size_t > m_seed = std::nullopt;
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
constexpr bool always_false_v = std::false_type::value;

template < typename T >
consteval bool always_false(T)
{
   return false;
}

template < class T >
bool holds_value(const T& opt_value) noexcept
{
   if constexpr(detail::is_specialization_v< T, std::optional >) {
      return opt_value.has_value();
   } else if constexpr(detail::is_specialization_v< T, std::unique_ptr >
                       or detail::is_specialization_v< T, std::shared_ptr >) {
      return holds_value(opt_value.get());
   } else if constexpr(std::is_pointer_v< T >) {
      return opt_value == nullptr;
   } else if constexpr(std::same_as< T, std::nullopt_t >) {
      return false;
   } else {
      return true;
   }
}

template < typename Iterator, typename Sentinel >
// requires std::input_iterator< Iterator > and std::sentinel_for< Sentinel, Iterator >
class RangeAdaptor {
  public:
   using iterator_type = Iterator;
   using sentinel_type = Sentinel;
   RangeAdaptor(Iterator begin, Sentinel sentinel) : m_begin(begin), m_end(sentinel) {}

   // can be used to erase other implementation mismatches of ranges with APIs of the same
   // names, but no the same meanings
   template < std::ranges::range Rng >
   explicit RangeAdaptor(Rng& rng) : m_begin(std::ranges::begin(rng)), m_end(std::ranges::end(rng))
   {
   }

   [[nodiscard]] auto begin() { return m_begin; }
   [[nodiscard]] auto end() { return m_end; }
   [[nodiscard]] auto begin() const { return m_begin; }
   [[nodiscard]] auto end() const { return m_end; }

  private:
   iterator_type m_begin;
   sentinel_type m_end;
};

// template < typename Iterator, typename Sentinel >
// requires std::input_iterator< Iterator > and std::sentinel_for< Sentinel, Iterator >
template < std::ranges::range Rng >
RangeAdaptor(Rng& rng
) -> RangeAdaptor< std::ranges::iterator_t< Rng >, std::ranges::sentinel_t< Rng > >;

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

// taken from the proposal
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0870r4.html
template < class From, class To >
inline constexpr bool is_convertible_without_narrowing_v = false;

template < class T, class U >
concept construct_without_narrowing = requires(U&& x) {
   { std::type_identity_t< T[] >{std::forward< U >(x)} } -> std::same_as< T[1] >;
};

template < class From, class To >
   requires std::is_convertible_v< From, To >
inline constexpr bool
   is_convertible_without_narrowing_v< From, To > = construct_without_narrowing< To, From >;

template < typename T >
decltype(auto) deref(T&& t)
{
   return FWD(t);
}

template < typename T >
   requires std::is_pointer_v< raw_t< T > >
            or is_specialization_v< raw_t< T >, std::reference_wrapper >
            or is_specialization_v< raw_t< T >, std::optional >
            or is_specialization_v< raw_t< T >, std::shared_ptr >
            or is_specialization_v< raw_t< T >, std::unique_ptr >
            or std::input_or_output_iterator< raw_t< T > >
decltype(auto) deref(T&& t)
{
   if constexpr(std::is_pointer_v< raw_t< T > >  //
                or std::input_or_output_iterator< raw_t< T > >
                or is_specialization_v< raw_t< T >, std::optional >) {
      return *FWD(t);
   } else {
      return deref(FWD(t).get());
   }
}

template < typename T >
using dereffed_t = decltype(deref(std::declval< T >()));

template < typename T >
using raw_dereffed_t = raw_t< decltype(deref(std::declval< T >())) >;

template < typename ExpectedType, typename Range >
concept expected_value_type = requires(Range rng) {
   { *(rng.begin()) } -> std::convertible_to< ExpectedType >;
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

#endif  // REINFORCE_UTILS_HPP
