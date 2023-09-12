
#ifndef REINFORCE_UTILS_HPP
#define REINFORCE_UTILS_HPP

#include <utility>

#include "range/v3/all.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

namespace force::detail {

template < typename Iter >
class RangeAdaptor {
  public:
   using type = Iter;
   RangeAdaptor(Iter begin, Iter end) : m_begin(begin), m_end(end) {}

   [[nodiscard]] auto begin() { return m_begin; }
   [[nodiscard]] auto end() { return m_end; }
   [[nodiscard]] auto begin() const { return m_begin; }
   [[nodiscard]] auto end() const { return m_end; }

  private:
   Iter m_begin, m_end;
};

template < typename Iter >
class SizedRangeAdaptor {
  public:
   using type = Iter;
   SizedRangeAdaptor(Iter begin, Iter end, size_t size) : m_begin(begin), m_end(end), m_size(size)
   {
   }

   [[nodiscard]] auto begin() { return m_begin; }
   [[nodiscard]] auto end() { return m_end; }
   [[nodiscard]] auto begin() const { return m_begin; }
   [[nodiscard]] auto end() const { return m_end; }
   [[nodiscard]] auto size() const { return m_size; }

  private:
   Iter m_begin, m_end;
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

/// is_specialization checks whether T is a specialized template class of 'Template'
/// This has the limitation of
/// usage:
///     constexpr bool is_vector = is_specialization< std::vector< int >, std::vector>;
///
/// Note that this trait has 2 limitations:
///  1) Does not work with non-type parameters.
///     (i.e. templates such as std::array will not be compatible with this type trait)
///  2) Generally, templated typedefs do not get captured as the underlying template but as the
///     typedef template. As such the following scenarios will return:
///          specialization<uptr<int>, uptr> << std::endl;            -> false
///          specialization<std::unique_ptr<int>, uptr>;              -> false
///          specialization<std::unique_ptr<int>, std::unique_ptr>;   -> true
///          specialization<uptr<int>, std::unique_ptr>;              -> true
///     for a typedef template template <typename T> using uptr = std::unique_ptr< T >;
template < class T, template < class... > class Template >
struct is_specialization: std::false_type {};

template < template < class... > class Template, class... Args >
struct is_specialization< Template< Args... >, Template >: std::true_type {};

template < class T, template < class... > class Template >
constexpr bool is_specialization_v = is_specialization< T, Template >::value;

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

}  // namespace force::detail

namespace ranges::views {

constexpr ::force::detail::deref_fn deref{};

}  // namespace ranges::views

#endif  // REINFORCE_UTILS_HPP
