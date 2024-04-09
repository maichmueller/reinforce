#ifndef REINFORCE_VIEWS_EXTENSION_HPP
#define REINFORCE_VIEWS_EXTENSION_HPP

#include <range/v3/all.hpp>

namespace force::detail {
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

template < typename T >
constexpr auto cast = ranges::views::transform([](auto n) { return static_cast< T >(n); });

}  // namespace ranges::views

#endif  // REINFORCE_VIEWS_EXTENSION_HPP
