
#ifndef REINFORCE_DISCRETE_HPP
#define REINFORCE_DISCRETE_HPP

#include <fmt/format.h>

#include <concepts>
#include <cstddef>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>

#include "reinforce/spaces/concepts.hpp"
#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

template < typename T >
   requires discrete_reqs< T >
class DiscreteSpace: public Space< T, DiscreteSpace< T >, xarray< T > > {
  public:
   friend class Space< T, DiscreteSpace< T >, xarray< T > >;
   using base = Space< T, DiscreteSpace< T >, xarray< T > >;
   using data_type = T;
   using typename base::value_type;
   using typename base::batch_value_type;
   using base::rng;
   using base::shape;

   explicit DiscreteSpace(T n, T start = 0, std::optional< size_t > seed = std::nullopt)
       : base({1}, seed), m_nr_values(n), m_start(start)
   {
      if constexpr(std::is_signed_v< T >) {
         if(n <= 0) {
            throw std::invalid_argument("Counts parameter `n` has to be positive");
         }
      }
   }

   bool operator==(const DiscreteSpace< T >& rhs) const
   {
      return m_nr_values == rhs.m_nr_values && m_start == rhs.m_start;
   }

   [[nodiscard]] std::string repr() const
   {
      if(m_start != 0) {
         return fmt::format("Discrete({}, start={})", m_nr_values, m_start);
      }
      return fmt::format("Discrete({})", m_nr_values);
   }

   auto start() const { return m_start; }
   auto n() const { return m_nr_values; }

  private:
   T m_nr_values;
   T m_start;

   /// for tag dispatch within this class
   struct internal_tag_t {};
   static constexpr internal_tag_t internal_tag{};

   [[nodiscard]] value_type _sample(std::nullopt_t /*unused*/ = std::nullopt) const
   {
      return xt::random::randint< T >({1}, m_start, m_start + m_nr_values, rng()).unchecked(0);
   }

   [[nodiscard]] value_type _sample(const xarray< bool >& mask) const
   {
      return xt::random::choice(
                xt::eval(xt::filter(xt::arange(m_start, m_start + m_nr_values), mask)),
                1,
                true,
                rng()
      )
         .unchecked(0);
   }

   [[nodiscard]] batch_value_type _sample(size_t batch_size) const;

   [[nodiscard]] batch_value_type _sample(size_t batch_size, std::nullopt_t) const;

   template < std::ranges::range MaskRange >
   [[nodiscard]] batch_value_type _sample(size_t batch_size, MaskRange&& mask) const;

   [[nodiscard]] batch_value_type
   _sample(internal_tag_t, size_t batch_size, const xarray< bool >& mask) const;

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return m_start <= value && value < m_start + m_nr_values;
   }
   [[nodiscard]] bool _contains(const batch_value_type& value) const
   {
      return xt::all(
         xt::greater_equal(value, m_start) and xt::less_equal(value, m_start + m_nr_values)
      );
   }
};

template < typename T >
   requires discrete_reqs< T >
auto DiscreteSpace< T >::_sample(size_t batch_size) const -> batch_value_type
{
   return xt::random::randint< T >({batch_size}, m_start, m_start + m_nr_values, rng());
}

template < typename T >
   requires discrete_reqs< T >
auto DiscreteSpace< T >::_sample(size_t batch_size, std::nullopt_t /*unused*/) const
   -> batch_value_type
{
   return _sample(batch_size);
}

template < typename T >
   requires discrete_reqs< T >
template < std::ranges::range MaskRange >
auto DiscreteSpace< T >::_sample(size_t batch_size, MaskRange&& mask) const -> batch_value_type
{
   using namespace ranges;
   if(batch_size == 0) {
      return xt::empty< value_type >({0});
   }

   if constexpr(std::ranges::sized_range< MaskRange >) {
      if(std::ranges::size(mask) != static_cast< size_t >(m_nr_values)) {
         throw std::invalid_argument(
            fmt::format("Mask size cannot be smaller than the number of elements ({})", m_nr_values)
         );
      }
   }

   if constexpr(detail::is_xarray< MaskRange >
                and std::integral< detail::value_t< detail::raw_t< MaskRange > > >) {
      return _sample(internal_tag, batch_size, mask);
   }

   std::vector< value_type > valid_indices;
   valid_indices.reserve(batch_size);
   auto&& flat_mask = xt::flatten(mask);
   [[maybe_unused]] size_t mask_size = 0;
   for(auto [i, selected] : views::zip(views::indices(0ul, batch_size), flat_mask)) {
      if(selected) {
         valid_indices.emplace_back(i);
      }
      ++mask_size;
   }
   if constexpr(not std::ranges::sized_range< MaskRange >) {
      if(mask_size != static_cast< size_t >(m_nr_values)) {
         throw std::invalid_argument(
            fmt::format("Mask size cannot be smaller than the number of elements ({})", m_nr_values)
         );
      }
   }

   auto samples = xt::empty< value_type >(xt::svector{batch_size});
   if(not valid_indices.empty()) {
      std::uniform_int_distribution< size_t > dist(0UL, valid_indices.size() - 1);
      for(auto i : views::iota(0UL, batch_size)) {
         samples.unchecked(i) = m_start + static_cast< value_type >(valid_indices[dist(rng())]);
      }
      return samples;
   }
   return {};
}

template < typename T >
   requires discrete_reqs< T >
auto DiscreteSpace< T >::_sample(internal_tag_t, size_t batch_size, const xarray< bool >& mask)
   const -> batch_value_type
{
   return xt::random::choice(
      xt::eval(xt::filter(xt::arange(m_start, m_start + m_nr_values), mask)),
      batch_size,
      true,
      rng()
   );
}

}  // namespace force

#endif  // REINFORCE_DISCRETE_HPP
