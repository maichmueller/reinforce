
#ifndef REINFORCE_DISCRETE_HPP
#define REINFORCE_DISCRETE_HPP

#include <fmt/format.h>
#include <pybind11/numpy.h>

#include <optional>
#include <random>
#include <stdexcept>
#include <vector>
#include <xtensor/xmath.hpp>

#include "space.hpp"

namespace force {

template < std::integral T >
class TypedDiscrete: public TypedSpace< T > {
  public:
   using base = TypedSpace< T >;
   using base::rng;

   explicit TypedDiscrete(T n, T start = 0, std::optional< size_t > seed = std::nullopt)
       : base({}, seed), m_nr_values(n), m_start(start)
   {
      if constexpr(std::is_signed_v< T >) {
         if(n <= 0) {
            throw std::invalid_argument("n (counts) have to be positive");
         }
      }
   }

   xarray< T > sample(const std::optional< xarray< bool > >& mask_opt = std::nullopt) override
   {
      if(mask_opt.has_value()) {
         const auto& mask = *mask_opt;
         if(mask.size() != m_nr_values) {
            throw std::invalid_argument(
               fmt::format("Mask size must match the number of elements ({})", m_nr_values)
            );
         }

         std::vector< int > valid_indices;
         valid_indices.reserve(mask.size());
         auto&& flat_mask = xt::flatten(mask);
         for(auto [i, selected] : ranges::views::enumerate(flat_mask)) {
            if(selected) {
               valid_indices.emplace_back(i);
            }
         }

         if(not valid_indices.empty()) {
            const std::uniform_int_distribution< size_t > dist(0UL, valid_indices.size() - 1);
            return m_start + static_cast< T >(valid_indices[dist(rng())]);
         }
         return m_start;
      }
      std::uniform_int_distribution< int > dist(0, m_nr_values - 1);
      return {m_start + dist(rng())};
   }

   bool contains(int value) { return m_start <= value && value < m_start + m_nr_values; }

   std::string repr()
   {
      if(m_start != 0) {
         return fmt::format("Discrete({}, start={})", m_nr_values, m_start);
      }
      return fmt::format("Discrete({})", m_nr_values);
   }

  private:
   T m_nr_values;
   T m_start;

   bool _equals(const TypedSpace< T >& rhs) const override
   {
      // we can safely use static-cast here, because the base checks for type-identity first and
      // only calls equals if the types of two compared objects are the same (hence
      // TypedDiscrete<T>)
      const auto& other_cast = static_cast< const TypedDiscrete< T >& >(rhs);
      return m_nr_values == other_cast.m_nr_values && m_start == other_cast.m_start;
   }
};

}  // namespace force

#endif  // REINFORCE_DISCRETE_HPP
