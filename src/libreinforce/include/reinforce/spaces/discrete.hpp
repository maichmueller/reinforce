
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

   explicit TypedDiscrete(int n, int start = 0, std::optional< size_t > seed = std::nullopt)
       : base({}, seed), m_nr_values(n), m_start(start)
   {
      if(n <= 0) {
         throw std::invalid_argument("n (counts) have to be positive");
      }
      if(start < 0) {
         throw std::invalid_argument("start must be non-negative");
      }
   }

   int sample(std::optional< xarray< bool > > mask_opt = std::nullopt)
   {
      if(mask_opt.has_value()) {
         const auto& mask = *mask_opt;
         if(mask.size() != m_nr_values) {
            throw std::invalid_argument("Mask size must match the number of elements (n)");
         }

         std::vector< int > valid_indices;
         valid_indices.reserve(mask.size());
         auto&& flat_mask = xt::flatten(mask);
         for(auto [i, m] : ranges::views::enumerate(flat_mask)) {
            if(m) {
               valid_indices.emplace_back(i);
            }
         }

         if(not valid_indices.empty()) {
            const std::uniform_int_distribution< size_t > dist(0UL, valid_indices.size() - 1);
            int selected_index = valid_indices[dist(rng())];
            return m_start + selected_index;
         }
         return m_start;
      }
      std::uniform_int_distribution< int > dist(0, m_nr_values - 1);
      return m_start + dist(rng());
   }

   bool contains(int value) { return m_start <= value && value < m_start + m_nr_values; }

   std::string repr()
   {
      if(m_start != 0) {
         return fmt::format("Discrete({}, start={})", m_nr_values, m_start);
      }
      return fmt::format("Discrete({})", m_nr_values);
   }

   bool operator==(const TypedDiscrete< T >& other)
   {
      return m_nr_values == other.m_nr_values && m_start == other.m_start;
   }

  private:
   int m_nr_values;
   int m_start;
};

}  // namespace force

#endif  // REINFORCE_DISCRETE_HPP
