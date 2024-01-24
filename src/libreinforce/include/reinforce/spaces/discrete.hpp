
#ifndef REINFORCE_DISCRETE_HPP
#define REINFORCE_DISCRETE_HPP

#include <fmt/format.h>

#include <optional>
#include <random>
#include <stdexcept>
#include <vector>
#include <xtensor/xmath.hpp>

#include "mono_space.hpp"

namespace force {

template < std::integral T >
class TypedDiscreteSpace: public TypedMonoSpace< T, TypedDiscreteSpace< T > > {
  public:
   using value_type = T;
   friend class TypedMonoSpace< T, TypedDiscreteSpace >;
   using base = TypedMonoSpace< T, TypedDiscreteSpace >;
   using base::rng;

   explicit TypedDiscreteSpace(T n, T start = 0, std::optional< size_t > seed = std::nullopt)
       : base({}, seed), m_nr_values(n), m_start(start)
   {
      if constexpr(std::is_signed_v< T >) {
         if(n <= 0) {
            throw std::invalid_argument("Counts parameter `n` has to be positive");
         }
      }
   }

   bool operator==(const TypedDiscreteSpace< T >& rhs) const
   {
      return m_nr_values == rhs.m_nr_values && m_start == rhs.m_start;
   }

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

   xarray< T > _sample(const std::optional< xarray< bool > >& mask_opt = std::nullopt)
   {
      return _sample(size_t{1}, mask_opt);
   }

   xarray< T >
   _sample(size_t nr_samples, const std::optional< xarray< bool > >& mask_opt = std::nullopt);

   bool _contains(const T& value) const
   {
      return m_start <= value && value < m_start + m_nr_values;
   }
};

template < std::integral T >
xarray< T >
TypedDiscreteSpace< T >::_sample(size_t nr_samples, const std::optional< xarray< bool > >& mask_opt)
{
   auto samples = xt::empty< T >(xt::svector{nr_samples});

   if(not mask_opt.has_value()) {
      std::uniform_int_distribution< int > dist(0, m_nr_values - 1);
      for(auto i : ranges::views::iota(0UL, nr_samples)) {
         samples.unchecked(i) = m_start + dist(rng());
      }
      return samples;
   }

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
      for(auto i : ranges::views::iota(0UL, nr_samples)) {
         samples.unchecked(i) = m_start + valid_indices[dist(rng())];
      }
      return samples;
   }
   return {};
}

}  // namespace force

#endif  // REINFORCE_DISCRETE_HPP
