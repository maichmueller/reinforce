#ifndef REINFORCE_SPACES_SEQUENCE_HPP
#define REINFORCE_SPACES_SEQUENCE_HPP

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <range/v3/all.hpp>
#include <range/v3/detail/prologue.hpp>
#include <range/v3/iterator/traits.hpp>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xset_operation.hpp>
#include <xtensor/xstorage.hpp>

#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

namespace detail {

template < typename T >
struct vector_if_not_xarray {
   using type = std::conditional_t< is_xarray< T >, T, std::vector< T > >;
};

template < typename T >
using vector_if_not_xarray_t = typename vector_if_not_xarray< T >::type;
}  // namespace detail

template < typename FeatureSpace >
class TypedSequenceSpace:
    public TypedSpace<
       detail::vector_if_not_xarray_t< typename FeatureSpace::value_type >,
       TypedSequenceSpace< FeatureSpace >,
       std::vector< typename FeatureSpace::multi_value_type > > {
   struct internal_tag {};

  public:
   friend class TypedSpace<
      detail::vector_if_not_xarray_t< typename FeatureSpace::value_type >,
      TypedSequenceSpace,
      std::vector< typename FeatureSpace::multi_value_type > >;
   using base = TypedSpace<
      detail::vector_if_not_xarray_t< typename FeatureSpace::value_type >,
      TypedSequenceSpace,
      std::vector< typename FeatureSpace::multi_value_type > >;
   using typename base::value_type;
   using typename base::multi_value_type;
   using base::shape;
   using base::rng;

   template < std::convertible_to< std::optional< size_t > > Int = std::nullopt_t >
      requires(not std::floating_point< Int >)
   TypedSequenceSpace(FeatureSpace space, Int seed = std::nullopt)
       : base(xt::svector< int >(), seed), m_feature_space(std::move(space))
   {
   }

   template < std::convertible_to< double > Float >
      requires(not std::integral< Float >)
   explicit TypedSequenceSpace(FeatureSpace space, Float geometric_probability)
       : base(xt::svector< int >()),
         m_feature_space(std::move(space)),
         m_geometric_prob(geometric_probability)
   {
   }

   explicit TypedSequenceSpace(
      FeatureSpace space,
      double geometric_probability,
      std::optional< size_t > seed
   )
       : base(xt::svector< int >(), seed),
         m_feature_space(std::move(space)),
         m_geometric_prob(geometric_probability)
   {
   }

   bool operator==(const TypedSequenceSpace& rhs) const = default;

   std::string repr() { return fmt::format("Sequence({}, stack=true)", m_feature_space); }

   auto& feature_space() const { return m_feature_space; }
   auto& feature_space() { return m_feature_space; }

  private:
   FeatureSpace m_feature_space;
   double m_geometric_prob = 0.25;

   template <
      template < typename... > class MaskTupleT = std::tuple,
      typename MaskT1 = std::nullopt_t,
      typename MaskT2 = std::nullopt_t >
      requires(
         (detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::tuple >
          or detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::pair >)
         or std::same_as< MaskT1, std::nullopt_t >
         or std::convertible_to< detail::raw_t< MaskT1 >, size_t >
         or (std::ranges::range< detail::raw_t< MaskT1 > > and std::convertible_to< ranges::value_type_t< detail::raw_t< MaskT1 > >, size_t >)
      )
   [[nodiscard]] multi_value_type
   _sample(size_t nr_samples, MaskTupleT< MaskT1, MaskT2 >&& mask_tuple = {}) const;

   [[nodiscard]] multi_value_type _sample(size_t nr_samples) const
   {
      return _sample(nr_samples, std::tuple{std::nullopt, std::nullopt});
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return m_feature_space.contains(value);
   }

   template < typename Range >
   [[nodiscard]] xarray< size_t > _compute_lengths(size_t nr_samples, Range&& lengths_rng) const;
};

// template implementations

template < typename FeatureSpace >
template < template < typename... > class MaskTupleT, typename MaskT1, typename MaskT2 >
   requires(
      (detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::tuple >
       or detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::pair >)
      or std::same_as< MaskT1, std::nullopt_t >
      or std::convertible_to< detail::raw_t< MaskT1 >, size_t >
      or (std::ranges::range< detail::raw_t< MaskT1 > > and std::convertible_to< ranges::value_type_t< detail::raw_t< MaskT1 > >, size_t >)
   )
auto TypedSequenceSpace< FeatureSpace >::_sample(
   size_t nr_samples,
   MaskTupleT< MaskT1, MaskT2 >&& mask_tuple
) const -> multi_value_type
{
   if(nr_samples == 0) {
      throw std::invalid_argument("`nr_samples` argument has to be greater than 0.");
   }
   auto&& [length_rng, feature_mask] = mask_tuple;
   // Compute the lenghts each sample should have. This is an array of potentially
   // differing integers which at index i indicates the sampled string size for sample i.
   auto lengths_per_sample = _compute_lengths(nr_samples, FWD(length_rng));
   SPDLOG_DEBUG(fmt::format("Lengths of each sample:\n{}", lengths_per_sample));
   return std::invoke([&] {
      return std::views::all(lengths_per_sample)  //
             | std::views::transform([&](auto nr_samples_concrete) {
                  if constexpr(std::same_as< MaskT2, std::nullopt_t >) {
                     return m_feature_space.sample(nr_samples_concrete);
                  } else {
                     return m_feature_space.sample(nr_samples_concrete, feature_mask);
                  }
               })  //
             | ranges::to_vector;
   });
}

template < typename FeatureSpace >
template < typename Range >
xarray< size_t >
TypedSequenceSpace< FeatureSpace >::_compute_lengths(size_t nr_samples, Range&& lengths_rng) const
{
   using namespace detail;
   if constexpr(std::same_as< raw_t< Range >, std::nullopt_t >) {
      // if none given, then sample the length from a geometric distribution
      return xt::random::geometric< size_t >(xt::svector{nr_samples}, m_geometric_prob, rng());
   } else if constexpr(std::convertible_to< raw_t< Range >, size_t >) {
      size_t length = static_cast< size_t >(lengths_rng);
      if(length == 0) {
         throw std::invalid_argument(
            fmt::format("Expecting a fixed length mask greater than 0. Given: {}", length)
         );
      }
      return xarray< size_t >{xt::adapt(
         make_carray< size_t >(nr_samples, length).first.release(),
         nr_samples,
         xt::acquire_ownership()
      )};
   } else {
      auto&& length_options = xt::cast< size_t >(std::invoke([&]() -> decltype(auto) {
         static constexpr bool is_already_xarray = is_xarray< raw_t< Range > >
                                                   or is_xarray_ref< raw_t< Range > >;
         if constexpr(is_already_xarray) {
            return FWD(lengths_rng);
         } else {
            // lengths is now confirmed to be a range
            xarray< size_t > arr = xt::empty< size_t >({std::ranges::distance(lengths_rng)});
            for(auto [index, value] : ranges::views::enumerate(lengths_rng)) {
               arr.unchecked(index) = static_cast< size_t >(value);
            }
            return arr;
         }
      }));
      return xt::random::choice(length_options, xt::svector{nr_samples}, true, rng());
   }
}

}  // namespace force

#endif  // REINFORCE_SPACES_SEQUENCE_HPP