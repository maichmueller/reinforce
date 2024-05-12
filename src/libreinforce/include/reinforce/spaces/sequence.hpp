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
#include "reinforce/utils/xtensor_extension.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

namespace detail {

template < typename Value, bool stacked >
struct stacked_value_type {
   static consteval auto type_selector()
   {
      using default_type = std::vector< Value >;
      if constexpr(stacked) {
         if constexpr(is_xarray< Value >) {
            return Value{};
         } else {
            return default_type{};
         }
      } else {
         return default_type{};
      }
   }
   using type = std::invoke_result_t< decltype(type_selector) >;
};

template < typename Value, bool stacked >
using stacked_value_type_t = typename stacked_value_type< Value, stacked >::type;
}  // namespace detail

template < typename FeatureSpace, bool stacked = true >
class SequenceSpace:
    public Space<
       detail::stacked_value_type_t< typename FeatureSpace::value_type, stacked >,
       SequenceSpace< FeatureSpace >,
       std::vector< typename FeatureSpace::batch_value_type > > {
   /// a tag for class-internal dispatch
   struct internal_tag_t {};
   static constexpr internal_tag_t internal_tag{};

   static constexpr bool _is_composite_space = true;

  public:
   friend class Space<
      detail::stacked_value_type_t< typename FeatureSpace::value_type, stacked >,
      SequenceSpace,
      std::vector< typename FeatureSpace::batch_value_type > >;
   using base = Space<
      detail::stacked_value_type_t< typename FeatureSpace::value_type, stacked >,
      SequenceSpace,
      std::vector< typename FeatureSpace::batch_value_type > >;
   using feature_space_type = FeatureSpace;
   using data_type = typename feature_space_type::data_type;
   using typename base::value_type;
   using typename base::batch_value_type;
   using base::seed;
   using base::shape;
   using base::rng;

   template < std::convertible_to< std::optional< size_t > > Int = std::nullopt_t >
      requires(not std::floating_point< Int >)
   explicit SequenceSpace(FeatureSpace space, Int seed_ = std::nullopt)
       : base(xt::svector< int >(), seed_), m_feature_space(std::move(space))
   {
      m_feature_space.seed(seed());
   }

   template < std::convertible_to< double > Float >
      requires(not std::integral< Float >)
   explicit SequenceSpace(FeatureSpace space, Float geometric_probability)
       : base(xt::svector< int >()),
         m_feature_space(std::move(space)),
         m_geometric_prob(geometric_probability)
   {
      m_feature_space.seed(seed());
   }

   explicit SequenceSpace(
      FeatureSpace space,
      double geometric_probability,
      std::optional< size_t > seed_
   )
       : base(xt::svector< int >(), seed_),
         m_feature_space(std::move(space)),
         m_geometric_prob(geometric_probability)
   {
      m_feature_space.seed(seed());
   }

   template < typename T >
   void seed(T value)
   {
      base::seed(value);
      m_feature_space.seed(value);
   }

   bool operator==(const SequenceSpace& rhs) const = default;

   [[nodiscard]] std::string repr() const
   {
      return fmt::format("Sequence({}, stack={})", m_feature_space, stacked);
   }

   auto& feature_space() const { return m_feature_space; }

  private:
   FeatureSpace m_feature_space;
   static constexpr double DEFAULT_GEOMETRIC_PROBABILITY = 0.25;
   double m_geometric_prob = DEFAULT_GEOMETRIC_PROBABILITY;

   template < typename MaskT1 = std::nullopt_t, typename MaskT2 = std::nullopt_t >
   [[nodiscard]] value_type _sample(const std::tuple< MaskT1, MaskT2 >& mask_tuple) const;

   [[nodiscard]] value_type _sample(std::nullopt_t /*unused*/ = std::nullopt) const
   {
      return _sample(std::tuple{std::nullopt, std::nullopt});
   }

   template < typename MaskT1 = std::nullopt_t, typename MaskT2 = std::nullopt_t >
   // requires(
   //    (detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::tuple >
   //     or detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::pair >)
   //    and (std::same_as< MaskT1, std::nullopt_t > or std::convertible_to< detail::raw_t< MaskT1
   //    >, size_t > or (std::ranges::range< detail::raw_t< MaskT1 > > and std::convertible_to<
   //    ranges::value_type_t< detail::raw_t< MaskT1 > >, size_t >))
   // )
   [[nodiscard]] batch_value_type
   _sample(size_t sub_batch_size, const std::tuple< MaskT1, MaskT2 >& mask_tuple = {}) const;

   [[nodiscard]] batch_value_type
   _sample(size_t batch_size, std::nullopt_t /*unused*/ = std::nullopt) const
   {
      return _sample(batch_size, std::tuple{std::nullopt, std::nullopt});
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return m_feature_space.contains(value);
   }

   template < typename Range >
   [[nodiscard]] xarray< size_t > _sample_lengths(size_t batch_size, Range&& lengths_rng) const;
};
template < typename FeatureSpace, bool stacked >
template < typename MaskT1, typename MaskT2 >
auto SequenceSpace< FeatureSpace, stacked >::_sample(const std::tuple< MaskT1, MaskT2 >& mask_tuple
) const -> value_type
{
   return nullptr;
}

// template implementations

template < typename FeatureSpace, bool stacked >
template < typename MaskT1, typename MaskT2 >
// requires(
//    (detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::tuple >
//     or detail::is_specialization_v< MaskTupleT< MaskT1, MaskT2 >, std::pair >)
//    and (std::same_as< MaskT1, std::nullopt_t > or std::convertible_to< detail::raw_t< MaskT1 >,
//    size_t > or (std::ranges::range< detail::raw_t< MaskT1 > > and std::convertible_to<
//    ranges::value_type_t< detail::raw_t< MaskT1 > >, size_t >))
// )
auto SequenceSpace< FeatureSpace, stacked >::_sample(
   size_t batch_size,
   const std::tuple< MaskT1, MaskT2 >& mask_tuple
) const -> batch_value_type
{
   if(batch_size == 0) {
      return batch_value_type{};
   }
   auto&& [length_rng, feature_mask] = mask_tuple;
   // Compute the lenghts each sample should have. This is an array of potentially
   // differing integers which at index i indicates the sampled string size for sample i.
   auto lengths_per_sample = _sample_lengths(batch_size, FWD(length_rng));
   SPDLOG_DEBUG(fmt::format("Lengths of each sample:\n{}", lengths_per_sample));
   return std::views::all(lengths_per_sample)  //
          | std::views::transform([&](auto sub_batch_size) {
               return m_feature_space.sample(sub_batch_size, feature_mask);
            })  //
          | ranges::to_vector;
}

template < typename FeatureSpace, bool stacked >
template < typename Range >
xarray< size_t > SequenceSpace< FeatureSpace, stacked >::_sample_lengths(
   size_t batch_size,
   Range&& lengths_rng
) const
{
   using namespace detail;
   if constexpr(std::same_as< raw_t< Range >, std::nullopt_t >) {
      // if none given, then sample the length from a geometric distribution
      return xt::random::geometric< size_t >(xt::svector{batch_size}, m_geometric_prob, rng());
   } else if constexpr(std::convertible_to< raw_t< Range >, size_t >) {
      auto length = static_cast< size_t >(lengths_rng);
      if(length == 0) {
         throw std::invalid_argument(
            fmt::format("Expecting a fixed length mask greater than 0. Given: {}", length)
         );
      }
      return xt::full(xt::svector{batch_size}, length);
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
      return xt::random::choice(length_options, xt::svector{batch_size}, true, rng());
   }
}

}  // namespace force

#endif  // REINFORCE_SPACES_SEQUENCE_HPP
