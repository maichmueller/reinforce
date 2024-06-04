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
#include <reinforce/utils/concatenate.hpp>
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

template < typename FeatureSpace, bool stacked >
using stacked_value_type = std::conditional_t<
   stacked,
   typename concatenate< FeatureSpace >::result_type,
   std::vector< detail::value_t< FeatureSpace > > >;

}  // namespace detail

template < typename FeatureSpace, bool stacked = true >
class SequenceSpace:
    public Space<
       detail::stacked_value_type< FeatureSpace, stacked >,
       SequenceSpace< FeatureSpace, stacked >,
       std::vector< typename FeatureSpace::batch_value_type > > {
   /// a tag for class-internal dispatch
   struct internal_tag_t {};
   static constexpr internal_tag_t internal_tag{};

   static constexpr bool _is_composite_space = true;

  public:
   friend class Space<
      detail::stacked_value_type< FeatureSpace, stacked >,
      SequenceSpace,
      std::vector< typename FeatureSpace::batch_value_type > >;
   using base = Space<
      detail::stacked_value_type< FeatureSpace, stacked >,
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
   [[nodiscard]] auto _lengths_sampler(size_t batch_size, Range&& lengths_mask_range) const;
};
template < typename FeatureSpace, bool stacked >
template < typename MaskT1, typename MaskT2 >
auto SequenceSpace< FeatureSpace, stacked >::_sample(const std::tuple< MaskT1, MaskT2 >& mask_tuple
) const -> value_type
{
   auto&& [length_rng, feature_mask] = mask_tuple;
   size_t minibatch_size = std::geometric_distribution< size_t >{m_geometric_prob}(rng());
   constexpr auto concat = concatenate< feature_space_type >{};
   return concat(
      m_feature_space,
      std::views::iota(0UL, minibatch_size)  //
         | std::views::transform([&](auto) { return m_feature_space.sample(feature_mask); })
   );
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
   return _lengths_sampler(batch_size, FWD(length_rng))  //
          | ranges::views::transform([&](auto sub_batch_size) {
               return m_feature_space.sample(sub_batch_size, feature_mask);
            })
          | ranges::to_vector;
}

template < typename FeatureSpace, bool stacked >
template < typename Range >
auto SequenceSpace< FeatureSpace, stacked >::_lengths_sampler(
   size_t batch_size,
   Range&& lengths_mask_range
) const
{
   using namespace detail;
   if constexpr(std::same_as< raw_t< Range >, std::nullopt_t >) {
      // if none given, then sample the length from a geometric distribution
      return ranges::views::indices(0ul, batch_size)
             | ranges::views::transform(
                [&,
                 dist = std::geometric_distribution< size_t >{m_geometric_prob}](auto&&) mutable {
                   return dist(rng());
                }
             );
   } else if constexpr(std::convertible_to< raw_t< Range >, size_t >) {
      auto length = static_cast< size_t >(lengths_mask_range);
      if(length == 0) {
         throw std::invalid_argument(
            fmt::format("Expecting a fixed length mask greater than 0. Given: {}", length)
         );
      }
      return ranges::views::repeat_n(length, static_cast< ptrdiff_t >(batch_size));
   } else {
      auto len_vec = ranges::to_vector(FWD(lengths_mask_range));
      size_t n = len_vec.size();
      // sample with replacement from the length vector
      return ranges::views::indices(0ul, batch_size)
             | ranges::views::transform([&,
                                         dist = std::uniform_int_distribution< size_t >{0, n},
                                         lens = std::move(len_vec)](auto&&) mutable {
                  return lens[dist(rng())];
               });
   }
}

}  // namespace force

#endif  // REINFORCE_SPACES_SEQUENCE_HPP
