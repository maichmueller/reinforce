
#ifndef REINFORCE_ONEOF_HPP
#define REINFORCE_ONEOF_HPP

#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>
#include <reinforce/utils/tuple_utils.hpp>
#include <tuple>
#include <variant>
#include <xtensor/xrandom.hpp>

#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/macro.hpp"
namespace force {

template < typename... Spaces >
class OneOf:
    public Space<
       std::pair< size_t, std::variant< detail::value_t< Spaces >... > >,
       OneOf< std::variant< detail::value_t< Spaces >... > >,
       std::pair< size_t, std::variant< detail::multi_value_t< Spaces >... > > > {
  public:
   friend class Space<
      std::pair< size_t, std::variant< detail::value_t< Spaces >... > >,
      OneOf< std::variant< detail::value_t< Spaces >... > >,
      std::pair< size_t, std::variant< detail::multi_value_t< Spaces >... > > >;
   using base = Space<
      std::pair< size_t, std::variant< detail::value_t< Spaces >... > >,
      OneOf< std::variant< detail::value_t< Spaces >... > >,
      std::pair< size_t, std::variant< detail::multi_value_t< Spaces >... > > >;
   using data_type = std::variant< typename Spaces::data_type... >;
   using typename base::value_type;
   using typename base::multi_value_type;
   using base::seed;
   using base::shape;
   using base::rng;

   using spaces_idx_seq = std::index_sequence_for< Spaces... >;

  private:
   std::tuple< Spaces... > m_spaces;

  public:
   template < std::integral T = size_t >
   explicit OneOf(T seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      SPDLOG_DEBUG("Called oneof space constructor with seed");
      seed(seed_);
   }

   template < typename OptionalT = std::optional< size_t > >
      requires detail::is_specialization_v< OptionalT, std::optional >
               and std::convertible_to< detail::value_t< OptionalT >, size_t >
   explicit OneOf(OptionalT seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      SPDLOG_DEBUG("Called oneof space constructor with optional-seed");
      seed(seed_);
   }

   explicit OneOf(Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      seed(std::optional< size_t >{});
   }

   template < typename T >
   void seed(T value)
   {
      base::seed(value);
      std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            (std::get< Is >(m_spaces).seed(rng()), ...);
         },
         spaces_idx_seq{}
      );
   }

   bool operator==(const OneOf& other) const
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return (std::get< Is >(m_spaces).operator==(std::get< Is >(other.m_spaces)) && ...);
         },
         spaces_idx_seq{}
      );
   }

   [[nodiscard]] std::string repr() const
   {
      return fmt::format(
         "OneOf({})",
         fmt::join(
            std::invoke(
               [&]< size_t... Is >(std::index_sequence< Is... >) {
                  return std::tuple{std::get< Is >(m_spaces)...};
               },
               spaces_idx_seq{}
            ),
            ", "
         )
      );
   }

   template < size_t N >
   auto& get_space() const
   {
      return std::get< N >(m_spaces);
   }
   template < size_t N >
   auto& get_space()
   {
      return std::get< N >(m_spaces);
   }

   [[nodiscard]] constexpr size_t size() const { return std::tuple_size_v< value_type >; }

  private:
   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
               and (std::tuple_size_v< detail::raw_t< MaskTuple > > == sizeof...(Spaces))
   [[nodiscard]] multi_value_type _sample(size_t nr_samples, MaskTuple&& mask_tuple) const
   {
      auto space_with_mask_tuple = zip_tuples(m_spaces, FWD(mask_tuple));
      size_t space_idx = xt::random::randint({1}, 0, sizeof...(Spaces), rng()).unchecked(0);
      return {
         space_idx,
         visit_at_unchecked(
            space_with_mask_tuple,
            space_idx,
            [=](const auto& space_and_mask) {
               auto&& [space, mask] = space_and_mask;
               return space.sample(nr_samples, mask);
            }
         )
      };
   }

   template < typename... MaskTs >
      requires(sizeof...(MaskTs) == sizeof...(Spaces))
   [[nodiscard]] multi_value_type _sample(size_t nr_samples, MaskTs&&... masks) const
   {
      return sample(nr_samples, std::tuple{FWD(masks)...});
   }

   [[nodiscard]] multi_value_type _sample(size_t nr_samples) const
   {
      size_t space_idx = xt::random::randint({1}, 0, sizeof...(Spaces), rng()).unchecked(0);
      return {space_idx, visit_at_unchecked(m_spaces, space_idx, [=](const auto& space) {
                 return space.sample(nr_samples);
              })};
   }
   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
               and (std::tuple_size_v< detail::raw_t< MaskTuple > > == sizeof...(Spaces))
   [[nodiscard]] value_type _sample(MaskTuple&& mask_tuple) const
   {
      auto space_with_mask_tuple = zip_tuples(m_spaces, FWD(mask_tuple));
      size_t space_idx = xt::random::randint({1}, 0, sizeof...(Spaces), rng()).unchecked(0);
      return {
         space_idx,
         visit_at_unchecked(
            space_with_mask_tuple,
            space_idx,
            [](const auto& space_and_mask) {
               auto&& [space, mask] = space_and_mask;
               return space.sample(mask);
            }
         )
      };
   }

   template < typename FirstMaskT, typename... TailMaskTs >
      requires(
         not std::is_integral_v< detail::raw_t< FirstMaskT > >
         and (sizeof...(TailMaskTs) == sizeof...(Spaces) - 1)
      )
   [[nodiscard]] value_type _sample(FirstMaskT&& mask1, TailMaskTs&&... tail_masks) const
   {
      return sample(std::forward_as_tuple(FWD(mask1), FWD(tail_masks)...));
   }

   [[nodiscard]] value_type _sample(std::nullopt_t = std::nullopt) const
   {
      size_t space_idx = xt::random::randint({1}, 0, sizeof...(Spaces), rng()).unchecked(0);
      return {space_idx, visit_at_unchecked(m_spaces, space_idx, [](const auto& space) {
                 return space.sample();
              })};
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      auto&& [space_idx, value_of_space] = value;
      return visit_at_unchecked(m_spaces, space_idx, [&]< typename SpaceT >(const SpaceT& space) {
         return space.contains(value_of_space);
      });
   }
}
};

}  // namespace force

#endif  // REINFORCE_ONEOF_HPP
