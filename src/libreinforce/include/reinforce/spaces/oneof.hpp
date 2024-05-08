
#ifndef REINFORCE_ONEOF_HPP
#define REINFORCE_ONEOF_HPP

#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>
#include <reinforce/utils/tuple_utils.hpp>
#include <tuple>
#include <variant>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xhistogram.hpp>
#include <xtensor/xrandom.hpp>

#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/macro.hpp"
namespace force {

template < typename... Spaces >
class OneOfSpace:
    public Space<
       std::pair< size_t, std::variant< detail::value_t< Spaces >... > >,
       OneOfSpace< Spaces... >,
       std::vector< std::pair< size_t, std::variant< detail::value_t< Spaces >... > > > > {
  public:
   friend class Space<
      std::pair< size_t, std::variant< detail::value_t< Spaces >... > >,
      OneOfSpace< Spaces... >,
      std::vector< std::pair< size_t, std::variant< detail::value_t< Spaces >... > > > >;
   using base = Space<
      std::pair< size_t, std::variant< detail::value_t< Spaces >... > >,
      OneOfSpace< Spaces... >,
      std::vector< std::pair< size_t, std::variant< detail::value_t< Spaces >... > > > >;
   using data_type = std::variant< typename Spaces::data_type... >;
   using value_variant_type = std::variant< detail::value_t< Spaces >... >;
   using typename base::value_type;
   using typename base::batch_value_type;
   using base::seed;
   using base::shape;
   using base::rng;

   using spaces_idx_seq = std::index_sequence_for< Spaces... >;

  private:
   std::tuple< Spaces... > m_spaces;

  public:
   template < std::integral T = size_t >
   explicit OneOfSpace(T seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      SPDLOG_DEBUG("Called oneof space constructor with seed");
      seed(seed_);
   }

   template < typename OptionalT = std::optional< size_t > >
      requires detail::is_specialization_v< OptionalT, std::optional >
               and std::convertible_to< detail::value_t< OptionalT >, size_t >
   explicit OneOfSpace(OptionalT seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      SPDLOG_DEBUG("Called oneof space constructor with optional-seed");
      seed(seed_);
   }

   explicit OneOfSpace(Spaces... spaces) : m_spaces{std::move(spaces)...}
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

   bool operator==(const OneOfSpace& other) const
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

   INJECT_STRUCTURED_BINDING_GETTERS(m_spaces)

   [[nodiscard]] constexpr size_t size() const { return std::tuple_size_v< value_type >; }

  private:
   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
               and (std::tuple_size_v< detail::raw_t< MaskTuple > > == sizeof...(Spaces))
   [[nodiscard]] batch_value_type _sample(size_t batch_size, MaskTuple&& mask_tuple) const
   {
      // generate how many samples we need from each space
      auto batch_size_per_space = xt::bincount(
         xt::random::randint< size_t >({batch_size}, 0, sizeof...(Spaces), rng())
      );
      // sample now from each space with the corresponding mask as many times as the bincount says
      // and stack the results together.
      batch_value_type result;
      result.reserve(batch_size);
      std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            (static_cast< void >(_append_samples< Is >(
                result,
                std::get< Is >(m_spaces).sample(batch_size_per_space.at(Is)),
                std::get< Is >(mask_tuple)
             )),
             ...);
         },
         spaces_idx_seq{}
      );
      // shuffle the result vector to ensure that the individual samples are not grouped by space
      ranges::shuffle(result, rng());
      return result;
   }

   template < typename... MaskTs >
      requires(sizeof...(MaskTs) == sizeof...(Spaces))
   [[nodiscard]] batch_value_type _sample(size_t batch_size, MaskTs&&... masks) const
   {
      return sample(batch_size, std::tuple{FWD(masks)...});
   }

   [[nodiscard]] batch_value_type _sample(size_t batch_size) const
   {
      // generate how many samples we need from each space
      auto batch_size_per_space = xt::bincount(
         xt::random::randint< size_t >({batch_size}, 0, sizeof...(Spaces), rng())
      );
      // sample now from each space with the corresponding mask as many times as the bincount says
      // and stack the results together.
      batch_value_type result;
      result.reserve(batch_size);
      std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            (static_cast< void >(_append_samples< Is >(
                result, std::get< Is >(m_spaces).sample(batch_size_per_space.at(Is))
             )),
             ...);
         },
         spaces_idx_seq{}
      );
      // shuffle the result vector to ensure that the individual samples are not grouped by space
      ranges::shuffle(result, rng());
      return result;
   }
   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
               and (std::tuple_size_v< detail::raw_t< MaskTuple > > == sizeof...(Spaces))
   [[nodiscard]] value_type _sample(MaskTuple&& mask_tuple) const
   {
      auto space_with_mask_tuple = zip_tuples(m_spaces, FWD(mask_tuple));
      size_t space_idx = xt::random::randint< size_t >({1}, 0, sizeof...(Spaces), rng())
                            .unchecked(0);
      return {space_idx, visit(space_with_mask_tuple, space_idx, [](const auto& space_and_mask) {
                 auto&& [space, mask] = space_and_mask;
                 return space.sample(mask);
              })};
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
      size_t space_idx = xt::random::randint< size_t >({1}, 0, sizeof...(Spaces), rng())
                            .unchecked(0);
      return {
         space_idx, visit(m_spaces, space_idx, [](const auto& space) { return space.sample(); })
      };
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      auto&& [space_idx, value_of_space] = value;
      return visit_at_unchecked(m_spaces, space_idx, [&]< typename SpaceT >(const SpaceT& space) {
         return space.contains(value_of_space);
      });
   }

   decltype(auto) visit(auto&&... args) const
   {
#ifndef NDEBUG
      return visit_at(FWD(args)...);
#else
      return visit_at_unchecked(FWD(args)...);
#endif
   }

   template < size_t space_idx >
   void _append_sample(batch_value_type& result, auto&& entry) const
   {
      result.push_back(
         {space_idx, value_variant_type{std::in_place_index< space_idx >, std::move(entry)}}
      );
   };

   template < size_t space_idx, typename Container >
   void _append_samples(batch_value_type& result, Container&& samples) const
   {
      if constexpr(detail::is_xarray< detail::raw_t< Container > >) {
         for(auto entry_iter = xt::axis_begin(samples, 0); entry_iter != xt::axis_end(samples, 0);
             entry_iter++) {
            _append_sample< space_idx >(result, *entry_iter);
         }
      } else {
         for(auto&& entry : FWD(samples)) {
            _append_sample< space_idx >(result, std::move(entry));
         }
      }
   };
};

}  // namespace force

#endif  // REINFORCE_ONEOF_HPP
