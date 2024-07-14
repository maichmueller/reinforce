
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

   static constexpr bool _is_composite_space = true;

  private:
   std::tuple< Spaces... > m_spaces;

  public:
   template < std::integral T = size_t >
   explicit OneOfSpace(T seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      seed(seed_);
   }

   template < typename OptionalT = std::optional< size_t > >
      requires detail::is_specialization_v< OptionalT, std::optional >
               and std::convertible_to< detail::value_t< OptionalT >, size_t >
   explicit OneOfSpace(OptionalT seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
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
                std::get< Is >(m_spaces).sample(
                   batch_size_per_space.at(Is), std::get< Is >(mask_tuple)
                )
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
      return _sample(batch_size, create_tuple< sizeof...(Spaces) >(std::nullopt));
   }

   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
               and (std::tuple_size_v< detail::raw_t< MaskTuple > > == sizeof...(Spaces))
   [[nodiscard]] value_type _sample(MaskTuple&& mask_tuple) const
   {
      size_t space_idx = std::uniform_int_distribution< size_t >{0, sizeof...(Spaces) - 1}(rng());
      return _sample_space_at< sizeof...(Spaces) - 1 >(space_idx, FWD(mask_tuple));
   }

   template < size_t I, typename MaskTuple >
   [[nodiscard]] value_type _sample_space_at(size_t space_idx, MaskTuple&& mask_tuple) const
   {
      if(space_idx == I) {
         const auto& space = std::get< I >(m_spaces);
         auto&& mask = std::get< I >(FWD(mask_tuple));
         return std::pair{I, value_variant_type{std::in_place_index< I >, space.sample(FWD(mask))}};
      }
      if constexpr(I == 0) {
         // this is unreachable if merely called from _sample. But we have to add this to stop the
         // compiler from generating a tuple_index_out_of_range error once I == 0 and hence
         // I - 1 == 18446744073709551615 (the size_t(-1) value on 64bit size_t systems).
         throw std::runtime_error("Invalid space index");
      } else {
         return _sample_space_at< I - 1 >(space_idx, FWD(mask_tuple));
      }
   }

   template < typename FirstMaskT, typename... TailMaskTs >
      requires(
         not std::is_integral_v< detail::raw_t< FirstMaskT > >
         and (sizeof...(TailMaskTs) == sizeof...(Spaces) - 1)
      )
   [[nodiscard]] value_type _sample(FirstMaskT&& mask1, TailMaskTs&&... tail_masks) const
   {
      return _sample(std::forward_as_tuple(FWD(mask1), FWD(tail_masks)...));
   }

   [[nodiscard]] value_type _sample(std::nullopt_t = std::nullopt) const
   {
      return _sample(create_tuple< sizeof...(Spaces) >(std::nullopt));
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      auto&& [space_idx, value_of_space] = value;
      return visit(m_spaces, space_idx, [&]< typename SpaceT >(const SpaceT& space) {
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
      result.emplace_back(
         space_idx, value_variant_type{std::in_place_index< space_idx >, std::move(entry)}
      );
   };

   template < size_t space_idx, typename Container >
   void _append_samples(batch_value_type& result, Container&& samples) const
   {
      if constexpr(detail::is_xarray< detail::raw_t< Container > >) {
         if constexpr(not detail::is_xarray< detail::value_t<
                         std::tuple_element_t< space_idx, decltype(m_spaces) > > >) {
            // if the space's value_type is not an xarray, but the batch_type is, then the batch
            // xarray container is merely a holder like a std::vector would. We can thus disregard
            // any nested xarray shapes we would preserve were we to sample according to e.g. the
            // 0th axis. If we didn't do this we would get errors of impossible conversions from
            // xt::strided_view to whatever the space's value type is, since axis_begin iterators
            // would wrap contained values in a view type of shape {1}.
            for(auto&& entry : FWD(samples)) {
               _append_sample< space_idx >(result, FWD(entry));
            }
         } else {
            for(auto entry_iter = xt::axis_begin(samples, 0);
                entry_iter != xt::axis_end(samples, 0);
                entry_iter++) {
               _append_sample< space_idx >(result, *entry_iter);
            }
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
