
#ifndef REINFORCE_SPACE_TUPLE_HPP
#define REINFORCE_SPACE_TUPLE_HPP

#include <concepts>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "reinforce/utils/type_traits.hpp"

namespace force {

template < typename... Spaces >
class TupleSpace:
    public Space<
       std::tuple< typename Spaces::value_type... >,
       TupleSpace< Spaces... >,
       std::tuple< typename Spaces::batch_value_type... > > {
  public:
   friend class Space<
      std::tuple< typename Spaces::value_type... >,
      TupleSpace,
      std::tuple< typename Spaces::batch_value_type... > >;
   using base = Space<
      std::tuple< typename Spaces::value_type... >,
      TupleSpace,
      std::tuple< typename Spaces::batch_value_type... > >;
   using data_type = std::tuple< typename Spaces::data_type... >;
   using spaces_tuple_type = std::tuple< Spaces... >;
   using typename base::value_type;
   using typename base::batch_value_type;
   using base::seed;
   using base::shape;
   using base::rng;

   using spaces_idx_seq = std::index_sequence_for< Spaces... >;

   static constexpr bool _is_composite_space = true;

  private:
   spaces_tuple_type m_spaces;

  public:
   template < std::integral T = size_t >
   explicit TupleSpace(T seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      SPDLOG_DEBUG("Called tuple space constructor with seed");
      seed(seed_);
   }

   template < typename OptionalT = std::optional< size_t > >
      requires detail::is_specialization_v< OptionalT, std::optional >
               and std::convertible_to< detail::value_t< OptionalT >, size_t >
   explicit TupleSpace(OptionalT seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      SPDLOG_DEBUG("Called tuple space constructor with optional-seed");
      seed(seed_);
   }

   explicit TupleSpace(Spaces... spaces) : m_spaces{std::move(spaces)...}
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

   bool operator==(const TupleSpace& other) const
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
         "Tuple({})",
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
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{
               std::get< Is >(m_spaces).sample(batch_size, std::get< Is >(FWD(mask_tuple)))...
            };
         },
         spaces_idx_seq{}
      );
   }

   template < typename... MaskTs >
      requires(sizeof...(MaskTs) == sizeof...(Spaces))
   [[nodiscard]] batch_value_type _sample(size_t batch_size, MaskTs&&... masks) const
   {
      return sample(batch_size, std::tuple{FWD(masks)...});
   }

   [[nodiscard]] batch_value_type _sample(size_t batch_size) const
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{std::get< Is >(m_spaces).sample(batch_size)...};
         },
         spaces_idx_seq{}
      );
   }

   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
               and (std::tuple_size_v< detail::raw_t< MaskTuple > > == sizeof...(Spaces))
   [[nodiscard]] value_type _sample(MaskTuple&& mask_tuple) const
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{std::get< Is >(m_spaces).sample(std::get< Is >(FWD(mask_tuple)))...};
         },
         spaces_idx_seq{}
      );
   }

   template < typename FirstMaskT, typename... MaskTs >
      requires(
         not std::is_integral_v< detail::raw_t< FirstMaskT > >
         and (sizeof...(MaskTs) == sizeof...(Spaces) - 1)
      )
   [[nodiscard]] value_type _sample(FirstMaskT&& mask1, MaskTs&&... tail_masks) const
   {
      return std::invoke(
         [&]<
            size_t... IsSpaces,
            size_t... IsMasks >(std::index_sequence< IsSpaces... >, std::index_sequence< IsMasks... >) {
            return std::tuple{std::get< IsSpaces >(m_spaces).sample(
               FWD(mask1), std::get< IsMasks >(FWD(tail_masks))
            )...};
         },
         std::index_sequence_for< Spaces... >{},
         std::index_sequence_for< MaskTs... >{}
      );
   }

   [[nodiscard]] value_type _sample(std::nullopt_t = std::nullopt) const
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{std::get< Is >(m_spaces).sample()...};
         },
         spaces_idx_seq{}
      );
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return (std::get< Is >(m_spaces).contains(std::get< Is >(value)) && ...);
         },
         spaces_idx_seq{}
      );
   }
};

}  // namespace force

namespace std {

template < typename... Spaces >
struct tuple_size< ::force::TupleSpace< Spaces... > >:
    integral_constant< size_t, sizeof...(Spaces) > {};

template < size_t idx, typename... Spaces >
struct tuple_element< idx, ::force::TupleSpace< Spaces... > > {
   using type = std::
      tuple_element_t< idx, typename ::force::TupleSpace< Spaces... >::spaces_tuple_type >;
};

}  // namespace std

#endif  // REINFORCE_SPACE_TUPLE_HPP
