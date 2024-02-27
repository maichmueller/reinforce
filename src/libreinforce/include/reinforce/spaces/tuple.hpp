
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
class TypedTupleSpace:
    public TypedSpace<
       std::tuple< typename Spaces::value_type... >,
       TypedTupleSpace< Spaces... >,
       std::tuple< typename Spaces::multi_value_type... > > {
  public:
   friend class TypedSpace<
      std::tuple< typename Spaces::value_type... >,
      TypedTupleSpace,
      std::tuple< typename Spaces::multi_value_type... > >;
   using base = TypedSpace<
      std::tuple< typename Spaces::value_type... >,
      TypedTupleSpace,
      std::tuple< typename Spaces::multi_value_type... > >;
   using typename base::value_type;
   using typename base::multi_value_type;
   using base::shape;
   using base::rng;

   using spaces_idx_seq = std::index_sequence_for< Spaces... >;

  private:
   std::tuple< Spaces... > m_spaces;

  public:
   template < std::integral T >
   explicit TypedTupleSpace(T seed_, Spaces... spaces) : m_spaces{std::move(spaces)...}
   {
      SPDLOG_DEBUG("Called tuple space constructor with seed");
      seed(seed_);
   }
   explicit TypedTupleSpace(Spaces... spaces) : m_spaces{std::move(spaces)...} {}

   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
   multi_value_type sample(size_t nr_samples, MaskTuple&& mask_tuple)
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{
               std::get< Is >(m_spaces).sample(nr_samples, std::get< Is >(FWD(mask_tuple)))...
            };
         },
         spaces_idx_seq{}
      );
   }

   template < typename... MaskTs >
   multi_value_type sample(size_t nr_samples, MaskTs&&... masks)
   {
      return sample(nr_samples, std::tuple{FWD(masks)...});
   }

   multi_value_type sample(size_t nr_samples)
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{std::get< Is >(m_spaces).sample(nr_samples)...};
         },
         spaces_idx_seq{}
      );
   }
   template < typename MaskTuple >
      requires detail::is_specialization_v< detail::raw_t< MaskTuple >, std::tuple >
   value_type sample(MaskTuple&& mask_tuple)
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{std::get< Is >(m_spaces).sample(std::get< Is >(FWD(mask_tuple)))...};
         },
         spaces_idx_seq{}
      );
   }

   template < typename FirstMaskT, typename... MaskTs >
      requires(not std::is_integral_v< detail::raw_t< FirstMaskT > >)
   value_type sample(FirstMaskT&& mask1, MaskTs&&... tail_masks)
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

   value_type sample()
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{std::get< Is >(m_spaces).sample()...};
         },
         spaces_idx_seq{}
      );
   }

   [[nodiscard]] bool contains(const value_type& value) const
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return (std::get< Is >(m_spaces).contains(std::get< Is >(value)) && ...);
         },
         spaces_idx_seq{}
      );
   }

   void seed(size_t value)
   {
      std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            (std::get< Is >(m_spaces).seed(value), ...);
         },
         spaces_idx_seq{}
      );
   }

   bool operator==(const TypedTupleSpace& other) const
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return (std::get< Is >(m_spaces).operator==(std::get< Is >(other.m_spaces)) && ...);
         },
         spaces_idx_seq{}
      );
   }

   std::string repr()
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
};

}  // namespace force

#endif  // REINFORCE_SPACE_TUPLE_HPP
