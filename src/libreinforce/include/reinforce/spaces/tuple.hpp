
#ifndef REINFORCE_SPACE_TUPLE_HPP
#define REINFORCE_SPACE_TUPLE_HPP

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
class TypedTupleSpace {
  public:
   using spaces_tuple = std::tuple< Spaces... >;
   using spaces_idx_seq = std::index_sequence_for< Spaces... >;

  private:
   spaces_tuple m_spaces;

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
   auto sample(size_t nr_samples, MaskTuple&& mask_tuple)
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{
               std::get< Is >(m_spaces).sample(nr_samples, std::get< Is >(mask_tuple))...
            };
         },
         spaces_idx_seq{}
      );
   }
   
   template < typename... MaskTs >
   auto sample(size_t nr_samples, MaskTs&&... masks)
   {
      return sample(nr_samples, std::tuple{FWD(masks)...});
   }

   auto sample(size_t nr_samples)
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
   auto sample(MaskTuple&& masks)
   {
      return sample(1, FWD(masks));
   }

   // Uncommenting these lines will cause a clang-16 compiler segfault crash
   template < typename FirstMaskT, typename... MaskTs >
      requires(not std::is_integral_v< detail::raw_t< FirstMaskT > >)
   auto sample(FirstMaskT&& mask1, MaskTs&&... tail_masks)
   {
      return sample(1, FWD(mask1), FWD(tail_masks)...);
   }

   auto sample() { return sample(1); }

   template < typename ValueTuple >
   bool contains(const ValueTuple& value) const
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
      // std::invoke(
      [&]< size_t... Is >(std::index_sequence< Is... >) {
         (std::get< Is >(m_spaces).seed(value), ...);
      }(spaces_idx_seq{});
      // );
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
};

}  // namespace force

#endif  // REINFORCE_SPACE_TUPLE_HPP
