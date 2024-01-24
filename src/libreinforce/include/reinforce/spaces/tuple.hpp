
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
   explicit TypedTupleSpace(Spaces... spaces, std::optional< size_t > seed_ = std::nullopt)
       : m_spaces{spaces...}
   {
      if(seed_.has_value()) {
         seed(*seed_);
      }
   }

   template < typename MaskOptionalTuple >
      requires detail::is_specialization_v< MaskOptionalTuple, std::tuple >
   auto sample(size_t nr_samples, const MaskOptionalTuple& mask_opt)
   {
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{
               std::get< Is >(m_spaces).sample(nr_samples, std::get< Is >(mask_opt))...
            };
         },
         spaces_idx_seq{}
      );
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
