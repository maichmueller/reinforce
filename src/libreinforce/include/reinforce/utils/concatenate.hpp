
#ifndef REINFORCE_CONCATENATE_HPP
#define REINFORCE_CONCATENATE_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xstorage.hpp>
#include <xtensor/xview.hpp>

#include "macro.hpp"
#include "reinforce/fwd.hpp"
#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/type_traits.hpp"

namespace force {

namespace detail {
template < typename MaybeSpaceT >
concept is_space_like = requires(MaybeSpaceT space_candidate) {
   { space_candidate.shape() } -> std::convertible_to< xt::svector< int > >;
   requires detail::has_value_type< MaybeSpaceT >;
   requires detail::has_batch_value_type< MaybeSpaceT >;
   requires detail::has_data_type< MaybeSpaceT >;
};
}  // namespace detail

template < typename T >
struct concatenate;

/// @brief generic specialization for undefined space types (similar to std::plus, std::minus, etc.)
///
/// will redirect to the respective implementation based on the provided space type.
template <>
struct concatenate< void > {
   template < typename Space, typename ValueRange >
   auto operator()(const Space& space, ValueRange&& items) const
   {
      return concatenate< Space >{}(space, FWD(items));
   }

   template < typename Space, typename ValueRange >
   auto& operator()(const Space& space, ValueRange&& items, auto& out) const
   {
      return concatenate< Space >{}(space, FWD(items), out);
   }
};

/// @brief default implementation for concatenating values in a space to a vector over value_type
template < typename Space >
   requires detail::is_space_like< Space >
struct concatenate< Space > {
   static_assert(detail::is_space_like< Space >, "Space must be a space-like type");

   using value_type = detail::value_t< Space >;
   using batch_value_type = detail::batch_value_t< Space >;
   using data_type = detail::data_t< Space >;
   using result_type = std::vector< value_type >;

   template < typename ValueRange >
   result_type operator()(const Space& /*space*/, ValueRange&& items) const
   {
      return ranges::to< result_type >(FWD(items));
   }

   template < typename ValueRange, typename OutT >
      requires detail::is_specialization_v< detail::raw_t< OutT >, std::vector >
   auto& operator()(const Space& /*space*/, ValueRange&& items, OutT& out) const
   {
      return ranges::move(out, std::ranges::begin(out), FWD(items));
   }
};

/// @brief specialization for spaces over xarrays.
///
/// concatenates range of xarrays to a single xarray of shape (n, ...)
template < typename Space >
   requires(detail::is_space_like< Space > and (detail::is_xarray< detail::value_t< Space > > or std::integral< detail::data_t< Space > > or std::floating_point< detail::data_t< Space > >) and detail::is_xarray< detail::batch_value_t< Space > >)
struct concatenate< Space > {
   using space_type = Space;
   using value_type = detail::value_t< Space >;
   using batch_value_type = detail::batch_value_t< Space >;
   using result_type = batch_value_type;
   using data_type = detail::data_t< Space >;

   template < std::ranges::sized_range ValueRange >
      requires std::ranges::forward_range< detail::raw_t< ValueRange > >
               and std::convertible_to<
                  std::ranges::range_reference_t< detail::raw_t< ValueRange > >,
                  value_type >
   batch_value_type operator()(const space_type& space, ValueRange&& items) const
   {
      batch_value_type out = xt::empty< data_type >(prepend(space.shape(), items.size()));
      stack(space, FWD(items), out);
      return out;
   }
   template < std::ranges::sized_range ValueRange >
      requires std::convertible_to< std::ranges::range_reference_t< ValueRange >, value_type >
   batch_value_type& operator()(const space_type& space, ValueRange&& items, batch_value_type& out)
      const
   {
      stack(space, FWD(items), out);
      return out;
   }

  private:
   template < typename ValueRange >
   void stack([[maybe_unused]] const space_type& space, ValueRange&& items, batch_value_type& out)
      const
   {
      size_t idx = 0;
      for(auto&& item : FWD(items)) {
         using actual_value_type = detail::raw_t< decltype(item) >;
         if constexpr(std::integral< actual_value_type >
                      or std::floating_point< actual_value_type >) {
            FORCE_DEBUG_ASSERT(xt::broadcastable(space.shape(), std::array{1}));
            xt::view(out, idx) = FWD(item);
         } else {
            if constexpr(detail::is_xarray< actual_value_type >) {
               FORCE_DEBUG_ASSERT(xt::broadcastable(space.shape(), item.shape()));
            }
            xt::strided_view(out, xt::xstrided_slice_vector{idx, xt::ellipsis()}) = FWD(item);
         }
         ++idx;
      }
   }
};

/// @brief Specialization for TupleSpace.
///
/// Concatenates range of tuples to a single tuple with individually
/// concatenated value types as concatenation goes for each subspace
template < typename Space >
   requires(detail::is_space_like< Space >
            and requires(Space space) {
                   Space::is_composite_space;  // has this member
                   requires Space::is_composite_space;  // member must be true
                   typename Space::spaces_tuple_type;  // has this member
                }
            and std::same_as< Space, TupleSpace< typename Space::spaces_tuple_type > >
   )
struct concatenate< Space > {
   using space_type = Space;
   using value_type = detail::value_t< Space >;
   using batch_value_type = detail::batch_value_t< Space >;
   using data_type = detail::data_t< Space >;
   using result_type = batch_value_type;

   static constexpr auto spaces_idx_seq = std::make_index_sequence<
      std::tuple_size_v< typename space_type::spaces_tuple_type > >{};

   template < std::ranges::sized_range ValueRange >
      requires std::ranges::forward_range< detail::raw_t< ValueRange > >
               and std::convertible_to<
                  std::ranges::range_reference_t< detail::raw_t< ValueRange > >,
                  value_type >
   batch_value_type operator()(const space_type& space, ValueRange&& items) const
   {
      constexpr auto concat = concatenate{};
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{
               concat(std::get< Is >(space.m_spaces), std::views::elements< Is >(items))...
            };
         },
         spaces_idx_seq
      );
   }
   template < std::ranges::sized_range ValueRange >
      requires std::convertible_to< std::ranges::range_reference_t< ValueRange >, value_type >
   batch_value_type& operator()(const space_type& space, ValueRange&& items, batch_value_type& out)
      const
   {
      constexpr auto concat = concatenate{};
      return std::invoke(
         [&]< size_t... Is >(std::index_sequence< Is... >) {
            return std::tuple{concat(
               std::get< Is >(space.m_spaces),
               std::views::elements< Is >(items),
               std::get< Is >(out)
            )...};
         },
         spaces_idx_seq
      );
   }
};

}  // namespace force
#endif  // REINFORCE_CONCATENATE_HPP
