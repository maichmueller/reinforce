
#ifndef REINFORCE_CONCATENATE_HPP
#define REINFORCE_CONCATENATE_HPP

#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "reinforce/fwd.hpp"
#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/type_traits.hpp"

namespace force {

template < typename T >
struct concatenate;

/// generic specialization for undefined space types (similar to std::plus, std::minus, etc.)
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

/// default implementation for concatenating values in a space to a vector over value_type
template < typename Space >
   requires detail::is_space< Space >
struct concatenate< Space > {
   static_assert(detail::is_space< Space >, "Space must be a space type");

   using value_type = detail::value_t< Space >;
   using batch_value_type = detail::batch_value_t< Space >;
   using data_type = detail::data_t< Space >;
   using result_type = std::vector< value_type >;

   template < typename ValueRange >
   result_type operator()(const Space& space, ValueRange&& items) const
   {
      return ranges::to< result_type >(FWD(items));
   }

   template < typename ValueRange, typename OutT >
      requires detail::is_specialization_v< detail::raw_t< OutT >, std::vector >
   auto& operator()(const Space& space, ValueRange&& items, OutT& out) const
   {
      return ranges::move(out, std::ranges::begin(out), (FWD(items)));
   }
};

/// specialization for spaces over xarrays
/// concatenates range of xarrays to a single xarray of shape (n, ...)
template < typename Space >
   requires(detail::is_space< Space >
            and detail::is_any_v< //
               Space, //
               DiscreteSpace< detail::data_t< Space > >,
               BoxSpace< detail::data_t< Space > >,
               MultiDiscreteSpace< detail::data_t< Space > >,
               MultiBinarySpace >
   )
struct concatenate< Space > {
   using space_type = Space;
   using value_type = detail::value_t< Space >;
   using batch_value_type = detail::batch_value_t< Space >;
   using data_type = detail::data_t< Space >;

   template < std::ranges::sized_range ValueRange >
      requires std::ranges::forward_range< detail::raw_t< ValueRange > >
               and std::convertible_to<
                  std::ranges::range_reference_t< detail::raw_t< ValueRange > >,
                  value_type >
   batch_value_type operator()(const space_type& space, ValueRange&& items)
   {
      value_type out = xt::empty< data_type >(prepend({items.size()}, space.shape()));
      stack(FWD(items), out);
      return out;
   }
   template < std::ranges::sized_range ValueRange >
      requires std::convertible_to< std::ranges::range_reference_t< ValueRange >, value_type >
   batch_value_type& operator()(const space_type& space, ValueRange&& items, batch_value_type& out)
   {
      stack(FWD(items), out);
      return out;
   }

  private:
   template < std::ranges::sized_range ValueRange >
   batch_value_type& stack(ValueRange&& items, batch_value_type& out)
   {
      size_t idx = 0;
      for(auto&& item : FWD(items)) {
         xt::view(out, idx, xt::ellipsis()) = FWD(item);
         ++idx;
      }
   }
};

/// specialization for TupleSpace
/// concatenates range of tuples to a single tuple with individually
/// concatenated value types as concatenation goes for each subspace
template < typename Space >
   requires(detail::is_space< Space >
            and Space::is_composite_space
            and requires(Space space) { typename Space::spaces_tuple_type; }
            and std::same_as< //
               Space, //
               TupleSpace< typename Space::spaces_tuple_type > >
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
   batch_value_type operator()(const space_type& space, ValueRange&& items)
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
