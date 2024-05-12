
#ifndef REINFORCE_CONCATENATE_HPP
#define REINFORCE_CONCATENATE_HPP

#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "reinforce/fwd.hpp"
#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/type_traits.hpp"

namespace force {

template < typename Space >
struct concatenate {
   template < typename ValueRange >
   auto operator()(const Space& space, ValueRange&& items) const
   {
      return ranges::to< std::vector< detail::batch_value_t< Space > > >(FWD(items));
   }

   template < typename ValueRange, typename OutT >
      requires detail::is_specialization_v< detail::raw_t< OutT >, std::vector >
   auto& operator()(const Space& space, ValueRange&& items, OutT& out) const
   {
      return ranges::move(out, std::ranges::begin(out), (FWD(items)));
   }
};

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

template < typename SpaceT >
   requires(detail::is_space< SpaceT >
            and detail::is_any_v< //
               SpaceT, //
               DiscreteSpace< detail::data_t< SpaceT > >,
               BoxSpace< detail::data_t< SpaceT > >,
               MultiDiscreteSpace< detail::data_t< SpaceT > >,
               MultiBinarySpace >
   )
struct concatenate< SpaceT > {
   using space_type = SpaceT;
   using value_type = detail::value_t< SpaceT >;
   using batch_value_type = detail::batch_value_t< SpaceT >;
   using data_type = detail::data_t< SpaceT >;

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

template < typename SpaceT >
   requires(detail::is_space< SpaceT >
            and SpaceT::is_composite_space
            and requires(SpaceT space) { typename SpaceT::spaces_tuple_type; }
            and std::same_as< //
               SpaceT, //
               TupleSpace< typename SpaceT::spaces_tuple_type > >
   )
struct concatenate< SpaceT > {
   using space_type = SpaceT;
   using value_type = detail::value_t< SpaceT >;
   using batch_value_type = detail::batch_value_t< SpaceT >;
   using data_type = detail::data_t< SpaceT >;

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

template < typename SpaceT >
   requires(detail::is_space< SpaceT >
            and detail::is_any_v< //
               SpaceT, //
               TupleSpace< detail::data_t< SpaceT > > >
   )
struct concatenate< SpaceT > {
   using space_type = SpaceT;
   using value_type = detail::value_t< SpaceT >;
   using batch_value_type = detail::batch_value_t< SpaceT >;
   using data_type = detail::data_t< SpaceT >;

   template < std::ranges::sized_range ValueRange >
      requires std::ranges::forward_range< detail::raw_t< ValueRange > >
               and std::convertible_to<
                  std::ranges::range_reference_t< detail::raw_t< ValueRange > >,
                  value_type >
   batch_value_type operator()(const space_type& space, ValueRange&& items)
   {
      value_type out = xt::empty< data_type >(prepend({items.size()}, space.shape()));
      stack(space, FWD(items), out);
      return out;
   }
   template < std::ranges::sized_range ValueRange >
      requires std::convertible_to< std::ranges::range_reference_t< ValueRange >, value_type >
   batch_value_type& operator()(const space_type& space, ValueRange&& items, batch_value_type& out)
   {
      stack(space, FWD(items), out);
      return out;
   }

  private:
   template < std::ranges::sized_range ValueRange >
   batch_value_type& stack(const space_type& space, ValueRange&& items, batch_value_type& out)
   {
      size_t idx = 0;
      for(auto&& item : FWD(items)) {
         xt::view(out, idx, xt::ellipsis()) = FWD(item);
         ++idx;
      }
   }
};

}  // namespace force
#endif  // REINFORCE_CONCATENATE_HPP
