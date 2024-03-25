#ifndef REINFORCE_MONO_SPACE_HPP
#define REINFORCE_MONO_SPACE_HPP

#include <fmt/core.h>

#include <concepts>
#include <cstddef>
#include <optional>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <xtensor/xarray.hpp>

#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

#define MASKED_SAMPLE_FUNC(type1, arg1, type2, arg2)                \
   multi_value_type sample(type1 arg1, type2 arg2) const            \
      requires requires(Derived self) { self._sample(arg1, arg2); } \
   {                                                                \
      return self()._sample(arg1, arg2);                            \
   }

namespace force {

namespace detail {
template < typename T >
concept has_getitem_operator = requires(T t, size_t idx) { t[idx]; };
}  // namespace detail

/// \brief The generic Space base class
///
/// The specifics are made to work as closely as possible to the corresponding internals of the
/// openai/gymnasium python class.
template < typename T, typename Derived, typename MultiT = T >
   requires(std::is_same_v< T, MultiT > or detail::has_getitem_operator< MultiT >)
class Space: public detail::rng_mixin {
  public:
   // the type of values returned by sampling or containment queries
   using value_type = T;
   // the type of values returned by multiple-sampling (i.e. sample-size > 1) queries and
   // containment queries for multiple elements at once
   using multi_value_type = MultiT;

   constexpr static bool mvt_is_container = not std::is_same_v< multi_value_type, value_type >
                                            and detail::has_getitem_operator< MultiT >;

   explicit Space(xt::svector< int > shape = {}, std::optional< size_t > seed = std::nullopt)
       : rng_mixin(seed), m_shape(std::move(shape))
   {
   }

   // Randomly sample an element of this space
   value_type sample() const
   {
      if constexpr(requires(Derived self) { self._sample(); }) {
         return self()._sample();
      } else if constexpr(mvt_is_container) {
         return sample(1)[0];
      } else {
         throw detail::not_implemented_error(fmt::format("_sample()"));
      }
   }

   template < typename U >
   value_type sample(const xarray< U >& mask) const
   {
      if constexpr(requires(Derived self) { self._sample(mask); }) {
         return self()._sample(mask);
      } else if constexpr(mvt_is_container) {
         return sample(1, mask)[0];
      } else {
         throw detail::not_implemented_error(
            fmt::format("_sample({})", detail::type_name< decltype(mask) >())
         );
      }
   }

   template < typename U >
   value_type sample(const std::vector< std::optional< xarray< U > > >& mask_vec) const
   {
      if constexpr(requires(Derived self) { self._sample(mask_vec); }) {
         return self()._sample(mask_vec);
      } else if constexpr(mvt_is_container) {
         return sample(1, mask_vec)[0];
      } else {
         throw detail::not_implemented_error(
            fmt::format("_sample({})", detail::type_name< decltype(mask_vec) >())
         );
      }
   }

   template < typename... Args >
   value_type sample(const std::tuple< Args... >& mask_tuple) const
   {
      if constexpr(requires(Derived self) { self._sample(mask_tuple); }) {
         return self()._sample(mask_tuple);
      } else if constexpr(mvt_is_container) {
         return sample(1, mask_tuple)[0];
      } else {
         throw detail::not_implemented_error(
            fmt::format("_sample({})", detail::type_name< decltype(mask_tuple) >())
         );
      }
   }

   multi_value_type sample(size_t nr) const
      requires requires(Derived self) { self._sample(nr); }
   {
      return self()._sample(nr);
   }

   template < typename U >
   MASKED_SAMPLE_FUNC(size_t, nr, const xarray< U >&, mask);

   template < typename... Args >
   MASKED_SAMPLE_FUNC(size_t, nr, const std::tuple< Args... >&, mask_tuple);

   template < typename U >
   MASKED_SAMPLE_FUNC(size_t, nr, const std::vector< std::optional< xarray< U > > >&, mask_vec);

   // Check if the value is a valid member of this space
   bool contains(const value_type& value) const { return self()._contains(value); }

   // Checks whether this space can be flattened to a Box
   [[nodiscard]] bool is_flattenable() const { return false; }

   bool operator==(const Space& rhs) const = default;

   // Return the shape of the space
   auto& shape() const { return m_shape; }

  protected:
   // mutable shape reference of the space
   auto& shape() { return m_shape; }

   /// tags for boundary inclusion or exclusion
   struct InclusiveTag {};
   struct ExclusiveTag {};
   /// @brief check whether an array of values aligns in terms of shape and is within the given
   /// (incl/excl) boundaries.
   ///
   /// @tparam DType the data type of the incoming values. Determines whether the check will be made
   /// on floats or ints.
   /// @tparam Rng1 range type over lower boundary values
   /// @tparam Rng2 range type over high boundary values
   /// @param values the actual value array to check containment for
   /// @param low_boundary range over lower boundaries. Has to align with the shape of the space.
   /// @param high_boundary range over lower boundaries. Has to align with the shape of the space.
   /// @return bool, whether all values are within the boundaries and the shape of the array aligns
   /// with the shape of the space.
   template <
      typename DType,
      std::ranges::range Rng1 = std::initializer_list< DType >,
      std::ranges::range Rng2 = std::initializer_list< DType >,
      typename BoundaryTag = InclusiveTag >
      requires(std::floating_point< DType > or std::integral< DType >)
   bool _isin_shape_and_bounds(
      const xarray< DType >& values,
      const Rng1& low_boundary,
      const Rng2& high_boundary,
      BoundaryTag /*boundary_tag*/ = {}
   ) const;

  private:
   xt::svector< int > m_shape;

   constexpr const auto& self() const { return static_cast< const Derived& >(*this); }
   constexpr auto& self() { return static_cast< Derived& >(*this); }
};

template < typename T, typename Derived, typename MultiT >
   requires(std::is_same_v< T, MultiT > or detail::has_getitem_operator< MultiT >)
template < typename DType, std::ranges::range Rng1, std::ranges::range Rng2, typename BoundaryTag >
   requires(std::floating_point< DType > or std::integral< DType >)
bool Space< T, Derived, MultiT >::_isin_shape_and_bounds(
   const xarray< DType >& values,
   const Rng1& low_boundary,
   const Rng2& high_boundary,
   BoundaryTag
) const
{
   if constexpr(detail::is_none_v< BoundaryTag, InclusiveTag, ExclusiveTag >) {
      static_assert(
         detail::is_any_v< std::tuple_element_t< 0, BoundaryTag >, InclusiveTag, ExclusiveTag >
            and detail::
               is_any_v< std::tuple_element_t< 1, BoundaryTag >, InclusiveTag, ExclusiveTag >,
         "Boundary Tag has to be either ExlcusiveTag, InclusiveTag, or a 2-arity pair-like of "
         "these."
      );
   }
   const auto& incoming_shape = values.shape();
   const auto& incoming_dim = incoming_shape.size();
   const auto space_dim = shape().size();

   if(incoming_dim < space_dim or space_dim + 1 < incoming_dim) {
      return false;
   }

   auto enum_bounds_view = ranges::views::enumerate(ranges::views::zip(low_boundary, high_boundary)
   );
   if(incoming_dim == space_dim + 1) {
      // zip to cut off the last entry (batch dim) in incoming_shape
      if(not ranges::all_of(ranges::views::zip(shape(), incoming_shape), [](auto pair) {
            return std::cmp_equal(std::get< 0 >(pair), std::get< 1 >(pair));
         })) {
         return false;
      }
      return ranges::any_of(enum_bounds_view, [&](const auto& idx_low_high) {
         const auto& [i, low_high] = idx_low_high;
         const auto& [low, high] = low_high;
         auto coordinates = xt::unravel_index(static_cast< int >(i), shape());
         const auto& vals = xt::strided_view(
            values, std::invoke([&] {
               xt::xstrided_slice_vector slice(coordinates.begin(), coordinates.end());
               slice.emplace_back(xt::all());
               return slice;
            })
         );
         constexpr auto compare =
            []< typename Comp, typename CompEq >(Comp cmp, CompEq cmp_eq, auto&&... args) {
               if constexpr(std::same_as< BoundaryTag, InclusiveTag >) {
                  return cmp_eq(FWD(args)...);
               } else if constexpr(std::same_as< BoundaryTag, ExclusiveTag >) {
                  return cmp(FWD(args)...);
               } else if constexpr(std::same_as<
                                      std::tuple_element_t< 0, BoundaryTag >,
                                      InclusiveTag >) {
                  return cmp_eq(FWD(args)...);
               } else {
                  return cmp(FWD(args)...);
               }
            };
         return xt::all(
            compare(AS_LAMBDA(xt::greater), AS_LAMBDA(xt::greater_equal), vals, low)
            and compare(AS_LAMBDA(xt::less), AS_LAMBDA(xt::less_equal), vals, high)
         );
      });
   }
   // we now know that shape().size() == incoming_shape.size()
   if(not ranges::equal(shape(), incoming_shape)) {
      return false;
   }
   return ranges::any_of(enum_bounds_view, [&](const auto& idx_low_high) {
      const auto& [i, low_high] = idx_low_high;
      const auto& [low, high] = low_high;
      auto coordinates = xt::unravel_index(static_cast< int >(i), shape());
      const auto& val = values.element(coordinates.begin(), coordinates.end());
      constexpr auto compare = []< typename Comp, typename CompEq >(
                                  Comp cmp, CompEq cmp_eq, auto&&... args
                               ) {
         if constexpr(std::same_as< BoundaryTag, InclusiveTag >) {
            return cmp_eq(FWD(args)...);
         } else if constexpr(std::same_as< BoundaryTag, ExclusiveTag >) {
            return cmp(FWD(args)...);
         } else if constexpr(std::same_as< std::tuple_element_t< 0, BoundaryTag >, InclusiveTag >) {
            return cmp_eq(FWD(args)...);
         } else {
            return cmp(FWD(args)...);
         }
      };
      return compare(std::greater{}, std::greater_equal{}, val, low)
             and compare(std::less{}, std::less_equal{}, val, high);
      return low <= val and val <= high;
   });
}

}  // namespace force

#undef MASKED_SAMPLE_FUNC

#endif  // REINFORCE_MONO_SPACE_HPP
