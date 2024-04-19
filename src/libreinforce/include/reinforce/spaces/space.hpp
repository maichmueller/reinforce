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

#include "reinforce/utils/exceptions.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

namespace detail {
template < typename T >
concept has_getitem_operator = requires(T t, size_t idx) { t[idx]; };
}  // namespace detail

/// \brief The generic Space base class
///
/// The specifics are made to work as closely as possible to the corresponding internals of the
/// openai/gymnasium python class.
template < typename T, typename Derived, typename MultiT = T, bool runtime_sample_throw = false >
   requires(std::is_same_v< T, MultiT > or detail::has_getitem_operator< MultiT >)
class Space: public detail::rng_mixin {
  private:
   /// for tag dispatch within this class
   struct internal_tag_t {};
   static constexpr internal_tag_t internal_tag{};

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

   template < typename MaskType = std::nullopt_t, typename... OtherArgs >
   value_type sample(internal_tag_t, MaskType mask_arg = std::nullopt, OtherArgs&... args) const
   {
      if constexpr(requires(Derived derived) { derived._sample(mask_arg, FWD(args)...); }) {
         // derived has the necessary sample function, so call it
         return derived()._sample(mask_arg, FWD(args)...);
      } else if constexpr(mvt_is_container and requires(Derived derived) {
                             derived._sample(1, mask_arg, FWD(args)...);
                          }) {
         // derived does not have a single sized sample function, but a multi-value-type sample
         // function that returns an indexable container. We trust that element 0 is simply an
         // element of value_type that corresponds with the first (and only) sample drawn.
         return sample(1, mask_arg, FWD(args)...)[0];
      } else {
         // neither options apply so we now decide between throwing a runtime exception (for dynamic
         // language support) or letting the call overload resolution fail at compile time.
         if constexpr(runtime_sample_throw) {
            if constexpr(sizeof...(OtherArgs) > 0) {
               throw not_implemented_error(
                  fmt::format("_sample({}, ...)", detail::type_name< MaskType >())
               );
            } else {
               throw not_implemented_error(
                  fmt::format("_sample({})", detail::type_name< MaskType >())
               );
            }
         } else {
            // trigger a compilation error
            return derived()._sample(mask_arg, FWD(args)...);
         }
      }
   }

   // Randomly sample an element of this space
   value_type sample() const { return sample(internal_tag); }

   template < typename... OtherArgs >
   value_type sample(std::nullopt_t, OtherArgs&&... extra_args) const
   {
      return sample(internal_tag, std::nullopt, FWD(extra_args)...);
   }

   template < typename U, typename... OtherArgs >
   value_type sample(std::optional< U > mask, OtherArgs&&... extra_args) const
   {
      if(mask.has_value()) {
         return sample(internal_tag, *mask, FWD(extra_args)...);
      }
      return sample(internal_tag, std::nullopt, FWD(extra_args)...);
   }

   template < typename U, typename... OtherArgs >
   value_type sample(const xarray< U >& mask, OtherArgs&&... extra_args) const
   {
      return sample(internal_tag, mask, FWD(extra_args)...);
   }

   template < typename U, typename... OtherArgs >
   value_type sample(
      const std::vector< std::optional< xarray< U > > >& mask_vec,
      OtherArgs&&... extra_args
   ) const
   {
      return sample(internal_tag, mask_vec, FWD(extra_args)...);
   }

   template < typename... Args, typename... OtherArgs >
   value_type sample(const std::tuple< Args... >& mask_tuple, OtherArgs&&... extra_args) const
   {
      return sample(internal_tag, mask_tuple, FWD(extra_args)...);
   }

   multi_value_type sample(size_t nr) const
      requires requires(Derived derived) { derived._sample(nr); }
   {
      return derived()._sample(nr);
   }

   template < std::integral T1, typename MaskType, typename... OtherArgs >
   multi_value_type sample(internal_tag_t, T1 arg1, MaskType&& mask_arg, OtherArgs&&... args) const
   {
      if constexpr(not requires(Derived derived) {
                      derived._sample(arg1, FWD(mask_arg), FWD(args)...);
                   }) {
         if constexpr(runtime_sample_throw) {
            if constexpr(sizeof...(OtherArgs) > 0) {
               throw not_implemented_error(fmt::format(
                  "_sample({}, {}, ...)", detail::type_name< T1 >(), detail::type_name< MaskType >()
               ));
            } else {
               throw not_implemented_error(fmt::format(
                  "_sample({}, {})", detail::type_name< T1 >(), detail::type_name< MaskType >()
               ));
            }
         } else {
            derived()._sample(arg1, FWD(mask_arg), FWD(args)...);
         }
      } else {
         return derived()._sample(arg1, FWD(mask_arg), FWD(args)...);
      }
   }

   template < typename U, typename... ExtraArgs >
   multi_value_type sample(size_t nr, const xarray< U >& mask, ExtraArgs&&... extra_args) const
   {
      return sample(internal_tag, nr, mask, FWD(extra_args)...);
   }
   template < typename... TupleArgs, typename... ExtraArgs >
   multi_value_type
   sample(size_t nr, const std::tuple< TupleArgs... >& mask_tuple, ExtraArgs&&... extra_args) const
   {
      return sample(internal_tag, nr, mask_tuple, FWD(extra_args)...);
   }
   template < typename U, typename... ExtraArgs >
   multi_value_type sample(
      size_t nr,
      const std::vector< std::optional< xarray< U > > >& mask_vec,
      ExtraArgs&&... extra_args
   ) const
   {
      return sample(internal_tag, nr, mask_vec, FWD(extra_args)...);
   }

   template < typename... ExtraArgs >
   multi_value_type sample(size_t nr, std::nullopt_t, ExtraArgs&&... extra_args) const
   {
      return sample(internal_tag, nr, std::nullopt, FWD(extra_args)...);
   }

   template < typename U, typename... ExtraArgs >
   value_type sample(size_t nr, std::optional< U > mask, ExtraArgs&&... extra_args) const
   {
      if(mask.has_value()) {
         return sample(internal_tag, nr, *mask, FWD(extra_args)...);
      }
      return sample(internal_tag, nr, std::nullopt, FWD(extra_args)...);
   }

   // Check if the value is a valid member of this space
   bool contains(const value_type& value) const { return derived()._contains(value); }

   // Checks whether this space can be flattened to a Box
   [[nodiscard]] bool is_flattenable() const
   {
      if constexpr(requires(const Derived derived) { derived._is_flattenable(); }) {
         return derived()._is_flattenable();
      } else {
         return false;
      }
   }

   bool operator<=>(const Space& rhs) const = default;

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
   /// @tparam BoundaryTag Tag for whether to include or exclude the boundaries. Alternatively as
   /// pair-like type can have differing options for either end.
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

   constexpr const auto& derived() const { return static_cast< const Derived& >(*this); }
   constexpr auto& derived() { return static_cast< Derived& >(*this); }
};

template < typename T, typename Derived, typename MultiT, bool runtime_sample_throw >
   requires(std::is_same_v< T, MultiT > or detail::has_getitem_operator< MultiT >)
template < typename DType, std::ranges::range Rng1, std::ranges::range Rng2, typename BoundaryTag >
   requires(std::floating_point< DType > or std::integral< DType >)
bool Space< T, Derived, MultiT, runtime_sample_throw >::_isin_shape_and_bounds(
   const xarray< DType >& values,
   const Rng1& low_boundary,
   const Rng2& high_boundary,
   BoundaryTag
) const
{
   using namespace ranges;
   using namespace detail;
   if constexpr(is_none_v< BoundaryTag, InclusiveTag, ExclusiveTag >) {
      static_assert(
         is_any_v< std::tuple_element_t< 0, BoundaryTag >, InclusiveTag, ExclusiveTag >
            and is_any_v< std::tuple_element_t< 1, BoundaryTag >, InclusiveTag, ExclusiveTag >,
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

   auto enum_bounds_view = views::enumerate(views::zip(low_boundary, high_boundary));
   if(incoming_dim == space_dim + 1) {
      // first dim assumed a batch size
      if(not ranges::equal(shape(), incoming_shape | ranges::views::drop(1))) {
         return false;
      }
      return ranges::any_of(enum_bounds_view, [&](const auto& idx_low_high) {
         const auto& [i, low_high] = idx_low_high;
         const auto& [low, high] = low_high;
         auto coordinates = xt::unravel_index(static_cast< int >(i), shape());
         const auto& vals = xt::strided_view(
            values,
            prepend(xt::xstrided_slice_vector(coordinates.begin(), coordinates.end()), xt::all())
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
      constexpr auto compare = []< typename Cmp, typename CmpEq >(
                                  Cmp cmp, CmpEq cmp_eq, auto&&... args
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
   });
}

}  // namespace force

#undef MASKED_SAMPLE_FUNC

#endif  // REINFORCE_MONO_SPACE_HPP
