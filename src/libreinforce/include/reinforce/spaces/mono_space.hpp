#ifndef REINFORCE_MONO_SPACE_HPP
#define REINFORCE_MONO_SPACE_HPP

#include <fmt/core.h>

#include <cstddef>
#include <optional>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <xtensor/xarray.hpp>

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
template < typename T, typename Derived, typename MultiT = T >
   requires std::is_same_v< T, MultiT > or detail::has_getitem_operator< MultiT >
class TypedMonoSpace: public detail::rng_mixin {
  public:
   // the type of values returned by sampling or containment queries
   using value_type = T;
   // the type of values returned by multiple-sampling (i.e. sample-size > 1) queries and
   // containment queries for multiple elements at once
   using multi_value_type = MultiT;

   constexpr static bool mvt_is_container = not std::is_same_v< multi_value_type, value_type >
                                            and detail::has_getitem_operator< MultiT >;

   explicit
   TypedMonoSpace(xt::svector< int > shape = {}, std::optional< size_t > seed = std::nullopt)
       : rng_mixin(seed), m_shape(std::move(shape))
   {
   }

   // Randomly sample an element of this space
   value_type sample()
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
   value_type sample(const xarray< U >& mask)
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
   value_type sample(const std::vector< std::optional< xarray< U > > >& mask_vec)
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
   value_type sample(const std::tuple< Args... >& mask_tuple)
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

   multi_value_type sample(size_t nr_samples)
   {
      if constexpr(requires(Derived self) { self._sample(nr_samples); }) {
         return self()._sample(nr_samples);
      } else {
         throw detail::not_implemented_error(fmt::format("_sample(size_t{{{}}})", nr_samples));
      }
   }

   template < typename U >
   multi_value_type sample(size_t nr_samples, const xarray< U >& mask)
   {
      if constexpr(requires(Derived self) { self._sample(nr_samples, mask); }) {
         return self()._sample(nr_samples, mask);
      } else {
         throw detail::not_implemented_error(fmt::format(
            "_sample(size_t{{{}}}, {})", nr_samples, detail::type_name< decltype(mask) >()
         ));
      }
   }

   template < typename U >
   multi_value_type
   sample(size_t nr_samples, const std::vector< std::optional< xarray< U > > >& mask_vec)
   {
      if constexpr(requires(Derived self) { self._sample(nr_samples, mask_vec); }) {
         return self()._sample(nr_samples, mask_vec);
      } else {
         throw detail::not_implemented_error(fmt::format(
            "_sample(size_t{{{}}}, {})", nr_samples, detail::type_name< decltype(mask_vec) >()
         ));
      }
   }

   template < typename... Args >
   multi_value_type sample(size_t nr_samples, const std::tuple< Args... >& mask_tuple)
   {
      if constexpr(requires(Derived self) { self._sample(nr_samples, mask_tuple); }) {
         return self()._sample(nr_samples, mask_tuple);
      } else {
         throw detail::not_implemented_error(fmt::format(
            "_sample(size_t{{{}}}, {})", nr_samples, detail::type_name< decltype(mask_tuple) >()
         ));
      }
   }

   // Check if the value is a valid member of this space
   bool contains(const value_type& value) const { return self()._contains(value); }

   // Checks whether this space can be flattened to a Box
   [[nodiscard]] bool is_flattenable() const { return false; }

   // Return the shape of the space
   auto& shape() const { return m_shape; }

  protected:
   // mutable shape reference of the space
   auto& shape() { return m_shape; }

  private:
   xt::svector< int > m_shape;

   constexpr const auto& self() const { return static_cast< const Derived& >(*this); }
   constexpr auto& self() { return static_cast< Derived& >(*this); }
};

}  // namespace force

#endif  // REINFORCE_MONO_SPACE_HPP
