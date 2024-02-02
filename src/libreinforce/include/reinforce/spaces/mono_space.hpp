#ifndef REINFORCE_MONO_SPACE_HPP
#define REINFORCE_MONO_SPACE_HPP

#include <optional>
#include <random>
#include <utility>
#include <vector>
#include <xtensor/xarray.hpp>

#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

/// \brief The generic Space base class
///
/// The specifics are made to work as closely as possible to the corresponding internals of the
/// openai/gymnasium python class.
template < typename T, typename Derived >
class TypedMonoSpace: public detail::rng_mixin {
  public:
   using value_type = T;

   explicit
   TypedMonoSpace(xt::svector< int > shape = {}, std::optional< size_t > seed = std::nullopt)
       : rng_mixin(seed), m_shape(std::move(shape))
   {
   }

   // Randomly sample an element of this space
   xarray< value_type > sample()
   {
      if constexpr(requires(Derived self) { self._sample(); }) {
         return self()._sample();
      } else {
         throw detail::not_implemented_error("sample()");
      }
   }

   xarray< value_type > sample(const xarray< bool >& mask)
   {
      if constexpr(requires(Derived self) { self._sample(mask); }) {
         return self()._sample(mask);
      } else {
         throw detail::not_implemented_error("sample(const xarray< bool >&)");
      }
   }

   xarray< value_type > sample(const std::vector< std::optional< xarray< bool > > >& mask_vec)
   {
      if constexpr(requires(Derived self) { self._sample(mask_vec); }) {
         return self()._sample(mask_vec);
      } else {
         throw detail::not_implemented_error("sample(const std::optional< xarray< bool > >&)");
      }
   }

   xarray< value_type > sample(size_t nr_samples)
   {
      if constexpr(requires(Derived self) { self._sample(nr_samples); }) {
         return self()._sample(nr_samples);
      } else {
         throw detail::not_implemented_error("sample(size_t)");
      }
   }

   xarray< value_type > sample(size_t nr_samples, const xarray< bool >& mask)
   {
      if constexpr(requires(Derived self) { self._sample(nr_samples, mask); }) {
         return self()._sample(nr_samples, mask);
      } else {
         throw detail::not_implemented_error("sample(size_t, const xarray< bool >&)");
      }
   }

   xarray< value_type >
   sample(size_t nr_samples, const std::vector< std::optional< xarray< bool > > >& mask_vec)
   {
      if constexpr(requires(Derived self) { self._sample(nr_samples, mask_vec); }) {
         return self()._sample(nr_samples, mask_vec);
      } else {
         throw detail::not_implemented_error(
            "sample(size_t, const std::vector< std::optional< xarray< bool > > >&)"
         );
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
