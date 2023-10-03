#ifndef REINFORCE_SPACE_HPP
#define REINFORCE_SPACE_HPP

#include <pybind11/numpy.h>

#include <optional>
#include <random>
#include <utility>
#include <vector>
#include <xtensor/xarray.hpp>

#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

/// \brief The generic Space base class
///
/// The specifics are made to work as closely as possible to the corresponding internals of the
/// openai/gymnasium python class.
template < typename T >
class TypedSpace {
  public:
   using value_type = T;

   explicit TypedSpace(xt::svector< int > shape = {}, std::optional< size_t > seed = std::nullopt)
       : m_shape(std::move(shape)), m_rng(seeded_rng(seed))
   {
   }
   virtual ~TypedSpace() = default;

   // Randomly sample an element of this space
   virtual xarray< T > sample(const std::optional< xarray< bool > >& mask = std::nullopt) = 0;
   virtual xarray< T >
   sample(size_t nr_samples, const std::optional< xarray< bool > >& mask = std::nullopt) = 0;

   // Seed the PRNG of this space
   void seed(size_t seed) { m_rng = seeded_rng(seed); }

   // Check if the value is a valid member of this space
   virtual bool contains(const T& value) const = 0;

   // Checks whether this space can be flattened to a Box
   virtual bool is_flattenable() const { return false; }

   // Return the shape of the space
   auto& shape() const { return m_shape; }
   /// const rng reference for external rng state inspection
   auto& rng() const { return m_rng; }

   bool operator==(const TypedSpace& rhs) const
   {
      return typeid(*this) == typeid(rhs) and shape() == rhs.shape() and _equals(rhs);
   }

  protected:
   // mutable shape reference of the space
   auto& shape() { return m_shape; }
   /// mutable rng reference for derived classes to forward random state
   auto& rng() { return m_rng; }

   virtual bool _equals(const TypedSpace& rhs) const = 0;

  private:
   xt::svector< int > m_shape;
   std::mt19937_64 m_rng;

   // Seed a PRNG
   auto seeded_rng(std::optional< size_t > seed)
   {
      return std::mt19937_64{seed.value_or(std::random_device{}())};
   }
};

}  // namespace force

#endif  // REINFORCE_SPACE_HPP
