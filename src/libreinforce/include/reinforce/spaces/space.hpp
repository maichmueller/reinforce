
#ifndef REINFORCE_SPACE_HPP
#define REINFORCE_SPACE_HPP

namespace force {

#include <pybind11/numpy.h>

#include <optional>
#include <random>
#include <utility>
#include <vector>
#include <xtensor/xarray.hpp>

#include "reinforce/utils/xtensor_typedefs.hpp"

/// \brief The generic Space base class
///
/// The specifics are almost 1:1 adapted from the corresponding definition of the openai/gymnasium
/// python class.
template < typename T >
class TypedSpace {
  public:
   using value_type = T;

   TypedSpace(xt::svector< int > shape = {}, std::optional< size_t > seed = std::nullopt)
       : m_shape(std::move(shape)), m_rng(seed_rng(seed))
   {
   }
   virtual ~TypedSpace() = default;

   // Randomly sample an element of this space
   virtual T sample(std::optional< xarray< int8_t > > mask = std::nullopt) = 0;

   // Seed the PRNG of this space
   void seed(size_t seed) { seed_rng(seed); }

   // Check if the value is a valid member of this space
   virtual bool contains(const T& value) const = 0;

   // Checks whether this space can be flattened to a Box
   virtual bool is_np_flattenable() const { return false; }

   // Return the shape of the space
   auto& shape() const { return m_shape; }
   /// const rng reference for external rng state inspection
   auto& rng() const { return m_rng; }

  protected:
   /// mutable rng reference for derived classes to forward random state
   auto& rng() { return m_rng; }

  private:
   xt::svector< int > m_shape;
   std::mt19937_64 m_rng;

   // Seed the PRNG
   void seed_rng(std::optional< size_t > seed)
   {
      m_rng = std::mt19937_64{seed.value_or(std::random_device{}())};
   }
};

}  // namespace force

#endif  // REINFORCE_SPACE_HPP
