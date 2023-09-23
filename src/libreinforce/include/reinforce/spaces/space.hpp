
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

// Define a generic Space class
template < typename T_cov >
class Space {
  public:
   Space(
      std::optional< std::vector< int > > shape = std::nullopt,
      std::optional< std::variant< std::string, pybind11::dtype > > dtype = std::nullopt,
      std::optional< size_t > seed = std::nullopt
   )
       : m_shape(std::move(shape)), m_dtype(std::move(dtype)), m_rng(seed_rng(seed))
   {
   }
   virtual ~Space() = default;

   // Randomly sample an element of this space
   virtual T_cov sample(std::optional< xarray< int8_t > > mask = std::nullopt) = 0;

   // Seed the PRNG of this space
   void seed(size_t seed) { seed_rng(seed); }

   // Check if the value is a valid member of this space
   virtual bool contains(const T_cov& value) const = 0;

   virtual std::vector< T_cov > to_jsonable(const std::vector< T_cov >& sample_n) const
   {
      return sample_n;
   }

   virtual std::vector< T_cov > from_jsonable(const std::vector< T_cov >& sample_n) const
   {
      return sample_n;
   }

   // Checks whether this space can be flattened to a Box
   virtual bool is_np_flattenable() const { return false; }

   // Return the shape of the space
   auto& shape() const { return m_shape; }
   /// the dtype variant reference
   auto& dtype() const { return m_dtype; }
   /// const rng reference for external rng state inspection
   auto& rng() const { return m_rng; }

  protected:
   /// mutable rng reference for derived classes to forward random state
   auto& rng() { return m_rng; }

  private:
   std::optional< std::vector< int > > m_shape;
   std::optional< std::variant< std::string, pybind11::dtype > > m_dtype;
   std::mt19937_64 m_rng;

   // Seed the PRNG
   void seed_rng(std::optional< size_t > seed)
   {
      m_rng = std::mt19937_64{seed.value_or(std::random_device{}())};
   }
};

}  // namespace force

#endif  // REINFORCE_SPACE_HPP
