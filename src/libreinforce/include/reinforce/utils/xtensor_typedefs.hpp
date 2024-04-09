#ifndef REINFORCE_XTENSOR_TYPEDEFS_HPP
#define REINFORCE_XTENSOR_TYPEDEFS_HPP

#include <cstddef>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>

namespace force {

constexpr auto layout = xt::layout_type::row_major;

template < typename T >
using xarray = xt::xarray< T, layout >;
template < typename T >
using pyarray = xt::pyarray< T, layout >;
template < typename T, size_t... shape >
using xstacktensor = xt::xtensor_fixed< T, xt::xshape< shape... >, layout >;

using idx_xarray = xt::xarray< size_t, layout >;

template < size_t dim >
using idx_xstacktensor = xstacktensor< size_t, dim >;
template < size_t dim >
using idx_xstackvector = xstacktensor< long, dim >;

using idx_pyarray = xt::pyarray< size_t, layout >;

}  // namespace force

#endif  // REINFORCE_XTENSOR_TYPEDEFS_HPP
