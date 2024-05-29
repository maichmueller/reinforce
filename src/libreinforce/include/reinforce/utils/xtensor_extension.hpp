
#ifndef XTENSOR_EXTENSION_HPP
#define XTENSOR_EXTENSION_HPP

#include <cstddef>

#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace xt {

template < typename T, typename Range >
xarray< T > full(Range&& shape, const T& value)
{
   auto arr = xt::empty< T >(FWD(shape));
   arr.fill(value);
   return arr;
}

template < class S1, class S2 >
inline bool reshapeable(const S1& src_shape, const S2& dst_shape)
{
   auto num_elems_src = std::accumulate(src_shape.cbegin(), src_shape.cend(), 1, std::multiplies{});
   auto num_elems_dst = std::accumulate(dst_shape.cbegin(), dst_shape.cend(), 1, std::multiplies{});
   return num_elems_src == num_elems_dst;
}

}  // namespace xt

#endif  // XTENSOR_EXTENSION_HPP
