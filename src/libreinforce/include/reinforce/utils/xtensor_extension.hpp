
#ifndef XTENSOR_EXTENSION_HPP
#define XTENSOR_EXTENSION_HPP

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

}  // namespace xt

#endif  // XTENSOR_EXTENSION_HPP
