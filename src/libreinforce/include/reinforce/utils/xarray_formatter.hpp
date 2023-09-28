#ifndef REINFORCE_XARRAY_FORMATTER_HPP
#define REINFORCE_XARRAY_FORMATTER_HPP

#include <fmt/core.h>
#include <fmt/std.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

template < class T, xt::layout_type L, class A, class SA >
struct fmt::formatter< xt::xarray< T, L, A, SA > >: fmt::ostream_formatter {};
template < class T, size_t N, xt::layout_type L, class A >
struct fmt::formatter< xt::xtensor< T, N, L, A > >: fmt::ostream_formatter {};
template < class T, class FSH, xt::layout_type L, bool S >
struct fmt::formatter< xt::xtensor_fixed< T, FSH, L, S > >: fmt::ostream_formatter {};

// Specify disambiguation specializations for xarray< T >.
// The specializations for this project's code are needed whenever a file includes <fmt/ranges.h>
// and "xarray_formatter.hpp". The formatting library fmt will then have an ambiguity between the
// generic formatter above for xarray< T, L, S, SA> and the formatter from fmt/ranges.h (as both are
// equally valid template choices).
// To use formatters in such situations we essentially need to lay out all the specializations we
// want to see formatting for in such a file. In the absence of fmt/range.h, it should not be
// necessary to have these as the generic xarray<...> formatter will apply.
// For types which need to carry
#ifndef XARRAY_FORMATTER
   #define XARRY_FORMATTER(T) \
      template <>             \
      struct fmt::formatter< xt::xarray< T > >: fmt::ostream_formatter {}
#endif
#ifndef XSTACKTENSOR_FORMATTER
   #define XSTACKTENSOR_FORMATTER(T, ...)                                         \
      template <>                                                                 \
      struct fmt::formatter< xt::xtensor_fixed< T, xt::xshape< __VA_ARGS__ > > >: \
          fmt::ostream_formatter {}
#endif

XARRY_FORMATTER(double);
XARRY_FORMATTER(float);
XARRY_FORMATTER(int);
XARRY_FORMATTER(unsigned int);
XARRY_FORMATTER(size_t);
XSTACKTENSOR_FORMATTER(size_t, 1);
XSTACKTENSOR_FORMATTER(size_t, 2);
XSTACKTENSOR_FORMATTER(size_t, 3);
XSTACKTENSOR_FORMATTER(size_t, 4);
XSTACKTENSOR_FORMATTER(size_t, 5);
XSTACKTENSOR_FORMATTER(size_t, 6);
XSTACKTENSOR_FORMATTER(size_t, 7);
XSTACKTENSOR_FORMATTER(size_t, 8);
XSTACKTENSOR_FORMATTER(size_t, 9);
XSTACKTENSOR_FORMATTER(size_t, 10);
XSTACKTENSOR_FORMATTER(size_t, 11);
XSTACKTENSOR_FORMATTER(size_t, 12);
XSTACKTENSOR_FORMATTER(size_t, 13);
XSTACKTENSOR_FORMATTER(size_t, 14);
XSTACKTENSOR_FORMATTER(size_t, 15);
XSTACKTENSOR_FORMATTER(size_t, 16);
XSTACKTENSOR_FORMATTER(size_t, 17);
XSTACKTENSOR_FORMATTER(size_t, 18);
XSTACKTENSOR_FORMATTER(size_t, 19);
XSTACKTENSOR_FORMATTER(size_t, 20);

#endif  // REINFORCE_XARRAY_FORMATTER_HPP