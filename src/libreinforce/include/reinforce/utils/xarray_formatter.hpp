#ifndef REINFORCE_XARRAY_FORMATTER_HPP
#define REINFORCE_XARRAY_FORMATTER_HPP

#include <fmt/core.h>
#include <fmt/std.h>

#include <xtensor/xarray.hpp>

template < class T, xt::layout_type L, class A, class SA >
struct fmt::formatter< xt::xarray< T, L, A, SA > >: fmt::ostream_formatter {};
template < class T, size_t N, xt::layout_type L, class A>
struct fmt::formatter< xt::xtensor< T, N, L, A > >: fmt::ostream_formatter {};
template < class T, class FSH, xt::layout_type L, bool S>
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

XARRY_FORMATTER(double);
XARRY_FORMATTER(float);
XARRY_FORMATTER(int);
XARRY_FORMATTER(unsigned int);
XARRY_FORMATTER(size_t);

#endif  // REINFORCE_XARRAY_FORMATTER_HPP
