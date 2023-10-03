#ifndef REINFORCE_XARRAY_FORMATTER_HPP
#define REINFORCE_XARRAY_FORMATTER_HPP

#include <fmt/core.h>
#include <fmt/std.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xstrided_view.hpp>

#include "reinforce/utils/type_traits.hpp"

template < class T, xt::layout_type L, class A, class SA >
struct fmt::formatter< xt::xarray< T, L, A, SA > >: fmt::ostream_formatter {};
template < class T, size_t N, xt::layout_type L, class A >
struct fmt::formatter< xt::xtensor< T, N, L, A > >: fmt::ostream_formatter {};
template < class T, class FSH, xt::layout_type L, bool S >
struct fmt::formatter< xt::xtensor_fixed< T, FSH, L, S > >: fmt::ostream_formatter {};

//template < typename T >
//   requires(force::detail::is_any_v< T, xt::xall_tag, xt::xellipsis_tag, xt::xnewaxis_tag, xt::placeholders::xtuph >)
//struct fmt::formatter< T > {
//   template < typename FormatContext >
//   auto format(const T& /*unused*/, FormatContext& ctx)
//   {
//      if constexpr(std::same_as< T, xt::xall_tag >) {
//         return format_to(ctx.out(), ":");
//      } else if constexpr(std::same_as< T, xt::xellipsis_tag >) {
//         return format_to(ctx.out(), "...");
//      } else if constexpr(std::same_as< T, xt::xnewaxis_tag >) {
//         return format_to(ctx.out(), "newaxis");
//      } else if constexpr(std::same_as< T, xt::placeholders::xtuph >) {
//         return format_to(ctx.out(), "_");
//      } else {
//         static_assert(force::detail::always_false(ctx), "tag does not have formatting string.");
//      }
//   }

//  static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
//};
template <>
struct fmt::formatter< xt::xall_tag > {
   template < typename FormatContext >
   auto format(const xt::xall_tag& /*unused*/, FormatContext& ctx) const
   {
      return format_to(ctx.out(), ":");
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};
template <>
struct fmt::formatter< xt::xellipsis_tag > {
   template < typename FormatContext >
   auto format(const xt::xellipsis_tag& /*unused*/, FormatContext& ctx) const
   {
      return format_to(ctx.out(), "...");
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};
template <>
struct fmt::formatter< xt::xnewaxis_tag > {
   template < typename FormatContext >
   auto format(const xt::xnewaxis_tag& /*unused*/, FormatContext& ctx) const
   {
      return format_to(ctx.out(), "newaxis");
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};
template <>
struct fmt::formatter< xt::placeholders::xtuph > {
   template < typename FormatContext >
   auto format(const xt::placeholders::xtuph& /*unused*/, FormatContext& ctx) const
   {
      return format_to(ctx.out(), "_");
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Custom formatter for xrange_adaptor
//
// template < typename T >
// struct fmt::formatter< xt::xrange_adaptor< xt::placeholders::xtuph, T, T > > {
//   template < typename FormatContext >
//   auto format(const xt::xrange_adaptor< T >& range, FormatContext& ctx)
//   {
//      return format_to(ctx.out(), "xrange<start={},stop=_,step={}>", range.start(), range.step());
//   }
//};
// template < typename T >
// struct fmt::formatter< xt::xrange_adaptor< T, xt::placeholders::xtuph, T > > {
//   template < typename FormatContext >
//   auto format(const xt::xrange_adaptor< T >& range, FormatContext& ctx)
//   {
//      return format_to(ctx.out(), "xrange<start=_,stop={},step={}>", range.stop(), range.step());
//   }
//};
// template < typename T >
// struct fmt::formatter< xt::xrange_adaptor< T, T, xt::placeholders::xtuph > > {
//   template < typename FormatContext >
//   auto format(const xt::xrange_adaptor< T >& range, FormatContext& ctx)
//   {
//      return format_to(ctx.out(), "xrange<start={},stop={},step=_>", range.start(), range.stop());
//   }
//};
// template < typename T >
// struct fmt::formatter< xt::xrange_adaptor< T, xt::placeholders::xtuph, xt::placeholders::xtuph >
// > {
//   template < typename FormatContext >
//   auto format(const xt::xrange_adaptor< T >& range, FormatContext& ctx)
//   {
//      return format_to(ctx.out(), "xrange<start={},stop=_,step=_>", range.start());
//   }
//};
// template < typename T >
// struct fmt::formatter< xt::xrange_adaptor< xt::placeholders::xtuph, T, xt::placeholders::xtuph >
// > {
//   template < typename FormatContext >
//   auto format(const xt::xrange_adaptor< T >& range, FormatContext& ctx)
//   {
//      return format_to(ctx.out(), "xrange<start=_,stop={},step=_>", range.stop());
//   }
//};
// template < typename T >
// struct fmt::formatter< xt::xrange_adaptor< xt::placeholders::xtuph, xt::placeholders::xtuph, T >
// > {
//   template < typename FormatContext >
//   auto format(const xt::xrange_adaptor< T >& range, FormatContext& ctx)
//   {
//      return format_to(ctx.out(), "xrange<start=_,stop=_,step={}>", range.step());
//   }
//};
// template <>
// struct fmt::formatter< xt::xrange_adaptor<
//   xt::placeholders::xtuph,
//   xt::placeholders::xtuph,
//   xt::placeholders::xtuph > > {
//   template < typename FormatContext >
//   auto format(const xt::xrange_adaptor< T >& range, FormatContext& ctx)
//   {
//      return format_to(ctx.out(), "xrange<start=_,stop=_,step=_>");
//   }
//};
template < typename A, typename B, typename C >
struct fmt::formatter< xt::xrange_adaptor< A, B, C > > {
   template < typename FormatContext >
   auto format(const xt::xrange_adaptor< A, B, C >& range, FormatContext& ctx) const
   {
      return format_to(
         ctx.out(), "xrange<start={},stop={},step={}>", range.start(), range.stop(), range.step()
      );
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Custom formatter for xrange
template < typename T >
struct fmt::formatter< xt::xrange< T > > {
   template < typename FormatContext >
   auto format(const xt::xrange< T >& range, FormatContext& ctx) const
   {
      return format_to(ctx.out(), "xrange<start={},size={}>", range(0), range.size());
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Custom formatter for xstepped_range
template < typename T >
struct fmt::formatter< xt::xstepped_range< T > > {
   template < typename FormatContext >
   auto format(const xt::xstepped_range< T >& range, FormatContext& ctx) const
   {
      return format_to(
         ctx.out(),
         "xstepped_range<start={},size={},step_size={}>",
         range(0),
         range.size(),
         range.step_size()
      );
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Custom formatter for xstrided_slice (which is a variant of the preceding types)
template < typename T >
struct fmt::formatter< xt::xstrided_slice< T > > {
   template < typename FormatContext >
   auto format(const xt::xstrided_slice< T >& slice, FormatContext& ctx) const
   {
      return mpark::visit([&](const auto& actual) { return format_to(ctx.out(), "{}", actual); }, slice);
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

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
XARRY_FORMATTER(std::string);
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
