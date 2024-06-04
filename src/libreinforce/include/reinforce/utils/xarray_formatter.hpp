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

// template < typename T >
//    requires(force::detail::is_any_v< T, xt::xall_tag, xt::xellipsis_tag, xt::xnewaxis_tag,
//    xt::placeholders::xtuph >)
// struct fmt::formatter< T > {
//    template < typename FormatContext >
//    auto format(const T& /*unused*/, FormatContext& ctx)
//    {
//       if constexpr(std::same_as< T, xt::xall_tag >) {
//          return format_to(ctx.out(), ":");
//       } else if constexpr(std::same_as< T, xt::xellipsis_tag >) {
//          return format_to(ctx.out(), "...");
//       } else if constexpr(std::same_as< T, xt::xnewaxis_tag >) {
//          return format_to(ctx.out(), "newaxis");
//       } else if constexpr(std::same_as< T, xt::placeholders::xtuph >) {
//          return format_to(ctx.out(), "_");
//       } else {
//          static_assert(force::detail::always_false(ctx), "tag does not have formatting string.");
//       }
//    }

//  static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
//};
template <>
struct fmt::formatter< xt::xall_tag > {
   template < typename FormatContext >
   auto format(const xt::xall_tag& /*unused*/, FormatContext& ctx) const
   {
      return fmt::format_to(ctx.out(), ":");
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};
template <>
struct fmt::formatter< xt::xellipsis_tag > {
   template < typename FormatContext >
   auto format(const xt::xellipsis_tag& /*unused*/, FormatContext& ctx) const
   {
      return fmt::format_to(ctx.out(), "...");
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};
template <>
struct fmt::formatter< xt::xnewaxis_tag > {
   template < typename FormatContext >
   auto format(const xt::xnewaxis_tag& /*unused*/, FormatContext& ctx) const
   {
      return fmt::format_to(ctx.out(), "newaxis");
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};
template <>
struct fmt::formatter< xt::placeholders::xtuph > {
   template < typename FormatContext >
   auto format(const xt::placeholders::xtuph& /*unused*/, FormatContext& ctx) const
   {
      return fmt::format_to(ctx.out(), "_");
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
      return fmt::format_to(
         ctx.out(), "xrange(start={},stop={},step={})", range.start(), range.stop(), range.step()
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
      return fmt::format_to(ctx.out(), "xrange(start={},size={})", range(0), range.size());
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Custom formatter for xstepped_range
template < typename T >
struct fmt::formatter< xt::xstepped_range< T > > {
   template < typename FormatContext >
   auto format(const xt::xstepped_range< T >& range, FormatContext& ctx) const
   {
      return fmt::format_to(
         ctx.out(),
         "xstepped_range(start={},size={},step_size={})",
         range(0),
         range.size(),
         range.step_size()
      );
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Custom formatter for xstrided_slice (which is a variant of the aforementioned types)
template < typename T >
struct fmt::formatter< xt::xstrided_slice< T > > {
   template < typename FormatContext >
   auto format(const xt::xstrided_slice< T >& slice, FormatContext& ctx) const
   {
      return mpark::visit(
         [&](const auto& actual) { return fmt::format_to(ctx.out(), "{}", actual); }, slice
      );
   }
   static constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
};

// Specify disambiguation specializations for xarray< T >.
// The specializations for this project's code are needed whenever a file includes <fmt/ranges.h>
// and "xarray_formatter.hpp". The formatting library fmt will then have an ambiguity between the
// generic formatter above for xarray< T, L, S, SA> and the formatter from fmt/ranges.h (as both
// are equally valid template choices). To use formatters in such situations we essentially need
// to lay out all the specializations we want to see formatting for in such a file. In the
// absence of fmt/range.h, it should not be necessary to have these as the generic xarray<...>
// formatter will apply. For types which need to carry
#ifndef XARRAY_FORMATTER
   #define XARRY_FORMATTER(T) \
      template <>             \
      struct fmt::formatter< xt::xarray< T > >: fmt::ostream_formatter {}
#endif

#ifndef XTENSOR_FORMATTER
   #define XTENSOR_FORMATTER(T, DIMS) \
      template <>                     \
      struct fmt::formatter< xt::xtensor< T, DIMS > >: fmt::ostream_formatter {}
#endif

#ifndef XSTACKTENSOR_FORMATTER
   #define XSTACKTENSOR_FORMATTER(T, ...)                                         \
      template <>                                                                 \
      struct fmt::formatter< xt::xtensor_fixed< T, xt::xshape< __VA_ARGS__ > > >: \
          fmt::ostream_formatter {}
#endif

XARRY_FORMATTER(bool);
XARRY_FORMATTER(double);
XARRY_FORMATTER(float);
XARRY_FORMATTER(long);
XARRY_FORMATTER(unsigned long);
XARRY_FORMATTER(long long);
XARRY_FORMATTER(unsigned long long);
XARRY_FORMATTER(int);
XARRY_FORMATTER(unsigned int);
XARRY_FORMATTER(short);
XARRY_FORMATTER(unsigned short);
XARRY_FORMATTER(char);
XARRY_FORMATTER(signed char);
XARRY_FORMATTER(unsigned char);
XARRY_FORMATTER(std::string);

XARRY_FORMATTER(xt::xarray< bool >);
XARRY_FORMATTER(xt::xarray< double >);
XARRY_FORMATTER(xt::xarray< float >);
XARRY_FORMATTER(xt::xarray< long >);
XARRY_FORMATTER(xt::xarray< unsigned long >);
XARRY_FORMATTER(xt::xarray< long long >);
XARRY_FORMATTER(xt::xarray< unsigned long long >);
XARRY_FORMATTER(xt::xarray< int >);
XARRY_FORMATTER(xt::xarray< unsigned int >);
XARRY_FORMATTER(xt::xarray< short >);
XARRY_FORMATTER(xt::xarray< unsigned short >);
XARRY_FORMATTER(xt::xarray< char >);
XARRY_FORMATTER(xt::xarray< signed char >);
XARRY_FORMATTER(xt::xarray< unsigned char >);
XARRY_FORMATTER(xt::xarray< std::string >);

namespace xt::detail {

template < class T >
struct printer<
   T,
   std::enable_if_t< force::detail::is_specialization_v< typename T::value_type, std::vector > > > {
   using value_type = std::decay_t< typename T::value_type >;
   using cache_type = std::vector< T >;
   using cache_iterator = typename cache_type::const_iterator;

   explicit printer(std::streamsize precision) : m_printer(precision) {}

   void init() { m_it = m_cache.cbegin(); }

   std::ostream& print_next(std::ostream& out)
   {
      for(const auto& i : *m_it | ranges::views::drop_last)
         fmt::format(out, "{:<{}}, ", width(), i);
      fmt::format(out, "{:<{}}", width(), m_it.back());
      ++m_it;
      return out;
   }

   void update(const value_type& val) { m_cache.push_back(val); }

   std::streamsize width() { return m_printer.width(); }

  private:
   printer< typename value_type::value_type > m_printer;
   cache_type m_cache;
   cache_iterator m_it;
};

}  // namespace xt::detail

XARRY_FORMATTER(std::vector< bool >);
XARRY_FORMATTER(std::vector< double >);
XARRY_FORMATTER(std::vector< float >);
XARRY_FORMATTER(std::vector< long >);
XARRY_FORMATTER(std::vector< unsigned long >);
XARRY_FORMATTER(std::vector< long long >);
XARRY_FORMATTER(std::vector< unsigned long long >);
XARRY_FORMATTER(std::vector< int >);
XARRY_FORMATTER(std::vector< unsigned int >);
XARRY_FORMATTER(std::vector< short >);
XARRY_FORMATTER(std::vector< unsigned short >);
XARRY_FORMATTER(std::vector< char >);
XARRY_FORMATTER(std::vector< signed char >);
XARRY_FORMATTER(std::vector< unsigned char >);
XARRY_FORMATTER(std::vector< std::string >);

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
