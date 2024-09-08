
#ifndef REINFORCE_MULTI_DISCRETE_HPP
#define REINFORCE_MULTI_DISCRETE_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <concepts>
#include <iterator>
#include <optional>
#include <random>
#include <range/v3/all.hpp>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xstorage.hpp>

#include "reinforce/spaces/concepts.hpp"
#include "reinforce/spaces/space.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

namespace detail {

/// If it is an l-value array then we want to pass it on as such and do nothing else.
template < typename T, class Array >
   requires is_xarray_ref_of< Array, T >
auto build_xarray(Array&& arr) -> decltype(auto)
{
   return FWD(arr);
}

/// if we are having an r-value ref passed in then we return normally
/// this is so that pr-values (those values attained from construction of the object as a temporary)
/// will be copy-elided all the way to the destination, instead of moved and moved and moved...
template < typename T, class Array >
   requires is_xarray_of< raw_t< Array >, T >
auto build_xarray(Array&& arr)
{
   return FWD(arr);
}

template < typename T, typename Array >
   requires std::ranges::range< raw_t< Array > > and (not detail::is_xarray_of< raw_t< Array >, T >)
auto build_xarray(Array&& arr)
{
   auto [data_storage, size] = detail::make_carray< T >(FWD(arr));
   xarray< T > new_xarray = xt::adapt(
      data_storage.get(), size, xt::acquire_ownership(), xt::svector{size}
   );
   data_storage.release();
   return new_xarray;
}

template < typename Rng >
concept is_mask_range = std::ranges::range< Rng >
                        and std::same_as<
                           std::ranges::range_value_t< Rng >,
                           std::optional< xarray< bool > > >;

}  // namespace detail

template < typename T >
   requires multidiscrete_reqs< T >
class MultiDiscreteSpace: public Space< xarray< T >, MultiDiscreteSpace< T > > {
  public:
   friend class Space< xarray< T >, MultiDiscreteSpace >;
   using base = Space< xarray< T >, MultiDiscreteSpace >;
   using data_type = T;
   using typename base::value_type;
   using base::shape;
   using base::rng;

   template < class Array1, class Array2, typename... Args >
      requires(
         std::ranges::range< Array1 > and std::ranges::range< Array2 >
         and (not detail::is_xarray< detail::raw_t< Array1 > > or not detail::is_xarray< detail::raw_t< Array2 > >)
      )
   MultiDiscreteSpace(Array1&& start, Array2&& end, Args&&... args)
       : MultiDiscreteSpace(
            detail::build_xarray< T >(FWD(start)),
            detail::build_xarray< T >(FWD(end)),
            FWD(args)...
         )
   {
   }

   template < class Array, typename FirstArg, typename... TailArgs >
   MultiDiscreteSpace(const Array& end, FirstArg&& any, TailArgs&&... args)
       : MultiDiscreteSpace(
            xt::zeros_like(detail::build_xarray< T >(end)),
            detail::build_xarray< T >(end),
            FWD(any),
            FWD(args)...
         )
   {
   }

   template < class Array >
   MultiDiscreteSpace(const Array& end)
       : MultiDiscreteSpace(
            xt::zeros_like(detail::build_xarray< T >(end)),
            detail::build_xarray< T >(end)
         )
   {
   }

   template < std::integral Int >
   MultiDiscreteSpace(value_type start, value_type end, Int seed)
       : MultiDiscreteSpace(
            std::move(start),
            std::move(end),
            std::optional{static_cast< size_t >(seed)}
         )
   {
   }

   MultiDiscreteSpace(
      value_type start,
      value_type end,
      std::optional< size_t > seed = std::nullopt
   );

   bool operator==(const MultiDiscreteSpace& rhs) const
   {
      return xt::all(xt::equal(m_start, rhs.m_start)) and xt::all(xt::equal(m_end, rhs.m_end));
   }

   [[nodiscard]] std::string repr() const
   {
      if(xt::any(xt::not_equal(m_start, 0))) {
         return fmt::format("MultiDiscrete({}, start={})", m_end, m_start);
      }
      return fmt::format("MultiDiscrete({})", m_end);
   }

   const auto& start() const { return m_start; }
   const auto& end() const { return m_end; }

  private:
   value_type m_start;
   value_type m_end;

   template < typename MaskRange = std::array< std::optional< xarray< bool > >, 0 > >
      requires detail::is_mask_range< MaskRange >
   [[nodiscard]] value_type _sample(const MaskRange& mask_range = {}) const;

   [[nodiscard]] value_type _sample(std::nullopt_t /**/) const { return _sample(); }

   template < typename MaskRange = std::array< std::optional< xarray< bool > >, 0 > >
      requires detail::is_mask_range< MaskRange >
   [[nodiscard]] value_type _sample(size_t batch_size, const MaskRange& mask_range = {}) const;

   [[nodiscard]] value_type _sample(size_t batch_size, std::nullopt_t /**/) const
   {
      return _sample(batch_size);
   }

   [[nodiscard]] bool _contains(const value_type& value) const
   {
      return base::_isin_shape_and_bounds(value, m_start, m_end);
   }
};

/// Deduction guides

template < class Array1, class Array2, typename... Args >
MultiDiscreteSpace(Array1&& start, Array2&& end, Args&&... args)
   -> MultiDiscreteSpace< std::ranges::range_value_t< Array1 > >;

template < class Array, typename FirstArg, typename... TailArgs >
MultiDiscreteSpace(const Array& end, FirstArg&& any, TailArgs&&... args)
   -> MultiDiscreteSpace< std::ranges::range_value_t< Array > >;

template < class Array >
MultiDiscreteSpace(const Array& end) -> MultiDiscreteSpace< std::ranges::range_value_t< Array > >;

/// Implementations

template < typename T >
   requires multidiscrete_reqs< T >
MultiDiscreteSpace< T >::MultiDiscreteSpace(
   value_type start,
   value_type end,
   std::optional< size_t > seed
)
    : base(
         xt::svector< int >(std::ranges::begin(start.shape()), std::ranges::end(start.shape())),
         seed
      ),
      m_start(std::move(start)),
      m_end(std::move(end))
{
   using namespace fmt::literals;

   auto start_shape = m_start.shape();
   auto end_shape = m_end.shape();

   SPDLOG_DEBUG(fmt::format(
      "Start shape {}, End shape: {}, specified shape: {}", start_shape, end_shape, shape()
   ));
   if(not ranges::equal(end_shape, start_shape)) {
      throw std::invalid_argument(fmt::format(
         "'Low' and 'High' bound arrays need to have the same shape. Given:\n{}\nand\n{}",
         start_shape,
         end_shape
      ));
   }
   SPDLOG_DEBUG(fmt::format("Bounds:\n{}", std::invoke([&] {
                               xarray< std::string > bounds = xt::empty< std::string >(shape());
                               for(auto [i, bound_string] : ranges::views::enumerate(
                                      ranges::views::zip(m_start, m_end)
                                      | ranges::views::transform([](auto pair) {
                                           return fmt::format(
                                              "{bracket_open}{lower},..,{upper}{bracket_close}",
                                              "bracket_open"_a = "{",
                                              "bracket_close"_a = "}",
                                              "lower"_a = pair.first,
                                              "upper"_a = pair.second
                                           );
                                        })
                                   )) {
                                  bounds.flat(i) = bound_string;
                               };
                               return bounds;
                            })));
}

template < typename T >
   requires multidiscrete_reqs< T >
template < typename MaskRange >
   requires detail::is_mask_range< MaskRange >
auto MultiDiscreteSpace< T >::_sample(size_t batch_size, const MaskRange& mask_range) const
   -> value_type
{
   switch(batch_size) {
      case 0: {
         return xt::empty< T >({0});
      }
      case 1: {
         return xt::expand_dims(_sample(mask_range), 0);
      }
      default: {
         xarray< T > samples = xt::empty< T >(prepend(shape(), static_cast< int >(batch_size)));
         SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples.shape()));

         auto mask_iter = std::ranges::begin(mask_range),
              mask_iter_end = std::ranges::end(mask_range);
         for(auto&& [i, bounds] : ranges::views::enumerate(ranges::views::zip(m_start, m_end))) {
            auto&& [start, end] = FWD(bounds);
            // convert the flat index i to an indexing list for the given shape
            auto coordinates = xt::unravel_index(static_cast< int >(i), shape());
            // add all entries of the variate's access in the shape
            // add all the sampling indices as if samples[:, ...] on a numpy array so that they can
            // be emplaced all at once
            auto index_stride = prepend(
               xt::xstrided_slice_vector{},
               detail::as_range(coordinates.begin(), coordinates.end()),
               xt::all()
            );
            SPDLOG_DEBUG(fmt::format("Strides: {}", index_stride));
            auto&& view = xt::strided_view(samples, index_stride);
            if(mask_iter != mask_iter_end and mask_iter->has_value()) {
               view = xt::random::choice(
                  xt::eval(xt::filter(xt::arange(start, end), **mask_iter)), batch_size, true, rng()
               );
            } else {
               view = xt::random::randint({batch_size}, start, end, rng());
            }
            std::ranges::advance(mask_iter, 1, mask_iter_end);
         }
         return samples;
      }
   }
}

template < typename T >
   requires multidiscrete_reqs< T >
template < typename MaskRange >
   requires detail::is_mask_range< MaskRange >
auto MultiDiscreteSpace< T >::_sample(const MaskRange& mask_range) const -> value_type
{
   xarray< T > samples = xt::empty< T >(shape());
   SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples.shape()));

   auto mask_iter = std::ranges::begin(mask_range), mask_iter_end = std::ranges::end(mask_range);
   for(auto&& [sample, start, end] : ranges::views::zip(samples, m_start, m_end)) {
      if(mask_iter != mask_iter_end and mask_iter->has_value()) {
         sample = xt::random::choice(
                     xt::eval(xt::filter(xt::arange(start, end), **mask_iter)), 1, true, rng()
         )
                     .unchecked(0);
      } else {
         sample = xt::random::randint({1}, start, end, rng()).unchecked(0);
      }
      std::ranges::advance(mask_iter, 1, mask_iter_end);
   }
   return samples;
}

/// this method is clearer, but there is a bug in range-v3 that causes undefined behaviour
/// when using the zip view over xarrays. This bug no longer occurs with std::ranges. But only from
/// c++26 onwards are all equivalents necessary implemented in the std to do so.
/// For more information check this godbolt example:
/// https://godbolt.org/z/Go1n6vjEz
// template < typename T >
//    requires multidiscrete_reqs< T >
// template < typename MaskRange >
//    requires detail::is_mask_range< MaskRange >
// auto MultiDiscreteSpace< T >::_sample(const MaskRange& mask_range) const -> value_type
// {
//    using namespace ranges;
//    xarray< T > samples = xt::empty< T >(shape());
//    SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples.shape()));
//
//    for(auto&& [sample, start, end, mask_opt] :
//        views::zip(samples, m_start, m_end, views::concat(mask_range,
//        views::repeat(std::nullopt)))))
//       {
//          {
//             sample = std::invoke([&] {
//                if(mask_opt.has_value()) {
//                   return xt::random::choice(
//                             xt::eval(xt::filter(xt::arange(start, end), *mask_opt)), 1, true,
//                             rng()
//                   )
//                      .unchecked(0);
//                } else {
//                   return xt::random::randint({1}, start, end, rng()).unchecked(0);
//                }
//             });
//          }
//       }
//    return samples;
// }

}  // namespace force

#endif  // REINFORCE_MULTI_DISCRETE_HPP
