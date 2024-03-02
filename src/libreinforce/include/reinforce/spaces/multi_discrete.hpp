
#ifndef REINFORCE_MULTI_DISCRETE_HPP
#define REINFORCE_MULTI_DISCRETE_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

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

}  // namespace detail

template < std::integral T >
class MultiDiscreteSpace: public Space< xarray< T >, MultiDiscreteSpace< T > > {
  public:
   friend class Space< xarray< T >, MultiDiscreteSpace >;
   using base = Space< xarray< T >, MultiDiscreteSpace >;
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

   std::string repr()
   {
      if(xt::any(xt::not_equal(m_start, 0))) {
         return fmt::format("MultiDiscrete({}, start={})", m_end, m_start);
      }
      return fmt::format("MultiDiscrete({})", m_end);
   }

  private:
   value_type m_start;
   value_type m_end;

   value_type _sample(const std::vector< std::optional< xarray< bool > > >& mask_vec = {}) const
   {
      return _sample(1, mask_vec);
   }

   value_type _sample(
      size_t nr_samples,
      const std::vector< std::optional< xarray< bool > > >& mask_vec = {}
   ) const;

   bool _contains(const T& value) const
   {
      return ranges::any_of(ranges::views::zip(m_start, m_end), [&](const auto& low_high) {
         auto&& [low, high] = low_high;
         return low <= value and high >= value;
      });
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

template < std::integral T >
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

template < std::integral T >
auto MultiDiscreteSpace< T >::_sample(
   size_t nr_samples,
   const std::vector< std::optional< xarray< bool > > >& mask_vec
) const -> value_type
{
   if(nr_samples == 0) {
      throw std::invalid_argument("`nr_samples` argument has to be greater than 0.");
   }
   xt::svector< int > samples_shape = shape();
   samples_shape.push_back(static_cast< int >(nr_samples));
   SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples_shape));
   xarray< T > samples = xt::empty< T >(std::move(samples_shape));

   for(auto&& [i, bounds] : ranges::views::enumerate(ranges::views::zip(m_start, m_end))) {
      auto&& [start, end] = bounds;
      // convert the flat index i to an indexing list for the given shape
      auto coordinates = xt::unravel_index(static_cast< int >(i), shape());
      // add all entries of the variate's access in the shape
      xt::xstrided_slice_vector index_stride(coordinates.begin(), coordinates.end());
      // add all the sampling indices as if samples[...,:] on a numpy array so that they can be
      // emplaced all at once
      index_stride.emplace_back(xt::all());
      SPDLOG_DEBUG(fmt::format("Strides: {}", index_stride));
      auto draw_shape = xt::svector{nr_samples};
      if(mask_vec.size() > i and mask_vec[i].has_value()) {
         xt::strided_view(samples, index_stride) = xt::random::choice(
            xt::eval(xt::filter(xt::arange(start, end), *mask_vec[i])), nr_samples, true, rng()
         );
      } else {
         xt::strided_view(samples, index_stride) = xt::random::randint(
            draw_shape, start, end, rng()
         );
      }
   }

   return samples;
}

}  // namespace force

#endif  // REINFORCE_MULTI_DISCRETE_HPP
