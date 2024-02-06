
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

#include "mono_space.hpp"
#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

namespace concepts {

template < class Array, typename T >
concept is_xarray = detail::is_any_v< Array, pyarray< T >, xarray< T > >;

template < class Array, typename T >
concept is_xarray_ref = is_xarray< detail::raw_t< Array >, T >
                        and std::is_lvalue_reference_v< Array >;

}  // namespace concepts

namespace detail {

/// If it is an l-value array then we want to pass it on as such and do nothing else.
template < typename T, class Array >
   requires concepts::is_xarray_ref< Array, T >
auto build_xarray(Array&& arr) -> decltype(auto)
{
   return FWD(arr);
}

/// if we are having an r-value ref passed in then we return normally
/// this is so that pr-values (those values attained from construction of the object as a temporary)
/// will be copy-elided all the way to the destination, instead of moved and moved and moved...
template < typename T, class Array >
   requires concepts::is_xarray< raw_t< Array >, T >
auto build_xarray(Array&& arr)
{
   return FWD(arr);
}

template < typename T, typename Array >
   requires std::ranges::range< raw_t< Array > > and (not concepts::is_xarray< raw_t< Array >, T >)
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
class TypedMultiDiscreteSpace: public TypedMonoSpace< T, TypedMultiDiscreteSpace< T > > {
  public:
   using value_type = T;
   friend class TypedMonoSpace< T, TypedMultiDiscreteSpace >;
   using base = TypedMonoSpace< T, TypedMultiDiscreteSpace >;
   using base::shape;
   using base::rng;

   template < class Array1, class Array2, typename... Args >
      requires(
         std::ranges::range< Array1 > and std::ranges::range< Array2 >
         and (not concepts::is_xarray< detail::raw_t< Array1 >, T > or not concepts::is_xarray< detail::raw_t< Array2 >, T >)
      )
   TypedMultiDiscreteSpace(Array1&& start, Array2&& end, Args&&... args)
       : TypedMultiDiscreteSpace(
            detail::build_xarray< T >(FWD(start)),
            detail::build_xarray< T >(FWD(end)),
            FWD(args)...
         )
   {
   }

   // template < class Array1, class Array2 >
   //    requires(
   //       not (concepts::is_xarray< detail::raw_t< Array1 >, T > and concepts::is_xarray<
   //       detail::raw_t< Array2 >, T >)
   //    )
   // TypedMultiDiscreteSpace(Array1&& start, Array2&& end)
   //     : TypedMultiDiscreteSpace(
   //          detail::build_xarray< T >(FWD(start)),
   //          detail::build_xarray< T >(FWD(end))
   //       )
   // {
   // }

   template < class Array, typename FirstArg, typename... TailArgs >
   TypedMultiDiscreteSpace(const Array& end, FirstArg&& any, TailArgs&&... args)
       : TypedMultiDiscreteSpace(
            xt::zeros_like(detail::build_xarray< T >(end)),
            detail::build_xarray< T >(end),
            FWD(any),
            FWD(args)...
         )
   {
   }

   template < class Array >
   TypedMultiDiscreteSpace(const Array& end)
       : TypedMultiDiscreteSpace(
            xt::zeros_like(detail::build_xarray< T >(end)),
            detail::build_xarray< T >(end)
         )
   {
   }

   template < std::integral Int >
   TypedMultiDiscreteSpace(xarray< T > start, xarray< T > end, Int seed)
       : TypedMultiDiscreteSpace(
            std::move(start),
            std::move(end),
            std::optional{static_cast< size_t >(seed)}
         )
   {
   }

   TypedMultiDiscreteSpace(
      xarray< T > start,
      xarray< T > end,
      std::optional< size_t > seed = std::nullopt
   );

   bool operator==(const TypedMultiDiscreteSpace< T >& rhs) const
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
   xarray< T > m_start;
   xarray< T > m_end;

   xarray< T > _sample(const std::vector< std::optional< xarray< bool > > >& mask_vec = {})
   {
      return _sample(1, mask_vec);
   }

   xarray< T >
   _sample(size_t nr_samples, const std::vector< std::optional< xarray< bool > > >& mask_vec = {});

   bool _contains(const T& value) const
   {
      return ranges::any_of(ranges::views::zip(m_start, m_end), [&](const auto& low_high) {
         auto&& [low, high] = low_high;
         return low <= value and high >= value;
      });
   }
};

template < std::integral T >
TypedMultiDiscreteSpace< T >::TypedMultiDiscreteSpace(
   xarray< T > start,
   xarray< T > end,
   std::optional< size_t > seed
)
    : base(xt::svector< int >(start.shape().begin(), start.shape().end()), std::move(seed)),
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
xarray< T > TypedMultiDiscreteSpace< T >::_sample(
   size_t nr_samples,
   const std::vector< std::optional< xarray< bool > > >& mask_vec
)
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
      // add all the sampling indices so that they can be emplaced all at once
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
