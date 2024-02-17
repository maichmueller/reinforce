#include "reinforce/spaces/text.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace force {

xarray< size_t > TextSpace::_compute_lengths(
   size_t nr_samples,
   const std::optional< std::variant< size_t, std::vector< size_t > > >& opt_len
)
{
   if(opt_len.has_value()) {
      return std::visit(
         detail::overload{
            [&](size_t len) {
               return xarray< size_t >{xt::adapt(
                  detail::make_carray< size_t >(nr_samples, len).first.release(),
                  nr_samples,
                  xt::acquire_ownership()
               )};
            },
            [&](const std::vector< size_t >& lens) {
               xarray< size_t > arr = xt::empty< size_t >({lens.size()});
               for(auto index : std::views::iota(0u, lens.size())) {
                  arr.unchecked(index) = lens[index];
               }
               return arr;
            }
         },
         *opt_len
      );
   }
   return xt::random::randint(xt::svector{nr_samples}, m_min_length, m_max_length + 1, rng());
}

const xt::xarray< char >& TextSpace::_default_chars()
{
   static auto arr = [] {
      constexpr static char
         default_charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
      // we should be able to do this next commented out line
      //
      //    static xarray< size_t > arr =  xt::adapt(default_charset, default_charset_len);
      //
      // but the xtensor lib runs into compilation error bugs with xadapt so we have to copy
      // manually instead...
      auto array = xt::xarray< char >::from_shape({sizeof(default_charset)});
      for(auto index : std::views::iota(0u, sizeof(default_charset))) {
         array.unchecked(index) = default_charset[index];
      }
      return array;
   }();
   return arr;
}

}  // namespace force