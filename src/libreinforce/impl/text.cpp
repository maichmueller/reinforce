#include "reinforce/spaces/text.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace force {


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