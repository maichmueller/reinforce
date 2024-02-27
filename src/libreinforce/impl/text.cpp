#include "reinforce/spaces/text.hpp"

namespace force {

const xarray< char >& TextSpace::_default_chars()
{
   static auto arr = [] {
      // we should be able to do this next commented out line
      //
      //    static xarray< size_t > arr =  xt::adapt(default_charset, default_charset_len);
      //
      // but the xtensor lib runs into compilation error bugs with xadapt so we have to copy
      // manually instead...
      auto array = xt::xarray< char >::from_shape({sizeof(default_characters)});
      for(auto index : std::views::iota(0u, sizeof(default_characters))) {
         array.unchecked(index) = default_characters[index];
      }
      return array;
   }();
   return arr;
}

std::unordered_map< char, size_t > TextSpace::make_charmap(const xarray< char >& chars)
{
   return chars  //
          | ranges::views::enumerate  //
          | std::views::transform([](auto&& pair) {
               return std::pair{pair.second, pair.first};  // index, char -> char, index
            })
          |  //
          ranges::to< std::unordered_map< char, size_t > >;
}

const std::unordered_map< char, size_t >& TextSpace::_default_charmap()
{
   static auto arr = make_charmap(_default_chars());
   return arr;
}

}  // namespace force