#include "reinforce/spaces/multi_binary.hpp"

#include <cstddef>
#include <optional>

namespace force {

xt::svector< int > MultiBinarySpace::samples_shape(size_t batch_size) const
{
   auto out = shape();
   if(batch_size > 1) {
      prepend(out, static_cast< int >(batch_size));
   }
   SPDLOG_DEBUG(fmt::format("Samples shape: {}", out));
   return out;
}

auto MultiBinarySpace::_sample(size_t batch_size, std::nullopt_t) const -> value_type
{
   if(batch_size == 0) {
      throw std::invalid_argument("`batch_size` argument has to be greater than 0.");
   }
   return xt::random::randint(samples_shape(batch_size), 0, 2, rng());
}

auto MultiBinarySpace::_sample(size_t batch_size, const value_type& mask) const -> value_type
{
   if(batch_size == 0) {
      throw std::invalid_argument("`batch_size` argument has to be greater than 0.");
   }
   if(not ranges::equal(mask.shape(), shape())) {
      throw std::invalid_argument(fmt::format(
         "Shape of the mask ({}) needs to match shape of the space ({}).", mask.shape(), shape()
      ));
   }
   if(not (xt::all(mask < 3))) {
      throw std::invalid_argument(
         fmt::format("All values of a mask should be 0, 1 or 2, actual values: {}", mask)
      );
   }
   auto samples = xt::empty< int8_t >(samples_shape(batch_size));
   for(auto i : std::views::iota(0, ranges::accumulate(shape(), 1, std::multiplies{}))) {
      // convert the flat index i to an indexing list for the given shape
      auto coordinates = xt::unravel_index(i, shape());
      auto mask_value = mask.element(coordinates.begin(), coordinates.end());
      xt::xstrided_slice_vector index_stride(coordinates.begin(), coordinates.end());
      if(batch_size > 1) {
         prepend(index_stride, xt::all());
      }
      SPDLOG_DEBUG(fmt::format("Strides: {}", index_stride));
      if(mask_value < 2) {
         xt::strided_view(samples, index_stride) = mask_value;
      } else {
         xt::strided_view(samples, index_stride) = xt::random::randint(
            xt::svector{batch_size}, 0, 2, rng()
         );
      }
   }
   return samples;
}

}  // namespace force