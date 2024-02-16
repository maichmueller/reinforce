#include "reinforce/spaces/multi_binary.hpp"

namespace force {

auto MultiBinarySpace::_sample(size_t nr_samples, const std::optional< value_type >& mask_opt)
   -> value_type
{
   if(nr_samples == 0) {
      throw std::invalid_argument("`nr_samples` argument has to be greater than 0.");
   }
   xt::svector< int > samples_shape = shape();
   samples_shape.push_back(static_cast< int >(nr_samples));
   SPDLOG_DEBUG(fmt::format("Samples shape: {}", samples_shape));
   if(not mask_opt.has_value()) {
      return {xt::random::randint(samples_shape, 0, 2, rng())};
   }
   const auto& mask = *mask_opt;
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
   auto samples = xt::empty< int8_t >(samples_shape);
   for(auto i : std::views::iota(0, ranges::accumulate(shape(), 1, std::multiplies{}))) {
      // convert the flat index i to an indexing list for the given shape
      auto coordinates = xt::unravel_index(i, shape());
      auto mask_value = mask.element(coordinates.begin(), coordinates.end());
      // add all entries of the variate's access in the shape
      xt::xstrided_slice_vector index_stride(coordinates.begin(), coordinates.end());
      // add all the sampling indices so that they can be emplaced all at once
      index_stride.emplace_back(xt::all());
      SPDLOG_DEBUG(fmt::format("Strides: {}", index_stride));
      if(mask_value < 2) {
         xt::strided_view(samples, index_stride) = mask_value;
      } else {
         xt::strided_view(samples, index_stride) = xt::random::randint(
            xt::svector{nr_samples}, 0, 2, rng()
         );
      }
   }
   return samples;
}
}  // namespace force