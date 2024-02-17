#include <spdlog/spdlog.h>

#include <ranges>
#include <string>
#include <xtensor/xset_operation.hpp>

#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/text.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Text_constructor)
{
   EXPECT_NO_THROW((TextSpace{5, 4583048}));
   EXPECT_NO_THROW((TextSpace{{.max_length = 5}, 56356739}));
   EXPECT_NO_THROW((TextSpace{{.max_length = 5, .char_set = "AEIOUaeiou"}, 56356739}));
   EXPECT_NO_THROW(
      (TextSpace{{.max_length = 10, .min_length = 3, .char_set = "AEIOUaeiou"}, 56356739})
   );
}

TEST(Spaces, Text_sample)
{
   auto space = TextSpace{{.max_length = 5, .char_set = "+=|/{}[]()<>"}, 56356739};
   constexpr int n = 50;
   auto samples = space.sample(n);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE((std::ranges::all_of(samples, [&](const std::string& sample) {
      return std::ranges::all_of(sample, [&](char chr) {
         return ranges::contains(space.characters(), chr);
      });
   })));
   EXPECT_EQ(samples.size(), n);
   for([[maybe_unused]] auto _ : ranges::views::iota(0, n)) {
      auto sample = space.sample();
      EXPECT_TRUE(std::ranges::all_of(sample, [&](char chr) {
         return ranges::contains(space.characters(), chr);
      }));
      EXPECT_LE(sample.size(), space.max_length());
      EXPECT_GE(sample.size(), space.min_length());
   }
}
//
// TEST(Spaces, MultiBinary_sample_masked)
// {
//    auto space = MultiBinarySpace{xt::svector{2, 3}};
//    // wrong mask specified
//    EXPECT_THROW(space.sample(xt::xarray< int8_t >{{0, 0, 3}, {-1, 2, 2}}),
//    std::invalid_argument);
//    // valid mask
//    xt::xarray< int8_t > mask = {{0, 0, 2}, {1, 2, 2}};
//    auto samples = space.sample(10000, mask);
//    SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
//    EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3, 10000}));
//
//    EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
//    // the entries that should be set to 0
//    EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, 0, 0, xt::all()), 0)));
//    EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, 0, 1, xt::all()), 0)));
//    // entries that should be set to 1
//    EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, 1, 0, xt::all()), 1)));
//    // entires which should sample normally
//    EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, 0, 2, xt::all()), {0, 1})));
//    EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, 1, 1, xt::all()), {0, 1})));
//    EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, 1, 2, xt::all()), {0, 1})));
//    for([[maybe_unused]] auto _ : ranges::views::iota(0, 1000)) {
//       samples = space.sample();
//       EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
//       EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3, 1}));
//    }
// }
//
// TEST(Spaces, MultiBinary_copy_construction)
// {
//    auto space = MultiBinarySpace{xt::svector{2, 3}};
//    auto space_copy = space;
//    EXPECT_EQ(space_copy, space);
//    // RNG state should still be aligned
//    EXPECT_EQ(space_copy.sample(), space.sample());
//    // advance the rng of the copy
//    space_copy.sample();
//    // --> not all the samples should be the same anymore (statistically veee...eery unlikely)
//    auto samples = space.sample(10000);
//    auto samples_copy = space_copy.sample(10000);
//    EXPECT_NE(samples, samples_copy);
// }
