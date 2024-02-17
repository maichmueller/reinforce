#include <spdlog/spdlog.h>
#include <xtensor/xset_operation.hpp>

#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/multi_binary.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, MultiBinary_constructor)
{
   EXPECT_NO_THROW((MultiBinarySpace{std::vector{3, 2, 1}}));
   EXPECT_NO_THROW((MultiBinarySpace{xt::svector{2, 3}}));
   EXPECT_NO_THROW((MultiBinarySpace{xt::xarray< size_t >{1, 2, 3}}));
   EXPECT_NO_THROW((MultiBinarySpace{{1, 2, 3, 4, 5, 6, 7, 8}}));
}

TEST(Spaces, MultiBinary_sample)
{
   auto space = MultiBinarySpace{xt::svector{2, 3}};
   auto samples = space.sample(10000);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
   EXPECT_FALSE(xt::all(xt::equal(samples, 0)));
   EXPECT_FALSE(xt::all(xt::equal(samples, 1)));
   EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3, 10000}));
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 1000)) {
      samples = space.sample();
      EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
      EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3, 1}));
   }
}

TEST(Spaces, MultiBinary_sample_masked)
{
   auto space = MultiBinarySpace{xt::svector{2, 3}};
   // wrong mask specified
   EXPECT_THROW(space.sample(xt::xarray< int8_t >{{0, 0, 3}, {-1, 2, 2}}), std::invalid_argument);
   // valid mask
   xt::xarray< int8_t > mask = {{0, 0, 2}, {1, 2, 2}};
   auto samples = space.sample(10000, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3, 10000}));

   EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
   // the entries that should be set to 0
   EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, 0, 0, xt::all()), 0)));
   EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, 0, 1, xt::all()), 0)));
   // entries that should be set to 1
   EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, 1, 0, xt::all()), 1)));
   // entires which should sample normally
   EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, 0, 2, xt::all()), {0, 1})));
   EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, 1, 1, xt::all()), {0, 1})));
   EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, 1, 2, xt::all()), {0, 1})));
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 1000)) {
      samples = space.sample();
      EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
      EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3, 1}));
   }
}

TEST(Spaces, MultiBinary_copy_construction)
{
   auto space = MultiBinarySpace{xt::svector{2, 3}};
   auto space_copy = space;
   EXPECT_EQ(space_copy, space);
   // RNG state should still be aligned
   EXPECT_EQ(space_copy.sample(), space.sample());
   // advance the rng of the copy
   space_copy.sample();
   // --> not all the samples should be the same anymore (statistically veee...eery unlikely)
   auto samples = space.sample(10000);
   auto samples_copy = space_copy.sample(10000);
   EXPECT_NE(samples, samples_copy);
}
