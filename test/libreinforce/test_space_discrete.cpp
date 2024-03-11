#include <spdlog/spdlog.h>

#include <stdexcept>
#include <xtensor/xset_operation.hpp>

#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Discrete_constructor)
{
   constexpr int n = 10;
   constexpr int start = 0;
   EXPECT_NO_THROW((DiscreteSpace{n, start}));
   EXPECT_THROW((DiscreteSpace{0, start}), std::invalid_argument);
}

TEST(Spaces, Discrete_sample)
{
   constexpr int n = 10;
   constexpr int start = 0;
   auto space = DiscreteSpace{n, start};
   auto samples = space.sample(10000);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(xt::all(samples >= start));
   EXPECT_TRUE(xt::all(samples < start + n));

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = space.sample();
      EXPECT_TRUE(xt::all(samples >= start));
      EXPECT_TRUE(xt::all(samples < start + n));
   }
}

TEST(Spaces, Discrete_sample_masked)
{
   constexpr int n = 10;
   constexpr int start = 0;
   auto space = DiscreteSpace{n, start};
   xarray< bool > mask = {false, false, true, true, false, true, true, false, false, false};
   auto samples = space.sample(10000, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(xt::all(xt::isin(samples, xt::xarray< int >{2, 3, 5, 6})));

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = space.sample(mask);
      EXPECT_TRUE(xt::all(xt::isin(samples, xt::xarray< int >{2, 3, 5, 6})));
   }
}

TEST(Spaces, Discrete_reseeding)
{
   constexpr size_t SEED = 6492374569235;
   constexpr int n = 10;
   constexpr int start = 0;
   auto space = DiscreteSpace{n, start, SEED};
   constexpr size_t nr = 100;
   auto samples1 = space.sample(nr);
   auto samples2 = space.sample(nr);
   EXPECT_FALSE(xt::all(xt::equal(samples1, samples2)));
   space.seed(SEED);
   auto samples3 = space.sample(nr);
   auto samples4 = space.sample(nr);
   SPDLOG_DEBUG(fmt::format("Sample1:\n{}", samples1));
   SPDLOG_DEBUG(fmt::format("Sample3:\n{}", samples3));
   SPDLOG_DEBUG(fmt::format("Sample2:\n{}", samples2));
   SPDLOG_DEBUG(fmt::format("Sample4:\n{}", samples4));
   EXPECT_TRUE(xt::all(xt::equal(samples1, samples3)));
   EXPECT_TRUE(xt::all(xt::equal(samples2, samples4)));
}

TEST(Spaces, Discrete_copy_construction)
{
   constexpr int n = 10;
   constexpr int start = 0;
   DiscreteSpace space{n, start};
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
