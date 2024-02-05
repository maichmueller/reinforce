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

TEST(Space, Discrete_constructor)
{
   constexpr int n = 10;
   constexpr int start = 0;
   EXPECT_NO_THROW((TypedDiscreteSpace{n, start}));
   EXPECT_THROW((TypedDiscreteSpace{0, start}), std::invalid_argument);
}

TEST(Space, Discrete_sample)
{
   constexpr int n = 10;
   constexpr int start = 0;
   auto space = TypedDiscreteSpace{n, start};
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

TEST(Space, Discrete_sample_masked)
{
   constexpr int n = 10;
   constexpr int start = 0;
   auto space = TypedDiscreteSpace{n, start};
   xarray< bool > mask = {false, false, true, true, false, true, true, false, false, false};
   auto samples = space.sample(10000, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(xt::all(xt::isin(samples, xt::xarray< int >{2, 3, 5, 6})));

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = space.sample(mask);
      EXPECT_TRUE(xt::all(xt::isin(samples, xt::xarray< int >{2, 3, 5, 6})));
   }
}

TEST(Space, Discrete_copy_construction)
{
   constexpr int n = 10;
   constexpr int start = 0;
   TypedDiscreteSpace space{n, start};
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
