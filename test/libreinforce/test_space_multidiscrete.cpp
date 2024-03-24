#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>

#include <xtensor/xset_operation.hpp>

#include "reinforce/spaces/multi_discrete.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, MultiDiscrete_constructor)
{
   auto start = xarray< int >{0, 0, -2};
   auto end = xarray< int >{10, 5, 3};
   EXPECT_NO_THROW((MultiDiscreteSpace{start, end}));
   EXPECT_NO_THROW((MultiDiscreteSpace{end}));
   // construct with seed
   EXPECT_NO_THROW((MultiDiscreteSpace{start, end, 42}));
   EXPECT_NO_THROW((MultiDiscreteSpace{end, 42}));
}

TEST(Spaces, MultiDiscrete_sample)
{
   auto start = xarray< int >{{-5, -4, -1}, {0, 0, 2}};
   auto end = xarray< int >{{-2, 4, 1}, {10, 5, 3}};
   auto space = MultiDiscreteSpace{start, end};
   constexpr auto n_samples = 1000;
   auto samples = space.sample(n_samples);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(xt::all(samples >= xt::expand_dims(start, 2)));
   EXPECT_TRUE(xt::all(samples < xt::expand_dims(end, 2)));

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = space.sample();
      auto samples_squeezed = xt::squeeze(samples, static_cast< size_t >(-1));
      SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples_squeezed));
      EXPECT_TRUE(xt::all(samples_squeezed >= start));
      EXPECT_TRUE(xt::all(samples_squeezed < end));
   }
}

TEST(Spaces, MultiDiscrete_sample_masked)
{
   constexpr auto n_samples = 10000;
   auto start = xarray< int >{0, 0, -2};
   auto end = xarray< int >{10, 5, 3};
   auto space = MultiDiscreteSpace{start, end};
   auto mask = std::vector< std::optional< xarray< bool > > >{
      xarray< bool >{false, false, false, false, false, true, true, true, true, true, true},
      std::nullopt,
      xarray< bool >{false, true, true, true, false}
   };
   auto samples = space.sample(n_samples, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE((
      xt::all(xt::isin(xt::strided_view(samples, {0, xt::all()}), xt::xarray< int >{5, 6, 7, 8, 9}))
   ));
   EXPECT_TRUE((
      xt::all(xt::isin(xt::strided_view(samples, {1, xt::all()}), xt::xarray< int >{0, 1, 2, 3, 4}))
   ));
   EXPECT_TRUE(
      (xt::all(xt::isin(xt::strided_view(samples, {2, xt::all()}), xt::xarray< int >{-1, 0, 1})))
   );

   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = space.sample(mask);
      SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
      EXPECT_GE(samples(0, 0), 5);
      EXPECT_LE(samples(0, 0), 9);
      EXPECT_GE(samples(1, 0), 0);
      EXPECT_LE(samples(1, 0), 4);
      EXPECT_GE(samples(2, 0), -1);
      EXPECT_LE(samples(2, 0), 1);
   }
}

TEST(Spaces, MultiDiscrete_reseeding)
{
   constexpr size_t SEED = 6492374569235;
   auto start = xarray< int >{0, 0, -2};
   auto end = xarray< int >{10, 5, 3};
   auto space = MultiDiscreteSpace{start, end, SEED};
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

TEST(Spaces, MultiDiscrete_contains)
{
   auto start = xarray< int >{0, 0, -2};
   auto end = xarray< int >{10, 5, 3};
   auto space = MultiDiscreteSpace{start, end};
   int n = 1000;
   xarray< int >
      contain_candidates = xt::vstack(std::tuple{
                                         xt::random::randint(xt::svector{1, n}, start(0), end(0)),
                                         xt::random::randint(xt::svector{1, n}, start(1), end(1)),
                                         xt::random::randint(xt::svector{1, n}, start(2), end(2))
                                      })
                              .reshape({n / 2, 3, n / 2});
   SPDLOG_DEBUG(fmt::format(
      "Incorrect containment candidates array (wrong shape): {}\n{}",
      contain_candidates.shape(),
      contain_candidates
   ));
   /// candidates are missing one dimension to be part of this space
   EXPECT_FALSE(space.contains(contain_candidates));
   /// not within bounds
   EXPECT_FALSE(space.contains(contain_candidates + xt::random::randint({1}, -100, 100)));
   /// reshape to correct shape to become part of this space
   SPDLOG_DEBUG(fmt::format(
      "Correct containment candidates array: {}\n{}",
      contain_candidates.reshape({3, -1}).shape(),
      contain_candidates.reshape({3, -1})
   ));
   EXPECT_TRUE(space.contains(contain_candidates.reshape({3, -1})));
}

TEST(Spaces, MultiDiscrete_copy_construction)
{
   auto start = xarray< int >{0, 0, -2};
   auto end = xarray< int >{10, 5, 3};
   MultiDiscreteSpace space{start, end};
   auto space_copy = space;
   EXPECT_EQ(space_copy, space);
   // RNG state should still be aligned
   EXPECT_EQ(space_copy.sample(), space.sample());
   // now the copy has an advanced rng
   space_copy.sample();
   // the samples now should no longer be the same
   EXPECT_NE(space_copy.sample(), space.sample());
}
