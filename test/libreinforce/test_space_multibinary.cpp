#include <spdlog/spdlog.h>

#include <xtensor/xset_operation.hpp>

#include "gtest/gtest.h"
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
   constexpr int n_samples = 10000;
   auto samples = space.sample(n_samples);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
   EXPECT_FALSE(xt::all(xt::equal(samples, 0)));
   EXPECT_FALSE(xt::all(xt::equal(samples, 1)));
   EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{n_samples, 2, 3}));
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 1000)) {
      samples = space.sample();
      EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
      EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3}));
   }
}

TEST(Spaces, MultiBinary_sample_masked)
{
   auto space = MultiBinarySpace{xt::svector{2, 3}};
   // wrong mask specified
   EXPECT_THROW(space.sample(xt::xarray< int8_t >{{0, 0, 3}, {-1, 2, 2}}), std::invalid_argument);
   // valid mask
   constexpr int n_samples = 10000;
   xt::xarray< int8_t > mask = {{0, 0, 2}, {1, 2, 2}};
   auto samples = space.sample(n_samples, mask);
   SPDLOG_DEBUG(fmt::format("Samples:\n{}", samples));
   EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{n_samples, 2, 3}));

   EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
   // the entries that should be set to 0
   EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, xt::all(), 0, 0), 0)));
   EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, xt::all(), 0, 1), 0)));
   // entries that should be set to 1
   EXPECT_TRUE(xt::all(xt::equal(xt::view(samples, xt::all(), 1, 0), 1)));
   // entrees which should sample normally
   EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, xt::all(), 0, 2), {0, 1})));
   EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, xt::all(), 1, 1), {0, 1})));
   EXPECT_TRUE(xt::all(xt::isin(xt::view(samples, xt::all(), 1, 2), {0, 1})));
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 1000)) {
      samples = space.sample();
      EXPECT_TRUE(xt::all(xt::isin(samples, {0, 1})));
      EXPECT_TRUE(ranges::equal(samples.shape(), xt::svector{2, 3}));
   }
}

TEST(Spaces, MultiBinary_reseeding)
{
   constexpr size_t SEED = 6492374569235;
   auto space = MultiBinarySpace{xt::svector{2, 3}, SEED};
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

TEST(Spaces, MultiBinary_contains)
{
   auto space = MultiBinarySpace{xt::svector{2, 2}};
   int n = 1000;
   xarray< int > contain_candidates = xt::vstack(std::tuple{
                                                    xt::random::randint(xt::svector{n, 1}, 0, 2),
                                                    xt::random::randint(xt::svector{n, 1}, 0, 2),
                                                    xt::random::randint(xt::svector{n, 1}, 0, 2)
                                                 })
                                         .reshape({-1, 1, 3});
   SPDLOG_DEBUG(fmt::format(
      "Incorrect containment candidates array: {}\n{}",
      contain_candidates.shape(),
      contain_candidates
   ));
   /// candidates are missing one dimension to be part of this space
   EXPECT_FALSE(space.contains(contain_candidates));
   /// add last dimension to candiates to become part of the space
   contain_candidates = xt::concatenate(
                           std::tuple{
                              std::move(contain_candidates),
                              xt::random::randint(xt::svector{n, 1, 1}, 0, 2)
                           },
                           2
   )
                           .reshape({-1, 2, 2});
   SPDLOG_DEBUG(fmt::format(
      "Correct containment candidates array: {}\n{}", contain_candidates.shape(), contain_candidates
   ));
   EXPECT_TRUE(space.contains(contain_candidates));
   /// candidates now are not in the correct range
   contain_candidates = xt::concatenate(
                           std::tuple{
                              std::move(contain_candidates),
                              xt::random::randint(xt::svector{n, 2, 2}, 0, 10)
                           },
                           0
   )
                           .reshape({-1, 2, 2});
   SPDLOG_DEBUG(fmt::format(
      "Incorrect containment candidates array: {}\n{}",
      contain_candidates.shape(),
      contain_candidates
   ));
   EXPECT_FALSE(space.contains(contain_candidates));
   contain_candidates = xt::vstack(std::tuple{
                                      xt::random::randint(xt::svector{1, n}, 0, 2),
                                      xt::random::randint(xt::svector{1, n}, 0, 2),
                                      xt::random::randint(xt::svector{1, n}, 0, 2)
                                   })
                           .reshape({1, 3, -1});

   /// candidates have batch dim at the back, not the front.
   SPDLOG_DEBUG(fmt::format(
      "Incorrect containment candidates array: {}\n{}",
      contain_candidates.shape(),
      contain_candidates
   ));
   EXPECT_FALSE(space.contains(contain_candidates));
   contain_candidates = xt::random::randint(xt::svector{2, 2}, 0, 2);
   SPDLOG_DEBUG(fmt::format(
      "Correct containment candidates array: {}\n{}", contain_candidates.shape(), contain_candidates
   ));
   EXPECT_TRUE(space.contains(contain_candidates));
   contain_candidates = xt::random::randint(xt::svector{2, 2}, 0, 10);
   SPDLOG_DEBUG(fmt::format(
      "Incorrect containment candidates array: {}\n{}",
      contain_candidates.shape(),
      contain_candidates
   ));
   EXPECT_FALSE(space.contains(contain_candidates));
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
