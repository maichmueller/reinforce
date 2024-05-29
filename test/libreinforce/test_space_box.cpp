#include <numbers>
#include <stdexcept>
#include <tuple>

#include "gtest/gtest.h"
#include "reinforce/spaces/box.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Box_singlevariate_constructor)
{
   EXPECT_NO_THROW((BoxSpace{-100, 100, xt::svector{10}}));
   EXPECT_NO_THROW((BoxSpace{-100, 100, xt::svector{10}, 524622}));
   EXPECT_NO_THROW((BoxSpace{-100UL, 100L, xt::svector{10}}));
   EXPECT_NO_THROW((BoxSpace{-100, short(100), xt::svector{10}, 524622}));
}

TEST(Spaces, Box_multivariates_constructor)
{
   const xarray< double > low{-inf<>, 0, -10};
   const xarray< double > high{0, inf<>, 10};
   EXPECT_NO_THROW((BoxSpace{low, high}));
   EXPECT_NO_THROW((BoxSpace{low, high, xt::svector< int >{}}));
   EXPECT_NO_THROW((BoxSpace{low, high, xt::svector{3}}));
   EXPECT_NO_THROW((BoxSpace{low, high, xt::svector{1, 3}}));
   EXPECT_NO_THROW((BoxSpace{low, high, xt::svector{1, 1, 3}}));
   EXPECT_NO_THROW((BoxSpace{low, high, xt::svector{3, 1}}));
   EXPECT_NO_THROW((BoxSpace{low, high, xt::svector{3, 1, 1}}));
   EXPECT_THROW((BoxSpace{low, high, xt::svector{0}}), std::invalid_argument);
   EXPECT_THROW((BoxSpace{low, high, xt::svector{1}}), std::invalid_argument);
   EXPECT_THROW((BoxSpace{low, high, xt::svector{2}}), std::invalid_argument);
   EXPECT_THROW((BoxSpace{low, high, xt::svector{3, 0}}), std::invalid_argument);
   EXPECT_THROW((BoxSpace{low, high, xt::svector{4, 1}}), std::invalid_argument);
}

TEST(Spaces, Box_multivariates_sample)
{
   const xarray< double > low{-inf<>, 0, -10};
   const xarray< double > high{0, inf<>, 10};
   auto box = BoxSpace< double >{low, high};
   auto samples = box.sample(10000);
   fmt::print("Samples:\n{}", samples);
   for(auto i : ranges::views::iota(0, 3)) {
      EXPECT_TRUE(xt::all(xt::view(samples, xt::all(), i) >= low(i)));
      EXPECT_TRUE(xt::all(xt::view(samples, xt::all(), i) <= high(i)));
   }
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = box.sample();
      for(auto i : ranges::views::iota(0, 3)) {
         EXPECT_TRUE(xt::all(xt::view(samples, i) >= low(i)));
         EXPECT_TRUE(xt::all(xt::view(samples, i) <= high(i)));
      }
   }
}

TEST(Spaces, Box_2D_multivariates_sample)
{
   const xarray< double > low{{-inf<>, 0, -1}, {-inf<>, 4, 1}};
   const xarray< double > high{{3, inf<>, 0}, {7, 5, 11}};
   auto box = BoxSpace{low, high};
   auto samples = box.sample(10000);
   fmt::print("Samples:\n{}", samples);
   for(auto [i, j] :
       ranges::views::cartesian_product(ranges::views::iota(0, 2), ranges::views::iota(0, 3))) {
      //      SPDLOG_DEBUG(fmt::format(
      //         "Low: {}, High: {}, values: {}", low(i, j), high(i, j), xt::view(samples,
      //         xt::all(), i, j)
      //      ));
      EXPECT_TRUE(xt::all(xt::view(samples, xt::all(), i, j) >= low(i, j)));
      EXPECT_TRUE(xt::all(xt::view(samples, xt::all(), i, j) <= high(i, j)));
   }
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = box.sample();
      for(auto [i, j] :
          ranges::views::cartesian_product(ranges::views::iota(0, 2), ranges::views::iota(0, 3))) {
         EXPECT_GE(samples(i, j), low(i, j));
         EXPECT_LE(samples(i, j), high(i, j));
      }
   }
}

TEST(Spaces, Box_bounds)
{
   const xarray< double > low{{-inf<>, 0, -1}, {-inf<>, 4, 1}};
   const xarray< double > high{{3, inf<>, 0}, {7, 5, 11}};
   auto box = BoxSpace< double >{low, high};
   EXPECT_EQ((std::pair{-inf<>, 3.}), box.bounds({0, 0}));
   EXPECT_EQ((std::pair{0., inf<>}), box.bounds({0, 1}));
   EXPECT_EQ((std::pair{-1., 0.}), box.bounds({0, 2}));
   EXPECT_EQ((std::pair{-inf<>, 7.}), box.bounds({1, 0}));
   EXPECT_EQ((std::pair{4., 5.}), box.bounds({1, 1}));
   EXPECT_EQ((std::pair{1., 11.}), box.bounds({1, 2}));

   EXPECT_EQ((std::pair{1., 11.}), box.bounds(std::array{1, 2}));
   EXPECT_EQ((std::pair{1., 11.}), box.bounds(std::vector{1, 2}));
}

TEST(Spaces, Box_reseeding)
{
   constexpr size_t SEED = 6492374569235;
   const xarray< double > low{-inf<>, 0, -10};
   const xarray< double > high{0, inf<>, 10};
   BoxSpace space{low, high, low.shape(), SEED};
   size_t n = 1000;
   auto samples1 = space.sample(n);
   auto samples2 = space.sample(n);
   EXPECT_FALSE(xt::all(xt::equal(samples1, samples2)));
   space.seed(SEED);
   auto samples3 = space.sample(n);
   auto samples4 = space.sample(n);
   SPDLOG_DEBUG(fmt::format("Sample1:\n{}", samples1));
   SPDLOG_DEBUG(fmt::format("Sample3:\n{}", samples3));
   SPDLOG_DEBUG(fmt::format("Sample2:\n{}", samples2));
   SPDLOG_DEBUG(fmt::format("Sample4:\n{}", samples4));
   EXPECT_TRUE(xt::all(xt::equal(samples1, samples3)));
   EXPECT_TRUE(xt::all(xt::equal(samples2, samples4)));
}

TEST(Spaces, Box_contains)
{
   constexpr size_t SEED = 6492374569235;
   const xarray< double > low{-inf<>, 0, 50, 1.1};
   const xarray< double > high{0, inf<>, 51, std::numbers::e};
   BoxSpace space{low, high, low.shape(), SEED};
   int n = 1000;
   xarray< double > contain_candidates = xt::hstack(
                                            std::tuple{
                                               -xt::random::exponential(xt::svector{n, 1}, 10.),
                                               xt::random::exponential(xt::svector{n, 1}, 100.),
                                               xt::random::rand(xt::svector{n, 1}, low(2), high(2))
                                            }
   )
                                            .reshape({n, 1, 3});
   SPDLOG_DEBUG(fmt::format(
      "Incorrect containment candidates array: {}\n{}",
      contain_candidates.shape(),
      contain_candidates
   ));
   /// candidates are missing one dimension to be part of this space
   EXPECT_FALSE(space.contains(contain_candidates));
   /// add last dimension to candiates to become part of the space
   contain_candidates = xt::concatenate(
                           xt::xtuple(
                              std::move(contain_candidates),
                              xt::random::rand(xt::svector{n, 1, 1}, low(3), high(3))
                           ),
                           2
   )
                           .reshape({-1, 4});
   SPDLOG_DEBUG(fmt::format(
      "Correct containment candidates array: {}\n{}", contain_candidates.shape(), contain_candidates
   ));
   EXPECT_TRUE(space.contains(contain_candidates));
}

TEST(Spaces, Box_copy_construction)
{
   const xarray< double > low{-inf<>, 0, -10};
   const xarray< double > high{0, inf<>, 10};
   BoxSpace< double > space{low, high};
   auto space_copy = space;
   EXPECT_EQ(space_copy, space);
   EXPECT_EQ(space.bounds({0}), space_copy.bounds({0}));
   // RNG state should still be aligned
   EXPECT_EQ(space_copy.sample(), space.sample());
   // now the copy has an advanced rng
   std::ignore = space_copy.sample();
   // the samples now should no longer be the same
   EXPECT_NE(space_copy.sample(), space.sample());
}
