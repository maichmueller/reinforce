#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/mono_space.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Space, Box_single_variates_constructor)
{
   const xarray< double > low{-infinity<>, 0, -10};
   const xarray< double > high{0, infinity<>, 10};
   EXPECT_NO_THROW((TypedBox< double >{low, high}));
   EXPECT_NO_THROW((TypedBox< double >{low, high, xt::svector< int >{}}));
   EXPECT_NO_THROW((TypedBox< double >{low, high, xt::svector{3}}));
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{0}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{1}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{2}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{3, 0}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{1, 3}}), std::invalid_argument);
}

TEST(Space, Box_single_variates_sample)
{
   const xarray< double > low{-infinity<>, 0, -10};
   const xarray< double > high{0, infinity<>, 10};
   auto box = TypedBox< double >{low, high};
   auto samples = box.sample(10000);
   fmt::print("Samples:\n{}", samples);
   for(auto i : ranges::views::iota(0, 3)) {
      EXPECT_TRUE(xt::all(xt::view(samples, i, xt::all()) >= low(i)));
      EXPECT_TRUE(xt::all(xt::view(samples, i, xt::all()) <= high(i)));
   }
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = box.sample();
      for(auto i : ranges::views::iota(0, 3)) {
         EXPECT_TRUE(xt::all(xt::view(samples, i) >= low(i)));
         EXPECT_TRUE(xt::all(xt::view(samples, i) <= high(i)));
      }
   }
}

TEST(Space, Box_multi_variates_constructor)
{
   const xarray< double > low{{-infinity<>, 0, -10}, {-infinity<>, 100, 10}};
   const xarray< double > high{{5, infinity<>, 10}, {0, 101, 11}};
   EXPECT_NO_THROW((TypedBox< double >{low, high}));
   EXPECT_NO_THROW((TypedBox< double >{low, high, xt::svector< int >{}}));
   EXPECT_NO_THROW((TypedBox< double >{low, high, xt::svector{2, 3}}));
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{0}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{1}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{2}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{3, 0}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{1, 3}}), std::invalid_argument);
}

TEST(Space, Box_multi_variates_sample)
{
   const xarray< double > low{{-infinity<>, 0, -1}, {-infinity<>, 4, 1}};
   const xarray< double > high{{3, infinity<>, 0}, {7, 5, 11}};
   auto box = TypedBox< double >{low, high};
   auto samples = box.sample(10000);
   fmt::print("Samples:\n{}", samples);
   for(auto [i, j] :
       ranges::views::cartesian_product(ranges::views::iota(0, 2), ranges::views::iota(0, 3))) {
      EXPECT_TRUE(xt::all(xt::view(samples, i, j, xt::all()) >= low(i, j)));
      EXPECT_TRUE(xt::all(xt::view(samples, i, j, xt::all()) <= high(i, j)));
   }
   for([[maybe_unused]] auto _ : ranges::views::iota(0, 100)) {
      samples = box.sample();
      for(auto [i, j] :
          ranges::views::cartesian_product(ranges::views::iota(0, 2), ranges::views::iota(0, 3))) {
         EXPECT_TRUE(xt::all(xt::view(samples, i, j, xt::all()) >= low(i, j)));
         EXPECT_TRUE(xt::all(xt::view(samples, i, j, xt::all()) <= high(i, j)));
      }
   }
}

TEST(Space, Box_bounds)
{
   const xarray< double > low{{-infinity<>, 0, -1}, {-infinity<>, 4, 1}};
   const xarray< double > high{{3, infinity<>, 0}, {7, 5, 11}};
   auto box = TypedBox< double >{low, high};
   EXPECT_EQ((std::pair{-infinity<>, 3.}), box.bounds({0, 0}));
   EXPECT_EQ((std::pair{0., infinity<>}), box.bounds({0, 1}));
   EXPECT_EQ((std::pair{-1., 0.}), box.bounds({0, 2}));
   EXPECT_EQ((std::pair{-infinity<>, 7.}), box.bounds({1, 0}));
   EXPECT_EQ((std::pair{4., 5.}), box.bounds({1, 1}));
   EXPECT_EQ((std::pair{1., 11.}), box.bounds({1, 2}));

   EXPECT_EQ((std::pair{1., 11.}), box.bounds(std::array{1, 2}));
   EXPECT_EQ((std::pair{1., 11.}), box.bounds(std::vector{1, 2}));
}

TEST(Space, Box_copy_construction)
{
   const xarray< double > low{-infinity<>, 0, -10};
   const xarray< double > high{0, infinity<>, 10};
   TypedBox< double > space{low, high};
   auto space_copy = space;
   EXPECT_EQ(space_copy, space);
   EXPECT_EQ(space.bounds({0}), space_copy.bounds({0}));
   // RNG state should still be aligned
   EXPECT_EQ(space_copy.sample(), space.sample());
   // now the copy has an advanced rng
   space_copy.sample();
   // the samples now should no longer be the same
   EXPECT_NE(space_copy.sample(), space.sample());

}
