#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/space.hpp"
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
   auto sample = box.sample();
   fmt::print("{}", sample);
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
   auto sample = box.sample(10);
   fmt::print("{}", sample);
}
