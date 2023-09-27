#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/space.hpp"
#include "reinforce/spaces/box.hpp"
#include "reinforce/spaces/discrete.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"
#include "reinforce/utils/math.hpp"

using namespace force;

TEST(Space, Box_constructor)
{
   const xarray<double> low{-infinity<>, 0, -10};
   const xarray<double> high{0, infinity<>, 10};
   EXPECT_NO_THROW((TypedBox< double >{low, high}));
   EXPECT_NO_THROW((TypedBox< double >{low, high, xt::svector<int>{}}));
   EXPECT_NO_THROW((TypedBox< double >{low, high, xt::svector{3}}));
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{0}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{1}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{2}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{3, 0}}), std::invalid_argument);
   EXPECT_THROW((TypedBox< double >{low, high, xt::svector{1, 3}}), std::invalid_argument);
}
