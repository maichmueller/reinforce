#include <cstddef>
#include <reinforce/spaces/box.hpp>
#include <reinforce/spaces/discrete.hpp>

#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/spaces/graph.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

using namespace force;

TEST(Spaces, Graph_Discrete_Discrete_construction)
{
   EXPECT_NO_THROW((GraphSpace{DiscreteSpace{10, 0}}));
   EXPECT_NO_THROW((GraphSpace{DiscreteSpace{10, 0}, DiscreteSpace{5, 0}}));
   EXPECT_NO_THROW((GraphSpace{DiscreteSpace{10, 0}, DiscreteSpace{5, 0}, 45466}));
}
