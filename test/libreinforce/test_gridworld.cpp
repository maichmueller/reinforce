#include "reinforce/reinforce.hpp"
#include "gtest/gtest.h"

using namespace force;

TEST(Gridworld, construction) {
   Gridworld<2>({2, 3}, idx_pyarray{}, idx_pyarray{}, 1.);
}