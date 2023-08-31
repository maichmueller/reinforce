#include "reinforce/reinforce.hpp"
#include "gtest/gtest.h"

using namespace force;

TEST(Gridworld, construction) {
   GridWorld<2>({2, 3}, np::index_array{}, np::index_array{}, 1.);
}