#include <gtest/gtest.h>

#include "pybind11/embed.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

int main(int argc, char **argv)
{
   const pybind11::scoped_interpreter guard{};

   // needs to be done once to ensure numpy is imported
   xt::import_numpy();

   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
