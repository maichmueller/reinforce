#define FORCE_IMPORT_ARRAY

#include <gtest/gtest.h>
#include "pybind11/embed.h"

int main(int argc, char **argv)
{
   const pybind11::scoped_interpreter guard{};
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
