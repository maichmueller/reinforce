#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <span>

#ifdef REINFORCE_USE_PYTHON
   #include "pybind11/embed.h"

   #define FORCE_IMPORT_ARRAY
   #include "xtensor-python/pyarray.hpp"
#endif

int main(int argc, char** argv)
{
   auto cmd_args = std::span{argv, static_cast< size_t >(argc)};
   // setting the level is needed on top of setting the level with the macro (for some reason)
   spdlog::set_level(spdlog::level::level_enum{SPDLOG_ACTIVE_LEVEL});
#ifdef REINFORCE_USE_PYTHON
   // start up a python interpreter to be used by xtensor-python and numpy calls
   const pybind11::scoped_interpreter guard{};
   // needs to be done once to ensure numpy is available
   xt::import_numpy();
#endif
   // now run all defined GTests
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}