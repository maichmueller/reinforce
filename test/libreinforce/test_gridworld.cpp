#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/reinforce.hpp"

using namespace force;

TEST(PythonImport, numpy)
{
   // the bug https://github.com/pybind/pybind11/issues/4654 may cause pybind11 to not find the
   // numpy installation when done for the user, e.g. as happens when numpy is installed via 'pip3
   // install numpy'. This call can place packages in a ~/.local/lib/python3.xx folder.
   // Such an import will then fail to find numpy, as it appears to only look in system-wide install
   // folders, such as /usr/local/lib/python3.xx/dist-packages etc. (see the GitHub issue).
   // A workaround is to install numpy system-wide via sudo if possible, i.e.
   //    'sudo pip3 install numpy',
   // but this may well be impossible on some systems. An alternative may be to pass the python
   // executable path into CMake to use a virtual env or conda-env at the given location. This may
   // automatically include the required folders, since it isn't a /usr/bin/python install.
   // However, the latter workaround has not been tested yet.
   ASSERT_NO_THROW(py::module_::import("numpy"));
   ASSERT_NO_THROW(py::module_::import("numpy.core.multiarray"));
}

TEST(Gridworld, construction)
{
   const idx_xarray goals{{0, 0, 2}, {0, 1, 1}};
   const idx_xarray starts{{0, 1, 0}};
   auto grid_env = Gridworld< 3 >(std::vector{1, 2, 3}, starts, goals, 1.);
}

class ThreeDimensionalGridworldParamsF:
    public ::testing::TestWithParam< std::tuple<
       size_t,  // index
       std::array< size_t, 3 >  // the expected coordinates
       > > {
  public:
   constexpr static std::array< size_t, 3 > shape = {3, 4, 5};

  protected:
   Gridworld< 3 > gridworld{shape, idx_pyarray{{0, 0, 2}}, idx_pyarray{{0, 1, 0}}, 1.};
};

TEST_P(ThreeDimensionalGridworldParamsF, index_to_coordinates)
{
   auto [index, expected_coordinates] = GetParam();
   auto computed_coordinates = gridworld.coord_state(index);
   EXPECT_EQ(computed_coordinates, expected_coordinates);
}

namespace {

using paramtuple = typename ThreeDimensionalGridworldParamsF::ParamType;

INSTANTIATE_TEST_SUITE_P(
   index_to_coordinates_3d,
   ThreeDimensionalGridworldParamsF,
   ::testing::ValuesIn(std::invoke([] {
      std::vector< paramtuple > values;
      constexpr auto shape = ThreeDimensionalGridworldParamsF::shape;
      values.reserve(ranges::accumulate(shape, size_t(0), std::plus{}));
      for(size_t i = 0; i < shape[0]; i++) {
         for(size_t j = 0; j < shape[1]; j++) {
            for(size_t k = 0; k < shape[2]; k++) {
               //               fmt::print("Pre emplacement: index={} + {} + {} = {}\n", i, j, k, i
               //               * (shape[1] * shape[2]) + j * shape[2] + k);
               values.emplace_back(
                  i * (shape[1] * shape[2]) + j * shape[2] + k, std::array< size_t, 3 >{i, j, k}
               );
               //               fmt::print(
               //                  "Input: {}, Expecting: {}\n",
               //                  std::get< 0 >(values.back()),
               //                  std::get< 1 >(values.back())
               //               );
            }
         }
      }
      return values;
   }))
);

}  // namespace
