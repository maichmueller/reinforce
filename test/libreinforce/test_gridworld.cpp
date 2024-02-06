#include <array>
#include <cstddef>
#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/reinforce.hpp"

using namespace force;
namespace py = pybind11;

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

class Gridworld3DimF {
  public:
   constexpr static std::array< size_t, 3 > shape = {3, 4, 5};

  protected:
   Gridworld< 3 > gridworld{shape, idx_pyarray{{0, 0, 2}}, idx_pyarray{{0, 1, 0}}, 1.};
};

class IndexCoordinatesMappingParamsF: public Gridworld3DimF, public ::testing::Test {};

class ParameterizedIndexCoordinatesMappingParamsF:
    public Gridworld3DimF,
    public ::testing::TestWithParam< std::tuple<
       size_t,  // index
       std::array< size_t, 3 >  // the associated coordinates
       > > {};

TEST_P(ParameterizedIndexCoordinatesMappingParamsF, index_to_coordinates)
{
   auto [index, expected_coordinates] = GetParam();
   auto computed_coordinates = gridworld.coord_state(index);
   EXPECT_TRUE(ranges::equal(computed_coordinates, expected_coordinates));
}

TEST_P(ParameterizedIndexCoordinatesMappingParamsF, coordinates_to_index)
{
   auto [expected_index, coordinates] = GetParam();
   auto computed_index = gridworld.index_state(coordinates);
   EXPECT_EQ(expected_index, computed_index);
}

auto idx_to_coords_parameters()
{
   std::vector< typename ParameterizedIndexCoordinatesMappingParamsF::ParamType > values;
   constexpr auto shape = IndexCoordinatesMappingParamsF::shape;
   values.reserve(ranges::accumulate(shape, size_t(0), std::plus{}));
   for(size_t i = 0; i < shape[0]; i++) {
      for(size_t j = 0; j < shape[1]; j++) {
         for(size_t k = 0; k < shape[2]; k++) {
            values.emplace_back(
               i * (shape[1] * shape[2]) + j * shape[2] + k, std::array< size_t, 3 >{i, j, k}
            );
         }
      }
   }
   return values;
}

INSTANTIATE_TEST_SUITE_P(
   index_to_coordinates_serial,
   ParameterizedIndexCoordinatesMappingParamsF,
   ::testing::ValuesIn(idx_to_coords_parameters())
);

TEST_F(IndexCoordinatesMappingParamsF, index_to_coordinates_batch)
{
   auto values = idx_to_coords_parameters();
   auto indices = ranges::to_vector(values | ranges::views::transform([](auto&& tuple) {
                                       return std::get< 0 >(tuple);
                                    }));
   auto computed_coordinates = gridworld.coord_state(indices);
   for(auto [res_idx, inp_idx] : ranges::views::enumerate(indices)) {
      EXPECT_TRUE(
         ranges::equal(xt::row(computed_coordinates, long(res_idx)), gridworld.coord_state(inp_idx))
      );
   }
}

class ParamterizedTerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       size_t,  // state index
       bool  // whether the state is terminal
       > > {
  public:
   constexpr static std::array< size_t, 3 > shape = {3, 4, 5};

  protected:
   Gridworld< 3 > gridworld{
      shape,
      idx_pyarray{{0, 0, 2}},
      /*goal_states=*/idx_pyarray{{1, 2, 3}, {2, 2, 2}},
      1.
   };
};

TEST_P(ParamterizedTerminalParamsF, is_terminal)
{
   auto [index, expected_terminal] = GetParam();
   auto answer_for_index = gridworld.is_terminal(index);
   auto answer_for_coords = gridworld.is_terminal(gridworld.coord_state(index));
   EXPECT_EQ(answer_for_index, answer_for_coords);
   EXPECT_EQ(answer_for_index, expected_terminal);
}

INSTANTIATE_TEST_SUITE_P(
   is_terminal_,
   ParamterizedTerminalParamsF,
   ::testing::ValuesIn(std::invoke([] {
      std::vector< typename ParamterizedTerminalParamsF::ParamType > values;
      const std::vector< std::array< size_t, 3 > > chosen_goals{{1, 2, 3}, {2, 2, 2}};
      constexpr auto shape = ParamterizedTerminalParamsF::shape;
      values.reserve(ranges::accumulate(shape, size_t(0), std::plus{}));
      for(size_t i = 0; i < shape[0]; i++) {
         for(size_t j = 0; j < shape[1]; j++) {
            for(size_t k = 0; k < shape[2]; k++) {
               values.emplace_back(
                  i * (shape[1] * shape[2]) + j * shape[2] + k,
                  ranges::contains(chosen_goals, std::array{i, j, k})
               );
            }
         }
      }
      return values;
   }))
);

class Gridworld2DimF {
  public:
   constexpr static std::array< size_t, 2 > shape = {4, 5};

  protected:
   Gridworld< 2 > gridworld{shape, idx_pyarray{{0, 2}}, idx_pyarray{{3, 0}}, 1.};
};

class Gridworld2DimStepF: public Gridworld2DimF, public ::testing::Test {};

TEST_F(Gridworld2DimStepF, step)
{
   SPDLOG_DEBUG(fmt::format(
      "Gridworld for `step` function test:\n"
      "Shape: {}\n"
      "Start states: {}\n"
      "Goal states: {}\n"
      "Transition Tensor: {}\n",
      gridworld.shape(),
      gridworld.start_states(),
      gridworld.goal_states(),
      gridworld.transition_tensor()
   ));
   // start at {0,2}
   // comments format: ACTION NEXT_LOCATION
   for(auto action : std::vector< size_t >{
          0,  // left  {0,2} (out of bounds)
          1,  // right {1,2}
          0,  // left  {0,2}
          2,  // down  {0,1}
          3,  // up    {0,2}
          3,  // up    {0,3}
          3,  // up    {0,4}
          1,  // right {1,4}
          1,  // right {2,4}
          1,  // right {3,4}
          1,  // right {4,4}
          2,  // down  {4,3}
          2,  // down  {4,2}
          2,  // down  {4,1}
          2,  // down  {4,0}
          0,  // left  {3,0}
          3,  // up    {3,1}
       }) {
      SPDLOG_DEBUG(fmt::format(
         "Location before: {}, Action index: {}, action name: {}\n",
         gridworld.location(),
         action,
         gridworld.action_name(action)
      ));
      auto [observation, reward, terminated, truncated] = gridworld.step(action);
      SPDLOG_DEBUG(fmt::format("Location after:  {}\n", gridworld.location()));
      if(terminated) {
         SPDLOG_DEBUG(fmt::format("Goal state reached. Terminated."));
         break;
      }
   }
}