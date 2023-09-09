
#ifndef REINFORCE_GRIDWORLD_HPP
#define REINFORCE_GRIDWORLD_HPP

#include <valarray>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "optional"
#include "pybind11/numpy.h"
#include "range/v3/all.hpp"
#include "reinforce/utils/utils.hpp"
#include "variant"

#include <xtensor/xaxis_slice_iterator.hpp>

#include "xtensor-python/pyarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

namespace force {

namespace py = pybind11;

template < typename T >
using xarray = xt::xarray< T, xt::layout_type::row_major >;
template < typename T >
using pyarray = xt::pyarray< T, xt::layout_type::row_major >;

using idx_xarray = xt::xarray< size_t, xt::layout_type::row_major >;
using idx_pyarray = xt::pyarray< size_t, xt::layout_type::row_major >;

// namespace helper {
// template < typename T, typename integer_constant >
//    requires utils::is_convertible_without_narrowing_v< typename integer_constant::type, size_t >
// struct array: std::array< T, integer_constant::value > {};
// }  // namespace helper

namespace np {

template < typename T >
using array = py::array_t< T >;

using index_array = py::array_t< size_t >;

}  // namespace np

template < typename ExpectedType, typename Range >
concept expected_value_type = requires(Range rng) {
   {
      *(rng.begin())
   } -> std::convertible_to< ExpectedType >;
};

struct CoordinateHasher {
   using is_transparent = std::true_type;

   template < ranges::range Range >
   size_t operator()(const Range &coords) const noexcept
      requires expected_value_type< size_t, Range >
   {
      constexpr auto string_hasher = std::hash< std::string >{};
      std::stringstream ss;
      std::copy(coords.begin(), coords.end(), std::ostream_iterator< size_t >(ss, ","));
      return string_hasher(ss.str());
   }
};

enum class StateType { default_ = 0, goal = 1, subgoal = 2, start = 3, restart = 4, obstacle = 5 };

template < size_t dim >
class Gridworld {
  public:
   using reward_map_type = std::
      unordered_map< std::vector< size_t >, std::pair< StateType, double >, CoordinateHasher >;

   /**
    * @brief Construct a GridWorld instance.
    *
    * @param num_rows The number of rows in the gridworld.
    * @param num_cols The number of columns in the gridworld.
    * @param start_states The start states of the gridworld.
    * @param goal_states The goal states for the gridworld (m <= n).
    * @param step_reward The reward for each step taken by the agent.
    * @param goal_reward The reward for reaching a goal state.
    * @param subgoal_state_reward The reward for transitioning to a subgoal state.
    * @param restart_state_reward The reward for transitioning to a restart state.
    * @param transition_matrix The probability or probability matrix of successfully transitioning
    * states.
    * @param bias The probability of transitioning left or right if not successful.
    * @param obstacle_states States the agent cannot enter.
    * @param subgoal_states States where the agent incurs high subgoal.
    * @param restart_states States where the agent transitions to start.
    */
   template < std::integral I, typename... Args >
   Gridworld(std::initializer_list< I > shape, Args &&...args)
       : Gridworld(shape.begin(), shape.end(), std::forward< Args >(args)...)
   {
   }
   template < std::forward_iterator FwdIter, typename... Args >
      requires std::convertible_to< size_t, std::iter_value_t< FwdIter > >
   Gridworld(FwdIter shape_begin, FwdIter shape_end, Args &&...args)
       : Gridworld(detail::RangeAdaptor{shape_begin, shape_end}, std::forward< Args >(args)...)
   {
   }
   template < ranges::range Range >
      requires expected_value_type< size_t, Range >
   Gridworld(
      const Range &shape,
      const idx_pyarray &start_states,
      const idx_pyarray &goal_states,
      std::variant< double, pyarray< double > > goal_reward,
      double step_reward = 0.,
      std::variant< double, pyarray< double > > transition_matrix = double{1.},
      std::optional< idx_pyarray > subgoal_states = {},
      std::variant< double, pyarray< double > > subgoal_states_reward = double{0.},
      std::optional< idx_pyarray > obs_states = {},
      std::optional< idx_pyarray > restart_states = {},
      double restart_states_reward = 0.
   );

   [[nodiscard]] auto coord_state(size_t state_index) const;
   [[nodiscard]] auto index_state(std::span< size_t > coordinates) const;

  private:
   constexpr static size_t m_num_actions = 2 * dim;
   // the lengths of each grid dimension
   std::array< size_t, dim > m_grid_shape;
   // the cumulative product shape from the last dimension to the 0th dimension
   std::array< size_t, dim > m_grid_shape_products;
   /// shape (n, DIM)
   idx_xarray m_start_states;
   idx_xarray m_goal_states;
   idx_xarray m_subgoal_states;
   idx_xarray m_obs_states;
   idx_xarray m_restart_states;
   /// shape (s_x1, s_x2, s_x..., s_xdim, a, s'_x1, s'_x2, ..., s'_xdim) where s, s' are
   /// the state and successor state to which `a`, the action, might lead.
   /// The entry in the matrix provides the probability of this transition.
   xarray< double > m_transition_tensor;
   reward_map_type m_reward_map;
   double m_step_reward;

   [[nodiscard]] constexpr std::span< double, std::dynamic_extent >
   c_array(const size_t size, const auto value) const
   {
      auto data = std::span{new double[size], size};
      std::fill(data.begin(), data.end(), value);
      return data;
   }

   template < typename T, size_t dimensions = dim >
   void assert_dimensions(const xarray< T > &arr) const
   {
      auto arr_shape = arr.shape();
      if(arr_shape.size() != 2) {
         throw std::invalid_argument(
            "Array is not exactly two dimensional. Actual dimensions: "
            + std::to_string(arr_shape.size())
         );
      }
      if(arr_shape[1] != dimensions) {
         std::stringstream sstream;
         sstream << "Dimension mismatch:\n"
                 << "Passed states array has coordinate dimensions: " << arr.dimension()
                 << "\nThe expected dimensions are: " << dimensions;
         throw std::invalid_argument(sstream.str());
      }
   }

   template < typename Array, typename Rng >
   void assert_shape(const Array &arr, Rng &&shape) const
   {
      auto arr_shape = arr.shape();
      if(not ranges::equal(arr_shape, shape)) {
         std::stringstream sstream;
         sstream << "Shape mismatch:\n"
                 << "Passed array has shape: " << fmt::format("{}", arr_shape)
                 << "\nThe required shape is: " << fmt::format("{}", std::forward< Rng >(shape));
         throw std::invalid_argument(sstream.str());
      }
   }

   std::array< size_t, dim > _verify_shape(const ranges::range auto &rng) const;

   xarray< double > _init_transition_tensor(
      std::variant< double, pyarray< double > > transition_matrix
   );

   auto _init_reward_map(
      const std::variant< double, pyarray< double > > &goal_reward,
      const std::variant< double, pyarray< double > > &subgoal_reward,
      double restart_reward
   );

   template < StateType state_type >
   void _enter_rewards(
      const std::variant< double, pyarray< double > > &reward,
      reward_map_type &reward_map
   ) const;

   template < typename Array >
   void rearrange_layout(Array &arr) const
   {
      if(arr.layout() != xt::layout_type::row_major) {
         arr.resize(arr.shape(), xt::layout_type::row_major);
      }
   }
};

}  // namespace force

#include "gridworld.tcc"

#endif  // REINFORCE_GRIDWORLD_HPP
