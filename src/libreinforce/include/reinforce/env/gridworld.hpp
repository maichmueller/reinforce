
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
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

namespace force {

namespace py = pybind11;

template < typename T >
using xarray = xt::xarray< T, xt::layout_type::row_major >;


using index_xarray = xt::xarray< size_t, xt::layout_type::row_major >;

namespace np {

template < typename T >
using array = py::array_t< T >;

using index_array = py::array_t< size_t >;

}  // namespace np

template < size_t dim >
class GridWorld {
  public:
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
   GridWorld(
      std::array< size_t, dim > dimensions,
      index_xarray start_states,
      index_xarray goal_states,
      std::variant< double, xarray< double > > goal_reward,
      double step_reward = 0.,
      std::variant< double, xarray< double > > transition_matrix = double{1.},
      std::optional< index_xarray > subgoal_states = {},
      std::variant< double, xarray< double > > subgoal_states_reward = double{0.},
      std::optional< index_xarray > obs_states = {},
      std::optional< index_xarray > restart_states = {},
      double restart_states_reward = 0.
   );

  private:
   constexpr static size_t m_num_actions = 2 * dim;
   std::array< size_t, dim > m_dimensions;
   /// shape (n, DIM)
   index_xarray m_start_states;
   index_xarray m_goal_states;
   index_xarray m_subgoal_states;
   index_xarray m_obs_states;
   index_xarray m_restart_states;
   /// shape (s_x1, s_x2, s_x..., s_xdim, a, s'_x1, s'_x2, ..., s'_xdim) where `s`, `s'` are
   /// the state and successor state to which `a`, the action, might lead.
   /// The entry in the matrix provides the probability of this transition.
   xarray< double > m_transition_tensor;
   xarray< double > m_reward_tensor;

   [[nodiscard]] constexpr std::span< double, std::dynamic_extent >
   c_array(const size_t size, const auto value) const
   {
      auto data = std::span{new double[size], size};
      std::fill(data.begin(), data.end(), value);
      return data;
   }

   template < typename T >
   auto array_like_lambda_gen(const xarray< T >& ref_array) const
   {
      return [&](double value) -> xarray< double > {
         const auto shape = std::span{ref_array.shape(), size_t(ref_array.ndim())};
         return xarray< double >{shape, c_array(size_t(ref_array.size()), value).data()};
      };
   }

   auto array_from_value_lambda() const
   {
      return [&](double value) {
         std::valarray< size_t > shape(dim * 2 + 1);
         ranges::copy(m_dimensions, ranges::begin(shape));
         shape[dim] = m_num_actions;
         ranges::copy(m_dimensions, std::next(ranges::begin(shape), dim + 1));
         std::array< size_t, dim * 2 + 1 > strides(sizeof(double));
         for(size_t i = 0; i < shape.size() - 1; i++) {
            // stride[i] = prod(shape[i+1:])
            strides[i] = std::accumulate(
               std::next(std::begin(shape), long(i) + 1),
               std::end(shape),
               size_t(1),
               std::multiplies{}
            );
         }
         return xarray< double >{
            shape, strides, c_array(ranges::accumulate(shape, size_t(0)), value).data()};
      };
   }

   template < typename T >
   void assert_dimensions(const xarray< T >& arr) const
   {
      if(arr.ndim() != dim) {
         std::stringstream sstream;
         sstream << "Dimension mismatch:\n"
                 << "Passed array has dimension: " << arr.ndim()
                 << "\nThe environment dimenion is: " << dim;
         throw std::invalid_argument(sstream.str());
      }
   }

   template < typename T, typename Rng >
   void assert_shape(const xarray< T >& arr, Rng&& shape) const
   {
      const auto arr_shape = std::span{arr.shape(), size_t(arr.ndim())};
      if(not ranges::equal(arr_shape, shape)) {
         std::stringstream sstream;
         sstream << "Shape mismatch:\n"
                 << "Passed array has shape: " << fmt::format("{}", arr_shape)
                 << "\nThe required shape is: " << fmt::format("{}", shape);
         throw std::invalid_argument(sstream.str());
      }
   }

   xarray< double > _init_transition_tensor(
      std::variant< double, xarray< double > > transition_matrix
   );
   xarray< double > _init_reward_tensor(
      std::variant< double, xarray< double > > goal_reward,
      double step_reward,
      std::variant< double, xarray< double > > subgoal_states_reward,
      double restart_states_reward
   );

   template < typename DType >
   std::vector< size_t > _strides_from_shape(std::vector< size_t >& shape) const;
};

template < size_t dim >
GridWorld< dim >::GridWorld(
   std::array< size_t, dim > dimensions,
   index_xarray start_states,
   index_xarray goal_states,
   std::variant< double, xarray< double > > goal_reward,
   double step_reward,
   std::variant< double, xarray< double > > transition_matrix,
   std::optional< index_xarray > subgoal_states,
   std::variant< double, xarray< double > > subgoal_states_reward,
   std::optional< index_xarray > obs_states,
   std::optional< index_xarray > restart_states,
   double restart_states_reward
)
    : m_dimensions(dimensions),
      m_start_states(std::move(start_states)),
      m_goal_states(std::move(goal_states)),
      m_subgoal_states(subgoal_states.has_value() ? std::move(*subgoal_states) : index_xarray{}),
      m_obs_states(obs_states.has_value() ? std::move(*obs_states) : index_xarray{}),
      m_restart_states(restart_states.has_value() ? std::move(*restart_states) : index_xarray{}),
      m_transition_tensor(_init_transition_tensor(transition_matrix)),
      m_reward_tensor(
         _init_reward_tensor(goal_reward, step_reward, subgoal_states_reward, restart_states_reward)
      )
{
   for(const auto& arr :
       {m_start_states, m_goal_states, m_obs_states, m_subgoal_states, m_restart_states}) {
      if(arr.size() != 0) {
         // if the array is not an empty one then we need to ensure that we are having the same
         // dimensions as expected by the env
         assert_dimensions(arr);
      }
   }
   assert_shape(
      m_transition_tensor,
      ranges::concat_view(m_dimensions, std::array{m_num_actions}, m_dimensions)
   );
}

template < size_t dim >
xarray< double > GridWorld< dim >::_init_transition_tensor(
   std::variant< double, xarray< double > > transition_matrix
)
{
   return std::visit(
      utils::overload{
         [&](double value) {
            std::vector< size_t > shape(dim * 2 + 1);
            ranges::copy(m_dimensions, ranges::begin(shape));
            shape[dim] = m_num_actions;
            ranges::copy(m_dimensions, std::next(ranges::begin(shape), dim + 1));
            std::vector< size_t > strides = _strides_from_shape< dim >(shape);
            return xarray< double >{
               shape,
               strides,
               c_array(ranges::accumulate(shape, size_t(1), std::multiplies{}), value).data()};
         },
         utils::identity},
      transition_matrix
   );
}
template < size_t dim >
template < typename DType >
std::vector< size_t > GridWorld< dim >::_strides_from_shape(std::vector< size_t >& shape) const
{
   std::vector< size_t > strides(dim * 2 + 1, sizeof(DType));
   for(size_t i = 0; i < shape.size() - 1; i++) {
      // stride[i] = prod(shape[i+1:])
      strides[i] = std::accumulate(
         std::next(std::begin(shape), long(i) + 1), std::end(shape), size_t(1), std::multiplies{}
      );
   }
   return strides;
}

template < size_t dim >
xarray< double > GridWorld< dim >::_init_reward_tensor(
   std::variant< double, xarray< double > > goal_reward,
   double step_reward,
   std::variant< double, xarray< double > > subgoal_states_reward,
   double restart_states_reward
)
{
   const size_t size = ranges::accumulate(m_dimensions, size_t(1), std::multiplies{});
   auto data = std::span{new double[size], size};
   std::fill(data.begin(), data.end(), step_reward);

   xarray< double > reward{m_dimensions, _strides_from_shape< double >(m_dimensions), data};
   std::visit(
      utils::overload{
         [&](double r_goal) { m_goal_states.mutable_unchecked< dim >(); },
         [&](const xarray< double >& r_goals) {

         }},
      goal_reward
   );
   auto reader = reward.mutable_unchecked< dim >();
}

// template < typename T, size_t current_dim, size_t dim >
// void iterate_array(const xarray< T >& array, size_t index)
//{
//    //   double sum = 0;
//    //   for(py::ssize_t i = 0; i < r.shape(0); i++)
//    //      for(py::ssize_t j = 0; j < r.shape(1); j++)
//    //         for(py::ssize_t k = 0; k < r.shape(2); k++)
//    //            sum += r(i, j, k);
//    if constexpr(current_dim == dim - 1) {
//       // Innermost loop: Process the element at the current indices
//       for(size_t i = 0; i < buffer.shape[current_dim]; i++) {
//          T value = buffer.ptr[index];
//          // Do something with 'value'
//       }
//       for(indices[current_dim] = 0; indices[current_dim] < array.shape(current_dim);
//           ++indices[current_dim]) {
//          T value = array(indices);
//          // Do something with 'value'
//       }
//    } else {
//       // Recursive loop: Iterate through the current dimension and move to the next dimension
//       for(indices[current_dim] = 0; indices[current_dim] < array.shape(current_dim);
//           ++indices[current_dim]) {
//          iterate_array(array, indices, current_dim + 1);
//       }
//    }
// }
//
// namespace detail {
// template < typename T, size_t current_dim, size_t dim >
// void _iterate_array(const py::buffer_info& buffer, size_t index)
//{
//    //   double sum = 0;
//    //   for(py::ssize_t i = 0; i < r.shape(0); i++)
//    //      for(py::ssize_t j = 0; j < r.shape(1); j++)
//    //         for(py::ssize_t k = 0; k < r.shape(2); k++)
//    //            sum += r(i, j, k);
//    if constexpr(current_dim == dim - 1) {
//       // Innermost loop: Process the element at the current indices
//       for(size_t i = 0; i < buffer.shape[current_dim]; i++) {
//          T value = reinterpret_cast< T* >(buffer.ptr)[index];
//          // Do something with 'value'
//       }
//       for(indices[current_dim] = 0; indices[current_dim] < array.shape(current_dim);
//           ++indices[current_dim]) {
//          T value = array(indices);
//          // Do something with 'value'
//       }
//    } else {
//       // Recursive loop: Iterate through the current dimension and move to the next dimension
//       for(indices[current_dim] = 0; indices[current_dim] < array.shape(current_dim);
//           ++indices[current_dim]) {
//          iterate_array(array, indices, current_dim + 1);
//       }
//    }
// }
// }  // namespace detail

}  // namespace force

#endif  // REINFORCE_GRIDWORLD_HPP
