
#ifndef REINFORCE_GRIDWORLD_HPP
#define REINFORCE_GRIDWORLD_HPP

#include "optional"
#include "pybind11/numpy.h"
#include "range/v3/all.hpp"
#include "reinforce/utils/utils.hpp"
#include "variant"

namespace force {

namespace py = pybind11;

class GridWorld {
   /**
    * @brief Construct a GridWorld instance.
    *
    * @param num_rows The number of rows in the gridworld.
    * @param num_cols The number of columns in the gridworld.
    * @param start_states The start states of the gridworld.
    * @param goal_states The goal states for the gridworld (m <= n).
    * @param step_reward The reward for each step taken by the agent.
    * @param goal_reward The reward for reaching a goal state.
    * @param bad_state_reward The reward for transitioning to a bad state.
    * @param restart_state_reward The reward for transitioning to a restart state.
    * @param transition_matrix The probability or probability matrix of successfully transitioning
    * states.
    * @param bias The probability of transitioning left or right if not successful.
    * @param obstacle_states States the agent cannot enter.
    * @param bad_states States where the agent incurs high penalty.
    * @param restart_states States where the agent transitions to start.
    */
   GridWorld(
      size_t n_rows,
      size_t n_cols,
      py::array_t< size_t > start_states,
      py::array_t< size_t > goal_states,
      std::variant< double, py::array_t< double > > goal_reward,
      double step_reward = 0.,
      std::variant< double, py::array_t< double > > transition_matrix = double{1.},
      py::array_t< size_t > bad_states = {},
      std::variant< double, py::array_t< double > > bad_states_reward = {},
      py::array_t< size_t > obs_states = {},
      py::array_t< size_t > restart_states = {},
      double restart_states_reward = 0.
   )
       : m_num_rows(n_rows),
         m_num_cols(n_cols),
         m_start_states(std::move(start_states)),
         m_goal_states(std::move(goal_states)),
         m_obs_states(std::move(obs_states)),
         m_bad_states(std::move(bad_states)),
         m_restart_states(std::move(restart_states)),
         m_transition_matrix(std::visit(
            utils::overload{
               [&](double prob) -> py::array_t< double > {
                  return py::array_t< double >{
                     std::array{m_num_rows, m_num_cols},
                     std::array{sizeof(double) * m_num_rows, sizeof(double)},
                     c_array(m_num_rows * m_num_cols, prob).data()};
               },
               utils::identity},
            transition_matrix
         )),
         m_reward_step(step_reward),
         m_reward_goal(std::visit(
            utils::overload{
               [&](double value) -> py::array_t< double > {
                  const auto *shape = m_goal_states.shape();
                  const auto size = m_goal_states.size();
                  return py::array_t< double >{
                     shape,
                     std::array{sizeof(double)},
                     c_array(static_cast<size_t>(size), value).data()};
               },
               utils::identity},
            goal_reward
         )),
         m_reward_bad(std::visit(
            utils::overload{
               [&](double value) -> py::array_t< double > {
                  const auto* shape = m_bad_states.shape();
                  const auto size = m_bad_states.size();
                  return py::array_t< double >{
                     shape,
                     std::array{sizeof(double) * m_num_rows, sizeof(double)},
                     c_array(static_cast<size_t>(size), value).data()};
               },
               utils::identity},
            bad_states_reward
         )),
         m_reward_restart(restart_states_reward)
   {
   }

  private:
   size_t m_num_rows;
   size_t m_num_cols;
   py::array_t< size_t > m_start_states;
   py::array_t< size_t > m_goal_states;
   py::array_t< size_t > m_obs_states;
   py::array_t< size_t > m_bad_states;
   py::array_t< size_t > m_restart_states;
   py::array_t< double > m_transition_matrix;
   double m_reward_step;
   py::array_t< double > m_reward_goal;
   py::array_t< double > m_reward_bad;
   double m_reward_restart;

   [[nodiscard]] constexpr std::span< double, std::dynamic_extent >
   c_array(const size_t size, const auto value) const
   {
      auto data = std::span{new double[size], size};
      std::fill(data.begin(), data.end(), value);
      return data;
   }
};

}  // namespace force

#endif  // REINFORCE_GRIDWORLD_HPP
