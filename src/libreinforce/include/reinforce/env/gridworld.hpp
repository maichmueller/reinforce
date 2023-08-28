
#ifndef REINFORCE_GRIDWORLD_HPP
#define REINFORCE_GRIDWORLD_HPP

#include "optional"
#include "pybind11/numpy.h"
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
      std::variant< double, py::array_t< double > > step_reward,
      std::variant< double, py::array_t< double > > transition_matrix = double{1.},
      std::optional< py::array_t< size_t > > bad_states = {},
      std::optional< std::variant< double, py::array_t< double > > > bad_states_reward = {},
      std::optional< py::array_t< size_t > > obs_states = {},
      std::optional< py::array_t< size_t > > restart_states = {},
      std::optional< std::variant< double, py::array_t< double > > > restart_states_reward = {}
   );

  private:
   int m_num_rows;
   int m_num_cols;
   py::array_t< size_t > m_start_states;
   py::array_t< size_t > m_goal_states;
   py::array_t< size_t > m_obs_states;
   py::array_t< size_t > m_bad_states;
   py::array_t< size_t > m_restart_states;
   py::array_t< double > m_transition_matrix;
   py::array_t< double > m_reward_step;
   py::array_t< double > m_reward_goal;
   py::array_t< double > m_reward_bad;
   py::array_t< double > m_reward_restart;
};

}  // namespace force

#endif  // REINFORCE_GRIDWORLD_HPP
