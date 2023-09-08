
#ifndef REINFORCE_GRIDWORLD_TCC
#define REINFORCE_GRIDWORLD_TCC

#include "gridworld.hpp"

namespace force {

template < size_t dim >
template < ranges::range Range >
   requires expected_value_type< size_t, Range >
Gridworld< dim >::Gridworld(
   const Range &shape,
   const idx_pyarray &start_states,
   const idx_pyarray &goal_states,
   std::variant< double, pyarray< double > > goal_reward,
   double step_reward,
   std::variant< double, pyarray< double > > transition_matrix,
   std::optional< idx_pyarray > subgoal_states,
   std::variant< double, pyarray< double > > subgoal_states_reward,
   std::optional< idx_pyarray > obs_states,
   std::optional< idx_pyarray > restart_states,
   double restart_states_reward
)
    : m_grid_shape(_verify_shape(shape)),
      m_grid_cumul_shape(std::invoke([&] {
         std::array< size_t, dim > grid_cumul_shape;
         // we set the last entry of the cumul shape to 1 as each shape must have at least
         grid_cumul_shape.back() = 1;
         size_t cumsum = 0;
         for(std::tuple< size_t &, size_t & > output_and_dimlen : ranges::views::zip(
                // we drop the first in reverse (i.e. the last entry) since we want entry `i` to
                // hold the cumsum up to and including `i-1`
                ranges::views::reverse(grid_cumul_shape) | ranges::views::drop(1),
                ranges::views::reverse(m_grid_shape)
             )) {
            auto &[output, dim_length] = output_and_dimlen;
            cumsum += dim_length;
            output = cumsum;
         }
         return grid_cumul_shape;
      })),
      m_start_states(start_states),
      m_goal_states(goal_states),
      m_subgoal_states(
         subgoal_states.has_value() ? idx_xarray(*subgoal_states)
                                    : xt::empty< size_t >(std::initializer_list< size_t >{0})
      ),
      m_obs_states(
         obs_states.has_value() ? idx_xarray(*obs_states)
                                : xt::empty< size_t >(std::initializer_list< size_t >{0})
      ),
      m_restart_states(
         restart_states.has_value() ? idx_xarray(*restart_states)
                                    : xt::empty< size_t >(std::initializer_list< size_t >{0})
      ),
      m_transition_tensor(_init_transition_tensor(transition_matrix)),
      m_reward_map(_init_reward_map(goal_reward, subgoal_states_reward, restart_states_reward)),
      m_step_reward(step_reward)
{
   size_t total_alloc = 0;
   uint counter = 0;
   for(const auto &arr :
       {std::ref(m_start_states),
        std::ref(m_goal_states),
        std::ref(m_obs_states),
        std::ref(m_subgoal_states),
        std::ref(m_restart_states)}) {
      auto size = arr.get().size();
      if(size != 0) {
         // if the array is not an empty one then we need to ensure that we are having the same
         // dimensions as expected by the env
         assert_dimensions(arr.get());
      }
      total_alloc += size;
      counter++;
   }
}
template < size_t dim >
std::array< size_t, dim > Gridworld< dim >::_verify_shape(const ranges::range auto &rng) const
{
   auto dist = ranges::distance(rng);
   constexpr long long_dim = long(dim);
   if(dist > long_dim) {
      std::stringstream ss;
      ss << "Expected a <=" << dim << "-dimensional shape parameter. Got " << dist << ".";
      throw std::invalid_argument(ss.str());
   }
   if(dist < long_dim) {
      // dimension of shape param is less than the dimension of the grid.
      // Pad length 1 for each unspecified dimension.
      std::array< size_t, dim > actual_shape{};
      ranges::copy(rng, actual_shape.begin());
      std::fill(std::next(actual_shape.begin(), dist), actual_shape.end(), 1);
      return actual_shape;
   }
   return std::invoke([&] {
      std::array< size_t, dim > out;
      ranges::copy(rng, out.begin());
      return out;
   });
}

template < size_t dim >
xarray< double > Gridworld< dim >::_init_transition_tensor(
   std::variant< double, pyarray< double > > transition_matrix
)
{
   return std::visit(
      detail::overload{
         [&](double value) {
            std::vector< size_t > shape(dim * 2 + 1);
            ranges::copy(m_grid_shape, ranges::begin(shape));
            shape[dim] = m_num_actions;
            ranges::copy(m_grid_shape, std::next(ranges::begin(shape), dim + 1));
            size_t total_size = ranges::accumulate(shape, size_t(1), std::multiplies{});
            return xarray< double >(xt::adapt< xt::layout_type::row_major >(
               c_array(total_size, value).data(), total_size, xt::acquire_ownership(), shape
            ));
         },
         [&](pyarray< double > arr) {
            auto shape = ranges::to_vector(
               ranges::concat_view(m_grid_shape, std::array{m_num_actions}, m_grid_shape)
            );
            assert_shape(arr, shape);
            xarray< double > allocated(shape);
            ranges::copy(arr, allocated.begin());
            return allocated;
         }},
      transition_matrix
   );
}

template < size_t dim >
auto Gridworld< dim >::_init_reward_map(
   const std::variant< double, pyarray< double > > &goal_reward,
   const std::variant< double, pyarray< double > > &subgoal_reward,
   double restart_reward
)
{
   reward_map_type reward_map;
   _enter_rewards< StateType::goal >(goal_reward, reward_map);
   _enter_rewards< StateType::subgoal >(subgoal_reward, reward_map);
   _enter_rewards< StateType::restart >(restart_reward, reward_map);
   return reward_map;
}

template < size_t dim >
template < StateType state_type >
void Gridworld< dim >::_enter_rewards(
   const std::variant< double, pyarray< double > > &reward,
   reward_map_type &reward_map
) const
{
   const auto reward_setter = [&](auto access_functor) {
      // iterate over axis 0 (the state index) to get a slice of the state coordinates
      auto coord_begin = xt::axis_slice_begin(std::as_const(m_goal_states), 0);
      auto coord_end = xt::axis_slice_end(std::as_const(m_goal_states), 0);
      for(auto [idx_iter, counter] = std::pair{coord_begin, size_t(0)}; idx_iter != coord_end;
          idx_iter++, counter++) {
         reward_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(idx_iter->cbegin(), idx_iter->cbegin()),
            std::forward_as_tuple(state_type, access_functor(counter))
         );
      }
   };

   const auto assert_shape = [&](const pyarray< double > &r_goals) {
      if(r_goals.shape(0) != m_goal_states.shape(0)) {
         std::stringstream ss;
         ss << "Length (" << r_goals.shape(0)
            << ") of passed goal state reward array does not match number of goal states ("
            << m_goal_states.shape(0) << ").";
         throw std::invalid_argument(ss.str());
      }
   };

   std::visit(
      detail::overload{
         [&](double r_goal) { reward_setter([&](auto) { return r_goal; }); },
         [&](const pyarray< double > &r_goals) {
            assert_shape(r_goals);
            reward_setter([&](auto index) { return r_goals(index); });
         }},
      reward
   );
}

template < size_t dim >
auto Gridworld< dim >::coord_state(size_t state_index) const
{
   std::array< size_t, dim > coords;
   for(auto _ : ranges::enumerate(ranges::views::zip(m_grid_shape, m_grid_cumul_shape))) {
      auto [i, shape_cumusum_pair] = _;
      auto [shape, cumsum] = shape_cumusum_pair;
      auto mod_result = std::div(state_index, cumsum);
      if(mod_result.quot > 0) {
         coords[i] = mod_result.quot;
         state_index -= mod_result.rem;
      } else {
         // only for the very last domain does a quotient of 0 mean that we enter the remainder as
         // coordinate value. Every dimension prior enters 0 and so passes the index on.
         coords[i] = mod_result.rem * (i == dim - 1);
      }
   }
   return coords;
}

}  // namespace force

#endif  // REINFORCE_GRIDWORLD_TCC
