
#ifndef REINFORCE_GRIDWORLD_TCC
#define REINFORCE_GRIDWORLD_TCC

#include <reinforce/utils/views_extension.hpp>
#include <utility>

#include "gridworld.hpp"

namespace force {

using namespace xt::placeholders;  // to enable `_` syntax in xt::range

template < size_t dim >
template < ranges::range Range >
   requires detail::expected_value_type< size_t, Range >
Gridworld< dim >::Gridworld(
   const Range& shape,
   const idx_pyarray& start_states,
   const idx_pyarray& goal_states,
   std::variant< double, pyarray< double > > goal_reward,
   double step_reward,
   std::optional< idx_pyarray > start_states_prob_weights,
   std::variant< double, pyarray< double > > transition_matrix,
   std::optional< idx_pyarray > subgoal_states,
   std::variant< double, pyarray< double > > subgoal_states_reward,
   std::optional< idx_pyarray > obs_states,
   std::optional< idx_pyarray > restart_states,
   double restart_states_reward
)
    : m_grid_shape(_adapt_coords(shape)),
      m_grid_shape_products(std::invoke([&] {
         idx_xstacktensor< dim > grid_cumul_shape;
         // we set the last entry of the cumul shape to 1 as each shape must have at least
         grid_cumul_shape.back() = 1;
         size_t cumprod = 1;
         for(std::tuple< size_t&, const size_t& > output_and_dimshape : ranges::views::zip(
                // we drop the first in reverse (i.e. the last entry) since we want entry `i` to
                // hold the cumprod up to and including `i-1`
                ranges::views::reverse(grid_cumul_shape) | ranges::views::drop(1),
                ranges::views::reverse(m_grid_shape)
             )) {
            auto& [output, dim_shape] = output_and_dimshape;
            cumprod *= dim_shape;
            output = cumprod;
         }
         return grid_cumul_shape;
      })),
      m_size(std::invoke([&] {
         return ranges::accumulate(m_grid_shape, size_t(1), std::multiplies{});
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
      m_start_state_distribution(
         start_states_prob_weights.has_value()
            ? std::discrete_distribution<
                 size_t >{(*start_states_prob_weights).begin(), (*start_states_prob_weights).end()}
            : std::invoke([&] {
                 std::vector< int > weights(m_start_states.shape(0), 1);  // makes it uniform
                 return std::discrete_distribution< size_t >{weights.begin(), weights.end()};
              })
      ),
      m_transition_tensor(_init_transition_tensor(transition_matrix)),
      m_reward_map(_init_reward_map(goal_reward, subgoal_states_reward, restart_states_reward)),
      m_step_reward(step_reward),
      m_action_space{0, m_num_actions - 1},
      m_obs_space{DiscreteSpace{m_size}, MultiDiscreteSpace< size_t >{m_grid_shape}},
      m_reward_range{std::invoke([&] {
         auto [min, max] = std::ranges::minmax(
            m_reward_map | std::views::values | std::views::values
         );
         return std::pair{min, max};
      })}
{
   // we will now count the storage of all our special state definitions to relocate the possibly
   // fragmented memory into one contiguous memory chunk in which segments are owned by the specific
   // state arrays. This is a tiny optimization for memory contiguousness and is possible because we
   // never change the setup of the grid after creating it.
   size_t total_alloc = 0;
   auto array_list = {
      std::ref(m_start_states),
      std::ref(m_goal_states),
      std::ref(m_obs_states),
      std::ref(m_subgoal_states),
      std::ref(m_restart_states)
   };
   for(auto& arr : array_list | ranges::views::deref) {
      auto size = arr.size();
      if(size != 0) {
         // if the array is not an empty one then ensure that we have the same dimensions as env
         assert_dimensions(arr);
      }
      total_alloc += size;
   }
   // use a unique ptr here, in order to avoid a memory leak in case of an exeception
   auto data_uptr = std::make_unique< size_t[] >(total_alloc);
   size_t* data_ptr_start = data_uptr.get();

   auto adapt = [&](auto& arr, auto ownership_token) {
      auto size = arr.size();
      ranges::copy(arr, data_ptr_start);
      arr = xt::adapt(data_ptr_start, size, ownership_token, arr.shape());
      data_ptr_start += size;
   };
   // first array takes ownership of all memory
   adapt(array_list.begin()->get(), xt::acquire_ownership{});
   // we now no longer need the uptr to control the memory block. Ownership has been transferred.
   data_uptr.release();
   // all others do not take ownership
   ranges::for_each(array_list | ranges::views::drop(1) | ranges::views::deref, [&](auto& arr) {
      adapt(arr, xt::no_ownership{});
   });
   reset();
}

template < size_t dim >
template < ranges::range Range >
idx_xstacktensor< dim > Gridworld< dim >::_verify_shape(const Range& coords_range) const
{
   static_assert(
      std::forward_iterator< ranges::iterator_t< Range > >,
      "Iterator type of range must be at least forward iterator (allow multiple passes)."
   );
   constexpr long long_dim = long(dim);
   auto n_coordinates = ranges::distance(coords_range);
   std::array< size_t, dim > actual_shape;
   if(n_coordinates > long_dim) {
      throw std::invalid_argument(
         fmt::format("Expected a <={}-dimensional shape parameter. Got {} .", dim, n_coordinates)
      );
   }
   ranges::copy(coords_range, actual_shape.begin());
   if(n_coordinates < long_dim) {
      // dimension of shape param is less than the dimension of the grid.
      // Pad length 1 at the end of the shape vector for each unspecified dimension.
      ranges::fill(actual_shape | ranges::views::drop(n_coordinates), 1);
   }
   return actual_shape;
}

template < size_t dim >
template < ranges::range Range >
idx_xstacktensor< dim > Gridworld< dim >::_adapt_coords(const Range& coords_range) const
{
   static_assert(
      std::forward_iterator< ranges::iterator_t< std::remove_cvref_t< Range > > >,
      "Iterator type of range must be at least forward iterator (allow multiple passes)."
   );
   constexpr long long_dim = long(dim);
   auto dist = ranges::distance(coords_range);
   idx_xstacktensor< dim > final_coords;
   // initialize the coordinates to all 0.
   ranges::fill(final_coords, 0);
   const long diff = dist - long_dim;
   // The condition diff > 0 checks how many surplus coordinates were provided. We drop those from
   // the beginning, keeping only the `dim`-dimensional tail of coordinates. In the case diff < 0,
   // we have fewer coordinates than `dim`, so for consistency’s sake we view the given coordinates
   // as the tail coordinates and pad/leave the unspecified coordinates as 0.
   auto [surplus_to_drop, copy_start_offset] = diff > 0 ? std::pair{diff, 0L}
                                                        : std::pair{0L, -diff};
   ranges::copy(
      coords_range | ranges::views::drop(surplus_to_drop),
      std::next(final_coords.begin(), copy_start_offset)
   );
   return final_coords;
}

template < size_t dim >
xarray< double > Gridworld< dim >::_init_transition_tensor(
   std::variant< double, pyarray< double > > transition_matrix
) const
{
   xt::svector< size_t > shape{m_size, m_num_actions, m_num_actions};
   auto tensor = xarray< double >::from_shape(shape);
   std::visit(
      detail::overload{
         [&](double value) {
            if(value > 1.) {
               throw std::invalid_argument(
                  fmt::format("Transition probability must be <= 1. Given: {}", value)
               );
            }
            for(auto [state, action_choice, action_realized] : ranges::views::cartesian_product(
                   ranges::views::iota(0UL, m_size),
                   ranges::views::iota(0UL, m_num_actions),
                   ranges::views::iota(0UL, m_num_actions)
                )) {
               double probability = value * (action_realized == action_choice)
                                    + (1. - value) / double(m_num_actions - 1.)
                                         * (action_choice != action_realized);
               tensor.unchecked(state, action_choice, action_realized) = probability;
            }
         },
         [&](pyarray< double > arr) {
            assert_shape(arr, shape);
            ranges::copy(arr, tensor.begin());
         }
      },
      transition_matrix
   );
   return tensor;
}

template < size_t dim >
Gridworld< dim >::RewardMap Gridworld< dim >::_init_reward_map(
   const std::variant< double, pyarray< double > >& goal_reward,
   const std::variant< double, pyarray< double > >& subgoal_reward,
   double restart_reward
)
{
   RewardMap reward_map;
   _enter_rewards< StateType::goal >(goal_reward, reward_map);
   _enter_rewards< StateType::subgoal >(subgoal_reward, reward_map);
   _enter_rewards< StateType::restart >(restart_reward, reward_map);
   return reward_map;
}

template < size_t dim >
template < StateType state_type >
void Gridworld< dim >::_enter_rewards(
   const std::variant< double, pyarray< double > >& reward_variant,
   RewardMap& reward_map
) const
{
   const auto& states = _states< state_type >();
   if(states.size() == 0) {
      SPDLOG_DEBUG(fmt::format(
         "State type ({}) has an empty associated array. Not adding any values to reward map.",
         state_type,
         states
      ));
      return;
   }
   SPDLOG_DEBUG(fmt::format(
      "State type's ({}) associated array: \n{}", state_type, states
   ));
   const auto reward_setter = [&](auto access_functor) {
      // iterate over axis 0 (the state index) to get a slice over state coordinates
      auto coord_begin = xt::axis_slice_begin(states, 1);
      auto coord_end = xt::axis_slice_end(states, 1);
      for(auto [idx_iter, counter] = std::pair{coord_begin, size_t{0}}; idx_iter != coord_end;
          ++idx_iter, ++counter) {
         SPDLOG_DEBUG(
            "State index: {}",
            index_state(detail::SizedRangeAdaptor{idx_iter->cbegin(), idx_iter->cend(), dim})
         );
         reward_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(
               index_state(detail::SizedRangeAdaptor{idx_iter->cbegin(), idx_iter->cend(), dim})
            ),
            std::forward_as_tuple(state_type, access_functor(counter))
         );
      }
   };

   const auto assert_shape = [&](const pyarray< double >& reward_arr) {
      if(reward_arr.shape(0) != states.shape(0)) {
         throw std::invalid_argument(fmt::format(
            "Length ({}) of goal state reward array does not match number of goal states ({}).",
            reward_arr.shape(0),
            states.shape(0)
         ));
      }
   };

   std::visit(
      detail::overload{
         [&](double reward_val) { reward_setter([&](auto) { return reward_val; }); },
         [&](const pyarray< double >& reward_arr) {
            assert_shape(reward_arr);
            reward_setter([&](auto index) { return reward_arr(index); });
         }
      },
      reward_variant
   );
}

template < size_t dim >
auto Gridworld< dim >::coord_state(size_t state_index) const
{
   idx_xstacktensor< dim > coords;
   // we need the same type in std::div to avoid ambiguity. So we use long for both inputs.
   long index = static_cast< long >(state_index);
   for(auto [i, shape] : ranges::views::reverse(ranges::views::enumerate(m_grid_shape))) {
      auto [quotient, remainder] = modulo(index, static_cast< long >(shape));
      coords[i] = static_cast< size_t >(remainder);
      index = quotient;
   }
   return coords;
}

template < size_t dim >
template < ranges::sized_range Range >
   requires detail::expected_value_type< size_t, Range >
auto Gridworld< dim >::coord_state(const Range& indices) const
{
   // create the output buffer first
   idx_xarray coords_out(/*shape=*/xt::svector< size_t >{indices.size(), dim});
   // process the indices one-by-one and emplace them in the data buffer
   for(auto [row_idx, state_idx] : ranges::views::enumerate(indices)) {
      auto coords = coord_state(state_idx);
      for(auto&& [col_idx, entry] : ranges::views::enumerate(coords)) {
         coords_out(row_idx, col_idx) = entry;
      }
   }
   return coords_out;
}

template < size_t dim >
template < ranges::sized_range Range >
   requires detail::expected_value_type< size_t, Range >
size_t Gridworld< dim >::index_state(const Range& coordinates) const
{
   auto size = ranges::distance(coordinates);
   long int diff = static_cast< long >(dim) - static_cast< long >(size);
   if(diff < 0) {
      throw std::invalid_argument(
         fmt::format("More arguments ({}) passed than dimensions in the grid ({}).", size, dim)
      );
   }
   idx_xstacktensor< dim > coords;
   // every dimension we have been given is used to fill up the coordinates from the end.
   // all dimensions from the start for which we do not have a value will be given coordinate 0
   // If coords={2,4,9} and dim = 5, then the actual passed coordinates are {0,0,2,4,9}
   ranges::copy(coordinates, std::next(coords.begin(), diff));
   ranges::fill(coords.begin(), std::next(coords.begin(), diff), 0);
   size_t state = 0;
   for(auto i : ranges::views::iota(0UL, dim)) {
      state += m_grid_shape_products.unchecked(i) * coords(i);
   }
   return state;
}

template < size_t dim >
constexpr std::array< long, dim > Gridworld< dim >::action_as_vector(const size_t action) const
{
   _assert_action_in_bounds(action);
   return _action_as_vector(action);
}

template < size_t dim >
constexpr std::array< long, dim > Gridworld< dim >::_action_as_vector(const size_t action) noexcept
{
   std::array< long, dim > vector;
   ranges::fill(vector, 0);
   auto [quot, rem] = modulo(action, 2);
   vector[static_cast< size_t >(quot)] = _direction_from_remainder(rem);
   return vector;
}

template < size_t dim >
std::tuple< typename Gridworld< dim >::obs_type, double, bool, bool > Gridworld< dim >::step(
   size_t action
)
{
   _assert_action_in_bounds(action);
   // a list of all action vectors that we can simply access without cost since they are computed at
   // compile time
   constexpr static auto actions_to_vectors{std::invoke(
      []< size_t... As >(std::index_sequence< As... >) {
         return std::array< std::array< long, dim >, num_actions() >{_action_as_vector(As)...};
      },
      std::make_index_sequence< num_actions() >{}
   )};

   auto transition_probs = xt::view(m_transition_tensor, location_idx(), action, xt::all());
   size_t chosen_action = xt::random::
      choice(xt::arange(num_actions()), 1, transition_probs, false, m_rng)(0);
   SPDLOG_DEBUG(fmt::format("Passed action: {}, selected action: {}", action, chosen_action));
   auto vector = to_xstackvector(actions_to_vectors[chosen_action]);
   auto next_position = xt::eval(location() + vector);
   auto next_position_index = index_state(next_position);
   auto next_state_attr = m_reward_map.find_or(next_position_index, 0.);
   for(auto [coordinate, shape] : ranges::views::zip(next_position, m_grid_shape)) {
      if(coordinate < 0 or coordinate >= shape) {
         // illegal move, we would be out of the grid bounds if we accepted it
         // --> action has no effect
         return std::tuple{m_location, 0., false, false};
      }
   };
   switch(next_state_attr.first) {
      case StateType::start:  // fall through to default_
      case StateType::default_: {
         m_location = std::pair{next_position_index, next_position};
         return std::tuple{m_location, m_step_reward + 0., false, false};
      }
      case StateType::subgoal: {
         m_location = std::pair{next_position_index, next_position};
         return std::tuple{m_location, m_step_reward + next_state_attr.second, false, false};
      }
      case StateType::goal: {
         m_location = std::pair{next_position_index, next_position};
         return std::tuple{m_location, m_step_reward + next_state_attr.second, true, false};
      }
      case StateType::obstacle: {
         // we do not move so no step reward and no change whatsoever
         return std::tuple{m_location, 0., false, false};
      }
      case StateType::restart: {
         reset();
         return std::tuple{m_location, m_step_reward + next_state_attr.second, false, false};
      }
   }
   throw std::logic_error(
      fmt::format("Switch statement did not handle case ({}).", next_state_attr.first)
   );
}

template < size_t dim >
[[nodiscard]] std::string Gridworld< dim >::action_name(size_t action) const
{
   _assert_action_in_bounds(action);
   if constexpr(dim == 2) {
      constexpr std::array< std::string_view, num_actions() > avail_actions{
         "left", "right", "down", "up"
      };
      return std::string{avail_actions[action]};
   }
   if constexpr(dim == 3) {
      constexpr std::array< std::string_view, num_actions() > avail_actions{
         "left", "right", "down", "up", "out", "in"
      };
      return std::string{avail_actions[action]};
   } else {
      auto [quot, rem] = modulo(action, dim);
      return fmt::format("<DIM: {}, DIRECTION: {}>", quot, _direction_from_remainder(rem));
   }
}

template < size_t dim >
constexpr void Gridworld< dim >::_assert_action_in_bounds(size_t action) const
{
   if(action >= num_actions()) {
      throw std::invalid_argument(
         fmt::format("Action ({}) is out of bounds ({})", action, num_actions())
      );
   }
}

template < size_t dim >
template < StateType state_type >
constexpr auto& Gridworld< dim >::_states() const
{
   if constexpr(state_type == StateType::goal) {
      return m_goal_states;
   } else if constexpr(state_type == StateType::subgoal) {
      return m_subgoal_states;
   } else if constexpr(state_type == StateType::restart) {
      return m_restart_states;
   } else if constexpr(state_type == StateType::start) {
      return m_start_states;
   } else if constexpr(state_type == StateType::obstacle) {
      return m_obs_states;
   } else {
      static_assert(detail::always_false(state_type), "State type not associated with any arrays.");
   }
}

template < size_t dim >
std::string Gridworld< dim >::render() const
{
   throw std::logic_error("Not implemented yet.");
   return fmt::format("Start states: ");
}

}  // namespace force

#endif  // REINFORCE_GRIDWORLD_TCC
