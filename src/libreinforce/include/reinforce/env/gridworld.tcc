
#ifndef REINFORCE_GRIDWORLD_TCC
#define REINFORCE_GRIDWORLD_TCC

#include "gridworld.hpp"

namespace force {

template < size_t dim >
template < ranges::range Range >
   requires expected_value_type< size_t, Range >
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
         idx_xtensor_stack< dim > grid_cumul_shape;
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
      m_step_reward(step_reward)
{
   // we will now count the storage of all our special state definitions to relocate the possibly
   // fragmented memory into one contiguous memory chunk in which segments are owned by the specific
   // state arrays.
   size_t total_alloc = 0;
   auto array_list = {
      std::ref(m_start_states),
      std::ref(m_goal_states),
      std::ref(m_obs_states),
      std::ref(m_subgoal_states),
      std::ref(m_restart_states)};
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
idx_xtensor_stack< dim > Gridworld< dim >::_verify_shape(Range&& rng) const
{
   static_assert(
      std::forward_iterator< ranges::iterator_t< Range > >,
      "Iterator type of range must be at least forward iterator (allow multiple passes)."
   );
   constexpr long long_dim = long(dim);
   auto dist = ranges::distance(rng);
   std::array< size_t, dim > actual_shape;
   if(dist > long_dim) {
      std::stringstream ss;
      ss << "Expected a <=" << dim << "-dimensional shape parameter. Got " << dist << ".";
      throw std::invalid_argument(ss.str());
   }
   ranges::copy(rng, actual_shape.begin());
   if(dist < long_dim) {
      // dimension of shape param is less than the dimension of the grid.
      // Pad length 1 for each unspecified dimension.
      std::fill(std::next(actual_shape.begin(), dist), actual_shape.end(), 1);
   }
   return actual_shape;
}

template < size_t dim >
template < ranges::range Range >
idx_xtensor_stack< dim > Gridworld< dim >::_adapt_coords(Range&& coords) const
{
   static_assert(
      std::forward_iterator< ranges::iterator_t< std::remove_cvref_t< Range > > >,
      "Iterator type of range must be at least forward iterator (allow multiple passes)."
   );
   constexpr long long_dim = long(dim);
   auto dist = ranges::distance(coords);
   idx_xtensor_stack< dim > final_coords;
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
      coords | ranges::views::drop(surplus_to_drop),
      std::next(final_coords.begin(), copy_start_offset)
   );
   return final_coords;
}

template < size_t dim >
xarray< double > Gridworld< dim >::_init_transition_tensor(
   std::variant< double, pyarray< double > > transition_matrix
)
{
   return std::visit(
      detail::overload{
         [&](double value) {
            size_t n_states = size();
            xt::svector< size_t > shape{n_states, m_num_actions, n_states};
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
Gridworld< dim >::reward_map_type Gridworld< dim >::_init_reward_map(
   const std::variant< double, pyarray< double > >& goal_reward,
   const std::variant< double, pyarray< double > >& subgoal_reward,
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
   const std::variant< double, pyarray< double > >& reward,
   reward_map_type& reward_map
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
            std::forward_as_tuple(index_state(detail::SizedRangeAdaptor{
               idx_iter->cbegin(), idx_iter->cend(), dim})),
            std::forward_as_tuple(state_type, access_functor(counter))
         );
      }
   };

   const auto assert_shape = [&](const pyarray< double >& r_goals) {
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
         [&](const pyarray< double >& r_goals) {
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
   // we need the same type in std::div to avoid ambiguity. So we use long for both inputs.
   long index = long(state_index);
   for(auto _ : ranges::views::enumerate(ranges::views::zip(m_grid_shape, m_grid_shape_products))) {
      auto [i, shape_and_product_pair] = _;
      auto [shape, product] = shape_and_product_pair;
      auto modulus_result = std::div(index, long(product));
      coords[i] = size_t(modulus_result.quot);
      index = modulus_result.rem;
   }
   return coords;
}

template < size_t dim >
template < ranges::sized_range Range >
   requires expected_value_type< size_t, Range >
auto Gridworld< dim >::coord_state(Range&& indices) const
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
   requires expected_value_type< size_t, Range >
size_t Gridworld< dim >::index_state(Range&& coordinates) const
{
   auto size = ranges::distance(coordinates);
   long int diff = long(dim) - long(size);
   if(diff < 0) {
      std::stringstream ss;
      ss << "More arguments (" << size << ") passed than dimensions in the grid (" << dim << ").";
      throw std::invalid_argument(ss.str());
   }
   idx_xtensor_stack< dim > coords;
   // every dimension we have been given is used to fill up the coordinates from the end.
   // all dimensions from the start for which we do not have a value will be given coordinate 0
   // If coords={2,4,9} and dim = 5, then the actual passed coordinates are {0,0,2,4,9}
   ranges::copy(coordinates, std::next(coords.begin(), diff));
   ranges::fill(coords.begin(), std::next(coords.begin(), diff), 0);
   size_t state = 0;
   for(auto i : ranges::views::iota(0UL, dim)) {
      state += m_grid_shape(i) * coords(i);
   }
   return state;
}

template < size_t dim >
template < ranges::range Range >
bool Gridworld< dim >::is_terminal(Range&& coordinates) const
{
   return std::any_of(
      xt::axis_slice_begin(m_goal_states, 1),
      xt::axis_slice_end(m_goal_states, 1),
      [&, adapted_coords = _adapt_coords(coordinates)](const auto& goal_coords) {
         // this should be ever so sightly more efficient than ranges::equal, since equal checks for
         // same length which we already know is the case, because the coords are adapted and the
         // goal coords are verified upon construction
         return ranges::all_of(
            ranges::views::zip(goal_coords, adapted_coords),
            [](const auto& coord_pair) {
               return std::get< 0 >(coord_pair) == std::get< 1 >(coord_pair);
            }
         );
      }
   );
}

}  // namespace force

#endif  // REINFORCE_GRIDWORLD_TCC
