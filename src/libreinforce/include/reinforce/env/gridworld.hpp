
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

#define FORCE_IMPORT_ARRAY

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

//namespace helper {
//template < typename T, typename integer_constant >
//   requires utils::is_convertible_without_narrowing_v< typename integer_constant::type, size_t >
//struct array: std::array< T, integer_constant::value > {};
//}  // namespace helper

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

enum class StateType { default_ = 0, goal = 1, subgoal = 2, restart = 3, obstacle = 4 };

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
   Gridworld(std::vector< I > shape, Args &&...args)
       : Gridworld(std::span{shape}, std::forward< Args >(args)...)
   {
   }
   template < std::integral I >
   Gridworld(
      std::span< I > shape,
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

  private:
   constexpr static size_t m_num_actions = 2 * dim;
   std::array< size_t, dim > m_grid_shape;
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

   std::array< size_t, dim > _verify_shape(const auto &rng) const;

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

template < size_t dim >
template < std::integral I >
Gridworld< dim >::Gridworld(
   std::span< I > shape,
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
      m_start_states(start_states),
      m_goal_states(goal_states),
      m_subgoal_states(
         subgoal_states.has_value() ? idx_xarray(*subgoal_states)
                                    : xt::empty< size_t >(std::vector{0})
      ),
      m_obs_states(
         obs_states.has_value() ? idx_xarray(*obs_states) : xt::empty< size_t >(std::vector{0})
      ),
      m_restart_states(
         restart_states.has_value() ? idx_xarray(*restart_states)
                                    : xt::empty< size_t >(std::vector{0})
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
std::array< size_t, dim > Gridworld< dim >::_verify_shape(const auto &rng) const
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
      utils::overload{
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
      utils::overload{
         [&](double r_goal) { reward_setter([&](auto) { return r_goal; }); },
         [&](const pyarray< double > &r_goals) {
            assert_shape(r_goals);
            reward_setter([&](auto index) { return r_goals(index); });
         }},
      reward
   );
}

}  // namespace force

#endif  // REINFORCE_GRIDWORLD_HPP
