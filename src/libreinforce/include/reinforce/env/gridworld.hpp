#ifndef REINFORCE_GRIDWORLD_HPP
#define REINFORCE_GRIDWORLD_HPP

#ifndef SPDLOG_ACTIVE_LEVEL
static_assert(false, "No logging level set.");
#endif

#include <fmt/format.h>
#include <fmt/std.h>
#include <fmt/ranges.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>

#include <optional>
#include <range/v3/all.hpp>
#include <valarray>
#include <variant>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"

namespace force {

constexpr auto layout = xt::layout_type::row_major;

template < typename T >
using xarray = xt::xarray< T, layout >;
template < typename T >
using pyarray = xt::pyarray< T, layout >;
template < typename T, size_t... shape >
using xstacktensor = xt::xtensor_fixed< T, xt::xshape< shape... >, layout >;

using idx_xarray = xt::xarray< size_t, layout >;

template < size_t dim >
using idx_xstacktensor = xstacktensor< size_t, dim >;
template < size_t dim >
using idx_xstackvector = xstacktensor< long, dim >;

using idx_pyarray = xt::pyarray< size_t, layout >;

enum class StateType { default_ = 0, goal = 1, subgoal = 2, start = 3, restart = 4, obstacle = 5 };

template < size_t dim >
class Gridworld {
   /// Inheriting from the non-polymorphic base of unordered map is fine, as long as no state is
   /// added (a non-polymorphic base has a non-virtual destructor, so the child state would never be
   /// deleted upon destruction of the object)
   class RewardMap: public std::unordered_map< size_t, std::pair< StateType, double > > {
     public:
      using base = std::unordered_map< size_t, std::pair< StateType, double > >;
      using base::base;

      constexpr auto find_or(const std::integral auto& key, const double& default_value) const
      {
         auto find_iter = base::find(key);
         if(find_iter != base::end()) {
            return (*find_iter).second;
         }
         return std::pair{StateType::default_, default_value};
      }
   };

  public:
   using self = Gridworld;
   using obs_type = std::pair< size_t, idx_xstacktensor< dim > >;

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
   Gridworld(std::initializer_list< I > shape, Args&&... args)
       : Gridworld(shape.begin(), shape.end(), std::forward< Args >(args)...)
   {
   }
   template < std::forward_iterator FwdIter, typename... Args >
      requires std::convertible_to< size_t, std::iter_value_t< FwdIter > >
   Gridworld(FwdIter shape_begin, FwdIter shape_end, Args&&... args)
       : Gridworld(detail::RangeAdaptor{shape_begin, shape_end}, std::forward< Args >(args)...)
   {
   }
   template < ranges::range Range >
      requires detail::expected_value_type< size_t, Range >
   Gridworld(
      const Range& shape,
      const idx_pyarray& start_states,
      const idx_pyarray& goal_states,
      std::variant< double, pyarray< double > > goal_reward,
      double step_reward = 0.,
      std::optional< idx_pyarray > start_states_prob_weights = {},
      std::variant< double, pyarray< double > > transition_matrix = double{1.},
      std::optional< idx_pyarray > subgoal_states = {},
      std::variant< double, pyarray< double > > subgoal_states_reward = double{0.},
      std::optional< idx_pyarray > obs_states = {},
      std::optional< idx_pyarray > restart_states = {},
      double restart_states_reward = 0.
   );

   [[nodiscard]] auto coord_state(size_t state_index) const;
   template < ranges::sized_range Range >
      requires detail::expected_value_type< size_t, Range >
   [[nodiscard]] auto coord_state(const Range& indices) const;

   template < ranges::sized_range Range >
      requires detail::expected_value_type< size_t, Range >
   [[nodiscard]] size_t index_state(const Range& coordinates) const;

   [[nodiscard]] bool is_terminal(size_t state_index) const
   {
      return m_reward_map.find_or(state_index, 0.).first == StateType::goal;
   }
   template < ranges::range Range >
   [[nodiscard]] bool is_terminal(const Range& coordinates) const
   {
      return is_terminal(index_state(coordinates));
   }

   std::tuple< obs_type, double, bool, bool > step(size_t action);

   [[nodiscard]] auto& start_states() const { return m_start_states; }
   [[nodiscard]] auto& goal_states() const { return m_goal_states; }
   [[nodiscard]] auto& subgoal_states() const { return m_subgoal_states; }
   [[nodiscard]] auto& obstacle_states() const { return m_obs_states; }
   [[nodiscard]] auto& restart_states() const { return m_restart_states; }
   [[nodiscard]] auto& location() const { return m_location; }
   [[nodiscard]] auto& step_reward() const { return m_step_reward; }
   [[nodiscard]] size_t size() const { return m_size; };
   [[nodiscard]] auto& shape() const { return m_grid_shape; };

   void reseed(std::mt19937_64::result_type seed) { m_rng = std::mt19937_64{seed}; }
   const obs_type& reset(std::optional< std::mt19937_64::result_type > seed = std::nullopt)
   {
      if(seed.has_value()) {
         reseed(*seed);
      }
      auto row_index = m_start_state_distribution(m_rng);
      idx_xstacktensor< dim > start_coordinates = xt::row(m_start_states, long(row_index));
      return (m_location = std::pair{index_state(start_coordinates), start_coordinates});
   }

   [[nodiscard]] std::string action_name(size_t action) const;

   [[nodiscard]] constexpr static auto num_actions() { return m_num_actions; }

   [[nodiscard]] constexpr idx_xstackvector< dim > action_as_vector(size_t action) const;

  private:
   /// the number of actions are dependant only on the grid dimensionality. 'Back' and 'Forth' are
   /// the actions that can be done in each dimension.
   constexpr static size_t m_num_actions = 2 * dim;
   /// the lengths of each grid dimension
   idx_xstacktensor< dim > m_grid_shape;
   /// the cumulative product shape from the last dimension to the 0th dimension
   idx_xstacktensor< dim > m_grid_shape_products;
   /// the total number of states in this grid
   size_t m_size;
   /// shape (n, DIM)
   idx_xarray m_start_states;
   /// shape (m, DIM)
   idx_xarray m_goal_states;
   /// shape (p, DIM)
   idx_xarray m_subgoal_states;
   /// shape (l, DIM)
   idx_xarray m_obs_states;
   /// shape (k, DIM)
   idx_xarray m_restart_states;
   /// shape (n,)
   std::discrete_distribution< size_t > m_start_state_distribution;
   /// shape (N, a, N) where N are the total number of states (state and successor state indices to
   /// which `a`, the action, might lead). The matrix value represents the transition probability.
   xarray< double > m_transition_tensor;
   /// the reward map for a given state
   RewardMap m_reward_map;
   /// the reward an agent achieves/pays per step
   double m_step_reward;
   /// the current position of the agent as index array and associated coordinates
   obs_type m_location{};
   /// the random number generator
   std::mt19937_64 m_rng{std::random_device{}()};

   [[nodiscard("Discarding this value will cause a memory leak.")]]  //
   constexpr std::span< double, std::dynamic_extent >
   c_array(const size_t size, const auto value) const
   {
      auto data = std::span{new double[size], size};
      ranges::fill(data, value);
      return data;
   }

   template < typename T, size_t dimensions = dim >
   void assert_dimensions(const xarray< T >& arr) const
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
   void assert_shape(const Array& arr, Rng&& shape) const
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

   template < ranges::range Range >
   idx_xstacktensor< dim > _adapt_coords(const Range& coords_range) const;

   template < ranges::range Range >
   idx_xstacktensor< dim > _verify_shape(const Range& coords_range) const;

   xarray< double > _init_transition_tensor(
      std::variant< double, pyarray< double > > transition_matrix
   );

   RewardMap _init_reward_map(
      const std::variant< double, pyarray< double > >& goal_reward,
      const std::variant< double, pyarray< double > >& subgoal_reward,
      double restart_reward
   );

   template < StateType state_type >
   void _enter_rewards(
      const std::variant< double, pyarray< double > >& reward_variant,
      RewardMap& reward_map
   ) const;

   template < typename Array >
   void rearrange_layout(Array& arr) const
   {
      if(arr.layout() != xt::layout_type::row_major) {
         arr.resize(arr.shape(), xt::layout_type::row_major);
      }
   }

   constexpr void _assert_action_in_bounds(size_t action) const;

   constexpr idx_xstackvector< dim > _action_as_vector(size_t action) const noexcept;

   constexpr long _direction_from_remainder(long remainder) const noexcept
   {
      return remainder == 0 ? -1 : 1;
   }

   template < bool skip_size_check = false, ranges::range Range >
      requires detail::expected_value_type< size_t, Range >
   constexpr bool contains(const idx_xarray& states_arr, const Range& coordinates) const
   {
      if constexpr(not skip_size_check) {
         if(auto size = ranges::distance(coordinates); size != dim) {
            return false;
         }
      }
      return std::any_of(
         xt::axis_slice_begin(states_arr, 1),
         xt::axis_slice_end(states_arr, 1),
         [&, adapted_coords = _adapt_coords(coordinates)](const auto& obs_coords) {
            return _equal_coords(obs_coords, adapted_coords);
         }
      );
   }

   /// Compares the two coordinate ranges for equal coordinates without checking for same length.
   /// If the input coordinates do not have the same dimension,
   /// then the outcome will only compare coordinates up to the smaller range's size.
   template < ranges::range R1, ranges::range R2 >
      requires(detail::expected_value_type< size_t, R1 > and detail::expected_value_type< size_t, R2 >)
   constexpr bool _equal_coords(const R1& r1, const R2& r2) const noexcept
   {
      // this should be ever so sightly more efficient than ranges::equal, since equal checks
      // for same length which we already know is the case, because the coords are adapted and
      // the goal coords are verified upon construction
      return ranges::all_of(ranges::views::zip(r1, r2), [](const auto& coord_pair) {
         return std::get< 0 >(coord_pair) == std::get< 1 >(coord_pair);
      });
   }

   template < StateType state_type >
   constexpr auto& _states() const
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
         static_assert(
            detail::always_false(state_type), "State type not associated with any arrays."
         );
      }
   }
};

}  // namespace force

#include "gridworld.tcc"

#endif  // REINFORCE_GRIDWORLD_HPP
