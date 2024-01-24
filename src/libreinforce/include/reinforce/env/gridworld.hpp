#ifndef REINFORCE_GRIDWORLD_HPP
#define REINFORCE_GRIDWORLD_HPP

#ifndef SPDLOG_ACTIVE_LEVEL
static_assert(false, "No logging level set. Please define the macro 'SPDLOG_ACTIVE_LEVEL'");
#endif

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <frozen/unordered_map.h>
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
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "reinforce/spaces/discrete.hpp"
#include "reinforce/spaces/multi_discrete.hpp"
#include "reinforce/spaces/tuple.hpp"
#include "reinforce/utils/format.hpp"
#include "reinforce/utils/math.hpp"
#include "reinforce/utils/utils.hpp"
#include "reinforce/utils/xarray_formatter.hpp"
#include "reinforce/utils/xtensor_typedefs.hpp"

namespace force {

enum class StateType { default_ = 0, goal = 1, subgoal = 2, start = 3, restart = 4, obstacle = 5 };

namespace detail {
constexpr frozen::unordered_map< StateType, std::string_view, 6 > state_type_names{
   {StateType::default_, "default"},
   {StateType::goal, "goal"},
   {StateType::subgoal, "subgoal"},
   {StateType::start, "start"},
   {StateType::restart, "restart"},
   {StateType::obstacle, "obstacle"}
};

template <>
inline std::string to_string(const StateType& state_type)
{
   return std::string{state_type_names.at(state_type)};
}

}  // namespace detail

template < size_t dim >
class Gridworld {
   /// Inheriting from the non-polymorphic base of unordered map is fine, as long as no state is
   /// added (a non-polymorphic base has a non-virtual destructor, so the child state would never
   /// be deleted upon destruction of the object)

   constexpr static auto _default_reward_pair = std::pair{StateType::default_, 0.};

   class RewardMap: public std::unordered_map< size_t, std::pair< StateType, double > > {
     public:
      using base = std::unordered_map< size_t, std::pair< StateType, double > >;
      using base::base;

      template < typename DefaultT >
      constexpr auto find_or(const std::integral auto& key, DefaultT&& default_value) const
      {
         using default_type = std::remove_cvref_t< DefaultT >;
         auto find_iter = base::find(key);
         if(find_iter != base::end()) {
            return (*find_iter).second;
         }
         if constexpr(std::same_as< typename base::mapped_type, default_type >) {
            return std::forward< DefaultT >(default_value);
         }
         if constexpr(std::convertible_to< default_type, double >) {
            return std::pair{StateType::default_, std::forward< DefaultT >(default_value)};
         } else {
            static_assert(
               detail::always_false(default_value), "Default value type not recognized."
            );
         }
      }
   };

  public:
   using self = Gridworld;
   using obs_type = std::pair< size_t, idx_xstacktensor< dim > >;

   /**
    * @brief Construct a GridWorld instance.
    *
    * @param shape The shape of the "box" representation of the gridworld.
    * @param start_states The start states of the gridworld.
    * @param goal_states The goal states for the gridworld (m <= n).
    * @param goal_reward The reward for reaching a goal state.
    * @param step_reward The reward for each step taken by the agent.
    * @param subgoal_states_reward The reward for transitioning to a subgoal state.
    * @param restart_states_reward The reward for transitioning to a restart state.
    * @param transition_matrix The probability or probability matrix of successfully
    * transitioning states.
    * @param obs_states States the agent cannot enter (obstacles).
    * @param subgoal_states States where the agent incurs subgoal rewards of any kind.
    * @param restart_states States where the agent transitions to start.
    */
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

   [[nodiscard]] auto& start_states() const { return m_start_states; }
   [[nodiscard]] auto& goal_states() const { return m_goal_states; }
   [[nodiscard]] auto& subgoal_states() const { return m_subgoal_states; }
   [[nodiscard]] auto& obstacle_states() const { return m_obs_states; }
   [[nodiscard]] auto& restart_states() const { return m_restart_states; }
   [[nodiscard]] auto& transition_tensor() const { return m_transition_tensor; }
   [[nodiscard]] auto& step_reward() const { return m_step_reward; }
   [[nodiscard]] size_t size() const { return m_size; };
   [[nodiscard]] auto& shape() const { return m_grid_shape; };
   [[nodiscard]] auto& location() const { return std::get< 1 >(m_location); };
   [[nodiscard]] auto& location_idx() const { return std::get< 0 >(m_location); };

   void reseed(std::mt19937_64::result_type seed) { m_rng = std::mt19937_64{seed}; }

   [[nodiscard]] std::string action_name(size_t action) const;

   [[nodiscard]] constexpr static auto num_actions() { return m_num_actions; }

   [[nodiscard]] constexpr std::array< long, dim > action_as_vector(size_t action) const;

   ///
   /// OPENAI Gymnasium API needs to be replicated on the c++ side.
   ///
   /// The following set of methods only fulfills the API.
   /// For more details see: https://gymnasium.farama.org/api/env/

   std::tuple< obs_type, double, bool, bool > step(size_t action);

   const obs_type& reset(std::optional< std::mt19937_64::result_type > seed = std::nullopt)
   {
      if(seed.has_value()) {
         reseed(*seed);
      }
      const auto row_index = m_start_state_distribution(m_rng);
      idx_xstacktensor< dim > start_coordinates = xt::row(
         m_start_states, static_cast< long >(row_index)
      );
      m_location = std::pair{index_state(start_coordinates), start_coordinates};
      return m_location;
   }

   /// the render mode of this environment is ANSI
   /// see https://gymnasium.farama.org/api/env/#gymnasium.Env.render for more info.
   [[nodiscard]] std::string render() const;

   /// a gridworld environment currently does not require any external streams to be opened.
   void close() const {}

   const auto& action_space() const { return m_action_space; }

   const auto& observation_space() const { return m_obs_space; }

   const auto& reward_range() const { return m_reward_range; }

  private:
   /// the number of actions are dependant only on the grid dimensionality. 'Back' and 'Forth'
   /// are the actions that can be done in each dimension.
   constexpr static size_t m_num_actions = 2 * dim;
   /// the lengths of each grid dimension
   idx_xstacktensor< dim > m_grid_shape;
   /// the cumulative product shape from the last dimension to the 0th dimension
   idx_xstacktensor< dim > m_grid_shape_products;
   /// the total number of states in this grid
   size_t m_size;
   /// shape (n, DIM)
   /// This member's data storage is shared by all the following states containers blow.
   /// Post construction of the gridworld opbject, do not modify this member's data buffer anymore
   /// or memory access faults will occur.
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
   /// shape (N, A, A) where N is the total number of states (state indices) and A the total number
   /// of actions (same at each state). The tensor value for an input (s, a, a') represents the
   /// probability of ultimately applying action a' when in state s and choosing action a.
   xarray< double > m_transition_tensor;
   /// the reward map for a given state
   RewardMap m_reward_map;
   /// the reward an agent achieves/pays per step
   double m_step_reward;
   /// the current position of the agent as index array and associated coordinates
   obs_type m_location{};
   /// the action space underlying this environment
   TypedDiscreteSpace< size_t > m_action_space;
   /// the observation space underlying this environment
   TypedTupleSpace< TypedDiscreteSpace< size_t >, TypedMultiDiscreteSpace< size_t > > m_obs_space;
   /// the minimum and maximum reward that can be expected from the environment
   std::pair< double, double > m_reward_range;

   /// the random number generator
   std::mt19937_64 m_rng{std::random_device{}()};

   [[nodiscard("Discarding this value will cause a memory leak.")]]  //
   constexpr static std::span< double, std::dynamic_extent >
   c_array(const size_t size, const auto value)
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
         throw std::invalid_argument(fmt::format(
            "Array is not exactly two dimensional. Actual dimensions: {}", arr_shape.size()
         ));
      }
      if(arr_shape[1] != dimensions) {
         throw std::invalid_argument(fmt::format(
            "Dimension mismatch:\n"
            "Passed states array has coordinate dimensions: {}\n"
            "The expected dimensions are: {}",
            arr.dimension(),
            dimensions
         ));
      }
   }

   template < typename Array, typename Rng >
   void assert_shape(const Array& arr, Rng&& shape) const
   {
      auto arr_shape = arr.shape();
      if(not ranges::equal(arr_shape, shape)) {
         throw std::invalid_argument(fmt::format(
            "Shape mismatch:\n"
            "Passed array has shape: {}\n"
            "The required shape is: {}",
            arr_shape,
            std::forward< Rng >(shape)
         ));
      }
   }

   template < ranges::range Range >
   idx_xstacktensor< dim > _adapt_coords(const Range& coords_range) const;

   template < ranges::range Range >
   idx_xstacktensor< dim > _verify_shape(const Range& coords_range) const;

   xarray< double > _init_transition_tensor(
      std::variant< double, pyarray< double > > transition_matrix
   ) const;

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
   static void rearrange_layout(Array& arr)
   {
      if(arr.layout() != xt::layout_type::row_major) {
         arr.resize(arr.shape(), xt::layout_type::row_major);
      }
   }

   constexpr void _assert_action_in_bounds(size_t action) const;

   constexpr static std::array< long, dim > _action_as_vector(size_t action) noexcept;

   template < std::integral T >
   constexpr static long _direction_from_remainder(T remainder) noexcept
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
   /// If the input coordinates do not have the same dimension, then the outcome will only
   /// compare coordinates up to the smaller range's size.
   ///
   /// Note, this essentially only foregoes the length check in ranges::equal.
   /// \tparam R1 the 1st input range type, needs to be a range over `size_t`
   /// \tparam R2 the 2nd input range type, needs to be a range over `size_t`
   /// \param rng1 reference to the 1st input range
   /// \param rng2 reference to the 2nd input range
   /// \return bool, do both coordinate ranges agree
   template < ranges::range R1, ranges::range R2 >
      requires(detail::expected_value_type< size_t, R1 > and detail::expected_value_type< size_t, R2 >)
   constexpr bool _equal_coords(const R1& rng1, const R2& rng2) const noexcept
   {
      // this should be sightly more efficient than ranges::equal, since `equal` checks
      // for same length which we already know is the case, because the coords are adapted and
      // the goal coords are verified upon construction
      return ranges::all_of(ranges::views::zip(rng1, rng2), [](const auto& coord_pair) {
         return std::get< 0 >(coord_pair) == std::get< 1 >(coord_pair);
      });
   }

   /// \brief returns the xarray associated holing all the states of the given state type.
   template < StateType state_type >
   constexpr auto& _states() const;

   template < ranges::range Range >
   idx_xstackvector< dim > to_xstackvector(const Range& range) const
   {
      idx_xstackvector< dim > vec;
      ranges::copy(range, vec.begin());
      return vec;
   }
};

}  // namespace force

#include "gridworld.tcc"

#endif  // REINFORCE_GRIDWORLD_HPP
