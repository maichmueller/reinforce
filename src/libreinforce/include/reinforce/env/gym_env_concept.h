
#ifndef REINFORCE_GYMNASIUM_ENV_CONCEPT_HPP
#define REINFORCE_GYMNASIUM_ENV_CONCEPT_HPP

#include <any>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#include "reinforce/utils/type_traits.hpp"
#include "reinforce/utils/utils.hpp"

namespace force {

template < typename Tuple, typename ObservationT >
concept matches_step_return_type = requires(Tuple tuple) {
   requires detail::is_specialization_v< Tuple, std::tuple >;
   // the gym api states that the return type should be a tuple of length 5
   requires std::tuple_size_v< Tuple > == 5;
   // whose first 4 entries are Observation
   // (ObservationT), Reward (double), Truncated (bool), Terminated (bool)
   {
      detail::tuple_slice(tuple, std::make_index_sequence< 3 >{})
   } -> std::same_as< std::tuple< ObservationT, double, bool, bool > >;
   // tuple element 5 is `info` (py::dict of any contents which we cannot enforce without using
   // py::dict ourselves in the c++ code)
};

template < typename Tuple, typename ObservationT >
concept matches_reset_return_type = requires(Tuple tuple) {
   requires detail::is_specialization_v< Tuple, std::tuple >;
   // the gym api states that the return type should be a tuple of length 2
   requires std::tuple_size_v< Tuple > == 2;
   {
      std::get< 0 >(tuple)
   } -> std::same_as< ObservationT >;
   // tuple element 2 is `info` (py::dict of any contents which we cannot enforce without using
   // py::dict ourselves in the c++ code)
};

template < typename Env, typename ActionT, typename ObservationT >
concept gym_env = requires(Env env, ActionT action, ObservationT observation) {
   {
      env.step(action)
   } -> std::same_as<
      std::tuple< ObservationT, double, bool, bool, std::unordered_map< std::string, std::any > > >;
   {
      env.reset(
         std::optional< ActionT >{}, std::optional< std::unordered_map< std::string, std::any > >{}
      )
   } -> std::same_as< std::tuple< ObservationT, std::unordered_map< std::string, std::any > > >;
};

}  // namespace force

#endif  // REINFORCE_GYMNASIUM_ENV_CONCEPT_HPP
