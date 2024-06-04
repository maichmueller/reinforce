
#ifndef REINFORCE_SPACE_FACTORY_HPP
#define REINFORCE_SPACE_FACTORY_HPP

#include "reinforce/utils/macro.hpp"
#include "reinforce/utils/type_traits.hpp"

namespace force {

template < bool stacked, typename FeatureSpace, typename... OtherArgs >
auto make_sequence_space(FeatureSpace&& space, OtherArgs&&... args)
{
   return SequenceSpace< detail::raw_t< FeatureSpace >, stacked >{FWD(space), FWD(args)...};
}

}  // namespace force

#endif  // REINFORCE_SPACE_FACTORY_HPP
