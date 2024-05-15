
#ifndef REINFORCE_FWD_HPP
#define REINFORCE_FWD_HPP

#include "reinforce/spaces/concepts.hpp"

namespace force {

template < typename Value, typename Derived, typename BatchValue, bool runtime_sample_throw >
class Space;

class MultiBinarySpace;

class TextSpace;

template < typename T >
   requires discrete_reqs< T >
class DiscreteSpace;

template < typename T >
   requires box_reqs< T >
class BoxSpace;

template < typename T >
   requires multidiscrete_reqs< T >
class MultiDiscreteSpace;

template < typename NS, typename ES >
class GraphSpace;

template < typename... Spaces >
class TupleSpace;

template < typename... Spaces >
class OneOfSpace;

template < typename FS, bool stacked >
class SequenceSpace;

}  // namespace force

#endif  // REINFORCE_FWD_HPP
