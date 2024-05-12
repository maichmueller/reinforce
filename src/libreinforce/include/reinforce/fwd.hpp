
#ifndef REINFORCE_FWD_HPP
#define REINFORCE_FWD_HPP

namespace force {

template <
   typename Value,
   typename Derived,
   typename BatchValue = Value,
   bool runtime_sample_throw = false >
class Space;

class MultiBinarySpace;

class TextSpace;

template < typename T >
class DiscreteSpace;

template < typename T >
class BoxSpace;

template < typename T >
class MultiDiscreteSpace;

template < typename... Spaces >
class TupleSpace;

template < typename... Spaces >
class OneOfSpace;

template < typename NS, typename ES >
class GraphSpace;

template < typename FS, bool stacked = true >
class SequenceSpace;

}  // namespace force

#endif  // REINFORCE_FWD_HPP
