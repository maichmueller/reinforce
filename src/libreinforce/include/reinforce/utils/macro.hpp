
#ifndef REINFORCE_MACRO_HPP
#define REINFORCE_MACRO_HPP

#ifndef FWD
   #define FWD(x) std::forward< decltype(x) >(x)
#endif  // FWD
#ifndef AS_LAMBDA
   #define AS_LAMBDA(func) [](auto&&... args) { return func(FWD(args)...); }
#endif  // AS_LAMBDA
/// creates a lambda wrapper around the given function with perfect-return
/// (avoids copies on return when references are passed back (and supposed to remain references)
#ifndef AS_PRFCT_LAMBDA
   #define AS_PRFCT_LAMBDA(func) [](auto&&... args) -> decltype(auto) { return func(FWD(args)...); }
#endif  // AS_PRFCT_LAMBDA

#endif  // REINFORCE_MACRO_HPP
