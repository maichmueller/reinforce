
#ifndef REINFORCE_MACRO_HPP
#define REINFORCE_MACRO_HPP

#ifndef FWD
   #define FWD(x) std::forward< decltype(x) >(x)
#endif  // FWD

#ifndef AS_LAMBDA
   #define AS_LAMBDA(func) [](auto&&... args) { return func(FWD(args)...); }
#endif  // AS_LAMBDA
/// creates a lambda wrapper around the given function and captures surrounding state by ref
#ifndef AS_CPTR_LAMBDA
   #define AS_CPTR_LAMBDA(func) [&](auto&&... args) { return func(FWD(args)...); }
#endif  // AS_CPTR_LAMBDA

/// creates a lambda wrapper around the given function with perfect-return
/// (avoids copies on return when references are passed back (and supposed to remain references)
#ifndef AS_PRFCT_LAMBDA
   #define AS_PRFCT_LAMBDA(func) [](auto&&... args) -> decltype(auto) { return func(FWD(args)...); }
#endif  // AS_PRFCT_LAMBDA

/// creates a lambda wrapper around the given function with perfect-return and captures surrounding
/// state by ref
#ifndef AS_PRFCT_CPTR_LAMBDA
   #define AS_PRFCT_CPTR_LAMBDA(func) \
      [&](auto&&... args) -> decltype(auto) { return func(FWD(args)...); }
#endif  // AS_PRFCT_CPTR_LAMBDA

#ifndef FORCE_DEBUG_ASSERT
   #define FORCE_DEBUG_ASSERT(expression)                                             \
      if(not (expression)) {                                                          \
         throw force_library_error{"Expression " #expression " evaluated to false."}; \
      }
#endif

#endif  // REINFORCE_MACRO_HPP
