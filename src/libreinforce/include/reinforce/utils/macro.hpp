
#ifndef REINFORCE_MACRO_HPP
#define REINFORCE_MACRO_HPP

#ifndef fwd_lambda
   #define fwd_lambda(func) \
      [](auto&&... args) -> decltype(auto) { return func(std::forward< decltype(args) >(args)...); }
#endif

#endif  // REINFORCE_MACRO_HPP
