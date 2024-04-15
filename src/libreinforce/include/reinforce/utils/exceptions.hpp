#ifndef REINFORCE_EXCEPTIONS_HPP
#define REINFORCE_EXCEPTIONS_HPP

#include <fmt/format.h>

#include <exception>
#include <string_view>

namespace force {

struct not_implemented_error: public std::logic_error {
   explicit not_implemented_error(std::string_view func_name)
       : std::logic_error(fmt::format("Function {} not implemented.", func_name)){};
};

struct force_library_error: public std::runtime_error {
   explicit force_library_error(std::string_view msg) : std::runtime_error(std::string(msg)){};
};

}  // namespace force

#endif  // REINFORCE_EXCEPTIONS_HPP
