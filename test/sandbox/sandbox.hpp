#ifndef NOR_SANDBOX_HPP
#define NOR_SANDBOX_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <utility>

namespace py = pybind11;

struct Messenger {
   explicit Messenger(std::string name_ = "DefaultObject") : name(std::move(name_))
   {
      std::cout << "CTOR - " << name << "\n";
   }
   ~Messenger() { std::cout << "DTOR - " << name << "\n"; }
   Messenger(const Messenger& other) : name(other.name) { std::cout << "COPY - " << name << "\n"; }
   Messenger(Messenger&& other) noexcept : name(std::move(other.name))
   {
      std::cout << "MOVE - " << name << "\n";
   }
   Messenger& operator=(const Messenger& other)
   {
      if(this == &other) {
         return *this;
      }
      name = other.name;
      std::cout << "COPY assignment - " << name << "\n";
      return *this;
   }
   Messenger& operator=(Messenger&& other) noexcept
   {
      if(this == &other) {
         return *this;
      }
      name = std::move(other.name);
      std::cout << "MOVE assignment - " << name << "\n";
      return *this;
   }

   std::string name;
};

#endif  // NOR_SANDBOX_HPP
