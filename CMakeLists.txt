cmake_minimum_required(VERSION 3.17)

project(
        reinforce
        VERSION 0.0.1
        LANGUAGES CXX
)

string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# options
option(ENABLE_BUILD_DOCS "Enable building the docs. Requires doxygen to be installed on the system" OFF)
option(ENABLE_BUILD_PYTHON_EXTENSION "Enable building the python extension." ON)
option(ENABLE_BUILD_BENCHMARK "Enable building of the benchmarks." OFF)
option(ENABLE_BUILD_SANDBOX "Enable building of the sandbox testbed (Only for development purposes)." ON)
option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
option(ENABLE_CACHE "Enable cache if available" ON)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)
option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_FAST_MATH "Enable Compiler specific breakway from IEEE compliance to speed up math computations" OFF)
option(ENABLE_INCLUDE_WHAT_YOU_USE "Enable static analysis with include-what-you-use" OFF)
option(ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
option(ENABLE_PCH "Enable Precompiled Headers" ON)
option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
option(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR "Enable undefined behavior sanitizer" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)
option(ENABLE_WARNING_PADDING "Warn when structs are memory padded" OFF)
option(USE_PYBIND11_FINDPYTHON "Use pybind11 to search for the python library" OFF)
option(INSTALL_PYMODULE "Configure installation for python module." OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# settings

# Use C++20
set(PROJ_CXX_STD_FEATURE cxx_std_20)
set(PROJ_CXX_STANDARD C++20)


set(reinforce-lib-type STATIC)
set(reinforce_lib reinforce++)
set(reinforce_pymodule reinforce)
set(reinforce_test reinforce_tests)
set(reinforce_benchmark reinforce_benchmark)

set(CONANFILE conanfile.txt)
set(DEPENDENCY_DIR deps) # has to be relative to CMAKE_CURRENT_SOURCE_DIR
set(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test")

set(PROJECT_REINFORCE_DIR "${PROJECT_SRC_DIR}/libreinforce")
set(PROJECT_PYREINFORCE_DIR "${PROJECT_SRC_DIR}/reinforce")
set(PROJECT_ENVS_DIR "${PROJECT_SRC_DIR}/envs")
set(PROJECT_BENCHMARK_DIR "${PROJECT_DIR}/benchmark")

set(PROJECT_REINFORCE_SRC_DIR "${PROJECT_REINFORCE_DIR}/impl")
set(PROJECT_REINFORCE_INCLUDE_DIR "${PROJECT_REINFORCE_DIR}/include")
set(PROJECT_COMMON_INCLUDE_DIR "${PROJECT_SRC_DIR}/common/include")
set(PROJECT_REINFORCE_BINDING_DIR "${PROJECT_PYREINFORCE_DIR}/binding")
set(PROJECT_REINFORCE_BENCHMARK_SRC_DIR "${PROJECT_BENCHMARK_DIR}")

set(_cmake_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(_docs_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/_build)
set(_docs_DOXYGEN_WORKING_DIR ${PROJECT_REINFORCE_INCLUDE_DIR}) # has to be relative to CMAKE_CURRENT_SOURCE_DIR

message(STATUS "SETTINGS PRINT start:")

message("CMake Generator: ${CMAKE_GENERATOR}")
message("C Compiler: ${CMAKE_C_COMPILER}")
message("C++ Compiler: ${CMAKE_CXX_COMPILER}")
message("C Compiler Info: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message("C++ Compiler Info: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message("CURRENT CMAKE BINARY DIR: ${CMAKE_CURRENT_BINARY_DIR}")

message("REINFORCE project files directory: ${PROJECT_SRC_DIR}")
message("REINFORCE src directory: ${PROJECT_REINFORCE_SRC_DIR}")
message("REINFORCE include directory: ${PROJECT_REINFORCE_INCLUDE_DIR}")
message("REINFORCE bindings directory: ${PROJECT_REINFORCE_BINDING_DIR}")

message("REINFORCE c++ library name: ${reinforce_lib}")
message("REINFORCE python extension name: ${reinforce_pymodule}")
message("BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message("FAST MATH SWITCH: ${ENABLE_FAST_MATH}")
message("USE PYBIND11 FIND-PYTHON: ${USE_PYBIND11_FINDPYTHON}")

message(STATUS "SETTINGS PRINT end.\n")

# Set this path if the conan install is not in the standard directory This doesn't work due to a conan internal error
# when conan is not found in the standard directory (/usr/bin on linux) It doesn't work in general when CONAN_COMMAND is
# set it seems. execute_process(COMMAND which conan OUTPUT_VARIABLE CONAN_PATH)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

target_compile_features(project_options INTERFACE ${PROJ_CXX_STD_FEATURE})
if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" AND ENABLE_BUILD_WITH_TIME_TRACE)
    add_compile_definitions(project_options INTERFACE -ftime-trace)
endif ()

include(${_cmake_DIR}/settings/Utilities.cmake)

include(${_cmake_DIR}/settings/CompilerOptions.cmake)
set_project_compiler_options(project_options)
# add dependencies
# the deps to be downloaded by cpm
include(${_cmake_DIR}/settings/CPM.cmake)
# let cpm try to find other deps locally first (find_package) to allow interplay with conan
set(CPM_USE_LOCAL_PACKAGES TRUE)

# set the standard project settings
include(${_cmake_DIR}/settings/StandardProjectSettings.cmake)

# load extra clang dev tools
include(${_cmake_DIR}/settings/Clang-cxx-dev-tools.cmake)

# enable cache system
include(${_cmake_DIR}/settings/Cache.cmake)

# standard compiler warnings
include(${_cmake_DIR}/settings/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(${_cmake_DIR}/settings/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
if (ENABLE_BUILD_DOCS)
    include(${_cmake_DIR}/settings/Doxygen.cmake)
endif ()

# allow for static analysis options
include(${_cmake_DIR}/settings/StaticAnalyzers.cmake)

if (ENABLE_PCH)
    target_precompile_headers(
            project_options
            INTERFACE
            <vector>
            <array>
            <string>
            <utility>
            <memory>
            <map>
            <unordered_map>
            <ranges>
    )
endif ()


if (USE_PYBIND11_FINDPYTHON)
    set(PYBIND11_FINDPYTHON TRUE)
else ()
    set(PYBIND11_FINDPYTHON FALSE)
    find_package(Python3 COMPONENTS Interpreter Development NumPy)
    if (NOT Python3_NumPy_FOUND AND Python3_Interpreter_FOUND)
        message("Could not find numpy with the associated python interpreter ${Python3_EXECUTABLE}.
        Attempting to install it with pip.")
        message(STATUS "Executing: ${Python3_EXECUTABLE} -m pip install numpy")
        execute_process(COMMAND "${Python3_EXECUTABLE} -m pip install numpy" RESULT_VARIABLE result ERROR_VARIABLE err)
        if (result EQUAL 0)
            message(STATUS "numpy installed successfully.")
        else ()
            message(FATAL_ERROR "Failed to install numpy. Error: ${err}. Please check if pip3 is installed and try again.")
        endif ()
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
    endif ()
endif ()
find_package(pybind11 REQUIRED)

message(STATUS "Python3_FOUND: ${Python3_FOUND}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

message(STATUS "Python3_Interpreter_FOUND: ${Python3_Interpreter_FOUND}")
message(STATUS "Python3_INTERPRETER_ID: ${Python3_INTERPRETER_ID}")

message(STATUS "Python3_Development_FOUND: ${Python3_Development_FOUND}")
message(STATUS "Python3_Development.Module_FOUND: ${Python3_Development.Module_FOUND}")

message(STATUS "Python3_NumPy_FOUND: ${Python3_NumPy_FOUND}")
message(STATUS "Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")
message(STATUS "Python3_NumPy_VERSION: ${Python3_NumPy_VERSION}")

find_package(pcg-cpp REQUIRED)
find_package(xtensor REQUIRED)
find_package(spdlog REQUIRED)
find_package(range-v3 REQUIRED)
find_package(fmt REQUIRED)
find_package(frozen REQUIRED)
find_package(OpenBLAS REQUIRED)
add_definitions(-DHAVE_CBLAS=1)
find_package(LAPACK REQUIRED)

# install all CPM installed dependencies
include(${DEPENDENCY_DIR}/CPM_deps.cmake)

message("Configuring Libraries Shared By Targets.")
include(${_cmake_DIR}/targets/shared_libs.cmake)
message("Configuring Core Library '${reinforce_lib}'.")
include(${_cmake_DIR}/targets/reinforce.cmake)
if (ENABLE_TESTING)
    find_package(GTest REQUIRED)
    enable_testing()
    message("Configuring Tests.")
    include(${_cmake_DIR}/targets/tests.cmake)
endif ()
if (ENABLE_BUILD_PYTHON_EXTENSION)
    message("Configuring Python Extension '${reinforce_pymodule}'.")
    include(${_cmake_DIR}/targets/pyreinforce.cmake)
endif ()
if (ENABLE_GAMES)
    message("Configuring Game Builds.")
    include(${_cmake_DIR}/targets/games.cmake)
endif ()
if (ENABLE_BUILD_BENCHMARK)
    message("Configuring Benchmark Build.")
    include(${_cmake_DIR}/targets/benchmark.cmake)
endif ()
if (ENABLE_BUILD_SANDBOX)
    message("Configuring Sandbox Build.")
    include(${_cmake_DIR}/targets/sandbox.cmake)
endif ()

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)

include(${_cmake_DIR}/settings/Installation.cmake)