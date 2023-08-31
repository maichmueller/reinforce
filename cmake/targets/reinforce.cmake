
set(
        LIBREINFORCE_SOURCES
        gridworld.cpp
)
list(TRANSFORM LIBREINFORCE_SOURCES PREPEND "${PROJECT_REINFORCE_SRC_DIR}/")

add_library(${reinforce_lib} ${reinforce-lib-type} ${LIBREINFORCE_SOURCES})

target_include_directories(
        ${reinforce_lib}
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_REINFORCE_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
        ${reinforce_lib}
        PUBLIC
        project_options
        CONAN_PKG::range-v3
        CONAN_PKG::fmt
        CONAN_PKG::xtensor
        pybind11::module
)

set_target_properties(${reinforce_lib} PROPERTIES CXX_VISIBILITY_PRESET hidden)
