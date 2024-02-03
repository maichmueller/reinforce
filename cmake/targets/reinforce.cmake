
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
        pcg-cpp::pcg-cpp
        range-v3::range-v3
        fmt::fmt
        xtensor
        spdlog::spdlog
        frozen::frozen
        xtensor-python
        xtensor-blas
        pybind11::module
        pybind11::lto
        "${Python3_LIBRARIES}"
)

set_target_properties(${reinforce_lib} PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_compile_definitions(
        ${reinforce_lib}
        PUBLIC
        XTENSOR_USE_XSIMD
        # turn off logging in release build, allow debug-level logging in debug build
        SPDLOG_ACTIVE_LEVEL=$<$<CONFIG:RELEASE>:SPDLOG_LEVEL_INFO>$<$<CONFIG:DEBUG>:SPDLOG_LEVEL_DEBUG>
)
