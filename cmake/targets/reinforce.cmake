
set(
        LIBREINFORCE_SOURCES
        multi_binary.cpp
        text.cpp
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
        $<$<BOOL:${TBB_FOUND}>:onetbb::onetbb>
        spdlog::spdlog
        frozen::frozen
#        xtensor-blas
)

set_target_properties(${reinforce_lib} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(
        ${reinforce_lib}
        PUBLIC
        XTENSOR_USE_XSIMD
        $<$<BOOL:${TBB_FOUND}>:XTENSOR_USE_TBB>
        # turn off logging in release build, allow debug-level logging in debug build
        SPDLOG_ACTIVE_LEVEL=$<$<CONFIG:RELEASE>:SPDLOG_LEVEL_INFO>$<$<CONFIG:DEBUG>:SPDLOG_LEVEL_DEBUG>
)


if(ENABLE_BUILD_PYTHON_EXTENSION)
    target_link_libraries(
            ${reinforce_lib}
            PUBLIC
            xtensor-python
            pybind11::module
            pybind11::lto
            "${Python3_LIBRARIES}"
    )
    target_compile_definitions(
            ${reinforce_lib}
            PUBLIC
            REINFORCE_USE_PYTHON
    )
endif ()