include("${_cmake_DIR}/settings/Utilities.cmake")

register_reinforce_target(
        ${reinforce_test}_gridworld
        test_gridworld.cpp
)
register_reinforce_target(
        ${reinforce_test}_spaces
        test_space_box.cpp
        test_space_discrete.cpp
        test_space_multidiscrete.cpp
        test_space_tuple.cpp
        test_space_multibinary.cpp
        test_space_text.cpp
        test_space_sequence.cpp
        test_space_graph.cpp
        test_space_oneof.cpp
)


# for the overall test executable we simply merge all other test files together
foreach (sources_list IN LISTS REGISTERED_TEST_SOURCES_LIST)
    list(APPEND REINFORCE_TEST_SOURCES ${${sources_list}})
endforeach ()
register_reinforce_target(${reinforce_test}_all ${REINFORCE_TEST_SOURCES})

# the test of all parts needs an extra linkage for the pybind11 components and Python
target_link_libraries(
        ${reinforce_test}_all
        PRIVATE
        project_options
)
if(ENABLE_BUILD_PYTHON_EXTENSION)
    target_link_libraries(
            ${reinforce_test}_all
            PUBLIC
            pybind11::module
            pybind11::embed
            $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>
    )
endif ()

