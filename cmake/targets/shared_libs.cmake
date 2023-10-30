add_library(common_testing_utils INTERFACE)

target_include_directories(common_testing_utils INTERFACE "${PROJECT_TEST_DIR}/shared_test_utils")

add_library(shared_test_libs INTERFACE)
target_link_libraries(
        shared_test_libs
        INTERFACE
        ${reinforce_lib}
        project_options
        project_warnings
        common_testing_utils
        GTest::gtest
        fmt::fmt-header-only
        range-v3::range-v3
        pybind11::module
)
