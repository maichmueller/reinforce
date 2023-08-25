set(BENCHMARK_SOURCES main.cpp)

list(TRANSFORM BENCHMARK_SOURCES PREPEND "${PROJECT_REINFORCE_BENCHMARK_SRC_DIR}/")

add_executable(${reinforce_benchmark} ${BENCHMARK_SOURCES})

target_link_libraries(${reinforce_benchmark} PRIVATE ${reinforce_lib}_envs CONAN_PKG::benchmark)
