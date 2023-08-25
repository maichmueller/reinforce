set(
        PYTHON_MODULE_SOURCES
        module.cpp
)
list(TRANSFORM PYTHON_MODULE_SOURCES PREPEND "${PROJECT_REINFORCE_BINDING_DIR}/")

if (ENABLE_BUILD_PYTHON_EXTENSION AND INSTALL_PYMODULE)
    set(_pyreinforce_exclude_from_all)
else ()
    set(_pyreinforce_exclude_from_all EXCLUDE_FROM_ALL)
endif ()

pybind11_add_module(
        ${reinforce_pymodule}
        MODULE
        ${_pyreinforce_exclude_from_all}
        ${LIBRARY_SOURCES}
        ${PYTHON_MODULE_SOURCES})

set_target_properties(
        ${reinforce_pymodule}
        PROPERTIES
        LIBRARY_OUTPUT_NAME _${reinforce_pymodule}
        CXX_VISIBILITY_PRESET hidden
)
target_link_libraries(
        ${reinforce_pymodule}
        PUBLIC
        ${reinforce_lib}
        pybind11::module
        $<$<NOT:$<BOOL:USE_PYBIND11_FINDPYTHON>>:Python3::Module>
)
