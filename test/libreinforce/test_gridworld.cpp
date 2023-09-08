#include "gtest/gtest.h"
#include "pybind11/embed.h"
#include "reinforce/reinforce.hpp"


using namespace force;

TEST(PythonImport, numpy) {
    // the bug https://github.com/pybind/pybind11/issues/4654 may cause pybind11 to not find the
    // numpy installation when made for the user, e.g. as happens when numpy is installed via 'pip3
    // install numpy'. This call places packages typically in a ~/.local/lib/python3.xx folder.
    // Such an import will then fail to find numpy, as it appears to only look in system wide install
    // folders, such as /usr/local/lib/python3.xx/dist-packages etc. (see the GitHub issue).
    // A workaround is to install numpy system-wide via sudo if possible, i.e.
    //    'sudo pip3 install numpy',
    // but this may well be impossible on some systems. An alternative may be to pass the python
    // executable path into CMake to use a virtual env or conda-env at the given location. This may
    // automatically include the required folders, since it isn't a /usr/bin/python install. This has
    // not been tested yet though.
    ASSERT_NO_THROW(py::module_::import("numpy"));
    ASSERT_NO_THROW(py::module_::import("numpy.core.multiarray"));
}

TEST(Gridworld, construction) {
    auto numpy = py::module_::import("numpy");
    xt::import_numpy();
    const idx_xarray goals{{2, 0, 0},
                           {2, 2, 1}};
    const idx_xarray starts{{0, 1, 2}};
    auto grid_env = Gridworld<3>({2, 3}, starts, goals, 1.);
}