from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain


def cmake_option_value(value):
    return "ON" if value else "OFF"


class Reinforce(ConanFile):
    name = "reinforce"
    version = "0.0.1.dev"
    package_type = "static-library"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    options = {
        "with_tbb": [True, False],
        "with_pymodule": [True, False],
        "with_fast_math": [True, False],
        "with_testing": [True, False],
    }
    default_options = {
        "with_tbb": False,
        "with_pymodule": False,
        "with_fast_math": False,
        "with_testing": False,
    }

    def requirements(self):
        requirements = self.conan_data.get("requirements", [])
        for requirement in requirements:
            self.requires(requirement)
        if self.options.with_tbb:
            self.requires("onetbb/2021.10.0")
        if self.options.with_pymodule:
            self.requires("pybind11/2.12.0")

        self.test_requires("gtest/[>=1.13.0]")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        # tc.user_presets_path = 'ConanPresets.json'
        tc.user_presets_path = False
        for key, value in dict(
            enable_build_docs=False,
            enable_build_python_extension=cmake_option_value(
                self.options.with_pymodule
            ),
            enable_build_benchmark=False,
            enable_build_with_time_trace=False,
            enable_cache=False,
            enable_clang_tidy=False,
            enable_coverage=False,
            enable_cppcheck=False,
            enable_fast_math=cmake_option_value(self.options.with_fast_math),
            enable_include_what_you_use=False,
            enable_ipo=False,
            enable_pch=False,
            enable_sanitizer_address=False,
            enable_sanitizer_leak=False,
            enable_sanitizer_memory=False,
            enable_sanitizer_thread=False,
            enable_sanitizer_undefined_behavior=False,
            enable_testing=cmake_option_value(self.options.with_testing),
            enable_warning_padding=False,
            use_tbb=cmake_option_value(self.options.with_tbb),
            use_pybind11_findpython=True,
            install_pymodule=False,
            warnings_as_errors=False,
        ).items():
            tc.variables[key.upper()] = value
        tc.generate()
