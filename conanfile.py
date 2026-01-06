import os

from conan import ConanFile
from conan.tools.cmake import cmake_layout


class Pkg(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = (
        "CMakeDeps",
        "CMakeToolchain",
    )
    default_options = {
        "llvm-core/*:targets": "X86",
        "boost/*:fPIC": "True",
        "boost/*:header_only": "False",
        "boost/*:without_contract": "True",
        "boost/*:without_fiber": "True",
        "boost/*:without_graph": "True",
        "boost/*:without_graph_parallel": "True",
        "boost/*:without_iostreams": "True",
        "boost/*:without_json": "True",
        "boost/*:without_locale": "True",
        "boost/*:without_log": "True",
        "boost/*:without_math": "True",
        "boost/*:without_mpi": "True",
        "boost/*:without_nowide": "True",
        "boost/*:without_python": "True",
        "boost/*:without_random": "True",
        "boost/*:without_regex": "True",
        "boost/*:without_stacktrace": "True",
        "boost/*:without_test": "True",
        "boost/*:without_timer": "True",
        "boost/*:without_type_erasure": "True",
        "boost/*:without_wave": "True",
    }

    def requirements(self):
        self.requires("fmt/8.0.1")
        self.requires("spdlog/1.9.2")
        self.requires("boost/1.85.0")
        self.requires("elfio/3.11")
        self.requires("lz4/1.9.3")
        self.requires("yaml-cpp/0.7.0")
        self.requires("jsoncpp/1.9.5")
        self.requires("zlib/1.3.1")
        self.requires("asmjit/cci.20240531")
        if "WITH_LLVM" in os.environ:
            self.requires("llvm-core/19.1.7")
        if os.path.isdir("dbt-rise-plugins"):
            self.requires("lua/5.4.3")

    def build_requirements(self):
        pass

    def layout(self):
        cmake_layout(self)
