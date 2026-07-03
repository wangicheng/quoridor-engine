from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "quoridor_core",
        sources=[
            "cpp/bindings.cpp",
            "cpp/board.cpp",
            "cpp/mcgs.cpp",
        ],
        cxx_std=17,
    ),
]

setup(
    name="quoridor-core",
    version="0.1.0",
    packages=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
