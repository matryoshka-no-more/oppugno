#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import shutil
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


def get_cuda_config():
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise EnvironmentError(
            "The nvcc binary could not be "
            "located in your $PATH. Either add it to your path, "
            "or set $CUDAHOME")
    home = os.path.dirname(os.path.dirname(nvcc))
    return {
        "nvcc": nvcc,
        "home": home,
        "lib64": os.path.join(home, "lib64"),
        "include": os.path.join(home, "include"),
    }


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it"s not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it"s kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn"t have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


CUDA = get_cuda_config()

ext = Extension(
    "oppugno_cuda",
    sources=[".cuda/cuda.cu", "cuda.pyx"],
    library_dirs=[CUDA["lib64"]],
    libraries=["cudart"],
    language="c++",
    runtime_library_dirs=[CUDA["lib64"]],
    # This syntax is specific to this build system
    # we"re only going to use certain compiler args with nvcc
    # and not with gcc the implementation of this trick is in
    # customize_compiler()
    extra_compile_args={
        "gcc": [],
        "nvcc": [
            "-arch=sm_30", "--ptxas-options=-v", "-c", "--compiler-options",
            "-fPIC"
        ]
    },
    include_dirs=[CUDA["include"], ".cuda"])

setup(name="oppugno_cuda",
      ext_modules=[ext],
      cmdclass={"build_ext": custom_build_ext},
      zip_safe=False)
