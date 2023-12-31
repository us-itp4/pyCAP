from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy
os.environ['CFLAGS'] = '-O3'

setup(
    ext_modules = cythonize("*.pyx"),
    include_dirs=[numpy.get_include()]
)
