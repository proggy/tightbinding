from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import numpy

extensions = [
    Extension('misc', sources=[os.path.join('src', 'misc.pyx')], include_dirs=[numpy.get_include()]),
    Extension('sc.core', sources=[os.path.join('src', 'sc', 'core.pyx')], include_dirs=[numpy.get_include()]),
]

setup(
    #package_dir={'cython_test': ''},
    ext_modules=cythonize(extensions, language_level=3)
)
