from setuptools import setup, Extension
from Cython.Build import cythonize
import os

extensions = [
    Extension('misc', sources=[os.path.join('src', 'tightbinding', 'misc.pyx')]),
    Extension('sc.core', sources=[os.path.join('src', 'tightbinding', 'sc', 'core.pyx')]),
]

setup(
    ext_modules=cythonize(extensions, language_level=3)
)
