from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
      name='core',
      ext_modules=cythonize("/project/6080355/shenranw/Matcha-TTS/matcha/utils/monotonic_align/core.pyx"),
      include_dirs=[numpy.get_include()]
)
