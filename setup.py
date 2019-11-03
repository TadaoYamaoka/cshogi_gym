from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

ext_modules = [
    Extension('cshogi_gym.features',
        ['cshogi_gym/features.pyx'],
        language='c++',
        include_dirs = [numpy.get_include()],
        ),
]

setup(
    name='cshogi_gym',
    version='0.0.0',
    packages=['cshogi_gym'],
    ext_modules=ext_modules,
)