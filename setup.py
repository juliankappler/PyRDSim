import numpy
import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True

extension = Extension("pyrdsim/*", ["pyrdsim/*.pyx"],
    include_dirs=[numpy.get_include()],
                    )

setup(
    name='PyRDSim',
    version='0.0.1',
    url='https://github.com/juliankappler/',
    author='Julian Kappler',
    license='MIT',
    description='python library for numerical simulation of the many-particle'\
                +' overdamped Langevin equation, including reactions',
    long_description='PyRDSim is a python library for the numerical'\
        + ' simulation of the many-particle overdamped Langevin equation.'\
        + 'The package includes the possibility for defining molecules, as'\
        +' well as for including reactions that change the interaction'\
        +' properties of particles.',
    platforms='works on all platforms',
    ext_modules=cythonize([ extension ],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    packages=['pyrdsim'],
    package_data={'pyrdsim': ['*.pxd']},
)
