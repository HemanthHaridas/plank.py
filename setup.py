from setuptools             import  setup, find_packages
from setuptools.extension   import  Extension
from Cython.Build           import  cythonize

import  numpy
import  os

os.environ["CPPFLAGS"]  =   os.getenv("CPPFLAGS","") + "-I" + numpy.get_include()

routines                =   [
                                Extension('plank.routines.oneelectron', ['./routines/oneelectron.pyx']),
                                Extension('plank.routines.twoelectron', ['./routines/twoelectron.pyx']),
                                Extension('plank.routines.hatreefock',  ['./routines/hatreefock.pyx'])
                            ]

setup(   
        name                =   'plank',
        version             =   '0.0.1',
        packages            =   find_packages(),
        license             =   'GPLv3',
        python_requires     =   '>=3.4',
        install_requires    =   ['cython','numpy','scipy','pyyaml'],
        ext_modules         =   cythonize(routines, compiler_directives =   {'linetrace' : True, 'language_level' : '3'}),
        include_dirs        =   [numpy.get_include()]
    )
