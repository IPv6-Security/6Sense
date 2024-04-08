from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Print IPs',
    ext_modules=cythonize("Generator/cython_print_ips.pyx"),
    zip_safe=False,
)