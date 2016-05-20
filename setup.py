#! /usr/bin/env python

# Copyright (C) 2008 Cournapeau David <cournape@gmail.com>
# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>
# Copyright (C) 2016 Antony Lee <anntzer.lee@gmail.com>

"""Sparse matrix tools.

This is a home for sparse matrix code in Python that plays well with
scipy.sparse, but that is somehow unsuitable for inclusion in scipy
proper. Usually this will be because it is released under the GPL.

So far we have a wrapper for the CHOLMOD library for sparse Cholesky
decomposition. Further contributions are welcome!
"""


DISTNAME            = 'scikit-sparse'
DESCRIPTION         = 'Scikits sparse matrix package'
LONG_DESCRIPTION    = __doc__
MAINTAINER          = 'Antony Lee',
MAINTAINER_EMAIL    = 'anntzer.lee@gmail.com',
URL                 = 'https://github.com/scikit-sparse/scikit-sparse/'
LICENSE             = 'GPL'
DOWNLOAD_URL        = 'https://github.com/scikit-sparse/scikit-sparse/downloads'
VERSION             = '0.3'


import json
import os
import subprocess
import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np


def is_running_conda():
    # See http://stackoverflow.com/questions/21282363/any-way-to-tell-if-users-python-environment-is-anaconda
    return "conda" in sys.version or "Continuum" in sys.version


def get_conda_info():
    return json.loads(subprocess.check_output(["conda", "info", "--json"],
                                              universal_newlines=True))


def base_prefix():
    if is_running_conda():
        prefix = get_conda_info()["default_prefix"]
        if os.name == "posix":
            return prefix
        elif os.name == "nt":
            return os.path.join(prefix, "Library")
    else:
        if os.name == "posix":
            return "/usr"
        elif os.name == "nt":
            raise ValueError("No default include path on Windows")


def suitesparse_include():
    return os.path.join(base_prefix(), "include", "suitesparse")


def library_dirs():
    return [os.path.join(base_prefix(), "lib"), os.path.join(base_prefix(), "bin")]


def suitesparse_libraries():
    if os.name == "posix":
        return "cholmod"
    elif os.name == "nt":
        return "libcholmod"


if __name__ == "__main__":
    setup(name=DISTNAME,
          version=VERSION,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          classifiers=
            [ 'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Topic :: Scientific/Engineering'],
          # You may specify the directory where CHOLMOD is installed using the
          # include_path and library_dirs distutils directives at the top of
          # cholmod.pyx.
          ext_modules=cythonize(
              "sksparse/cholmod.pyx",
              aliases={"NUMPY_INCLUDE": np.get_include(),
                       "SUITESPARSE_INCLUDE": suitesparse_include(),
                       "LIBRARY_DIRS": library_dirs(),
                       "SUITESPARSE_LIB": suitesparse_libraries()}),
          install_requires=['numpy', 'scipy'],
          packages=find_packages(),
          package_data={
              "": ["test_data/*.mtx.gz"],
              },
          test_suite="nose.collector")
