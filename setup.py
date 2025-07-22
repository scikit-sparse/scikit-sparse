# Copyright (C) 2008-2017 The scikit-sparse developers:
#
# 2008        David Cournapeau        <cournape@gmail.com>
# 2009-2015   Nathaniel Smith         <njs@pobox.com>
# 2010        Dag Sverre Seljebotn    <dagss@student.matnat.uio.no>
# 2014        Leon Barrett            <lbarrett@climate.com>
# 2015        Yuri                    <yuri@tsoft.com>
# 2016-2017   Antony Lee              <anntzer.lee@gmail.com>
# 2016        Alex Grigorievskiy      <alex.grigorievskiy@gmail.com>
# 2016-2017   Joscha Reimer           <jor@informatik.uni-kiel.de>
# 2021-       Justin Ellis            <justin.ellis18@gmail.com>
# 2022-       Aaron Johnson           <aaron9035@gmail.com>

import os
import subprocess
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

INCLUDE_DIRS = [
    np.get_include(),
    sys.prefix + "/include",
    # Debian's suitesparse-dev installs to
    "/usr/include/suitesparse",
]
LIBRARY_DIRS = []

# check if suitesparse is installed via homebrew
homebrew_suitesparse_dir = (
    subprocess.run(
        "readlink -f $(brew --prefix suitesparse)",
        shell=True,
        stdout=subprocess.PIPE,
    )
    .stdout.decode()
    .strip()
)

# empty string if not found (because error is printed to stderr)
if homebrew_suitesparse_dir:
    INCLUDE_DIRS.append(
        # Include directory for homebrew-installed suitesparse
        homebrew_suitesparse_dir
        + "/include/suitesparse/",
    )
    LIBRARY_DIRS.append(
        # Library directory for homebrew-installed suitesparse
        homebrew_suitesparse_dir
        + "/lib"
    )

user_include_dir = os.getenv("SUITESPARSE_INCLUDE_DIR")
user_library_dir = os.getenv("SUITESPARSE_LIBRARY_DIR")

if user_include_dir:
    INCLUDE_DIRS.append(user_include_dir)

if user_library_dir:
    LIBRARY_DIRS.append(user_library_dir)

setup(
    # You may specify the directory where CHOLMOD is installed using the
    # library_dirs and include_dirs keywords in the lines below.
    ext_modules=cythonize(
        Extension(
            "sksparse.cholmod",
            ["sksparse/cholmod.pyx"],
            include_dirs=INCLUDE_DIRS,
            library_dirs=LIBRARY_DIRS,
            libraries=["cholmod"],
        )
    ),
)
