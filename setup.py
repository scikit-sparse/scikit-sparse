# Copyright (C) 2008-2025 The scikit-sparse developers:
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
# 2025-       Bernard Roesler         <bernard.roesler@gmail.com>

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup


def get_numpy_include():
    """Get the include directory for NumPy."""
    try:
        import numpy as np  # noqa: PLC0415

        return np.get_include()
    except ImportError:
        return []


INCLUDE_DIRS = []
LIBRARY_DIRS = []

numpy_include = get_numpy_include()
if numpy_include:
    INCLUDE_DIRS.append(numpy_include)

# Check user SuiteSparse directories first
user_include_dir = os.getenv("SUITESPARSE_INCLUDE_DIR")
user_library_dir = os.getenv("SUITESPARSE_LIBRARY_DIR")

if user_include_dir:
    INCLUDE_DIRS.append(user_include_dir)

if user_library_dir:
    LIBRARY_DIRS.append(user_library_dir)

# Check if suitesparse is installed via conda
conda_prefix = os.getenv("CONDA_PREFIX")

if conda_prefix:
    conda_include = Path(conda_prefix) / "include" / "suitesparse"
    conda_lib = Path(conda_prefix) / "lib"
    if conda_include.is_dir():
        INCLUDE_DIRS.append(str(conda_include))
    if conda_lib.is_dir():
        LIBRARY_DIRS.append(str(conda_lib))

# Check if suitesparse is installed via homebrew
try:
    homebrew_prefix = (
        subprocess.run(
            "readlink -f $(brew --prefix suitesparse)",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,  # raise an error if command fails
        )
        .stdout.decode()
        .strip()
    )
    brew_include = Path(homebrew_prefix) / "include" / "suitesparse"
    brew_lib = Path(homebrew_prefix) / "lib"
    if brew_include.is_dir():
        INCLUDE_DIRS.append(str(brew_include))
    if brew_lib.is_dir():
        LIBRARY_DIRS.append(str(brew_lib))
except Exception:
    pass

# Check system-wide directories
INCLUDE_DIRS.append(str(Path(sys.prefix) / "include"))
INCLUDE_DIRS.append("/usr/include/suitesparse")  # Linux default path

extension_names = ["cholmod", "amd", "btf", "camd", "colamd", "ccolamd", "umfpack"]

extensions = [
    Extension(
        f"sksparse.{name}",
        [f"src/sksparse/{name}.pyx"],
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=[name],
    )
    for name in extension_names
]

# No need to call "cythonize" here. Rely on pyproject.toml.
setup(ext_modules=extensions)
