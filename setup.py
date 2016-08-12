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

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import versioneer

if __name__ == "__main__":
    setup(install_requires = ['numpy', 'scipy'],
          packages = find_packages(),
          package_data = {
              "": ["test_data/*.mtx.gz"],
              },
          test_suite = "nose.collector",
          name = DISTNAME,
          version = versioneer.get_version(),
          cmdclass = versioneer.get_cmdclass(),
          maintainer = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          url = URL,
          long_description = LONG_DESCRIPTION,
          classifiers =
            [ 'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Topic :: Scientific/Engineering'],
          # You may specify the directory where CHOLMOD is installed using the
          # include_path and library_dirs distutils directives at the top of
          # cholmod.pyx.
          ext_modules = cythonize(
              "sksparse/cholmod.pyx",
              aliases={"NP_GET_INCLUDE": np.get_include()}))
