#! /usr/bin/env python

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

import sys

import numpy as np
from setuptools import setup, find_packages, Extension
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
            ['Development Status :: 3 - Alpha',
             'Environment :: Console',
             'Intended Audience :: Developers',
             'Intended Audience :: Science/Research',
             'License :: OSI Approved :: BSD License',
             'Programming Language :: Cython',
             'Topic :: Scientific/Engineering',
             'Topic :: Scientific/Engineering :: Mathematics'],
          setup_requires = ['setuptools>=18.0', 'numpy', 'cython'],
          # You may specify the directory where CHOLMOD is installed using the
          # library_dirs and include_dirs keywords in the lines below.
          ext_modules = [
              Extension("sksparse.cholmod", ["sksparse/cholmod.pyx"],
                        include_dirs=[np.get_include(),
                                      sys.prefix + "/include",
                                      # Debian's suitesparse-dev installs to
                                      # /usr/include/suitesparse
                                      "/usr/include/suitesparse"],
                        library_dirs=[],
                        libraries=['cholmod'])],
          )
