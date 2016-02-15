#! /usr/bin/env python

# Copyright (C) 2008 Cournapeau David <cournape@gmail.com>
# Copyright (C) 2009 Nathaniel Smith <njs@pobox.com>

descr   = """Sparse matrix tools.

This is a home for sparse matrix code in Python that plays well with
scipy.sparse, but that is somehow unsuitable for inclusion in scipy
proper. Usually this will be because it is released under the GPL.

So far we have a wrapper for the CHOLMOD library for sparse cholesky
decomposition. Further contributions are welcome!
"""

import os
import sys

DISTNAME            = 'scikits.sparse'
DESCRIPTION         = 'Scikits sparse matrix package'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Nathaniel Smith',
MAINTAINER_EMAIL    = 'njs@pobox.com',
URL                 = 'https://github.com/njsmith/scikits-sparse/'
LICENSE             = 'GPL'
DOWNLOAD_URL        = "https://github.com/njsmith/scikits-sparse/downloads"
VERSION             = '0.2+dev'

from numpy.distutils.core import setup, Extension
from numpy.distutils.system_info import lapack_info, lapack_mkl_info

lapack_libs = []
lapack_lib_dirs = []
lapack_include_dirs = []
for l in [lapack_mkl_info().get_info(), lapack_info().get_info()]:
  try:
    lapack_libs += l['libraries']
    lapack_lib_dirs += l['library_dirs']
    lapack_include_dirs += l['include_dirs']
    break
  except:
    pass

import numpy as np
# the monkey patch trick so that cython is called on pyx files.
import monkey

if __name__ == "__main__":
    setup(install_requires = ['numpy', 'scipy'],
          namespace_packages = ['scikits'],
          packages = ['scikits.sparse'], #find_packages(),
          package_data = {
              "": ["test_data/*.mtx.gz"],
              },
          test_suite="nose.collector",
          # Well, technically zipping the package will work, but since it's
          # all compiled code it'll just get unzipped again at runtime, which
          # is pointless:
          zip_safe = False,
          name = DISTNAME,
          version = VERSION,
          maintainer = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          url = URL,
          download_url = DOWNLOAD_URL,
          long_description = LONG_DESCRIPTION,
          classifiers =
            [ 'Development Status :: 3 - Alpha',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License (GPL)',
              'Topic :: Scientific/Engineering'],
          ext_modules = [
              Extension("scikits.sparse.cholmod",
                        ["scikits/sparse/cholmod.pyx"],
                        libraries=["cholmod", "colamd", "amd", "suitesparseconfig", 'rt'] + lapack_libs,
                        include_dirs=[np.get_include()] + lapack_include_dirs,
                        # If your CHOLMOD is in a funny place, you may need to
                        # add some LDFLAGS and CFLAGS before running setup
                        library_dirs=lapack_lib_dirs,
                        ),
              ],
          )
