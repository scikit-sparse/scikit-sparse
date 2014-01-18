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
VERSION             = '0.2'

# Add our fake Pyrex at the end of the Python search path
# in order to fool setuptools into allowing compilation of
# pyx files to C files. Importing Cython.Distutils then
# makes Cython the tool of choice for this rather than
# (the possibly nonexisting) Pyrex.
project_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(project_path, 'fake_pyrex'))

from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy as np

if __name__ == "__main__":
    setup(install_requires = ['numpy', 'scipy'],
          namespace_packages = ['scikits'],
          packages = find_packages(),
          package_data = {
              "": ["*.mtx.gz"],
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
          cmdclass = {"build_ext": build_ext},
          ext_modules = [
              Extension("scikits.sparse.cholmod",
                        ["scikits/sparse/cholmod.pyx"],
                        libraries=["cholmod"],
                        include_dirs=[np.get_include()],
                        # If your CHOLMOD is in a funny place, you may need to
                        # add something like this:
                        #library_dirs=["/opt/suitesparse/lib"],
                        # And modify include_dirs above in a similar way.
                        ),
              ],
          )
