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

"""
===================================
Scikit Sparse API (:mod:`sksparse`)
===================================

.. currentmodule:: sksparse

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   sksparse.amd <amd>
   sksparse.btf <btf>
   sksparse.camd <camd>
   sksparse.ccolamd <ccolamd>
   sksparse.cholmod <cholmod>
   sksparse.colamd <colamd>

Provides sparse matrix algorithms not found in SciPy, for use with SciPy's
sparse matrix classes in :mod:`scipy.sparse`.


Submodules
==========

.. autosummary::

   amd
   btf
   camd
   ccolamd
   cholmod
   colamd


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse GitHub <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scikit-sparse-dev")
except PackageNotFoundError:
    # package is not installed, so we set a default version
    __version__ = "0.0.0.dev0"

from . import amd
from . import btf
from . import camd
from . import ccolamd
from . import cholmod
from . import colamd

__all__ = ["amd", "btf", "camd", "ccolamd", "cholmod", "colamd"]
