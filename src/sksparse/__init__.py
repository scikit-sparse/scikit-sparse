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
   sksparse.klu <klu>
   sksparse.spqr <spqr>
   sksparse.umfpack <umfpack>

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
   klu
   spqr
   umfpack


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse GitHub <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_
"""

from importlib.metadata import version, PackageNotFoundError
from importlib import import_module

try:
    __version__ = version("scikit-sparse")
except PackageNotFoundError:
    # package is not installed, so we set a default version
    __version__ = "0.0.0.dev0"

__all__ = [
    "amd",
    "btf",
    "camd",
    "ccolamd",
    "cholmod",
    "colamd",
    "klu",
    "spqr",
    "umfpack",
]


def __getattr__(name):
    """Lazy import submodules.

    This function allows users to import a single submodule, e.g. ``from
    sksparse import cholmod``, without importing the entire package. It is
    helpful in cases where the entire SuiteSparse package is not installed.
    """
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module  # cache the naame
        return module
    else:
        raise AttributeError(f"module {repr(__name__)} has no attribute {repr(name)}")


def __dir__():
    return __all__
