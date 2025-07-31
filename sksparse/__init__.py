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

"""Sparse matrix tools.

This is a home for sparse matrix code in Python that plays well with
scipy.sparse, but that is somehow unsuitable for inclusion in scipy
proper. Usually this will be because it is released under the GPL.

So far we have a wrapper for the CHOLMOD library for sparse Cholesky
decomposition. Further contributions are welcome!
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scikit-sparse")
except PackageNotFoundError:
    # package is not installed, so we set a default version
    __version__ = "0.0.0.dev0"

from . import amd
from . import colamd
from . import cholmod

__all__ = ["amd", "colamd", "cholmod"]
