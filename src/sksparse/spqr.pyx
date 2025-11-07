# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: spqr.pyx
#  Created: 2025-11-06 20:19
# =============================================================================

"""
==============================================
Sparse QR Decomposition (:mod:`sksparse.spqr`)
==============================================

.. currentmodule:: sksparse.spqr

.. versionadded:: 0.5.0


An interface to the SuiteSparse `SPQR
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR>`_
package, which computes the QR factorization and solves systems of equations
for sparse, possibly non-square, non-symmetric, indefinite matrices.


Function Interface
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    spqr_solve - Solve a linear system using the SPQR factorization.


Object Interface
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    spqr_factor - Compute the QR factorization of a sparse matrix.
    SPQRFactor - An object-oriented interface to SPQR.
    SPQRInfo - A dataclass to return SPQR info.
    SPQRControl - A dataclass to set SPQR control parameters.


.. spqr-exceptions:

Warnings and Exceptions
-----------------------

.. autosummary::
    :toctree: generated/

    SPQRWarning
    SPQRSingularMatrixWarning

    SPQRError
    SPQROutOfMemoryError
    SPQRInvalidError
    SPQROverflowError


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse SPQR <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/SPQR>`_
"""

cimport cython

from sksparse.cholmod cimport cholmod_l_start
from sksparse.cholmod import _cholmod_sparse_from_csc

import numpy as np
from scipy.sparse import issparse, csc_array
import warnings

from .utils import validate_csc_input


def spqr(A):
    """Compute the sparse QR factorization of a matrix."""
    A, _, _ = validate_csc_input(A)
    A.indptr = A.indptr.astype(np.int64)
    A.indices = A.indices.astype(np.int64)

    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    cholmod_l_start(cm)

    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix

    cdef int stype = -1

    # Get sparse *pattern*
    _cholmod_sparse_from_csc(
        A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
    )

    cdef SuiteSparseQR_factorization[double, int64_t]* qr
    cdef int ordering = 0  # natural ordering
    cdef double tol = 0.0  # default tolerance
    qr = SuiteSparseQR_factorize[double, int64_t](ordering, tol, Ac, cm)
    print(f"{qr.rank=}")
    print(f"{qr.narows=}, {qr.nacols=}, {qr.bncols=}")
