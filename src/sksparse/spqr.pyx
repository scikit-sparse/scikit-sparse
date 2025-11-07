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
from cython cimport doublecomplex

from sksparse.cholmod cimport (
    CHOLMOD_OK,
    CHOLMOD_INT,
    CHOLMOD_REAL,
    cholmod_start,
    cholmod_l_start,
    cholmod_finish,
    cholmod_l_finish,
)

import numpy as np
from scipy.sparse import issparse, csc_array
import warnings

from sksparse.cholmod import _cholmod_sparse_from_csc

from .utils import validate_csc_input


__all = [
    "SPQRFactor",
]


# Define specific instantiations
ctypedef SuiteSparseQR_factorization[double, int32_t] spqr_fact_di
ctypedef SuiteSparseQR_factorization[double, int64_t] spqr_fact_dl
ctypedef SuiteSparseQR_factorization[doublecomplex, int32_t] spqr_fact_zi
ctypedef SuiteSparseQR_factorization[doublecomplex, int64_t] spqr_fact_zl


# -------------------------------------------------------------------------------------
#         Error Handling
# -------------------------------------------------------------------------------------
class SPQRError(Exception):
    """Base class for SPQR exceptions."""
    pass


# -------------------------------------------------------------------------------------
#         SPQR Factor Class
# -------------------------------------------------------------------------------------
cdef class SPQRFactor:
    """The main object used for creating and manipulating SPQR factorizations.

    The constructor computes the sybolic analysis of the matrix and determines
    a fill-reducing ordering such that:

    .. math::

        Q R = A E

    where :math:`E` is a column permutation matrix, :math:`Q` is an orthogonal
    matrix, and :math:`R` is an upper-triangular matrix. The actual numerical
    factorization is computed when calling :meth:`.factorize`.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    use_singletons : bool, optional
        If True, directly compute the numeric factorization to exploit singleton rows.
        Otherwise, only perform symbolic analysis. Default is False.
    """

    cdef:
        cholmod_common _common
        cholmod_common *_cm
        spqr_fact_di *_fact_di
        spqr_fact_dl *_fact_dl
        spqr_fact_zi *_fact_zi
        spqr_fact_zl *_fact_zl
        bint _use_int32
        bint _is_real
        readonly object itype
        readonly object dtype

    def __init__(
        self,
        object A,
        *,
        bint use_singletons=False,
        object order=None,
        object norm_tol=None,
    ):
        """Initialize the SPQRFactor object and perform symbolic analysis."""
        A, _, _ = validate_csc_input(A)

        # Promote single to double precision
        if not (
            np.issubdtype(A.dtype, np.float64) or np.issubdtype(A.dtype, np.complex128)
        ):
            if np.issubdtype(A.dtype, np.floating):
                A = A.astype(np.promote_types(A.dtype, np.float64))
            elif np.issubdtype(A.dtype, np.complexfloating):
                A = A.astype(np.promote_types(A.dtype, np.complex128))

        # Validate inputs
        cdef int ordering
        cdef double tol

        if order is None:
            ordering = SPQR_ORDERING_DEFAULT
        else:
            # TODO validate order input
            raise NotImplementedError("ordering methods not yet implemented")

        if norm_tol is None:
            tol = SPQR_DEFAULT_TOL
        else:
            tol = <double>norm_tol

        # Get the input matrix into CHOLMOD format
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix
        cdef int stype = 0  # assume matrix is not symmetric

        _cholmod_sparse_from_csc(
            A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
        )

        self._use_int32 = (Ac.itype == CHOLMOD_INT)
        self._is_real = (Ac.xtype == CHOLMOD_REAL)

        # Initialize the common object
        self._cm = &self._common

        if self._use_int32:
            cholmod_start(self._cm)
        else:
            cholmod_l_start(self._cm)

        # TODO pass as input?
        cdef bint allow_tol = True  # if False, do not perform rank detection

        if use_singletons:
            # Perform both symbolic and numeric factorization
            if self._is_real:
                if self._use_int32:
                    self._fact_di = SuiteSparseQR_factorize[double, int32_t](
                        ordering, tol, Ac, self._cm
                    )
                else:
                    self._fact_dl = SuiteSparseQR_factorize[double, int64_t](
                        ordering, tol, Ac, self._cm
                    )
            else:
                if self._use_int32:
                    self._fact_zi = SuiteSparseQR_factorize[doublecomplex, int32_t](
                        ordering, tol, Ac, self._cm
                    )
                else:
                    self._fact_zl = SuiteSparseQR_factorize[doublecomplex, int64_t](
                        ordering, tol, Ac, self._cm
                    )
        else:
            # Perform symbolic analysis only
            if self._is_real:
                if self._use_int32:
                    self._fact_di = SuiteSparseQR_symbolic[double, int32_t](
                        ordering, allow_tol, Ac, self._cm
                    )
                else:
                    self._fact_dl = SuiteSparseQR_symbolic[double, int64_t](
                        ordering, allow_tol, Ac, self._cm
                    )
            else:
                if self._use_int32:
                    self._fact_zi = SuiteSparseQR_symbolic[doublecomplex, int32_t](
                        ordering, allow_tol, Ac, self._cm
                    )
                else:
                    self._fact_zl = SuiteSparseQR_symbolic[doublecomplex, int64_t](
                        ordering, allow_tol, Ac, self._cm
                    )

        # TODO proper error handling
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"Error {self._cm.status}")

        self.itype = np.dtype(np.int32 if self._use_int32 else np.int64)
        self.dtype = np.dtype(np.float64 if self._is_real else np.complex128)

    def __dealloc__(self):
        """Free the SPQR factorization and common objects."""
        if self._is_real:
            if self._use_int32:
                assert SuiteSparseQR_free[double, int32_t](&self._fact_di, self._cm)
            else:
                assert SuiteSparseQR_free[double, int64_t](&self._fact_dl, self._cm)
        else:
            if self._use_int32:
                assert SuiteSparseQR_free[doublecomplex, int32_t](&self._fact_zi, self._cm)
            else:
                assert SuiteSparseQR_free[doublecomplex, int64_t](&self._fact_zl, self._cm)

        if self._use_int32:
            cholmod_finish(self._cm)
        else:
            cholmod_l_finish(self._cm)

    # ---------------------------------------------------------------------------------
    #         Properties
    # ---------------------------------------------------------------------------------
    cdef inline int _require_symbolic(self) except -1:
        """Raise an error if the symbolic factorization has not been computed yet."""
        if self._is_real:
            if self._use_int32:
                assert self._fact_di is not NULL
            else:
                assert self._fact_dl is not NULL
        else:
            if self._use_int32:
                assert self._fact_zi is not NULL
            else:
                assert self._fact_zl is not NULL

    cdef inline int _require_numeric(self) except -1:
        """Raise an error if the numeric factorization has not been computed yet."""
        self._require_symbolic()
        if self._is_real:
            if self._use_int32:
                assert self._fact_di.QRnum is not NULL
            else:
                assert self._fact_dl.QRnum is not NULL
        else:
            if self._use_int32:
                assert self._fact_zi.QRnum is not NULL
            else:
                assert self._fact_zl.QRnum is not NULL

    @property
    def rank(self):
        """The rank of the matrix as determined by SPQR."""
        cdef int rank
        self._require_symbolic()
        if self._is_real:
            if self._use_int32:
                return self._fact_di.rank
            else:
                return self._fact_dl.rank
        else:
            if self._use_int32:
                return self._fact_zi.rank
            else:
                return self._fact_zl.rank

    # ---------------------------------------------------------------------------------
    #         Public API
    # ---------------------------------------------------------------------------------
