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
    _ndarray_copy_from_intptr,
)

import numpy as np
import warnings

from sksparse.cholmod import _cholmod_sparse_from_csc

from .utils import validate_csc_input


__all = [
    "SPQRFactor",
    "spqr_factor",
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
    matrix, and :math:`R` is an upper-triangular matrix.

    The numerical factorization is computed in one of two ways:

    1. by setting ``use_singletons=True`` in the constructor, which computes
        both the symbolic and numeric factorizations at once, or
    2. by calling :meth:`.factorize(A)`, which computes the numeric factorization
        after symbolic analysis has been performed.

    The first method is useful when factoring a single matrix, but solving multiple
    right-hand sides.

    The second method is useful when factoring multiple matrices with the same sparsity
    pattern but different numerical values.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    use_singletons : bool, optional
        If True, directly compute the numeric factorization to exploit singleton rows.
        Otherwise, only perform symbolic analysis. Default is False.
    order : int, optional
        The ordering strategy to use.
    tol : float, optional
        If the 2-norm of a column in ``A`` is less than ``tol``, that column is
        considered to be a zero column. If ``None``, the default tolerance is used.

    Properties
    ----------
    is_numeric : bool
        Whether the numeric factorization has been computed.
    shape : tuple
        The shape of the input matrix (M, N).
    itype : dtype
        The integer type used for indices (``int32`` or ``int64``).
    dtype : dtype
        The data type of the matrix (``float64`` or ``complex128``).
    Qshape : tuple
        The shape of the orthogonal matrix Q.
    Rshape : tuple
        The shape of the upper-triangular matrix R.
    rank : int
        The rank of the matrix as determined by SPQR.
    perm : ndarray of int
        The combined singleton and fill-reducing column permutation vector.

    See Also
    --------
    spqr_factor, spqr_solve

    Notes
    -----
    This object is an interface to the SuiteSparse SPQR library [#spqr_url]_.


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#spqr_url] SuiteSparse SPQR
        https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR
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
        bint _econ
        int _M
        int _N
        readonly object itype
        readonly object dtype
        double _tol

    def __init__(
        self,
        object A,
        *,
        bint use_singletons=False,
        object order=None,
        object tol=None,
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

        if order is None:
            ordering = SPQR_ORDERING_DEFAULT
        else:
            # TODO validate order input
            raise NotImplementedError("ordering methods not yet implemented")

        self._tol = <double>tol if tol is not None else SPQR_DEFAULT_TOL

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

        cdef bint allow_tol = True  # if False, do not perform rank detection

        if use_singletons:
            # Perform both symbolic and numeric factorization
            if self._is_real:
                if self._use_int32:
                    self._fact_di = SuiteSparseQR_factorize[double, int32_t](
                        ordering, self._tol, Ac, self._cm
                    )
                else:
                    self._fact_dl = SuiteSparseQR_factorize[double, int64_t](
                        ordering, self._tol, Ac, self._cm
                    )
            else:
                if self._use_int32:
                    self._fact_zi = SuiteSparseQR_factorize[doublecomplex, int32_t](
                        ordering, self._tol, Ac, self._cm
                    )
                else:
                    self._fact_zl = SuiteSparseQR_factorize[doublecomplex, int64_t](
                        ordering, self._tol, Ac, self._cm
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
        self._M = Ac.nrow
        self._N = Ac.ncol
        self._econ = False  # TODO econ mode (default to full Q and R shapes)

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

    def __repr__(self):
        cls_name = self.__class__.__name__
        factor_type = 'numeric' if self.is_numeric else 'symbolic'
        # TODO nnz of Q and R
        return (
            f"<{cls_name} {factor_type} factor of dtype '{self.dtype}' "
            f"with '{self.itype}' indices:\n"
            f"    Q: {self.Qshape} with XXX stored elements\n"
            f"    R: {self.Rshape} with XXX stored elements>"
        )

    def __str__(self):
        return self.__repr__()

    # ---------------------------------------------------------------------------------
    #         Properties
    # ---------------------------------------------------------------------------------
    @property
    def is_numeric(self):
        try:
            self._require_numeric()
            return True
        except AssertionError:
            return False

    @property
    def shape(self):
        return (self._M, self._N)

    @property
    def Qshape(self):
        return (self._M, self._N) if self._econ else (self._M, self._M)

    @property
    def Rshape(self):
        return (self._N, self._N) if self._econ else (self._M, self._N)

    @property
    def rank(self):
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

    @property
    def perm(self):
        self._require_symbolic()

        cdef void* ptr
        if self._is_real:
            if self._use_int32:
                ptr = <void*>self._fact_di.Q1fill
            else:
                ptr = <void*>self._fact_dl.Q1fill
        else:
            if self._use_int32:
                ptr = <void*>self._fact_zi.Q1fill
            else:
                ptr = <void*>self._fact_zl.Q1fill

        return _ndarray_copy_from_intptr(ptr, self._N, self._use_int32)

    # ---------------------------------------------------------------------------------
    #         Public API
    # ---------------------------------------------------------------------------------
    def factorize(self, object A, *, object tol=None):
        """Compute the numeric factorization of the matrix.

        Parameters
        ----------
        A : (M, N) array_like or sparse array, optional
            An array convertible to a sparse matrix. If None, the numeric factorization
            is computed for the matrix used in the constructor. If ``A`` is provided,
            it must have the same sparsity pattern as the matrix used in the
            constructor.
        tol : float, optional
            If the 2-norm of a column in ``A`` is less than ``tol``, that column is
            considered to be a zero column. If ``None``, tolerance used in the
            constructor is used.

        Returns
        -------
        SPQRFactor
            The current object with the numeric factorization computed.
        """
        A, _, itype = validate_csc_input(A)
        self._check_input_matrix(A, itype)

        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix
        cdef int stype = 0  # assume matrix is not symmetric

        _cholmod_sparse_from_csc(
            A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
        )

        if tol is not None:
            if self.is_numeric and <double>tol != self._tol:
                warnings.warn(
                    "The tolerance has been changed from the one used "
                    "during a previous numeric factorization. This may lead to "
                    "inconsistent rank determination.",
                    UserWarning,
                )
            self._tol = <double>tol

        # Perform numeric factorization
        if self._is_real:
            if self._use_int32:
                SuiteSparseQR_numeric[double, int32_t](
                    self._tol, Ac, self._fact_di, self._cm
                )
            else:
                SuiteSparseQR_numeric[double, int64_t](
                    self._tol, Ac, self._fact_dl, self._cm
                )
        else:
            if self._use_int32:
                SuiteSparseQR_numeric[doublecomplex, int32_t](
                    self._tol, Ac, self._fact_zi, self._cm
                )
            else:
                SuiteSparseQR_numeric[doublecomplex, int64_t](
                    self._tol, Ac, self._fact_zl, self._cm
                )

        # TODO proper error handling
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"Error {self._cm.status}")

        return self


    # ---------------------------------------------------------------------------------
    #         Private API
    # ---------------------------------------------------------------------------------
    cdef inline int _require_symbolic(self) except -1:
        """Raise an error if the symbolic factorization has not been computed yet."""
        if self._is_real:
            if self._use_int32:
                assert self._fact_di is not NULL and self._fact_di.QRsym is not NULL
            else:
                assert self._fact_dl is not NULL and self._fact_dl.QRsym is not NULL
        else:
            if self._use_int32:
                assert self._fact_zi is not NULL and self._fact_zi.QRsym is not NULL
            else:
                assert self._fact_zl is not NULL and self._fact_zl.QRsym is not NULL

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

    def _check_input_matrix(self, object A, object itype):
        """Check that the input matrix matches the existing factorization."""
        if A.shape != self.shape:
            raise ValueError(
                "The shape of the input matrix does not match "
                "the one used for symbolic factorization. "
                f"Expected {self.shape}, got {A.shape}."
            )

        if itype != self.itype:
            raise ValueError(
                "The integer size of the input matrix does not match "
                "the one used for symbolic factorization. "
                f"Expected '{self.itype}', got '{itype}'."
            )

        if A.dtype != self.dtype:
            raise ValueError(
                "The data type of the input matrix does not match "
                "the one used for symbolic factorization. "
                f"Expected '{self.dtype}', got '{A.dtype}'."
            )


# -------------------------------------------------------------------------------------
#         Convenience Functions
# -------------------------------------------------------------------------------------
def spqr_factor(A, *, use_singletons=False, order=None, tol=None):
    """Compute the SPQR factorization of a sparse matrix.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    use_singletons : bool, optional
        If True, directly compute the numeric factorization to exploit singleton rows.
        Otherwise, only perform symbolic analysis. Default is False, so that the factor
        can be reused efficiently for multiple numeric factorizations.
    order : int, optional
        The ordering strategy to use.
    tol : float, optional
        If the 2-norm of a column in ``A`` is less than ``tol``, that column is
        considered to be a zero column. If ``None``, the default tolerance is used.

    Returns
    -------
    SPQRFactor
        The SPQR factorization of the input matrix.
    """
    if use_singletons:
        return SPQRFactor(A, use_singletons=True, order=order, tol=tol)
    else:
        return SPQRFactor(A, use_singletons=False, order=order, tol=tol).factorize(A)


