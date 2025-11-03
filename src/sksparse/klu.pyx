# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: klu.pyx
#  Created: 2025-10-30 21:02
# =============================================================================

"""
=================================================
Clark Kent LU Decomposition (:mod:`sksparse.klu`)
=================================================

.. currentmodule:: sksparse.klu

.. versionadded:: 0.5.0


An interface to the SuiteSparse `KLU
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/KLU>`_
package, which computes the LU factorization and solves systems of equations
for sparse, possibly non-symmetric, indefinite matrices.


Function Interface
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    klu_solve - Solve a linear system using the KLU factorization.


Object Interface
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    klu_factor - Compute the LU factorization of a sparse matrix.
    KLUFactor - An object-oriented interface to KLU.
    KLUInfo - A dataclass to return KLU info.
    KLUControl - A dataclass to set KLU control parameters.


.. klupack-exceptions:

Warnings and Exceptions
-----------------------

.. autosummary::
    :toctree: generated/

    KLUWarning
    KLUSingularMatrixWarning

    KLUError
    KLUOutOfMemoryError
    KLUInvalidError
    KLUOverflowError


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse KLU <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/KLU>`_
"""

cimport cython
cimport numpy as cnp

import numpy as np
from scipy.sparse import issparse, csc_array
import warnings

from .utils import validate_csc_input


# -----------------------------------------------------------------------------
#         Define types
# -----------------------------------------------------------------------------
ctypedef fused index_t:
    int32_t
    int64_t


ctypedef fused value_t:
    double
    double complex


# -------------------------------------------------------------------------------------
#         Warnings and Errors
# -------------------------------------------------------------------------------------
class KLUWarning(Warning):
    """Base warning for KLU-related warnings."""
    pass


class KLUSingularMatrixWarning(KLUWarning):
    """Warning raised when a singular matrix is encountered."""
    pass


class KLUError(Exception):
    """Base exception for KLU-related errors."""
    pass


class KLUOutOfMemoryError(MemoryError, KLUError):
    """Exception raised when KLU runs out of memory."""
    pass


class KLUInvalidError(KLUError):
    """Exception raised for invalid inputs to KLU."""
    pass


class KLUOverflowError(OverflowError, KLUError):
    """Exception raised when KLU encounters an overflow."""
    pass


# Known Errors
cdef dict _ERROR_INDEX = {
    KLU_SINGULAR: (KLUSingularMatrixWarning, "The matrix is singular."),
    KLU_OUT_OF_MEMORY: (KLUOutOfMemoryError, "KLU ran out of memory."),
    KLU_INVALID: (KLUInvalidError, "An invalid input was provided to KLU."),
    KLU_TOO_LARGE: (KLUOverflowError, "The matrix is too large for KLU to handle."),
}


cdef int _handle_errors(int status) except -1 with gil:
    """Handle KLU errors by raising Python exceptions or warnings.

    This function should be called with the return ``status`` after any KLU
    C function that may fail.

    Parameters
    ----------
    status : int
        The KLU exit status code.

    Returns
    -------
    None

    Raises
    ------
    :exc:`KLUWarning` or subclass
        Raises a warning for non-critical issues.
    :exc:`KLUError` or subclass
        Raises an appropriate Python exception based on the KLU status code.
    """
    if status == KLU_OK:
        return 0

    # Fallback to generic error for unknown codes
    exc_class, msg = _ERROR_INDEX.get(
        status,
        (KLUError, "An unknown KLU error occurred.")
    )
    full_msg = f"{msg} (code {status:d})"

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class)
    else:
        raise exc_class(full_msg)


# -------------------------------------------------------------------------------------
#         KLU Control and Info Classes
# -------------------------------------------------------------------------------------
cdef object _get_row_scale_string(int scale):
    """Convert KLU row scaling integer to string."""
    if scale == -1:
        return "none_no_check"
    elif scale == 0:
        return "none"
    elif scale == 1:
        return "sum"
    elif scale == 2:
        return "max"
    else:
        return "unknown"


cdef object _get_ordering_string(int ordering):
    """Convert KLU ordering integer to string."""
    if ordering == 0:
        return "AMD"
    elif ordering == 1:
        return "COLAMD"
    elif ordering == 2:
        return "user-given"
    elif ordering == 3:
        return "user function"
    else:
        return "unknown"


@cython.dataclasses.dataclass(frozen=True)
cdef class KLUInfo:
    """A dataclass to store KLU information."""
    noffdiag : int | None = None
    nrealloc : int | None = None
    rcond : double | None = None
    singular_col : int | None = None
    rgrowth : double | None = None
    flops : int | None = None
    nblocks : int | None = None
    ordering : str | None = None
    row_scale : str | None = None
    lnz : int | None = None
    unz : int | None = None
    nzoff : int | None = None
    tol : double | None = None
    memory : int | None = None

    cdef KLUInfo update_from_klu(
        self, klu_symbolic* symbolic, klu_numeric* numeric, klu_common* cm
    ):
        """Update a KLUInfo object from KLU structs."""
        if cm is not NULL:
            self.noffdiag = cm.noffdiag
            self.nrealloc = cm.nrealloc
            self.rcond = cm.rcond
            self.singular_col = cm.singular_col
            self.rgrowth = cm.rgrowth
            self.flops = <int>cm.flops
            self.ordering = _get_ordering_string(cm.ordering)
            self.row_scale = _get_row_scale_string(cm.scale)
            self.tol = cm.tol
            self.memory = <int>cm.memusage

        if symbolic is not NULL:
            self.nblocks = symbolic.nblocks

        if numeric is not NULL:
            self.lnz = <int>numeric.lnz
            self.unz = <int>numeric.unz
            self.nzoff = <int>numeric.nzoff

        return self

    cdef KLUInfo update_from_l_klu(
        self, klu_l_symbolic* symbolic, klu_l_numeric* numeric, klu_l_common* cm
    ):
        """Create a KLUInfo object from KLU structs."""
        if cm is not NULL:
            self.noffdiag = cm.noffdiag
            self.nrealloc = cm.nrealloc
            self.rcond = cm.rcond
            self.singular_col = cm.singular_col
            self.rgrowth = cm.rgrowth
            self.flops = <int>cm.flops
            self.ordering = _get_ordering_string(cm.ordering)
            self.row_scale = _get_row_scale_string(cm.scale)
            self.tol = cm.tol
            self.memory = <int>cm.memusage

        if symbolic is not NULL:
            self.nblocks = symbolic.nblocks

        if numeric is not NULL:
            self.lnz = <int>numeric.lnz
            self.unz = <int>numeric.unz
            self.nzoff = <int>numeric.nzoff

        return self



# -------------------------------------------------------------------------------------
#         KLU Class Interface
# -------------------------------------------------------------------------------------
cdef dict _ROW_SCALE_INDEX = {
    "none_no_check": -1,
    "none": 0,
    "sum": 1,
    "max": 2,
}

# TODO add docs note on difference of row scaling vs KLU
cdef class KLUFactor:
    """Class to compute and store the KLU factorization of a sparse matrix.

    The constructor computes the symbolic analysis of a sparse matrix :math:`A`
    and determines a fill-reducing ordering such that:

    .. math::
        L U + F = R^{-1} P A Q.

    The numeric factorization is not computed until :meth:`.factorize` is called.

    Attributes
    ----------
    N : int
        The number of rows/columns in the matrix.
    L : scipy.sparse.csc_array
        The :math:`L` factor as a sparse CSC matrix.
    U : scipy.sparse.csc_array
        The :math:`U` factor as a sparse CSC matrix.
    perm_r, perm_c : numpy.ndarray
        The row and column permutation arrays, :math:`P` and :math:`Q`.

    Notes
    -----
    This object is an interface to the SuiteSparse KLU library [#klu_url]_.


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#klu_url] SuiteSparse KLU
        https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/KLU
    """

    cdef:
        readonly Py_ssize_t _N
        readonly object itype
        readonly object dtype
        bint _use_int32
        bint _is_real
        # settings + output info
        klu_common _common
        klu_common* _cm
        klu_l_common _l_common
        klu_l_common* _l_cm
        KLUInfo _info
        # Symbolic analysis
        klu_symbolic* _symbolic
        klu_l_symbolic* _l_symbolic
        # Numeric factorization
        klu_numeric* _numeric
        klu_l_numeric* _l_numeric
        # Cached factor objects
        object _L, _U, _F, _P, _Q, _Rs, _R

    def __init__(self, A, *, row_scale=None):
        """Compute the KLU factorization of a sparse matrix.

        Parameters
        ----------
        A : (N, N) numpy.ndarray or sparse array
            The input matrix. Any object that can be converted to
            a :class:`~scipy.sparse.csc_array` is accepted.
        row_scale : str, optional
            The row-scaling method. Accepted values are:

            * ``none_no_check`` : no scaling, and no error check.
            * ``none`` : no scaling, but check for zero rows.
            * ``sum`` : scale rows to have unit 1-norm.
            * ``max`` : scale rows to have unit infinity-norm.

            Default is ``None``, which uses the ``KLU`` default setting of ``max``
            (subject to change).

        """
        A, self._use_int32, _ = validate_csc_input(A, require_square=True)

        self._N = A.shape[0]

        scale = None
        if row_scale is not None:
            try:
                scale = _ROW_SCALE_INDEX[row_scale]
            except KeyError:
                raise ValueError(
                    f"Invalid value for row_scale: {scale}. "
                    f"Expected one of {list(_ROW_SCALE_INDEX.keys())}"
                )

        if self._use_int32:
            self._cm = &self._common
            assert klu_defaults(self._cm)
            if scale is not None:
                self._cm.scale = scale
        else:
            self._l_cm = &self._l_common
            assert klu_l_defaults(self._l_cm)
            if scale is not None:
                self._l_cm.scale = scale

        self._init_symbolic(self._N, A.indptr, A.indices, A.data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _init_symbolic(
        self,
        Py_ssize_t N,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data,
    ):
        """Compute the symbolic factorization.

        Parameters
        ----------
        N : int
            Number of rows and columns of the matrix.
        indptr : 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : 1D array of index_t
            The row indices array of the CSC matrix.
        """
        self._is_real = value_t is double

        if self._use_int32:
            self._symbolic = klu_analyze(
                N,
                <int32_t*>&indptr[0],
                <int32_t*>&indices[0],
                self._cm
            )
            _handle_errors(self._cm.status)
        else:
            self._l_symbolic = klu_l_analyze(
                N,
                <int64_t*>&indptr[0],
                <int64_t*>&indices[0],
                self._l_cm
            )
            _handle_errors(self._l_cm.status)

        self.itype = np.dtype(np.int32 if self._use_int32 else np.int64)
        self.dtype = np.dtype(np.float64 if self._is_real else np.complex128)

    def __dealloc__(self):
        """Deallocate KLU objects."""
        if self._use_int32:
            if self._symbolic is not NULL:
                klu_free_symbolic(&self._symbolic, self._cm)
            if self._numeric is not NULL:
                if self._is_real:
                    klu_free_numeric(&self._numeric, self._cm)
                else:
                    klu_z_free_numeric(&self._numeric, self._cm)
        else:
            if self._l_symbolic is not NULL:
                klu_l_free_symbolic(&self._l_symbolic, self._l_cm)
            if self._l_numeric is not NULL:
                if self._is_real:
                    klu_l_free_numeric(&self._l_numeric, self._l_cm)
                else:
                    klu_zl_free_numeric(&self._l_numeric, self._l_cm)

    def __iter__(self):
        for attr in ['L', 'U', 'perm_r', 'perm_c', 'rscale', 'F', 'rblocks']:
            yield getattr(self, attr)

    # ---------------------------------------------------------------------------------
    #         Properties
    # ---------------------------------------------------------------------------------
    @property
    def is_numeric(self):
        if self._use_int32:
            return self._symbolic is not NULL and self._numeric is not NULL
        else:
            return self._l_symbolic is not NULL and self._l_numeric is not NULL

    @property
    def lnz(self):
        if self._use_int32:
            if self._numeric is NULL:
                return None
            val = self._numeric.lnz
        else:
            if self._l_numeric is NULL:
                return None
            val = self._l_numeric.lnz
        return int(val) if val >= 0 else None

    @property
    def unz(self):
        if self._use_int32:
            if self._numeric is NULL:
                return None
            val = self._numeric.unz
        else:
            if self._l_numeric is NULL:
                return None
            val = self._l_numeric.unz
        return int(val) if val >= 0 else None

    @property
    def nzoff(self):
        if self._use_int32:
            if self._numeric is NULL:
                return None
            val = self._numeric.nzoff
        else:
            if self._l_numeric is NULL:
                return None
            val = self._l_numeric.nzoff
        return int(val) if val >= 0 else None

    @property
    def nblocks(self):
        if self._use_int32:
            if self._symbolic is NULL:
                return None
            val = self._symbolic.nblocks
        else:
            if self._l_symbolic is NULL:
                return None
            val = self._l_symbolic.nblocks
        return int(val) if val >= 0 else None

    @property
    def nnz(self):
        if self.lnz is None or self.unz is None:
            return None
        return int(self.lnz + self.unz)

    @property
    def shape(self):
        return (self._N, self._N)

    @property
    def L(self):
        if self._L is None:
            self._get_numeric()
        return self._L

    @property
    def U(self):
        if self._U is None:
            self._get_numeric()
        return self._U

    @property
    def F(self):
        if self._F is None:
            self._get_numeric()
        return self._F

    @property
    def perm_r(self):
        if self._P is None:
            self._get_numeric()
        return self._P

    @property
    def perm_c(self):
        if self._Q is None:
            self._get_numeric()
        return self._Q

    @property
    def rscale(self):
        if self._Rs is None:
            self._get_numeric()
        return self._Rs

    @property
    def rblocks(self):
        if self._R is None:
            self._get_numeric()
        return self._R

    @property
    def info(self):
        """Get information about the factorization and solve process."""
        if self._info is None:
            self._info = KLUInfo()

        # Compute flops to store in info (needs "is_real" info)
        if self._use_int32:
            if self._is_real:
                klu_flops(self._symbolic, self._numeric, self._cm)
            else:
                klu_z_flops(self._symbolic, self._numeric, self._cm)
        else:
            if self._is_real:
                klu_l_flops(self._l_symbolic, self._l_numeric, self._l_cm)
            else:
                klu_zl_flops(self._l_symbolic, self._l_numeric, self._l_cm)

        if self._use_int32:
            self._info.update_from_klu(self._symbolic, self._numeric, self._cm)
        else:
            self._info.update_from_l_klu(self._l_symbolic, self._l_numeric, self._l_cm)

        return self._info

    # ---------------------------------------------------------------------------------
    #         Public API
    # ---------------------------------------------------------------------------------
    def factorize(self, object A):
        """Compute the numeric factorization of the matrix.

        Computes the numeric factorization of a sparse matrix :math:`A`
        and determines a fill-reducing ordering such that:

        .. math::
            L U + F = R P A Q.

        If given, the matrix :math:`A` must have the same shape and nonzero pattern as
        the one used to create this :class:`KLUFactor` object, but need not have the
        same values. 

        .. warning::

            No check is made on the non-zero structure of the input matrix, so if it is
            different from the one used for the symbolic analysis, the results will be
            incorrect without raising an error.

        Parameters
        ----------
        A : (N, N) numpy.ndarray or sparse array
            The input matrix. Must have the same shape and nonzero pattern as
            the matrix used to create this :class:`KLUFactor` object. If not
            provided, the original matrix given to the constructor will be
            used.

        Returns
        -------
        :class:`KLUFactor`
            The current object, for method chaining.
        """
        _msg = "Symbolic analysis not present. Cannot perform numeric factorization."
        if self._use_int32:
            assert self._symbolic is not NULL, _msg
        else:
            assert self._l_symbolic is not NULL, _msg

        A, _, itype = validate_csc_input(A, require_square=True)
        self._check_input_matrix(A, itype)

        # Clear cached factor objects
        self._L = None
        self._U = None
        self._F = None
        self._P = None
        self._Q = None
        self._Rs = None
        self._R = None

        if self._numeric is not NULL or self._l_numeric is not NULL:
            # Refactorize with existing numeric struct
            self._refactorize(A.indptr, A.indices, A.data)
        else:
            # Allocate and compute new numeric struct
            self._factorize(A.indptr, A.indices, A.data)

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _factorize(
        self,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data,
    ):
        """Compute the numeric factorization given the CSC arrays.

        Parameters
        ----------
        indptr : contiguous 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : contiguous 1D array of index_t
            The row indices array of the CSC matrix.
        data : contiguous 1D array of value_t
            The data array of the CSC matrix.
        """
        # Compute the numeric factorization
        if self._use_int32:
            if self._is_real:
                self._numeric = c_klu_factor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._cm
                )
            else:
                self._numeric = klu_z_factor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._cm
                )
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                self._l_numeric = klu_l_factor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_cm
                )
            else:
                self._l_numeric = klu_zl_factor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_cm
                )
            _handle_errors(self._l_cm.status)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _refactorize(
        self,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data,
    ):
        """Re-compute the numeric factorization given the CSC arrays.

        Parameters
        ----------
        indptr : contiguous 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : contiguous 1D array of index_t
            The row indices array of the CSC matrix.
        data : contiguous 1D array of value_t
            The data array of the CSC matrix.
        """
        # Compute the numeric factorization
        if self._use_int32:
            if self._is_real:
                klu_refactor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._numeric,
                    self._cm
                )
            else:
                klu_z_refactor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._numeric,
                    self._cm
                )
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_refactor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_numeric,
                    self._l_cm
                )
            else:
                klu_zl_refactor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_numeric,
                    self._l_cm
                )
            _handle_errors(self._l_cm.status)

    def solve(self, object b):
        """Solve a linear system using the KLU factorization.

        This method solves the linear system:

        .. math::
            A x = b

        using the LU factorization of :math:`A` previously computed by
        :meth:`.factorize`.

        Parameters
        ----------
        b : (N,) or (N, K) numpy.ndarray
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) numpy.ndarray or sparse array
            The solution vector or matrix. If ``b`` is a 1D array, then ``x`` is
            returned as a 1D array. If ``b`` is a 2D array with ``K`` columns,
            then ``x`` is returned as a 2D array with ``K`` columns. If ``b``
            is a sparse array, then ``x`` is also returned as a sparse array.
        """
        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")

        if b.dtype != self.dtype:
            raise ValueError(
                f"LHS and RHS dtypes do not match. {self.dtype=} and {b.dtype=}"
            )

        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        cdef bint return_1D = b.ndim == 1
        cdef size_t K = 1 if b.ndim == 1 else b.shape[1]

        if b.shape[0] != self._N:
            raise ValueError(
                "Right-hand side b must have the same number of rows as A."
            )

        # Check the condition number
        self._check_rcond()

        cdef bint return_sparse = issparse(b)

        if return_sparse:
            b = b.toarray()
        else:
            b = np.asarray(b)

        # Ensure columns are contiguous for multiple RHS
        b = np.asfortranarray(b)

        # TODO allow overwrite_b=True
        # klu_solve expects B as a column-oriented 1D array
        # The klu_solve function overwrites the input with the output
        x = b.copy().reshape(-1, order='F')

        self._solve(K, x)

        # Reshape X into a  2D array
        x = x.reshape(self._N, K, order='F')

        if return_sparse:
            x = csc_array(x, dtype=b.dtype)
            x.indptr = x.indptr.astype(self.itype)
            x.indices = x.indices.astype(self.itype)

        if return_1D:
            x = x[:, 0]

        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _solve(self, size_t K, value_t[::1] x):
        """Solve Ax = b.

        Parameters
        ----------
        K : int
            The number of right-hand sides to solve.
        x : (N * K,) array_like
            The right-hand side matrix on input, in column-oriented form, solution on output.
        """
        cdef double *x_ptr = <double*>&x[0]

        if self._use_int32:
            if self._is_real:
                c_klu_solve(self._symbolic, self._numeric, self._N, K, x_ptr, self._cm)
            else:
                klu_z_solve(self._symbolic, self._numeric, self._N, K, x_ptr, self._cm)
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_solve(
                    self._l_symbolic, self._l_numeric, self._N, K, x_ptr, self._l_cm
                )
            else:
                klu_zl_solve(
                    self._l_symbolic, self._l_numeric, self._N, K, x_ptr, self._l_cm
                )
            _handle_errors(self._l_cm.status)

    # ---------------------------------------------------------------------------------
    #         Private API
    # ---------------------------------------------------------------------------------
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

    cdef int _check_rcond(self) except -1:
        """Check a crude estimate of the condition number.

        Computes ``min(abs(U.diagonal())) / max(abs(U.diagonal()))``. See
        ``klu_condest`` for more accurate estimate from the full LU decomposition.
        """
        # Compute the condition number estimate
        if self._use_int32:
            if self._is_real:
                klu_rcond(self._symbolic, self._numeric, self._cm)
            else:
                klu_z_rcond(self._symbolic, self._numeric, self._cm)
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_rcond(self._l_symbolic, self._l_numeric, self._l_cm)
            else:
                klu_zl_rcond(self._l_symbolic, self._l_numeric, self._l_cm)
            _handle_errors(self._l_cm.status)

        cdef double rcond = self._cm.rcond if self._use_int32 else self._l_cm.rcond
        cdef double eps = np.finfo(np.float64).eps

        if rcond == 0:
            raise KLUError("Matrix is indefinite or singular to working precision.")
        elif rcond < eps:
            warnings.warn(
                "Matrix is nearly singular."
                f"  Results may be inaccurate (rcond={rcond:.2e}).",
                KLUSingularMatrixWarning
            )

    cdef void _get_numeric(self) except *:
        """Extract and cache the numeric factors from the klu_numeric struct."""
        if (self._use_int32 and self._numeric is NULL) or (
            not self._use_int32 and self._l_numeric is NULL
        ):
            raise KLUError(
                "Numeric factorization not present. Run `KLUFactor.factorize(A)` first."
            )

        # Create output arrays
        Lp = np.empty(self._N + 1, dtype=self.itype)
        Li = np.empty(self.lnz, dtype=self.itype)
        Lx = np.empty(self.lnz, dtype=np.float64)

        Up = np.empty(self._N + 1, dtype=self.itype)
        Ui = np.empty(self.unz, dtype=self.itype)
        Ux = np.empty(self.unz, dtype=np.float64)

        Fp = np.empty(self._N + 1, dtype=self.itype)
        Fi = np.empty(self.nzoff, dtype=self.itype)
        Fx = np.empty(self.nzoff, dtype=np.float64)

        self._P = np.empty(self._N, dtype=self.itype)
        self._Q = np.empty(self._N, dtype=self.itype)
        self._Rs = np.empty(self._N, dtype=np.float64)  # always real
        self._R = np.empty(self.nblocks + 1, dtype=self.itype)

        if self._is_real:
            self._dispatch_get_numeric(
                Lp, Li, Lx,
                Up, Ui, Ux,
                Fp, Fi, Fx,
                self._P,
                self._Q,
                self._Rs,
                self._R
            )

            self._L = csc_array((Lx, Li, Lp), shape=self.shape)
            self._U = csc_array((Ux, Ui, Up), shape=self.shape)
            self._F = csc_array((Fx, Fi, Fp), shape=self.shape)
        else:
            # Allocate imaginary parts
            Lz = np.empty(self.lnz, dtype=np.float64)
            Uz = np.empty(self.unz, dtype=np.float64)
            Fz = np.empty(self.nzoff, dtype=np.float64)

            self._dispatch_get_z_numeric(
                Lp, Li, Lx, Lz,
                Up, Ui, Ux, Uz,
                Fp, Fi, Fx, Fz,
                self._P,
                self._Q,
                self._Rs,
                self._R
            )

            self._L = csc_array((Lx + 1j * Lz, Li, Lp), shape=self.shape)
            self._U = csc_array((Ux + 1j * Uz, Ui, Up), shape=self.shape)
            self._F = csc_array((Fx + 1j * Fz, Fi, Fp), shape=self.shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _dispatch_get_numeric(
        self,
        index_t[::1] Lp, index_t[::1] Li, value_t[::1] Lx,
        index_t[::1] Up, index_t[::1] Ui, value_t[::1] Ux,
        index_t[::1] Fp, index_t[::1] Fi, value_t[::1] Fx,
        index_t[::1] P,
        index_t[::1] Q,
        double[::1] Rs,
        index_t[::1] R,
    ):
        """Call the appropriate KLU extract function.

        Parameters
        ----------
        Lp, Li, Lx : arrays for the L factor
            The output arrays for the L factor in CSC format.
        Up, Ui, Ux : arrays for the U factor
            The output arrays for the U factor in CSC format.
        Fp, Fi, Fx : arrays for the F factor
            The output arrays for the F factor in CSC format.
        P : array of index_t
            The output row permutation array.
        Q : array of index_t
            The output column permutation array.
        Rs : array of double
            The output row scaling factors.
        R : array of index_t
            The output block boundaries.
        """
        # Extract the numeric factorization
        if self._use_int32:
            klu_extract(
                self._numeric,
                self._symbolic,
                <int32_t*>&Lp[0], <int32_t*>&Li[0], <double*>&Lx[0],
                <int32_t*>&Up[0], <int32_t*>&Ui[0], <double*>&Ux[0],
                <int32_t*>&Fp[0], <int32_t*>&Fi[0], <double*>&Fx[0],
                <int32_t*>&P[0],
                <int32_t*>&Q[0],
                <double*>&Rs[0],
                <int32_t*>&R[0],
                self._cm
            )
            _handle_errors(self._cm.status)
        else:
            klu_l_extract(
                self._l_numeric,
                self._l_symbolic,
                <int64_t*>&Lp[0], <int64_t*>&Li[0], <double*>&Lx[0],
                <int64_t*>&Up[0], <int64_t*>&Ui[0], <double*>&Ux[0],
                <int64_t*>&Fp[0], <int64_t*>&Fi[0], <double*>&Fx[0],
                <int64_t*>&P[0],
                <int64_t*>&Q[0],
                <double*>&Rs[0],
                <int64_t*>&R[0],
                self._l_cm
            )
            _handle_errors(self._l_cm.status)

    # TODO may be able to *just* use this call but pass "None"/NULL for imaginary parts
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _dispatch_get_z_numeric(
        self,
        index_t[::1] Lp, index_t[::1] Li, value_t[::1] Lx, value_t[::1] Lz,
        index_t[::1] Up, index_t[::1] Ui, value_t[::1] Ux, value_t[::1] Uz,
        index_t[::1] Fp, index_t[::1] Fi, value_t[::1] Fx, value_t[::1] Fz,
        index_t[::1] P,
        index_t[::1] Q,
        double[::1] Rs,
        index_t[::1] R,
    ):
        """Call the appropriate KLU extract function.

        Parameters
        ----------
        Lp, Li, Lx, Lz : arrays for the L factor
            The output arrays for the L factor in CSC format.
        Up, Ui, Ux, Uz : arrays for the U factor
            The output arrays for the U factor in CSC format.
        Fp, Fi, Fx, Fz : arrays for the F factor
            The output arrays for the F factor in CSC format.
        P : array of index_t
            The output row permutation array.
        Q : array of index_t
            The output column permutation array.
        Rs : array of double
            The output row scaling factors.
        R : array of index_t
            The output block boundaries.
        """
        cdef int status

        # Extract the numeric factorization
        if self._use_int32:
            klu_z_extract(
                self._numeric,
                self._symbolic,
                <int32_t*>&Lp[0], <int32_t*>&Li[0], <double*>&Lx[0], <double*>&Lz[0],
                <int32_t*>&Up[0], <int32_t*>&Ui[0], <double*>&Ux[0], <double*>&Uz[0],
                <int32_t*>&Fp[0], <int32_t*>&Fi[0], <double*>&Fx[0], <double*>&Fz[0],
                <int32_t*>&P[0],
                <int32_t*>&Q[0],
                <double*>&Rs[0],
                <int32_t*>&R[0],
                self._cm
            )
            _handle_errors(self._cm.status)
        else:
            klu_zl_extract(
                self._l_numeric,
                self._l_symbolic,
                <int64_t*>&Lp[0], <int64_t*>&Li[0], <double*>&Lx[0], <double*>&Lz[0],
                <int64_t*>&Up[0], <int64_t*>&Ui[0], <double*>&Ux[0], <double*>&Uz[0],
                <int64_t*>&Fp[0], <int64_t*>&Fi[0], <double*>&Fx[0], <double*>&Fz[0],
                <int64_t*>&P[0],
                <int64_t*>&Q[0],
                <double*>&Rs[0],
                <int64_t*>&R[0],
                self._l_cm
            )
            _handle_errors(self._l_cm.status)


# -----------------------------------------------------------------------------
#         Convenience Functions
# -----------------------------------------------------------------------------
def klu_factor(A, *, row_scale=None):
    """Compute the LU factorization of a sparse matrix using KLU.

    This is a convenience function that creates a :class:`KLUFactor` object,
    computes the numeric factorization, and returns the resulting object.

    Parameters
    ----------
    A : (M, N) numpy.ndarray or sparse array
        The input matrix to factorize.

    Returns
    -------
    :class:`KLUFactor`
        The LU factorization of the input matrix.

    Raises
    ------
    :exc:`KLUSingularMatrixWarning`
        If the matrix is exactly singular.

    See Also
    --------
    KLUFactor, klu_solve


    .. versionadded:: 0.5.0
    """
    return KLUFactor(A, row_scale=row_scale).factorize(A)


def klu_solve(A, b, *, row_scale=None):
    """Solve a linear system using KLU.

    This is a convenience function that creates a :class:`KLUFactor` object,
    computes the numeric factorization, and solves the linear system.

    Parameters
    ----------
    A : (N, N) numpy.ndarray or sparse array
        The input matrix to factorize.
    b : (N,) or (N, K) numpy.ndarray
        The right-hand side vector or matrix.

    Returns
    -------
    x : (N,) or (N, K) numpy.ndarray or sparse array
        The solution vector or matrix of the same type and shape as the input
        right-hand side ``b``.

    See Also
    --------
    KLUFactor, klu_solve


    .. versionadded:: 0.5.0
    """
    # factorize() and solve() will each warn for a singular matrix,
    # so we catch the warnings from factorize() and re-raise only once.
    with warnings.catch_warnings(record=True) as ws:
        x = KLUFactor(A, row_scale=row_scale).factorize(A).solve(b)

    # Raise only the latest singular matrix warning from solve
    if ws:
        w = ws[-1]
        warnings.warn(w.message, w.category)

    return x
