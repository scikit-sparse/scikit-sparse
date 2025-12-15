# Part of the scikit-sparse project.
# Copyright (C) 2008-2025 The scikit-sparse developers. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: cholmod.pyx
#  Created: 2025-08-11 14:49
# =============================================================================

"""
================================================
Cholesky Decomposition (:mod:`sksparse.cholmod`)
================================================

.. currentmodule:: sksparse.cholmod

.. versionadded:: 0.1.0

.. versionchanged:: 0.5.0
   Major API updates to more closely resemble the :func:`scipy.linalg.cholesky`
   dense interface, and incorporate more functions from the CHOLMOD MATLAB
   interface.


An interface to the SuiteSparse `CHOLMOD
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/CHOLMOD>`_
package, which computes basic linear algebra operations for sparse, symmetric,
positive-definite matrices.


Function Interface
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    cholesky - Computes the Cholesky factorization of a sparse matrix.
    ldl - Computes the LDL.T factorization of a sparse matrix.


Object Interface
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    cho_factor - Computes the Cholesky factorization of a sparse matrix.
    ldl_factor - Computes the LDL.T factorization of a sparse matrix.
    CholeskyFactor - Class representing a Cholesky factorization.


Symbolic Analysis
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    symbfact - Computes the symbolic factorization of a sparse matrix.
    etree - Computes the elimination tree of a sparse matrix.


Graph Partitioning
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    bisect - Bisects a graph using nested dissection.
    metis - Computes a fill-reducing ordering using METIS.
    nesdis - Computes a fill-reducing ordering using NESDIS.
    SeparatorTree - Class representing a separator tree.


.. _cholmod-exceptions:

Exceptions and Warnings
-----------------------

.. autosummary::
    :toctree: generated/

    CholmodWarning
    CholmodSmallDiagonalWarning

    CholmodError
    CholmodNotPositiveDefiniteError
    CholmodNotInstalledError
    CholmodOutOfMemoryError
    CholmodOverflowError
    CholmodInvalidInputError
    CholmodGpuProblemError


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse CHOLMOD <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD>`_
"""

from cpython.ref cimport Py_INCREF
cimport cython
cimport numpy as cnp

import numpy as np

from scipy.sparse import csc_array, diags_array, eye_array, issparse
import warnings

from .utils import validate_csc_input

__all__ = [
    "CholeskyFactor",
    "CholmodError",
    "CholmodGpuProblemError",
    "CholmodInvalidInputError",
    "CholmodNotInstalledError",
    "CholmodNotPositiveDefiniteError",
    "CholmodOutOfMemoryError",
    "CholmodOverflowError",
    "CholmodSmallDiagonalWarning",
    "CholmodWarning",
    "SeparatorTree",
    "bisect",
    "cho_factor",
    "cholesky",
    "etree",
    "ldl",
    "ldl_factor",
    "metis",
    "nesdis",
    "symbfact",
]


# Define constants for the mode of cholmod_transpose (see cholmod.h)
cdef int CHOLMOD_TRANS_PATTERN = 0    # transpose only the pattern
cdef int CHOLMOD_TRANS_NOCONJ = 1  # numeric (no conjugate)
cdef int CHOLMOD_TRANS_CONJ = 2  # numeric (conjugate transpose)


# -----------------------------------------------------------------------------
#         Error Handling
# -----------------------------------------------------------------------------
class CholmodError(Exception):
    """Base class for CHOLMOD-related errors."""
    pass


class CholmodNotPositiveDefiniteError(CholmodError):
    """Raised when the input matrix is not positive definite."""
    pass


class CholmodNotInstalledError(CholmodError):
    """Raised when the CHOLMOD library is not installed."""
    pass


class CholmodOutOfMemoryError(MemoryError, CholmodError):
    """Raised when CHOLMOD runs out of memory."""
    pass


class CholmodOverflowError(CholmodError):
    """Raised when CHOLMOD encounters an integer overflow."""
    pass


class CholmodInvalidInputError(CholmodError):
    """Raised when CHOLMOD receives invalid input."""
    pass


class CholmodGpuProblemError(CholmodError):
    """Raised when CHOLMOD encounters a problem with CUDA."""
    pass


class CholmodWarning(Warning):
    """Base class for CHOLMOD-related warnings."""
    pass


class CholmodSmallDiagonalWarning(CholmodWarning):
    """Warning for small diagonal entries."""
    pass


# Known Errors
cdef dict _ERROR_MAP = {
    CHOLMOD_NOT_INSTALLED: (
        CholmodNotInstalledError,
        "CHOLMOD library is not installed or not found."
    ),
    CHOLMOD_OUT_OF_MEMORY: (
        CholmodOutOfMemoryError,
        "CHOLMOD ran out of memory."
    ),
    CHOLMOD_TOO_LARGE: (
        CholmodOverflowError,
        "CHOLMOD encountered an integer overflow."
    ),
    CHOLMOD_INVALID: (
        CholmodInvalidInputError,
        "CHOLMOD received invalid input."
    ),
    CHOLMOD_GPU_PROBLEM: (
        CholmodGpuProblemError,
        "CHOLMOD encountered a problem with CUDA."
    ),
    CHOLMOD_NOT_POSDEF: (
        CholmodNotPositiveDefiniteError,
        "Input matrix is not positive definite."
    ),
    CHOLMOD_DSMALL: (
        CholmodSmallDiagonalWarning,
        "A diagonal entry is very small, which may lead to numerical instability."
    ),
}


cdef int _handle_errors(int status, minor=None) except -1 with gil:
    """Handle CHOLMOD errors by raising Python exceptions or warnings.

    This function should be called with cholmod_common->status after any
    CHOLMOD C function that may fail.

    .. note::

        It is not a safe practice to pass a function like this as the
        "error_handler" member of the cholmod_common struct, because CHOLMOD
        may call it from C code that does not hold the Python GIL.

    Parameters
    ----------
    status : int
        The CHOLMOD status code, from the cholmod_common.status field.
    minor : int, optional
        The column index that caused the error, if applicable.

    Returns
    -------
    None

    Raises
    ------
    :exc:`CholmodWarning`
        Raises a warning for non-critical issues.
    :exc:`CholmodError` or subclass
        Raises an appropriate Python exception based on the CHOLMOD status code.
    """
    if status == CHOLMOD_OK:
        return 0

    status_msg = f"(code {status:d})"

    # Fallback to generic error for unknown codes
    exc_class, msg = _ERROR_MAP.get(status, CholmodError)
    full_msg = msg + " " + status_msg

    if minor is not None:
        full_msg += f" Failed at column {minor}."

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class, stacklevel=2)
    else:
        raise exc_class(full_msg)


# -----------------------------------------------------------------------------
#         CSC <==> CHOLMOD Sparse
# -----------------------------------------------------------------------------
cdef inline int _single_or_double(floating_t _=0) noexcept:
    """Return the CHOLMOD dtype number for a given NumPy dtype."""
    if floating_t is float or floating_t is cython.floatcomplex:
        return CHOLMOD_SINGLE
    else:
        return CHOLMOD_DOUBLE


cdef inline int _real_or_complex(floating_t _=0) noexcept:
    """Return the CHOLMOD xtype number for a given NumPy dtype."""
    if floating_t is float or floating_t is double:
        return CHOLMOD_REAL
    else:
        return CHOLMOD_COMPLEX


@cython.boundscheck(False)
@cython.wraparound(False)
def _cholmod_sparse_from_csc(
    tuple shape not None,
    index_t[::1] indptr not None,
    index_t[::1] indices not None,
    floating_t[::1] data not None,
    int stype,
    uintptr_t A_static,  # cholmod_sparse* as an int
):
    """Create a CHOLMOD sparse matrix from a scipy.sparse.csc_array.

    See the CHOLMOD MATLAB interface for details [#sputil_get_sparse]_.

    Parameters
    ----------
    shape : (2,) tuple of int
        The shape of the CSC matrix.
    indptr : (N+1,) array of index_t
        The column pointer array of the CSC matrix.
    indices : (nnz,) array of index_t
        The row indices of the non-zero entries of the matrix.
    data : (nnz,) array of floating_t
        The numerical values of the non-zero entries of the matrix.
    stype : int
        The assumed symmetry type of ``A``:
        * -1: lower triangular,
        *  0: unsymmetric,
        *  1: upper triangular.
    A_static : cholmod_sparse*
        Pointer to a preallocated CHOLMOD sparse matrix structure. Contents
        need not be initialized. Contains the CHOLMOD sparse matrix on output.

    Returns
    -------
    res : csc_array
        A reference to ``A``. There is no use for the output of this
        function, except to keep the underlying data from being garbage
        collected until the cholmod_sparse object is freed.

    References
    ----------
    .. [#sputil_get_sparse] ``sputil2.c`` - CHOLMOD MATLAB utilities
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/sputil2.c
    """
    # Initialize the CHOLMOD sparse matrix
    cdef cholmod_sparse* A = <cholmod_sparse*>A_static
    memset(A, 0, sizeof(cholmod_sparse))

    assert len(shape) == 2

    # Set the matrix dimensions and properties
    A.nrow = shape[0]
    A.ncol = shape[1]
    A.nzmax = indices.shape[0]
    A.packed = True
    A.sorted = True  # NOTE requires input indices to be sorted
    A.itype = CHOLMOD_INT if index_t is int32_t else CHOLMOD_LONG
    A.stype = -1 if stype < 0 else (0 if stype == 0 else 1)
    A.dtype = _single_or_double[floating_t]()
    A.z = NULL

    # Declare dummy variables to avoid empty array issues
    cdef index_t dummy_index
    cdef floating_t dummy_value

    # Create the index arrays
    A.p = &indptr[0]  # always has size > 0 for a valid csc_array
    A.i = &indices[0] if indices.size > 0 else &dummy_index

    # Get the numerical values of A
    A.xtype = _real_or_complex[floating_t]()
    A.x = &data[0] if data.size > 0 else &dummy_value


cdef class _CholmodSparseDestructor:
    """A destructor for CHOLMOD sparse matrices.

    This class is used as a base for NumPy arrays that are views on CHOLMOD
    sparse matrices. It ensures that the CHOLMOD sparse matrix is properly
    freed when the NumPy array is no longer in use.

    Attributes
    ----------
    _sparse : cholmod_sparse*
        The CHOLMOD sparse matrix to be freed.
    _common : cholmod_common
        The CHOLMOD common structure used for memory management.
    _cm : cholmod_common*
        A pointer to the CHOLMOD common structure.
    """

    cdef cholmod_sparse* _sparse
    cdef cholmod_common _common
    cdef cholmod_common* _cm

    # NOTE __cinit__ is a *Python*-level constructor, cannot take C pointers
    cdef void init(self, cholmod_sparse* A, cholmod_common* common):
        assert A is not NULL
        assert common is not NULL
        self._sparse = A

        self._cm = &self._common

        if self._sparse.itype == CHOLMOD_INT:
            cholmod_start(self._cm)
        else:
            cholmod_l_start(self._cm)

        _copy_cholmod_common(self._cm, common)

    def __dealloc__(self):
        if self._sparse.itype == CHOLMOD_INT:
            cholmod_free_sparse(&self._sparse, self._cm)
            cholmod_finish(self._cm)
        else:
            cholmod_l_free_sparse(&self._sparse, self._cm)
            cholmod_l_finish(self._cm)


cdef inline int _np_itypenum_from_cholmod(int itype) noexcept:
    """Get the NumPy typenum corresponding to the given CHOLMOD itype.

    Parameters
    ----------
    itype : int
        The CHOLMOD itype of the matrix.

    Returns
    -------
    np_itypenum : int
        The corresponding NumPy typenum.
    """
    return cnp.NPY_INT32 if itype == CHOLMOD_INT else cnp.NPY_INT64


cdef inline int _np_dtypenum_from_cholmod(int xtype, int dtype) noexcept:
    """Get the NumPy typenum corresponding to the given CHOLMOD xtype and dtype.

    Parameters
    ----------
    xtype : int
        The CHOLMOD xtype of the matrix.
    dtype : int
        The CHOLMOD dtype of the matrix.

    Returns
    -------
    np_dtypenum : int
        The corresponding NumPy typenum.
    """
    if xtype == CHOLMOD_REAL:
        return cnp.NPY_FLOAT32 if dtype == CHOLMOD_SINGLE else cnp.NPY_FLOAT64
    elif xtype == CHOLMOD_COMPLEX:
        return cnp.NPY_COMPLEX64 if dtype == CHOLMOD_SINGLE else cnp.NPY_COMPLEX128
    elif xtype == CHOLMOD_PATTERN:
        return cnp.NPY_BOOL
    else:
        return cnp.NPY_OBJECT


cdef object _csc_from_cholmod_sparse(cholmod_sparse* A, cholmod_common* common):
    """Create a csc_array that is a view onto a cholmod_sparse object.

    Parameters
    ----------
    A : cholmod_sparse*
        A pointer to the CHOLMOD sparse matrix to convert to a csc_array.
    common : cholmod_common*
        A pointer to the CHOLMOD common structure used for memory management.

    Returns
    -------
    res : csc_array
        A scipy.sparse.csc_array that is a view onto the CHOLMOD sparse matrix.
        The array has a base with a destructor that frees the CHOLMOD sparse
        matrix when the array is no longer in use.
    """
    cdef int np_itypenum = _np_itypenum_from_cholmod(A.itype)
    cdef int np_dtypenum = _np_dtypenum_from_cholmod(A.xtype, A.dtype)

    # convert to NumPy arrays
    indptr = cnp.PyArray_SimpleNewFromData(1, [A.ncol + 1], np_itypenum, A.p)
    indices = cnp.PyArray_SimpleNewFromData(1, [A.nzmax], np_itypenum, A.i)
    data = cnp.PyArray_SimpleNewFromData(1, [A.nzmax], np_dtypenum, A.x)

    # Take ownership of the data
    cdef _CholmodSparseDestructor base = _CholmodSparseDestructor()
    base.init(A, common)

    # NOTE need to increment reference count of base manually for each array,
    # since the PyArray_SetBaseObject steals a reference.
    for arr in (indptr, indices, data):
        Py_INCREF(base)
        if cnp.PyArray_SetBaseObject(arr, base) < 0:
            raise MemoryError("Failed to set base object for array.")

    return csc_array((data, indices, indptr), shape=(A.nrow, A.ncol))


cdef object _csc_view_from_cholmod_factor(CholeskyFactor py_factor, object ldl=None):
    """Create a sparse matrix from a CHOLMOD factor.

    This function is similar to _csc_from_cholmod_sparse, but builds the matrix
    directly from the factor, without the intermediate cholmod_factor_to_sparse
    call.

    Parameters
    ----------
    py_factor : CholeskyFactor
        The input cholmod_factor and cholmod_common objects, wrapped in
        a Python object.
    ldl : None or bool, optional
        If True, return the LDL.T form, otherwise return the LL.T form. Default
        is to use the form of the existing factor ``py_factor._factor.is_ll``.

    Returns
    -------
    res : csc_array
        L scipy.sparse.csc_array that is a view onto the CHOLMOD factor. This
        array is *read-only*, so attempts to modify it will raise an error.

    Notes
    -----
    The ``cholmod_factor_to_sparse`` function moves the memory from the
    ``cholmod_factor`` to the newly-created ``cholmod_sparse`` struct, and sets
    the ``xtype`` of the factor to ``CHOLMOD_PATTERN``. This behavior is fine
    for standalone functions that return a matrix and no longer need the
    factor. For our :obj:`CholeskyFactor` class, however, we need to keep the
    factor intact for future updates or conversions to LL or LDL formats.
    Therefore, we use this function to create a view onto the factor without
    destroying it.
    """
    cdef cholmod_factor *L = py_factor._factor
    cdef cholmod_common *common = py_factor._cm

    if L is NULL:
        raise ValueError("The factor pointer is NULL.")

    if L.xtype == CHOLMOD_PATTERN:
        raise ValueError("The factor has no numerical values.")

    # Ensure the factor is in simplicial, packed, monotonic format
    cdef int to_ll = L.is_ll if ldl is None else not ldl
    cdef int to_super = False  # simplicial format
    cdef int to_packed = True
    cdef int to_monotonic = True

    if L.itype == CHOLMOD_INT:
        cholmod_change_factor(
            L.xtype, to_ll, to_super, to_packed, to_monotonic, L, common
        )
    else:
        cholmod_l_change_factor(
            L.xtype, to_ll, to_super, to_packed, to_monotonic, L, common
        )

    _handle_errors(common.status)

    # Create numpy arrays
    cdef int np_itypenum = _np_itypenum_from_cholmod(L.itype)
    cdef int np_dtypenum = _np_dtypenum_from_cholmod(L.xtype, L.dtype)

    indptr = cnp.PyArray_SimpleNewFromData(1, [L.n + 1], np_itypenum, L.p)
    indices = cnp.PyArray_SimpleNewFromData(1, [L.nzmax], np_itypenum, L.i)
    data = cnp.PyArray_SimpleNewFromData(1, [L.nzmax], np_dtypenum, L.x)

    # NOTE need to increment reference count of base manually for each array,
    # since the PyArray_SetBaseObject steals a reference.
    # Take ownership of the data
    for arr in (indptr, indices, data):
        Py_INCREF(py_factor)
        if cnp.PyArray_SetBaseObject(arr, py_factor) < 0:
            raise MemoryError("Failed to set base object for array.")
        cnp.PyArray_CLEARFLAGS(arr, cnp.NPY_ARRAY_WRITEABLE)  # make read-only

    return csc_array((data, indices, indptr), shape=(L.n, L.n))


cdef cholmod_sparse* _cholesky_pattern(
    cholmod_sparse *A,
    cholmod_sparse *F,
    size_t N,
    int32_t *Parent,
    int32_t *ColCount,
    bint col_etree,
    cholmod_common *cm
):
    """Compute the Cholesky pattern from the given matrices.

    Parameters
    ----------
    A, F : cholmod_sparse*
        Pointers to the sparse matrices to analyze.
    N : size_t
        The number of rows or columns in A.
    Parent : int32_t*
        Pointer to the array of the elimination tree.
    ColCount : int32_t*
        Pointer to the array of column counts of the Cholesky factor.
    col_etree : bint
        If True, analyze the column case F @ F.T. Otherwise, determine the case
        from ``A->stype``.
    cm : cholmod_common*
        Pointer to a CHOLMOD common structure for configuration and status.

    Returns
    -------
    L : cholmod_sparse*
        A pointer to the array containing the pattern of the Cholesky factor.
    """
    if A is NULL or F is NULL or cm is NULL:
        raise ValueError("Input pointer is NULL.")

    cdef cholmod_sparse *A_in = NULL
    cdef cholmod_sparse *F_in = NULL

    if A.stype == 1:
        A_in = A
    elif A.stype == -1:
        A_in = F
    elif col_etree:
        # column case: analyze F @ F.T
        A_in = F
        F_in = A
    else:
        # row case: analyze A @ A.T
        A_in = A
        F_in = F

    # Count the total number of entries in L
    cdef int32_t lnz = 0
    cdef size_t j

    for j in range(N):
        lnz += ColCount[j]

    # Initialize the CHOLMOD sparse matrix for L
    cdef cholmod_sparse *L = cholmod_allocate_sparse(
        N, N, lnz, True, True, 0, CHOLMOD_PATTERN, cm
    )

    cdef int32_t *Lp = <int32_t*>L.p
    cdef int32_t *Li = <int32_t*>L.i

    # Initialize column pointers
    lnz = 0

    for j in range(N):
        Lp[j] = lnz
        lnz += ColCount[j]

    Lp[N] = lnz

    # Create a copy of the column pointers
    cdef int32_t *W = <int32_t*>cholmod_malloc(N, sizeof(int32_t), cm)
    memcpy(W, Lp, N * sizeof(int32_t))

    # Get workspace for computing one row of L
    cdef cholmod_sparse *R = cholmod_allocate_sparse(
        N, 1, N, False, True, 0, CHOLMOD_PATTERN, cm
    )

    cdef int32_t *Rp = <int32_t*>R.p
    cdef int32_t *Ri = <int32_t*>R.i
    cdef size_t k
    cdef size_t p
    cdef size_t idx

    for k in range(N):
        # Get the kth row of L and store in the columns of L
        cholmod_row_subtree(A_in, F_in, k, Parent, R, cm)

        for p in range(Rp[1]):
            idx = W[Ri[p]]
            Li[idx] = k
            W[Ri[p]] += 1

        # Add the diagonal entry
        idx = W[k]
        Li[idx] = k
        W[k] += 1

    # Free the workspace
    cholmod_free(N, sizeof(int32_t), W, cm)
    cholmod_free_sparse(&R, cm)

    return L


cdef cholmod_sparse* _cholesky_l_pattern(
    cholmod_sparse *A,
    cholmod_sparse *F,
    size_t N,
    int64_t *Parent,
    int64_t *ColCount,
    bint col_etree,
    cholmod_common *cm
):
    """Compute the Cholesky pattern from the given matrices.

    Parameters
    ----------
    A, F : cholmod_sparse*
        Pointers to the sparse matrices to analyze.
    N : size_t
        The number of rows or columns in A.
    Parent : int64_t*
        Pointer to the array of the elimination tree.
    ColCount : int64_t*
        Pointer to the array of column counts of the Cholesky factor.
    col_etree : bint
        If True, analyze the column case F @ F.T. Otherwise, determine the case
        from ``A->stype``.
    cm : cholmod_common*
        Pointer to a CHOLMOD common structure for configuration and status.

    Returns
    -------
    L : cholmod_sparse*
        A pointer to the array containing the pattern of the Cholesky factor.
    """
    if A is NULL or F is NULL or cm is NULL:
        raise ValueError("Input pointer is NULL.")

    cdef cholmod_sparse *A_in = NULL
    cdef cholmod_sparse *F_in = NULL

    if A.stype == 1:
        A_in = A
    elif A.stype == -1:
        A_in = F
    elif col_etree:
        # column case: analyze F @ F.T
        A_in = F
        F_in = A
    else:
        # row case: analyze A @ A.T
        A_in = A
        F_in = F

    # Count the total number of entries in L
    cdef int64_t lnz = 0
    cdef size_t j

    for j in range(N):
        lnz += ColCount[j]

    # Initialize the CHOLMOD sparse matrix for L
    cdef cholmod_sparse *L = cholmod_l_allocate_sparse(
        N, N, lnz, True, True, 0, CHOLMOD_PATTERN, cm
    )

    cdef int64_t *Lp = <int64_t*>L.p
    cdef int64_t *Li = <int64_t*>L.i

    # Initialize column pointers
    lnz = 0

    for j in range(N):
        Lp[j] = lnz
        lnz += ColCount[j]

    Lp[N] = lnz

    # Create a copy of the column pointers
    cdef int64_t *W = <int64_t*>cholmod_l_malloc(N, sizeof(int64_t), cm)
    memcpy(W, Lp, N * sizeof(int64_t))

    # Get workspace for computing one row of L
    cdef cholmod_sparse* R = cholmod_l_allocate_sparse(
        N, 1, N, False, True, 0, CHOLMOD_PATTERN, cm
    )

    cdef int64_t *Rp = <int64_t*>R.p
    cdef int64_t *Ri = <int64_t*>R.i
    cdef size_t k, p, idx

    for k in range(N):
        # Get the kth row of L and store in the columns of L
        cholmod_l_row_subtree(A_in, F_in, k, Parent, R, cm)

        for p in range(Rp[1]):
            idx = W[Ri[p]]
            Li[idx] = k
            W[Ri[p]] += 1

        # Add the diagonal entry
        idx = W[k]
        Li[idx] = k
        W[k] += 1

    # Free the workspace
    cholmod_l_free_sparse(&R, cm)

    return L


# -----------------------------------------------------------------------------
#         CSC <==> CHOLMOD Dense
# -----------------------------------------------------------------------------
cdef void _cholmod_dense_from_ndarray(
    floating_t[::1, :] Xd,
    cholmod_dense *X_static
) noexcept:
    """Create a CHOLMOD dense matrix from a numpy.ndarray.

    See the CHOLMOD MATLAB interface for details [#sputil_get_dense]_.

    Parameters
    ----------
    Xd : (M, N) ndarray
        The input dense matrix. Must be 2-dimensional in column-major (Fortran)
        order.
    X_static : cholmod_sparse*
        Pointer to a preallocated CHOLMOD sparse matrix structure. Contents
        need not be initialized. Contains the CHOLMOD sparse matrix on output.

    Returns
    -------
    res : ndarray
        A reference to the memoryview ``Xd``. There is no use for the output of
        this function, except to keep the underlying data from being garbage
        collected until the cholmod_dense object is freed.

    References
    ----------
    .. [#sputil_get_dense] ``sputil2.c`` - CHOLMOD MATLAB utilities
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/sputil2.c
    """
    # Initialize the CHOLMOD dense matrix
    cdef cholmod_dense* X = X_static
    memset(X, 0, sizeof(cholmod_dense))

    X.nrow = Xd.shape[0]
    X.ncol = Xd.shape[1]
    X.d = X.nrow
    X.nzmax = X.nrow * X.ncol
    X.dtype = _single_or_double[floating_t]()
    X.z = NULL

    # Get the numerical values of X
    cdef floating_t dummy_value
    X.xtype = _real_or_complex[floating_t]()
    X.x = &Xd[0, 0] if Xd.size > 0 else &dummy_value


cdef class _CholmodDenseDestructor:
    """A destructor for CHOLMOD dense matrices.

    This class is used as a base for NumPy arrays that are views on CHOLMOD
    dense matrices. It ensures that the CHOLMOD dense matrix is properly
    freed when the NumPy array is no longer in use.

    Attributes
    ----------
    _dense : cholmod_dense*
        The CHOLMOD dense matrix to be freed.
    _use_int32 : bint
        Whether to use 32-bit or 64-bit integers.
    _common : cholmod_common
        The CHOLMOD common structure used for memory management.
    _cm : cholmod_common*
        A pointer to the CHOLMOD common structure.
    """

    cdef cholmod_dense* _dense
    cdef cholmod_common _common
    cdef cholmod_common* _cm
    cdef bint _use_int32

    cdef void init(self, cholmod_dense* A, bint use_int32, cholmod_common* common):
        assert A is not NULL
        assert common is not NULL
        self._dense = A
        self._use_int32 = use_int32

        self._cm = &self._common

        if self._use_int32:
            cholmod_start(self._cm)
        else:
            cholmod_l_start(self._cm)

        _copy_cholmod_common(self._cm, common)

    def __dealloc__(self):
        if self._use_int32:
            cholmod_free_dense(&self._dense, self._cm)
            cholmod_finish(self._cm)
        else:
            cholmod_l_free_dense(&self._dense, self._cm)
            cholmod_l_finish(self._cm)


cdef cnp.ndarray _ndarray_from_cholmod_dense(
    cholmod_dense* X, bint use_int32, cholmod_common* common
):
    """Create a numpy.ndarray that is a view onto a cholmod_dense object.

    Parameters
    ----------
    X : cholmod_dense*
        The CHOLMOD dense matrix to convert to a NumPy array.
    use_int32 : bint
        Whether to use 32-bit or 64-bit integers.
    common : cholmod_common*
        The CHOLMOD common structure used for memory management.

    Returns
    -------
    res : ndarray
        A NumPy array that is a view onto the CHOLMOD dense matrix. The array
        has a base with a destructor that frees the CHOLMOD dense matrix when
        the array is no longer in use.
    """
    cdef int np_dtypenum = _np_dtypenum_from_cholmod(X.xtype, X.dtype)

    # convert to NumPy array
    arr = cnp.PyArray_SimpleNewFromData(1, [X.nrow * X.ncol], np_dtypenum, X.x)

    # set destructor and check if writeable
    cdef _CholmodDenseDestructor base = _CholmodDenseDestructor()
    base.init(X, use_int32, common)

    # NOTE need to increment reference count of base manually, since the
    # PyArray_SetBaseObject steals a reference.
    Py_INCREF(base)
    if cnp.PyArray_SetBaseObject(arr, base) < 0:
        raise MemoryError("Failed to set base object for array.")

    # Cholmod dense matrices are stored in column-major order, so reshape
    return arr.reshape((X.nrow, X.ncol), order="F")


cdef cnp.ndarray _ndarray_copy_from_intptr(void* ptr, size_t N, bint use_int32):
    """Create a NumPy array from a pointer to an integer array.

    Parameters
    ----------
    ptr : void*
        A pointer to the C array.
    N : size_t
        The size of the vector.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for the array indices.

    Returns
    -------
    p : ndarray
        A copy of the vector as a NumPy array.
    """
    if ptr is NULL:
        raise ValueError("ptr is NULL, cannot get array")

    cdef int np_itypenum = cnp.NPY_INT32 if use_int32 else cnp.NPY_INT64
    p = cnp.PyArray_SimpleNewFromData(1, [N], np_itypenum, ptr)
    return p.copy()  # return a copy in case ptr is freed


cdef cnp.ndarray _ndarray_int_view_from_factor(
    void* ptr, size_t N, CholeskyFactor py_factor
):
    """Create a NumPy array from the an integer vector in a CHOLMOD factor.

    Parameters
    ----------
    ptr : void*
        A pointer to the C array, *e.g.* ``cholmod_factor.Perm``.
    N : size_t
        The size of the vector.
    py_factor : CholeskyFactor
        The CholeskyFactor object from which to extract the permutation.

    Returns
    -------
    p : ndarray
        The permutation vector as a NumPy array. This array is *read-only*,
        so attempts to modify it will raise an error.
    """
    cdef cholmod_factor *L = py_factor._factor

    if L is NULL:
        raise ValueError("The factor pointer is NULL.")

    if ptr is NULL:
        raise ValueError("The input pointer is NULL.")

    cdef int np_itypenum = _np_itypenum_from_cholmod(L.itype)
    p = cnp.PyArray_SimpleNewFromData(1, [N], np_itypenum, ptr)

    # NOTE need to increment reference count of base manually,
    # since the PyArray_SetBaseObject steals a reference.
    Py_INCREF(py_factor)
    if cnp.PyArray_SetBaseObject(p, py_factor) < 0:
        raise MemoryError("Failed to set base object for array.")
    cnp.PyArray_CLEARFLAGS(p, cnp.NPY_ARRAY_WRITEABLE)  # set to be read-only

    return p


# -----------------------------------------------------------------------------
#         Utilities
# -----------------------------------------------------------------------------
cdef dict _supernodal_modes = {
    "auto": CHOLMOD_AUTO,
    "simplicial": CHOLMOD_SIMPLICIAL,
    "supernodal": CHOLMOD_SUPERNODAL,
}


cdef dict _ordering_methods = {
    "default": None,
    "best": None,
    "natural": CHOLMOD_NATURAL,
    "given": CHOLMOD_GIVEN,
    "amd": CHOLMOD_AMD,
    "metis": CHOLMOD_METIS,
    "nesdis": CHOLMOD_NESDIS,
    "colamd": CHOLMOD_COLAMD,
    "postordered": CHOLMOD_POSTORDERED,
}


cdef dict _ordering_methods_inv = {
    v: k for k, v in _ordering_methods.items() if v is not None
}


cdef void _set_ordering_method(object order, cholmod_common* cm):
    """Set the ordering method in the CHOLMOD common struct.

    This function sets the values of ``cm->nmethods``, and possibly
    ``cm->method[0].ordering`` and ``cm->postorder``.

    Parameters
    ----------
    order : None or  str in {"default", "best", "natural", "metis", \
            "nesdis", "amd", "colamd", "postordered"}
        The desired ordering method.
    cm : cholmod_common*
        Pointer to a CHOLMOD common structure for configuration and status.
        Contains the ordering method on output.
    """
    if order == "default":
        cm.nmethods = 0
    elif order == "best":
        cm.nmethods = CHOLMOD_MAXMETHODS
    else:
        # CHOLMOD_POSTORDERED is not an input, but an output flag. We treat it
        # as "natural" + postordering, per cholmod.h description.
        ordering = "natural" if (order is None or order == "postordered") else order
        cm.nmethods = 1
        cm.method[0].ordering = _ordering_methods.get(ordering, CHOLMOD_NATURAL)
        cm.postorder = (
            order == "postordered"
            or ordering not in ["natural", "given"]
        )


cdef inline object _npitype_class_from_itype(int itype):
    """Get the NumPy integer dtype class corresponding to the given CHOLMOD itype.

    Parameters
    ----------
    itype : int
        The CHOLMOD itype of the matrix.

    Returns
    -------
    np_itype_class : type
        The corresponding NumPy integer dtype class.
    """
    return np.int32 if itype == CHOLMOD_INT else np.int64


cdef inline object _npdtype_class_from_xdtype(int xtype, int dtype):
    """Get the NumPy dtype class corresponding to the given CHOLMOD xtype and dtype.

    Parameters
    ----------
    xtype : int
        The CHOLMOD xtype of the matrix.
    dtype : int
        The CHOLMOD dtype of the matrix.

    Returns
    -------
    np_dtype_class : type
        The corresponding NumPy dtype class.
    """
    if xtype == CHOLMOD_REAL:
        return np.float32 if dtype == CHOLMOD_SINGLE else np.float64
    elif xtype == CHOLMOD_COMPLEX:
        return np.complex64 if dtype == CHOLMOD_SINGLE else np.complex128
    elif xtype == CHOLMOD_PATTERN:
        return np.bool_
    else:
        return object


# -----------------------------------------------------------------------------
#         CholeskyFactor Object
# -----------------------------------------------------------------------------
cdef int _copy_cholmod_common(cholmod_common* dest, cholmod_common* src) except -1:
    """Copy the contents of one cholmod_common struct to another."""
    assert dest is not NULL
    assert src is not NULL
    cdef int i

    dest.dbound = src.dbound
    dest.grow0 = src.grow0
    dest.grow1 = src.grow1
    dest.grow2 = src.grow2

    dest.maxrank = src.maxrank

    dest.supernodal_switch = src.supernodal_switch
    dest.supernodal = src.supernodal

    dest.final_asis = src.final_asis
    dest.final_super = src.final_super
    dest.final_ll = src.final_ll
    dest.final_pack = src.final_pack
    dest.final_monotonic = src.final_monotonic
    dest.final_resymbol = src.final_resymbol

    if src.zrelax is not NULL:
        for i in range(3):
            dest.zrelax[i] = src.zrelax[i]

    if src.nrelax is not NULL:
        for i in range(3):
            dest.nrelax[i] = src.nrelax[i]

    dest.prefer_zomplex = src.prefer_zomplex
    dest.prefer_upper = src.prefer_upper
    dest.quick_return_if_not_posdef = src.quick_return_if_not_posdef
    dest.prefer_binary = src.prefer_binary

    dest.print = src.print
    dest.precise = src.precise
    dest.try_catch = src.try_catch

    dest.error_handler = src.error_handler

    # Ordering
    dest.nmethods = src.nmethods
    dest.current = src.current
    dest.selected = src.selected

    if src.method is not NULL:
        for i in range(src.nmethods):
            dest.method[i].lnz = src.method[i].lnz
            dest.method[i].fl = src.method[i].fl
            dest.method[i].prune_dense = src.method[i].prune_dense
            dest.method[i].prune_dense2 = src.method[i].prune_dense2
            dest.method[i].nd_oksep = src.method[i].nd_oksep
            dest.method[i].nd_small = src.method[i].nd_small
            dest.method[i].aggressive = src.method[i].aggressive
            dest.method[i].order_for_lu = src.method[i].order_for_lu
            dest.method[i].nd_compress = src.method[i].nd_compress
            dest.method[i].nd_camd = src.method[i].nd_camd
            dest.method[i].nd_components = src.method[i].nd_components
            dest.method[i].ordering = src.method[i].ordering

    dest.postorder = src.postorder
    dest.default_nesdis = src.default_nesdis

    dest.metis_memory = src.metis_memory
    dest.metis_dswitch = src.metis_dswitch
    dest.metis_nswitch = src.metis_nswitch

    # Workspace -- not copied since CHOLMOD manages workspace internally
    # dest.nrow = src.nrow
    # dest.mark = src.mark
    # dest.iworksize = src.iworksize
    # dest.xworkbytes = src.xworkbytes
    # Flag, Head, Xwork, Iwork are set to NULL by cholmod_start, and kept as cleared
    # arrays between CHOLMOD calls, so we do not copy them here.

    dest.itype = src.itype
    dest.no_workspace_reallocate = src.no_workspace_reallocate

    # Output Statistics
    dest.status = src.status
    dest.fl = src.fl
    dest.lnz = src.lnz
    dest.anz = src.anz
    dest.modfl = src.modfl

    dest.malloc_count = src.malloc_count
    dest.memory_usage = src.memory_usage
    dest.memory_inuse = src.memory_inuse

    dest.nrealloc_col = src.nrealloc_col
    dest.nrealloc_factor = src.nrealloc_factor
    dest.ndbounds_hit = src.ndbounds_hit

    dest.rowfacfl = src.rowfacfl
    dest.aatfl = src.aatfl

    dest.called_nd = src.called_nd
    dest.blas_ok = src.blas_ok

    # SPQR parameters and statistics
    dest.SPQR_grain = src.SPQR_grain
    dest.SPQR_small = src.SPQR_small
    dest.SPQR_shrink = src.SPQR_shrink
    dest.SPQR_nthreads = src.SPQR_nthreads

    dest.SPQR_flopcount = src.SPQR_flopcount
    dest.SPQR_analyze_time = src.SPQR_analyze_time
    dest.SPQR_factorize_time = src.SPQR_factorize_time
    dest.SPQR_solve_time = src.SPQR_solve_time
    dest.SPQR_flopcount_bound = src.SPQR_flopcount_bound
    dest.SPQR_tol_used = src.SPQR_tol_used
    dest.SPQR_norm_E_fro = src.SPQR_norm_E_fro

    if src.SPQR_istat is not NULL:
        for i in range(8):
            dest.SPQR_istat[i] = src.SPQR_istat[i]

    # New as of CHOLMOD v5.0
    dest.nsbounds_hit = src.nsbounds_hit
    dest.sbound = src.sbound

    # Skip GPU related fields for now

    return 0


cdef class CholeskyFactor:
    """The main object used for creating and manipulating a Cholesky factor.

    The constructor computes the symbolic analysis of the matrix and
    determines a fill-reducing ordering (if ``order`` is not ``None`` or
    ``"natural"``) such that:

    .. math ::

        L L^{\\top} = P A P^{\\top}.

    The numeric factorization is not computed until :meth:`.factorize` is
    called.

    Parameters
    ----------
    A : (N, N) array_like or sparse array
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format.
    sym_kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization.  Only the upper or lower triangular part of
          the matrix is used (depending on ``lower``), and no check is made for
          symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.

        The resulting matrix must be square and symmetric positive definite.

    supernodal_mode : str in {"auto", "simplicial", "supernodal"}, optional
        The type of factorization to use:

        * ``auto``: Automatically select the factorization type.
        * ``simplicial``: Use a simplicial factorization.
        * ``supernodal``: Use a supernodal factorization.

        Default is ``auto``. This mode also applies to any subsequent calls to
        :meth:`.factorize`. Note that the ``simplicial`` mode may be slow for
        large matrices.

    lower : bool, optional
        If True, use the lower triangular part of ``A``.
    order : str in {"default", "best", "natural", "metis", \
            "nesdis", "amd", "colamd", "postordered"}, optional
        The permutation algorithm to use for the factorization. Options are:

        * ``default``: Use the default method, which first tries AMD, then METIS.
        * ``best``: Automatically select the best ordering based on the input.
        * ``metis``: Use the METIS library for graph partitioning.
        * ``nesdis``: Use the NESDIS library for nested dissection.
        * ``amd``: Use the Approximate Minimum Degree (AMD) algorithm.
        * ``colamd``: Use the Approximate Minimum Degree (AMD) algorithm
          for the symmetric case, or the COLAMD algorithm for the
          unsymmetric case (:math:`A A^{{\\top}}` or :math:`A^{{\\top}} A`).
        * ``postordered``: Use natural ordering followed by postordering.
        * ``natural`` or ``None``: No permutation is applied (identity permutation).

        By default, methods other than ``natural`` will also be
        postordered.

        .. warning::

            The ordering method ``best`` may be quite slow for large
            matrices, but if the factorization is reused many times, it can
            be worth it.

    Attributes
    ----------
    N : int
        The number of rows and columns in the factor.
    is_ll : bool
        Whether the factor is in ``LL.T`` form (True) or ``LDL.T`` form (False).
    is_super : bool
        Whether the factor is in supernodal (True) or simplicial (False) format.
    itype : :obj:`numpy.int32` or :obj:`numpy.int64`
        The integer type used for indices and indptr in the factor.
    dtype : numpy.dtype
        The data type used for numerical values in the factor.
    sym_kind : str
        The symmetry kind used for the factorization.
    rcond : float
        A rough estimate of the reciprocal of the condition number of the matrix,
        defined as ``(L.diagonal().min() / L.diagonal().max())**2`` for an ``LL.T``
        factorization. Estimated during the numeric factorization. If
        :meth:`.factorize` has not yet been called, this value is ``-1.0``.
    colcount : *(N,)* :obj:`numpy.ndarray` of int
        The number of nonzeros in each column of the factor.
    nnz : int
        The number of nonzeros in the factor.
    order : str or int
        The ordering method used for the factorization. If an unknown ordering
        was used, returns the integer value.
    perm : *(N,)* :obj:`numpy.ndarray` of int
        A read-only view of the permutation vector used for the factorization.
    factor : :obj:`~scipy.sparse.csc_array`
        A view of the the Cholesky factor in Compressed Sparse Column (CSC)
        format. If ``self.is_ll``, the returned matrix is lower triangular.
        Otherwise, the matrix view contains the lower triangular and the
        diagonal factors combined.

        .. note::

            The view is always in lower triangular form, even if the factor was
            created using ``lower=False``. To get the upper triangular factor,
            use :obj:`get_factor` with ``lower=False``. To get the split `L`
            and `D` factors, use :obj:`get_factor` with ``kind="LDL"``.

        .. warning::

            The returned matrix is a view on the internal data of the CHOLMOD
            factor. It will be modified if the factor is modified (*e.g.*, by
            calling :meth:`.factorize`). To get a copy, use
            :meth:`.get_factor`.
    L : :class:`~scipy.sparse.csc_array`
        The lower triangular factor in Compressed Sparse Column (CSC) format.
    R : :class:`~scipy.sparse.csc_array`
        The upper triangular factor in Compressed Sparse Column (CSC) format.
    D : :class:`~scipy.sparse.dia_array`
        A view of the diagonal factor in DIAgonal format. If ``self.is_ll``,
        this is the identity matrix.

    Raises
    ------
    CholmodNotPositiveDefiniteError
        If the input matrix is structurally singular (*e.g.*, if it is the zero
        matrix). The input *may* be numerically indefinite, but this property
        is not checked until :meth:`.factorize` is called.

    See Also
    --------
    cholesky, ldl, cho_factor, ldl_factor

    Notes
    -----
    The symbolic analysis follows that of the SuiteSparse CHOLMOD ``analyze``
    MATLAB function [#analyze_c]_.

    .. warning::

        Calling ``CholeskyFactor.__new__(CholeskyFactor)`` will leave the object in an
        "unsafe" state, since the internal CHOLMOD structures will not be initialized.
        Always use the constructor ``CholeskyFactor(...)`` to create a new object.


    References
    ----------
    .. [#analyze_c] ``analyze.c`` - CHOLMOD MATLAB analyze function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/analyze.c
    """

    cdef:
        cholmod_common _Common
        cholmod_common *_cm
        cholmod_factor *_factor
        bint _use_int32
        int _stype
        readonly bint is_lower
        readonly object sym_kind
        readonly double rcond
        # Cached factors
        cnp.ndarray _perm
        cnp.ndarray _colcount
        object _L
        object _R
        object _D
        object _factor_view

    def __init__(
        self,
        object A,
        *,
        bint lower=True,
        object order="default",
        object sym_kind=None,
        object supernodal_mode=None,
    ):
        A, self._use_int32, _ = validate_csc_input(A, ensure_double=False)

        if sym_kind is None:
            sym_kind = "sym"

        if supernodal_mode is None:
            supernodal_mode = "auto"

        if sym_kind not in {"sym", "row", "col"}:
            raise ValueError(
                f"Unknown symmetry kind: {sym_kind}. "
                "Must be one of 'sym', 'row', 'col'."
            )

        if sym_kind == "sym" and A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected square matrix. Got {A.shape}.")

        if supernodal_mode not in _supernodal_modes:
            raise ValueError(
                f"Unknown factorization mode: {supernodal_mode}. "
                f"Must be one of {set(_supernodal_modes.keys())}."
            )

        # Check the input ordering method
        if order is not None and order not in _ordering_methods:
            raise ValueError(
                f"Unknown ordering method: {order}. "
                f"Must be one of {set(_ordering_methods.keys())}."
            )

        # Matrix of all zeros
        if A.shape[0] > 0 and A.nnz == 0:
            raise CholmodNotPositiveDefiniteError("Input matrix not positive definite.")

        # Get the input matrix into CHOLMOD format
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix
        cdef cholmod_sparse *C

        # Use lower or upper triangular part of A
        self.is_lower = lower
        self.sym_kind = sym_kind
        self.rcond = -1.0
        cdef int stype = -1 if self.is_lower else 1
        cdef bint transpose = False

        if self.sym_kind in ["row", "col"]:
            stype = 0                        # unsymmetric A @ A.T or A.T @ A
            transpose = (self.sym_kind == "col")  # A.T @ A

        _cholmod_sparse_from_csc(
            A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
        )

        self._stype = Ac.stype

        # Initialize the Common object
        self._cm = &self._Common

        if self._use_int32:
            cholmod_start(self._cm)
        else:
            cholmod_l_start(self._cm)

        self._cm.supernodal = _supernodal_modes[supernodal_mode]
        _set_ordering_method(order, self._cm)

        # Analyze the matrix, but do not factorize yet
        if transpose:
            if self._use_int32:
                C = cholmod_transpose(Ac, CHOLMOD_TRANS_PATTERN, self._cm)
                self._factor = cholmod_analyze(C, self._cm)
                cholmod_free_sparse(&C, self._cm)
            else:
                C = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, self._cm)
                self._factor = cholmod_l_analyze(C, self._cm)
                cholmod_l_free_sparse(&C, self._cm)
        else:
            if self._use_int32:
                self._factor = cholmod_analyze(Ac, self._cm)
            else:
                self._factor = cholmod_l_analyze(Ac, self._cm)

        # Check for errors
        cdef int minor = -1 if self._factor is NULL else self._factor.minor
        _handle_errors(self._cm.status, minor)

    def __dealloc__(self):
        """Deallocate memory used by the CholeskyFactor."""
        if self._use_int32:
            cholmod_free_factor(&self._factor, self._cm)
            cholmod_finish(self._cm)
        else:
            cholmod_l_free_factor(&self._factor, self._cm)
            cholmod_l_finish(self._cm)

    def _require_factorized(self):
        """Raise an error if the factor is symbolic only."""
        if not self.is_numeric:
            raise CholmodError("Factor is symbolic. Call `factorize` before updating.")

    def _clear_cache(self):
        """Clear cached properties."""
        self._factor_view = None
        self._perm = None
        self._L = None
        self._R = None
        self._D = None

    def __repr__(self):
        return (
            f"CholeskyFactor("
            f"N={self.N}, "
            f"nnz={self.nnz}, "
            f"is_ll={self.is_ll}, "
            f"is_super={self.is_super}, "
            f"itype=np.{self.itype.name}, "
            f"dtype=np.{self.dtype.name}, "
            f"order={self.order}"
            ")"
        )

    def __str__(self):
        lines = [
            f"Cholesky factorization of size {self.N}x{self.N}",
            f"  Nonzeros: {self.nnz}",
            f"  Form:     {'LL.T' if self.is_ll else 'LDL.T'}",
            f"  Triangle: {'lower' if self.is_lower else 'upper'}",
            f"  Storage:  {'supernodal' if self.is_super else 'simplicial'}",
            f"  itype:    np.{self.itype.name}",
            f"  dtype:    np.{self.dtype.name}",
            f"  order:    {self.order}",
        ]
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    #         Properties
    # -------------------------------------------------------------------------
    @property
    def is_ll(self):
        return bool(self._factor.is_ll)

    @property
    def is_super(self):
        return bool(self._factor.is_super)

    @property
    def is_numeric(self):
        return self._factor.xtype != CHOLMOD_PATTERN

    @property
    def itype(self):
        return np.dtype(_npitype_class_from_itype(self._factor.itype))

    @property
    def dtype(self):
        # "np.int32" etc. are dtype classes, not actual dtypes. numpy handles
        # both well, but be explicit and return a dtype object.
        return np.dtype(
            _npdtype_class_from_xdtype(self._factor.xtype, self._factor.dtype)
        )

    @property
    def N(self):
        return self._factor.n

    @property
    def colcount(self):
        if self._colcount is None:
            self._colcount = _ndarray_int_view_from_factor(
                self._factor.ColCount, self._factor.n, self
            )
        return self._colcount

    @property
    def nnz(self):
        return np.sum(self.colcount)

    @property
    def order(self):
        cdef int iorder = self._factor.ordering
        return _ordering_methods_inv.get(iorder, iorder)

    @property
    def perm(self):
        if self._perm is None:
            self._perm = _ndarray_int_view_from_factor(self._factor.Perm, self._factor.n, self)
        return self._perm

    @property
    def factor(self):
        if self._factor_view is None:
            self._factor_view = _csc_view_from_cholmod_factor(self)
        return self._factor_view

    @property
    def L(self):
        if self._L is None:
            if self.is_ll:
                self._L = self.get_factor(kind="LL", lower=self.is_lower)
            else:
                self._L, self._D = self.get_factor(kind="LDL", lower=self.is_lower)
        return self._L

    @property
    def R(self):
        if self._R is None:
            if self._L is None:
                if self.is_ll:
                    self._L = self.get_factor(kind="LL", lower=self.is_lower)
                else:
                    self._L, self._D = self.get_factor(kind="LDL", lower=self.is_lower)
            self._R = self._L.T.conj()
        return self._R

    @property
    def D(self):
        if self._D is None:
            if self.is_ll:
                self._D = eye_array(self.N, dtype=self.dtype)
            else:
                self._L, self._D = self.get_factor(kind="LDL", lower=self.is_lower)
        return self._D

    # -------------------------------------------------------------------------
    #         Public Methods
    # -------------------------------------------------------------------------
    def copy(self):
        """Return a copy of the CholeskyFactor object.

        This method creates a deep copy of the CholeskyFactor object,
        including the CHOLMOD common struct and the factor itself.

        This method does not copy *all* of the underlying `cholmod_common`
        struct, only the parts that are necessary for using the factor.

        Returns
        -------
        CholeskyFactor
            A deep copy of the CholeskyFactor object.
        """
        cdef CholeskyFactor cf = CholeskyFactor.__new__(CholeskyFactor)

        cf._cm = &cf._Common

        # Allocate and copy the CHOLMOD common struct
        if self._use_int32:
            cholmod_start(cf._cm)
        else:
            cholmod_l_start(cf._cm)

        _copy_cholmod_common(cf._cm, self._cm)

        # Copy the factor
        if self._use_int32:
            cf._factor = cholmod_copy_factor(self._factor, cf._cm)
        else:
            cf._factor = cholmod_l_copy_factor(self._factor, cf._cm)

        _handle_errors(cf._cm.status)

        cf._use_int32 = self._use_int32
        cf.is_lower = self.is_lower
        cf._stype = self._stype
        cf.sym_kind = self.sym_kind
        cf.rcond = self.rcond

        return cf

    def change_factor(self, kind=None):
        """Change the type of factorization used by the object.

        This method changes the type of factorization used by the object
        between ``LL.T`` and ``LDL.T``. The symbolic analysis is reused,
        so this method does not require refactorization of the matrix.

        Parameters
        ----------
        kind : str in {'LL', 'LDL'}, optional
            The type of factorization to use. If ``LL``, use the Cholesky
            factor `L` such that :math:`L L^{\\top} = P A P^{\\top}`. If
            ``LDL``, use the combined `LD` factor such that
            :math:`L D L^{\\top} = P A P^{\\top}`. Default is None, which
            switches to the other type of factorization.

        See Also
        --------
        factor, get_factor
        """
        self._require_factorized()

        if kind is None:
            kind = "LDL" if self.is_ll else "LL"

        if kind not in ("LL", "LDL"):
            raise ValueError("kind must be 'LL' or 'LDL'.")

        # Clear cached properties
        self._clear_cache()

        cdef int to_ll = (kind == "LL")
        cdef int to_super = self._factor.is_super
        cdef int to_packed = True
        cdef int to_monotonic = self._factor.is_monotonic

        if self._factor.itype == CHOLMOD_INT:
            cholmod_change_factor(
                self._factor.xtype,
                to_ll,
                to_super,
                to_packed,
                to_monotonic,
                self._factor,
                self._cm,
            )
        else:
            cholmod_l_change_factor(
                self._factor.xtype,
                to_ll,
                to_super,
                to_packed,
                to_monotonic,
                self._factor,
                self._cm,
            )

        _handle_errors(self._cm.status)

        return self

    def get_factor(self, kind=None, lower=None):
        """Return a copy of the Cholesky factor in the specified format.

        Parameters
        ----------
        kind : None or str in {'LL', 'LDL'}, optional
            The type of factor to return. If ``LL``, return the Cholesky
            factor `L` such that :math:`L L^{\\top} = P A P^{\\top}`. If
            ``LDL``, return the combined `LD` factor such that
            :math:`L D L^{\\top} = P A P^{\\top}`. Default is None, which
            uses the kind with which ``factorize`` was called.
        lower : None or bool, optional
            If True, return the lower triangular factor `L`. If False, return
            the upper triangular factor `R`. If None (default), return the
            factor in the same triangular form as it was created with
            ``factorize``.

        Returns
        -------
        L : csc_array
            The Cholesky factor in Compressed Sparse Column (CSC) format.
        D : diags_array, optional
            If ``kind="LDL"``, also returns the `D` factor.
        """
        if kind is None:
            kind = "LL" if self.is_ll else "LDL"

        if kind not in ("LL", "LDL"):
            raise ValueError("kind must be 'LL' or 'LDL'.")

        if lower is None:
            lower = self.is_lower

        # Drop explicit zeros from returned copies
        if kind == "LL":
            L = _csc_view_from_cholmod_factor(self, ldl=False).copy()
            L.eliminate_zeros()
            if not lower:
                L = L.T.conj()
            return L
        else:
            # Extract L and D from combined LD factor
            L = _csc_view_from_cholmod_factor(self, ldl=True).copy()
            D = diags_array(L.diagonal())
            L.setdiag(1.0)
            L.eliminate_zeros()
            if not lower:
                L = L.T.conj()
            return L, D

    def get_perm(self):
        """Return a copy of the permutation vector used in the factorization.

        Returns
        -------
        p : ndarray
            The permutation vector `p` such that :math:`P A P^{\\top}` is the
            matrix that was factorized, where `P` is the permutation matrix
            corresponding to `p`, *i.e.*, ``P = I[p]``.
        """
        return self.perm.copy()

    def factorize(self, object A, object ldl=None, double beta=0.0):
        """Compute the numerical Cholesky factorization of a sparse matrix.

        This method computes the numerical values of :math:`P A P^{\\top}
        = R^{\\top} R` or :math:`P A P^{\\top} = L L^{\\top}` decomposition of
        a Hermitian positive-definite matrix `A`, with fill-reducing
        permutation `P`.

        Parameters
        ----------
        A : (N, N) {array_like, sparse array}
            An array convertible to a sparse matrix in Compressed Sparse Column
            (CSC) format. The matrix must be square and symmetric positive
            definite. Only the upper or lower triangular part of the matrix is
            used, and no check is made for symmetry. This matrix can be
            numericaly different from the matrix used to initialize the
            :obj:`CholeskyFactor` object, but it must have the same sparsity
            pattern.
        ldl : bool, optional
            If True, compute the LDL factorization instead of the Cholesky
            factorization. Default is None, which uses the same type of
            factorization as the previous call to :meth:`.factorize`, or False
            if this is the first call.
        beta : float, optional
            The scalar value to add to the diagonal of the matrix before
            factorization. Default is 0.

        Notes
        -----
        If ``ldl=False``, this function computes the Cholesky factorization of
        a symmetric positive definite matrix `A`:

        .. math::

            R^{\\top} R = P A P^{\\top},

        where `R` is an upper triangular matrix. Only the upper triangular part
        of `A` is used. If ``self.is_lower`` is True, the lower triangular
        factor `L` is computed instead, such that:

        .. math::

            L L^{\\top} = P A P^{\\top}.

        In this case, only the lower triangular part of `A` is used.

        If ``ldl=True``, compute the factorization:

        .. math::

            R^{\\top} D R = P A P^{\\top},

        or

        .. math::

            L D L^{\\top} = P A P^{\\top},

        respectively.

        If ``beta`` is not None, the factorization is computed for the matrix:

        .. math::

            P A P^{\\top} + \\beta I.

        Note that if the :obj:`CholeskyFactor` was initialized with ``sym_kind``
        equal to ``"row"`` or ``"col"``, the factorization is computed for
        :math:`P A A^{\\top} P^{\\top}` or :math:`P A^{\\top} A P^{\\top}`,
        respectively. Similarly, ``beta`` is added to the diagonal of these
        matrices.
        """
        assert self._factor is not NULL, "The factor has not been initialized."

        A, _, _ = validate_csc_input(A, ensure_double=False)

        if ldl is None:
            if self.is_numeric:
                ldl = not self.is_ll  # use the existing factor type
            else:
                ldl = False           # default to LL for first factorization

        if not isinstance(ldl, bool):
            raise ValueError("ldl must be a boolean value.")

        # Clear cached properties
        self._clear_cache()

        # See CHOLMOD/MATLAB/ldlchol.c and/or lchol.c for details
        self._cm.final_asis = False
        self._cm.final_super = False
        self._cm.final_ll = not ldl  # LL.T for Cholesky, LDL.T for LDL
        self._cm.final_pack = True
        self._cm.final_monotonic = True

        # We do *not* drop numerically zero entries from the symbolic
        # pattern, so that the resulting factor can be updated by `.update`.
        # We *do* drop entries that result from supernodal amalgamation.
        self._cm.final_resymbol = True

        self._cm.quick_return_if_not_posdef = True

        # Get the input matrix into CHOLMOD format
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix

        _cholmod_sparse_from_csc(
            A.shape, A.indptr, A.indices, A.data, self._stype, <uintptr_t>Ac
        )

        # Set beta
        if not np.isscalar(beta):
            raise ValueError("beta must be a scalar value.")

        cdef double betac[2]
        betac[0] = beta
        betac[1] = 0.0

        # If the symbolic analysis was for the unsymmetric case, determine
        # whether to factorize A @ A.T or A.T @ A
        cdef bint transpose = False
        if self._stype == 0:
            # Unsymmetric case: factorize A @ A.T or A.T @ A
            transpose = (self.sym_kind == "col")

        # Factorize the matrix
        if transpose:
            if self._use_int32:
                C = cholmod_transpose(Ac, CHOLMOD_TRANS_CONJ, self._cm)
                cholmod_factorize_p(C, betac, NULL, 0, self._factor, self._cm)
                cholmod_free_sparse(&C, self._cm)
            else:
                C = cholmod_l_transpose(Ac, CHOLMOD_TRANS_CONJ, self._cm)
                cholmod_l_factorize_p(C, betac, NULL, 0, self._factor, self._cm)
                cholmod_l_free_sparse(&C, self._cm)
        else:
            if self._use_int32:
                cholmod_factorize_p(Ac, betac, NULL, 0, self._factor, self._cm)
            else:
                cholmod_l_factorize_p(Ac, betac, NULL, 0, self._factor, self._cm)

        # Check for errors
        _handle_errors(self._cm.status, self._factor.minor)

        # Update rcond
        self._rcond()

        return self  # for method chaining

    def solve(self, b):
        """Solve the linear system :math:`A x = b` for `x`, using the
        factorization.

        Parameters
        ----------
        b : (N,) or (N, K) ndarray or sparse matrix
            The right-hand side vector or matrix. Must be a type that can be safely
            cast to the data type of the factor. The number of rows in ``b`` must be
            equal to the size of the factor.

        Returns
        -------
        x : (N,) or (N, K) ndarray or sparse matrix
            The solution vector or matrix, returned in the same format as ``b``.

        Raises
        ------
        CholmodNotPositiveDefiniteError
            If the matrix `A` is exactly singular, or singular to working
            precision.

        Notes
        -----
        This function solves the linear system:

        .. math::

            R^{\\top} R x = b,

        where `R` is the upper triangular factor from the Cholesky factorization
        of `A`. The input `b` is either dense or sparse, vector or matrix.

        If ``order`` was not ``natural`` when the factorization was computed,
        solve the system:

        .. math::

            P^{\\top} R^{\\top} R P x = b

        where `P` is the permutation matrix corresponding to the permutation
        vector. Similarly, if ``lower`` was True when the factorization was
        computed, the system solved is:

        .. math::

            P^{\\top} L L^{\\top} P x = b.

        If the factorization is in LDL form, the system solved is:

        .. math::

            P^{\\top} L D L^{\\top} P x = b.

        This function uses the CHOLMOD library to solve the linear system. It
        is intended to combine the MATLAB interfaces ``cholmod2.m``
        [#cholmod_c]_, and ``ldlsolve.m`` [#ldlsolve_c]_.

        .. versionadded:: 0.5.0

        References
        ----------
        .. [#cholmod_c] ``cholmod2.c`` - CHOLMOD MATLAB interface
            https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/cholmod2.c
        .. [#ldlsolve_c] ``ldlsolve.c`` - CHOLMOD MATLAB interface
            https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/ldlsolve.c
        """
        self._require_factorized()

        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")

        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        cdef size_t N = self._factor.n

        if b.shape[0] != N:
            raise ValueError(
                "Right-hand side b must have the same number of rows as L."
            )

        if not np.can_cast(b.dtype, self.dtype):
            raise TypeError(f"Cannot safely cast {b.dtype=} to {self.dtype=}.")
        else:
            b = b.astype(self.dtype, copy=False)

        # Special case: zero-dimension matrix
        if N == 0:
            return type(b)(b.shape, dtype=b.dtype)

        # Check the condition number before solving
        self._check_rcond()

        cdef bint return_1D = b.ndim == 1

        # CHOLMOD requires a 2D array
        if b.ndim == 1:
            b = b.reshape((N, 1))

        if issparse(b):
            X = self._solve_sparse(b.tocsc())
        else:
            X = self._solve_dense(np.asfortranarray(b))

        # Convert to 1D array if input b is 1D
        if return_1D:
            X = X[:, 0]

        return X

    cdef object _solve_sparse(self, object b):
        """Solve the system A x = b with a sparse right-hand side."""
        # Get the b vector or matrix into CHOLMOD format
        cdef cholmod_sparse Bspmatrix
        cdef cholmod_sparse* Bs = &Bspmatrix

        cdef int stype = 0

        b, _, _ = validate_csc_input(b, ensure_double=False)

        _cholmod_sparse_from_csc(
            b.shape, b.indptr, b.indices, b.data, stype, <uintptr_t>&Bspmatrix
        )

        # Solve the system
        cdef cholmod_sparse* Xs

        if self._use_int32:
            Xs = cholmod_spsolve(CHOLMOD_A, self._factor, Bs, self._cm)
        else:
            Xs = cholmod_l_spsolve(CHOLMOD_A, self._factor, Bs, self._cm)

        _handle_errors(self._cm.status)

        return _csc_from_cholmod_sparse(Xs, self._cm)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _solve_dense(self, floating_t[::1, :] b not None):
        """Solve the system A x = b with a dense right-hand side."""
        # Get the b vector or matrix into CHOLMOD format
        cdef cholmod_dense Bmatrix
        cdef cholmod_dense* Bd = &Bmatrix

        _cholmod_dense_from_ndarray(b, Bd)

        # Solve the system
        cdef cholmod_dense* Xd

        if self._use_int32:
            Xd = cholmod_solve(CHOLMOD_A, self._factor, Bd, self._cm)
        else:
            Xd = cholmod_l_solve(CHOLMOD_A, self._factor, Bd, self._cm)

        _handle_errors(self._cm.status)

        return _ndarray_from_cholmod_dense(Xd, self._use_int32, self._cm)

    cdef int _rcond(self) except -1:
        """Compute the reciprocal condition number."""
        if self._use_int32:
            self.rcond = cholmod_rcond(self._factor, self._cm)
        else:
            self.rcond = cholmod_l_rcond(self._factor, self._cm)
        _handle_errors(self._cm.status)
        return 0

    cdef int _check_rcond(self) except -1:
        """Check the condition number."""
        cdef double thresh = self._factor.n * np.finfo(self.dtype).eps

        if self.rcond == 0:
            raise CholmodNotPositiveDefiniteError(
                "Matrix is indefinite or singular to working precision."
            )
        elif self.rcond < thresh:
            warnings.warn(
                "Matrix is nearly singular."
                f"  Results may be inaccurate (rcond={self.rcond:.2e}).",
                CholmodWarning,
            )

        return 0

    def update(self, C):
        return self._update(C, updown="up")

    def downdate(self, C):
        return self._update(C, updown="down")

    def _update(self, C, updown="up"):
        assert updown in ("up", "down")

        self._require_factorized()

        if not issparse(C) or C.ndim not in {1, 2}:
            raise ValueError(f"Update matrix C is type {type(C)}. "
                             "Expected a 1D or 2D sparse array.")

        cdef size_t N = self._factor.n

        if C.shape[0] != N:
            raise ValueError("Update matrix C must have the same number of rows as L.")

        # Clear cached properties
        self._clear_cache()

        # Ensure C is in CSC format
        if C.ndim == 1:
            C = C.reshape((-1, 1)).tocsc()  # (N, 1)

        cdef int stype = 0  # use all of C
        C, _, _ = validate_csc_input(C, ensure_double=False)

        cdef cholmod_sparse Cmatrix
        cdef cholmod_sparse* Cc = &Cmatrix

        _cholmod_sparse_from_csc(
            C.shape, C.indptr, C.indices, C.data, stype, <uintptr_t>&Cmatrix
        )

        # Permute C so it is accepted in "matrix" space.
        # From Modify/cholmod_updown.c:
        #   Note that the fill-reducing permutation L->Perm is NOT used.  The row
        #   indices of C refer to the rows of L, not A.  If your original system is
        #   LDL' = PAP' (where P = L->Perm), and you want to compute the LDL'
        #   factorization of A+CC', then you must permute C first.  That is:
        #
        #        PAP' = LDL'
        #        P(A+CC')P' = PAP'+PCC'P' = LDL' + (PC)(PC)' = LDL' + Cnew*Cnew'
        #        where Cnew = P*C.
        #
        #   You can use the cholmod_submatrix routine in the MatrixOps module
        #   to permute C, with:
        #
        #   Cnew = cholmod_submatrix (C, L->Perm, L->n, NULL, -1, TRUE, TRUE, Common) ;
        #
        #   Note that the sorted input parameter to cholmod_submatrix must be TRUE,
        #   because cholmod_updown requires C with sorted columns.
        cdef cholmod_sparse *C_perm

        if self._use_int32:
            C_perm = cholmod_submatrix(
                Cc, <int32_t*>self._factor.Perm, N, NULL, -1, True, True, self._cm
            )
        else:
            C_perm = cholmod_l_submatrix(
                Cc, <int64_t*>self._factor.Perm, N, NULL, -1, True, True, self._cm
            )

        # Compute the update or downdate
        cdef int update = updown == "up"
        cdef int ok

        if self._use_int32:
            ok = cholmod_updown(update, C_perm, self._factor, self._cm)
        else:
            ok = cholmod_l_updown(update, C_perm, self._factor, self._cm)

        if self._use_int32:
            cholmod_free_sparse(&C_perm, self._cm)
        else:
            cholmod_l_free_sparse(&C_perm, self._cm)

        if not ok:
            raise CholmodError("Update or downdate failed.")

        return self

    def rowadd(self, k, C):
        r"""Add a row to a sparse LDL factorization.

        Compute a rank-1 update of a sparse LDL factorization. This method
        "adds" a row by setting the :math:`k^{th}` row and column of the
        original matrix to ``C``.

        Parameters
        ----------
        k : int :math:`\in [0, N)`
            The row/column index to modify.
        C : (N, 1) csc_array, optional
            If given, change the factorization such that row and column ``k``
            of the original matrix equal ``C``. The number of rows must match
            that of ``L`` and ``D``.

        Returns
        -------
        CholeskyFactor
            The current object, for method chaining.

        .. versionadded:: 0.5.0
        """
        self._require_factorized()

        if not (0 <= k < self.N):
            raise IndexError(
                f"Row index k={k} is out of bounds for matrix of size {self.N}."
            )

        if not issparse(C) or C.ndim not in {1, 2}:
            raise ValueError(
                f"Update matrix C is type {type(C)}."
                "Expected a 1D or 2D sparse array."
            )

        if C.shape[0] != self.N:
            raise ValueError(
                "Update matrix C must have the same number of rows as L."
            )

        # Clear cached properties
        self._clear_cache()

        # Get C Matrix
        cdef cholmod_sparse Cmatrix
        cdef cholmod_sparse* Cc = &Cmatrix
        cdef int stype = 0  # use all of C

        # Ensure C is in CSC format
        if C.ndim == 1:
            C = C.reshape((-1, 1)).tocsc()  # (N, 1)

        C, _, _ = validate_csc_input(C, ensure_double=False)
        _cholmod_sparse_from_csc(
            C.shape, C.indptr, C.indices, C.data, stype, <uintptr_t>&Cmatrix
        )

        # Compute the Update
        cdef int ok

        if self._use_int32:
            ok = cholmod_rowadd(k, Cc, self._factor, self._cm)
        else:
            ok = cholmod_l_rowadd(k, Cc, self._factor, self._cm)

        if not ok:
            raise CholmodError("cholmod_rowadd failed.")

        return self

    def rowdel(self, k):
        r"""Delete a row from a sparse LDL factorization.

        Compute a rank-1 update of a sparse LDL factorization. This method
        "deletes" a row by setting the :math:`k^{th}` row and column of the
        original matrix to the identity.

        Parameters
        ----------
        k : int :math:`\in [0, N)`
            The row/column index to modify.

        Returns
        -------
        CholeskyFactor
            The current object, for method chaining.

        .. versionadded:: 0.5.0
        """
        self._require_factorized()

        if not (0 <= k < self.N):
            raise IndexError(
                f"Row index k={k} is out of bounds for matrix of size {self.N}."
            )

        # Clear cached properties
        self._clear_cache()

        cdef int ok

        if self._use_int32:
            ok = cholmod_rowdel(k, NULL, self._factor, self._cm)
        else:
            ok = cholmod_l_rowdel(k, NULL, self._factor, self._cm)

        if not ok:
            raise CholmodError("cholmod_rowdel failed.")

        return self

    def resymbol(self, object A, bint is_permuted=True):
        """Recompute the symbolic Cholesky factorization of a sparse matrix.

        This function is useful after a series of downdates via
        :meth:`.update` or :meth:`.rowdel`, since downdates do not remove
        any entries in ``L`` [#resymbol_c]_.

        Parameters
        ----------
        A : (N, N) csc_array
            The input matrix in Compressed Sparse Column (CSC) format. Must be
            square and symmetric. Only the lower triangular part of ``A`` is
            used, and no check is made for symmetry. The numerical values of
            ``A`` are ignored. Only its non-zero pattern is used.

            .. note :: The input matrix ``A`` is expected to be the
                permuted matrix :math:`P A P^{\\top}`, where `P` is the
                permutation matrix corresponding to the permutation vector
                returned by :meth:`.get_perm`.

        is_permuted : bool
            If True (default), the input matrix ``A`` is assumed to be
            permuted by the fill-reducing permutation used in the factorization:
            :math:`P A P^{\\top}`. If False, ``A`` is assumed to be in the
            original ordering.

        Returns
        -------
        :class:`.CholeskyFactor`
            The current object, for method chaining.

        See Also
        --------
        cholesky, ldl, update, rowadd, rowdel

        References
        ----------
        .. [#resymbol_c] ``resymbol.c`` - CHOLMOD MATLAB resymbolization function
            https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/resymbol.c

        .. versionadded:: 0.5.0
        """
        self._require_factorized()

        cdef bint A_use_int32
        # TODO support non-square matrices with sym_kind='row'
        A, A_use_int32, _ = validate_csc_input(A, require_square=True, ensure_double=False)

        if A.shape[0] != self.N:
            raise ValueError(
                "Input matrix A must have the same number of rows as L."
            )

        if self._use_int32 and not A_use_int32:
            raise ValueError(
                "A and factor must have the same integer type. "
                f"Got {A.indptr.dtype=}, but {self._use_int32=}."
            )

        # Special Cases
        if self.N == 0:
            return self

        if A.nnz == 0:
            raise CholmodNotPositiveDefiniteError("Input matrix not positive definite.")

        # Clear cached properties
        self._clear_cache()

        # Get sparse *pattern*
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse* Ac = &Amatrix
        cdef int stype = -1  # use tril(A) only

        _cholmod_sparse_from_csc(
            A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
        )
        Ac.xtype = CHOLMOD_PATTERN
        Ac.x = NULL

        # NOTE the "noperm" version expects that A is already permuted by
        # self._factor.Perm (i.e., A = A[p][:, p]). The regular version uses
        # self._factor.Perm to permute A internally.
        if self._use_int32:
            if is_permuted:
                cholmod_resymbol_noperm(Ac, NULL, 0, True, self._factor, self._cm)
            else:
                cholmod_resymbol(Ac, NULL, 0, True, self._factor, self._cm)
        else:
            if is_permuted:
                cholmod_l_resymbol_noperm(Ac, NULL, 0, True, self._factor, self._cm)
            else:
                cholmod_l_resymbol(Ac, NULL, 0, True, self._factor, self._cm)

        return self

    def logdet(self):
        """Compute the (natural) log-determinant of the matrix from its
        Cholesky factorization.

        Returns
        -------
        logdet : float
            The natural logarithm of the determinant of the matrix `A` that was
            factorized.

        See Also
        --------
        slogdet, det, numpy.linalg.slogdet, numpy.linalg.det, scipy.linalg.det

        Notes
        -----
        This function computes the log-determinant of the matrix `A` from its
        Cholesky factorization. If the factorization is in :math:`LL^T` form,
        the determinant is computed as:

        .. math::

            \\log \\det(A) = 2 \\sum_i \\log L_{ii}.

        If the factorization is in :math:`LDL^{\\top}` form, the determinant is
        computed as:

        .. math::

            \\log \\det(A) = \\sum_i \\log D_{ii}.

        .. versionadded:: 0.2
        """
        self._require_factorized()
        if self.is_ll:
            L = self.get_factor()
            return 2 * np.sum(np.log(L.diagonal()))
        else:
            _, D = self.get_factor()
            return np.sum(np.log(D.diagonal()))

    def slogdet(self):
        """Compute the sign and (natural) log-determinant of the matrix from
        its Cholesky factorization.

        Returns
        -------
        sign : int
            The sign of the determinant of the matrix `A` that was
            factorized. This is always 1 for a positive definite matrix.
        logdet : float
            The natural logarithm of the absolute value of the determinant of
            the matrix `A` that was factorized.

        See Also
        --------
        logdet, det, numpy.linalg.slogdet, numpy.linalg.det, scipy.linalg.det

        Notes
        -----
        This function computes the sign and log-determinant of the matrix `A`
        from its Cholesky factorization. If the factorization is in
        :math:`LL^T` form, the determinant is computed as:

        .. math::

            \\log \\det(A) = 2 \\sum_i \\log L_{ii}.

        If the factorization is in :math:`LDL^{\\top}` form, the determinant is
        computed as:

        .. math::

            \\log \\det(A) = \\sum_i \\log D_{ii}.

        .. versionadded:: 0.2
        """
        return (self.dtype.type(1.0), self.logdet())

    def det(self):
        """Compute the determinant of the matrix from its Cholesky
        factorization.

        .. versionadded:: 0.2

        .. warning::

            This function may overflow or underflow for large matrices. Use
            :meth:`.logdet` or :meth:`.slogdet` instead.

        Returns
        -------
        det : float
            The determinant of the matrix `A` that was factorized.

        See Also
        --------
        logdet, slogdet, numpy.linalg.det, numpy.linalg.slogdet, scipy.linalg.det
        """
        return np.exp(self.logdet())

    def inv(self):
        """Compute the inverse of the matrix from its Cholesky factorization.

        .. warning:: For most purposes, it is better to use :meth:`.solve`
            instead of computing the inverse explicitly. The
            following two lines of code are mathematically equivalent::

                x = f.solve(b)
                x = f.inv() @ b  # DO NOT USE

            but the first line is both faster and more numerically stable.

        Returns
        -------
        Ainv : csc_array
            The inverse of the matrix `A` that was factorized.

        See Also
        --------
        numpy.linalg.inv, scipy.linalg.inv

        Notes
        -----
        This function computes the inverse of the matrix `A` from its Cholesky
        factorization. If the factorization is in :math:`LL^T` form, the
        inverse is computed as:

        .. math::

            A^{-1} = P^{\\top} L^{-\\top} L^{-1} P,

        where `P` is the permutation matrix corresponding to the permutation
        vector returned by :meth:`.get_perm`. If the factorization is in
        :math:`LDL^{\\top}` form, the inverse is computed as:

        .. math::

            A^{-1} = P^{\\top} L^{-\\top} D^{-1} L^{-1} P.

        .. versionadded:: 0.2
        """
        return self.solve(eye_array(self.N, format='csc', dtype=self.dtype))


# -----------------------------------------------------------------------------
#         Docstrings for CholeskyFactor
# -----------------------------------------------------------------------------
_DOC_UPDATE_TEMPLATE = """
Multiple-rank {direction} of a sparse LDL factorization.

Compute a {direction} to the factorization of a sparse matrix `A` [#{tag}]_:

.. math::

    L' D' L'^{{\\top}} = P (A {sign} C C^{{\\top}}) P^{{\\top}}

where `L` is a lower triangular matrix with unit diagonal, and `D` is
a diagonal matrix. The input ``C`` is a sparse matrix representing the
{direction} to the factorization. The fill-reducing permutation is *not*
recomputed from the original `A`.

Parameters
----------
C : (N, K) csc_array
    The sparse matrix representing the rank-`k` update or downdate to
    the matrix.

Returns
-------
CholeskyFactor
    The current object, for method chaining.

References
----------
.. [#{tag}] ``cholmod_updown.c`` - CHOLMOD up/downdate function
    https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Modify/cholmod_updown.c

.. versionadded:: 0.5.0
"""

# Assign the docstrings
CholeskyFactor.update.__doc__ = _DOC_UPDATE_TEMPLATE.format(
    direction="update", sign="+", tag="update_c",
)

CholeskyFactor.downdate.__doc__ = _DOC_UPDATE_TEMPLATE.format(
    direction="downdate", sign="-", tag="downdate_c",
)


# -----------------------------------------------------------------------------
#         Convenience functions
# -----------------------------------------------------------------------------
def cho_factor(
    A, beta=0.0, *, lower=False, order="default", sym_kind=None, supernodal_mode=None
):
    return CholeskyFactor(
        A, lower=lower, order=order, sym_kind=sym_kind, supernodal_mode=supernodal_mode
    ).factorize(A, ldl=False, beta=beta)


def ldl_factor(
    A, beta=0.0, *, lower=True, order="default", sym_kind=None, supernodal_mode=None
):
    return CholeskyFactor(
        A, lower=lower, order=order, sym_kind=sym_kind, supernodal_mode=supernodal_mode
    ).factorize(A, ldl=True, beta=beta)


# csc_arrays from the factorization, and optionally the permutation
def cholesky(
    A, beta=0.0, *, lower=False, order="default", sym_kind=None, supernodal_mode=None
):
    f = cho_factor(
        A,
        beta=beta,
        lower=lower,
        order=order,
        sym_kind=sym_kind,
        supernodal_mode=supernodal_mode,
    )
    R = f.get_factor()
    p = f.get_perm()
    return R if order is None else (R, p)


def ldl(A, beta=0.0, *, lower=True, order="default", sym_kind=None, supernodal_mode=None):
    f = ldl_factor(
        A,
        beta=beta,
        lower=lower,
        order=order,
        sym_kind=sym_kind,
        supernodal_mode=supernodal_mode,
    )
    R, D = f.get_factor()
    p = f.get_perm()
    return (R, D) if order is None else (R, D, p)


# -----------------------------------------------------------------------------
#         Docstring Template
# -----------------------------------------------------------------------------
_CHOLMOD_DOC_TEMPLATE = """
{intro}
Parameters
----------
A : (N, N) {{array_like, sparse array}}
    An array convertible to a sparse matrix in Compressed Sparse Column
    (CSC) format. The matrix must be square and symmetric positive definite.
    Only the upper or lower triangular part of the matrix is used, and no check
    is made for symmetry.
beta : float, optional
    The scalar value to add to the diagonal of the matrix before factorization.
lower : bool, optional
    If True, return the lower triangular factor `L`, otherwise return the
    upper triangular factor `R`.
order : None or str in {{"default", "best", "natural", "metis", "nesdis", \
        "amd", "colamd", "postordered"}}, optional
    The permutation algorithm to use for the factorization. Options are:

    * ``default``: Use the default method, which first tries AMD, then METIS.
    * ``best``: Automatically select the best ordering based on the input.
    * ``metis``: Use the METIS library for graph partitioning.
    * ``nesdis``: Use the NESDIS library for nested dissection.
    * ``amd``: Use the Approximate Minimum Degree (AMD) algorithm.
    * ``colamd``: Use the Approximate Minimum Degree (AMD) algorithm for the
        symmetric case, or the COLAMD algorithm for the unsymmetric case
        (:math:`A A^{{\\top}}` or :math:`A^{{\\top}} A`).
    * ``postordered``: Use natural ordering followed by postordering.
    * ``natural`` or ``None``: No permutation is applied (identity permutation).

    By default, methods other than ``natural`` will also be postordered.

    .. warning::

        The ordering method ``best`` may be quite slow for large matrices,
        but if the factorization is reused many times, it can be worth it.

sym_kind : str in {{"sym", "row", "col"}}, optional
    The type of factorization for which to analyze the matrix:

    * ``sym``: Symmetric factorization. No check is made for symmetry.
    * ``row``: Unsymmetric factorization of :math:`A A^{{\\top}}`.
    * ``col``: Unsymmetric factorization of :math:`A^{{\\top}} A`.

supernodal_mode : str in {{"auto", "simplicial", "supernodal"}}, optional
    The type of factorization to use:

    * ``auto``: Automatically select the factorization type.
    * ``simplicial``: Use a simplicial factorization.
    * ``supernodal``: Use a supernodal factorization.

    Note that the ``simplicial`` mode may be slow for large matrices.

Returns
-------
{returns}

Raises
------
:exc:`CholmodNotPositiveDefiniteError`
    If the input matrix is not positive definite.

See Also
--------
{see_also}

Notes
-----
This function is an interface to the CHOLMOD library, which is part of
the SuiteSparse collection by Timothy A. Davis. For more details, see the
documentation in the header file [{doc_tag}]_.


{version_notes}

References
----------
.. [{doc_tag}] ``cholmod.h`` - SuiteSparse CHOLMOD header file.
    https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Include/cholmod.h

Examples
--------
{example}
"""


_CHOLESKY_RETURNS = """R : csc_array
    The triangular factor of the Cholesky decomposition. The data type will
    match that of ``A``.
{ldl_D_output}
p : ndarray of int, optional
    The permutation vector used in the factorization. Only returned if the
    ordering is not ``None``.
"""


# -----------------------------------------------------------------------------
#         Cholesky Docstring
# -----------------------------------------------------------------------------
_cholesky_intro = """Compute the Cholesky factorization of a sparse matrix.

This function computes the Cholesky factorization of a symmetric positive
definite matrix `A`:

.. math::

    R^{\\top} R = P A P^{\\top},

where `R` is an upper triangular matrix. Only the upper triangular part of
`A` is used. If ``lower`` is True, the lower triangular factor `L` is
returned instead, such that:

.. math::

    L L^{\\top} = P A P^{\\top}.

In this case, only the lower triangular part of `A` is used.

If ``beta`` is a scalar value, compute the factorization of:

.. math::

    P A P^{\\top} + \\beta I,

where `I` is the identity matrix.
"""

_cho_factor_returns = """CholeskyFactor
    The factorization object. Use its methods to solve linear systems
    and manipulate the factorization.
"""

_cholesky_example = """
>>> import numpy as np
>>> from scipy.sparse import coo_array
>>> from sksparse.cholmod import cholesky, cho_factor
>>> # Create a symmetric positive definite matrix from (Davis, Eqn 2.1)
>>> N = 11
>>> rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
>>> cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
>>> rng = np.random.default_rng(56)
>>> vals = rng.random(len(rows), dtype=np.float64)
>>> L = coo_array((vals, (rows, cols)), shape=(N, N))
>>> A = L + L.T   # make it symmetric
>>> A.setdiag(N)  # make it strongly positive definite
>>> A = A.tocsc()
>>> L, p = cholesky(A, order='amd', lower=True)
>>> L
<Compressed Sparse Column sparse array of dtype 'float64'
        with 30 stored elements and shape (11, 11)>
>>> p
array([ 4,  8,  6,  0,  3,  5,  1,  2,  9, 10,  7])
>>> f = cho_factor(A, order='amd', lower=True)
>>> f
CholeskyFactor(N=11, nnz=30, is_ll=True, is_super=False, itype=np.int64,
    dtype=np.float64, order=natural)
>>> np.allclose(L.toarray(), f.get_factor().toarray(), atol=1e-15)
True
>>> np.array_equal(p, f.get_perm())
True
>>> # Solve a linear system
>>> expect_x = np.arange(N, dtype=np.float64)
>>> b = A @ expect_x
>>> x = f.solve(b)
>>> np.allclose(x, expect_x)
True
"""


cho_factor.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_cholesky_intro,
    returns=_cho_factor_returns,
    see_also="cholesky, ldl, ldl_factor",
    version_notes=".. versionadded:: 0.5.0",
    doc_tag="#cho_factor_h",
    example=_cholesky_example,
)


_cholesky_version_notes=""".. versionadded:: 0.1.0
.. versionchanged:: 0.5.0
    The function now returns the matrix directly instead of a ``Factor``
    object, and the permutation vector when an ordering method is specified.
"""

cholesky.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_cholesky_intro,
    returns=_CHOLESKY_RETURNS.format(ldl_D_output=""),
    see_also="cho_factor, ldl, ldl_factor",
    version_notes=_cholesky_version_notes,
    doc_tag="#cholesky_h",
    example=_cholesky_example,
)

# -----------------------------------------------------------------------------
#         LDL Docstring
# -----------------------------------------------------------------------------
_ldl_intro = """
Compute the LDL factorization of a sparse matrix.

This function computes the LDL factorization of a symmetric matrix `A`:

.. math::

    L D L^{\\top} = P A P^{\\top},

where `L` is a lower triangular matrix with unit diagonal, and `D` is
a diagonal matrix. Only the lower triangular part of `A` is used. If
``lower`` is False, the upper triangular factor `R` is returned instead,
such that:

.. math::

    R^{\\top} D R = P A P^{\\top}.

In this case, only the upper triangular part of `A` is used.

If ``beta`` is a scalar value, compute the factorization of:

.. math::

    P A P^{\\top} + \\beta I,

where `I` is the identity matrix.
"""

_ldl_D_output = """D : dia_array
    The diagonal matrix `D` of the factorization, in sparse DIA format.
    The data type will match that of ``A``."""


_ldl_example = """
>>> import numpy as np
>>> from scipy.sparse import coo_array
>>> from sksparse.cholmod import ldl, ldl_factor
>>> # Create a symmetric positive definite matrix from (Davis, Eqn 2.1)
>>> N = 11
>>> rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
>>> cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
>>> rng = np.random.default_rng(56)
>>> vals = rng.random(len(rows), dtype=np.float64)
>>> L = coo_array((vals, (rows, cols)), shape=(N, N))
>>> A = L + L.T   # make it symmetric
>>> A.setdiag(N)  # make it strongly positive definite
>>> A = A.tocsc()
>>> L, D, p = ldl(A, order='amd')
>>> L
<Compressed Sparse Column sparse array of dtype 'float64'
        with 30 stored elements and shape (11, 11)>
>>> D
<DIAgonal sparse array of dtype 'float64'
        with 11 stored elements (1 diagonals) and shape (11, 11)>
>>> p
array([ 4,  8,  6,  0,  3,  5,  1,  2,  9, 10,  7])
>>> f = ldl_factor(A, order='amd')
>>> f
CholeskyFactor(N=11, nnz=30, is_ll=False, is_super=False, itype=np.int64,
    dtype=np.float64, order=amd)
>>> Lf, Df = f.get_factor()
>>> np.allclose(L.toarray(), Lf.toarray(), atol=1e-15)
True
>>> np.allclose(D.toarray(), Df.toarray(), atol=1e-15)
True
>>> np.array_equal(p, f.get_perm())
True
>>> # Solve a linear system
>>> expect_x = np.arange(N, dtype=np.float64)
>>> b = A @ expect_x
>>> x = f.solve(b)
>>> np.allclose(x, expect_x)
True
"""


ldl_factor.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_ldl_intro,
    returns=_cho_factor_returns,
    see_also="ldl, cholesky, cho_factor",
    version_notes=".. versionadded:: 0.5.0",
    doc_tag="#ldl_factor_h",
    example=_ldl_example,
)


ldl.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_ldl_intro,
    returns=_CHOLESKY_RETURNS.format(ldl_D_output=_ldl_D_output),
    see_also="ldl_factor, cholesky, cho_factor",
    version_notes=".. versionadded:: 0.5.0",
    doc_tag="#ldl_h",
    example=_ldl_example,
)


# -----------------------------------------------------------------------------
#         Symbolic Functions
# -----------------------------------------------------------------------------
def symbfact(A, *, kind=None, bint lower=False, bint return_factor=False):
    """Symbolic factorization of a sparse matrix for Cholesky or LDL.

    This function performs the symbolic factorization of a sparse matrix ``A``
    for either Cholesky or LDL factorization. It computes the elimination
    tree and analyzes the sparsity pattern of the matrix [#symbfact_c]_.

    Parameters
    ----------
    A : (N, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square and symmetric. No check is made for symmetry, so the upper (or
        lower) triangular part of the matrix is used for the factorization, depending
        on the ``lower`` parameter.
    kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization. Only the upper triangular part of
          ``A`` is used, and no check is made for symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.
        * ``lo``: Lower triangular factorization. Same as ``symbfact(A.T)``.
          Only the lower triangular part of ``A`` is used, and no check is made
          for symmetry.

        If ``kind`` is None, it defaults to ``sym``.
    lower : bool, optional
        If True, the symbolic factorization is performed on the lower
        triangular part of the matrix. If False, the upper triangular part is
        used. Default is False (upper triangular).
    return_factor : bool, optional
        If True, the symbolic factorization returns the structure of the
        Cholesky factor `L` (or `LD` for LDL factorization) as a sparse matrix.
        Default is False.

    Returns
    -------
    count : (N,) ndarray of int
        The count of nonzeros in each column of the Cholesky factor.
    h : int
        The height of the elimination tree.
    parent : (N,) ndarray of int
        The parent of each node in the elimination tree. The root has no parent
        (parent[0] = -1).
    post : (N,) ndarray of int
        The postorder of the elimination tree. The first node in the postorder
        is the root of the tree.
    L : (N, N) csc_array
        The symbolic factorization of the matrix. Only returned if
        ``return_factor`` is True.

    See Also
    --------
    etree


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#symbfact_c] ``symbfact2.c`` - CHOLMOD MATLAB symbolic factorization function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/symbfact2.c

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> from sksparse.cholmod import cholesky, symbfact
    >>> # Create a symmetric positive definite matrix from (Davis, Eqn 2.1)
    >>> N = 11
    >>> rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    >>> cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    >>> rng = np.random.default_rng(56)
    >>> vals = rng.random(len(rows), dtype=np.float64)
    >>> L = coo_array((vals, (rows, cols)), shape=(N, N))
    >>> A = L + L.T   # make it symmetric
    >>> A.setdiag(N)  # make it strongly positive definite
    >>> A = A.tocsc()
    >>> L = cholesky(A, lower=True)
    >>> count, h, parent, post = symbfact(A)
    >>> count
    array([3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1])
    >>> np.array_equal(count, np.count_nonzero(L.toarray(), axis=0))
    True
    >>> h
    6
    >>> parent
    array([ 5,  2,  7,  5,  7,  6,  8,  9,  9, 10, -1])
    >>> post
    array([ 1,  2,  4,  7,  0,  3,  5,  6,  8,  9, 10])
    """
    cdef bint use_int32
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col", "lo"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t N = A.shape[1]

    if kind not in ["row", "col"] and M != N:
        raise ValueError(f"Input matrix A must be square, got shape {A.shape}.")

    # Special Cases
    # sym: A = (0, 0)
    # row: AA.T = (0, N) * (N, 0) = (0, 0)
    # col: A.TA = (0, M) * (M, 0) = (0, 0)
    if kind == "row" and M == 0 or N == 0:
        empty = np.array([], dtype=out_itype)
        count, h, parent, post, L = empty, 0, empty, empty, A.copy()
        if return_factor:
            return count, h, parent, post, L
        else:
            return count, h, parent, post

    if A.nnz == 0:
        D = N if kind == "col" else M
        count = np.zeros(D, dtype=out_itype)
        h = 1
        parent = np.full(D, -1, dtype=out_itype)
        post = np.arange(D, dtype=out_itype)
        L = eye_array(D, dtype=A.dtype)
        if return_factor:
            return count, h, parent, post, L
        else:
            return count, h, parent, post

    # -------------------------------------------------------------------------
    #         Start the Analysis
    # -------------------------------------------------------------------------
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix

    N = A.shape[0]
    cdef int stype = 1  # default kind="sym" uses triu(A) only
    cdef bint col_etree = False

    if kind == "row":
        stype = 0  # use A * A.T
    elif kind == "col":
        N = A.shape[1]
        stype = 0  # use A.T * A
        col_etree = True
    elif kind == "lo":
        stype = -1  # use tril(A) only

    # Get sparse *pattern*
    _cholmod_sparse_from_csc(
        A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
    )
    Ac.xtype = CHOLMOD_PATTERN
    Ac.x = NULL

    # -------------------------------------------------------------------------
    #         Compute the Outputs
    # -------------------------------------------------------------------------
    cdef void *Parent
    cdef void *Post
    cdef void *ColCount
    cdef void *First
    cdef void *Level

    if use_int32:
        Parent = cholmod_malloc(N, sizeof(int32_t), cm)
        Post = cholmod_malloc(N, sizeof(int32_t), cm)
        ColCount = cholmod_malloc(N, sizeof(int32_t), cm)
        First = cholmod_malloc(N, sizeof(int32_t), cm)
        Level = cholmod_malloc(N, sizeof(int32_t), cm)
    else:
        Parent = cholmod_l_malloc(N, sizeof(int64_t), cm)
        Post = cholmod_l_malloc(N, sizeof(int64_t), cm)
        ColCount = cholmod_l_malloc(N, sizeof(int64_t), cm)
        First = cholmod_l_malloc(N, sizeof(int64_t), cm)
        Level = cholmod_l_malloc(N, sizeof(int64_t), cm)

    cdef cholmod_sparse *Fc
    cdef cholmod_sparse *Aup
    cdef cholmod_sparse *Alo

    if use_int32:
        Fc = cholmod_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
    else:
        Fc = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)

    if Ac.stype == 1 or col_etree:
        Aup = Ac
        Alo = Fc
    else:
        Aup = Fc
        Alo = Ac

    if use_int32:
        cholmod_etree(Aup, <int32_t*>Parent, cm)
    else:
        cholmod_l_etree(Aup, <int64_t*>Parent, cm)

    _handle_errors(cm.status)

    if use_int32:
        if cholmod_postorder(<int32_t*>Parent, N, NULL, <int32_t*>Post, cm) != N:
            raise CholmodError("Postordering failed.")
    else:
        if cholmod_l_postorder(<int64_t*>Parent, N, NULL, <int64_t*>Post, cm) != N:
            raise CholmodError("Postordering failed.")

    if use_int32:
        cholmod_rowcolcounts(
            Alo,
            NULL,
            0,
            <int32_t*>Parent,
            <int32_t*>Post,
            NULL,
            <int32_t*>ColCount,
            <int32_t*>First,
            <int32_t*>Level,
            cm
        )
    else:
        cholmod_l_rowcolcounts(
            Alo,
            NULL,
            0,
            <int64_t*>Parent,
            <int64_t*>Post,
            NULL,
            <int64_t*>ColCount,
            <int64_t*>First,
            <int64_t*>Level,
            cm
        )

    _handle_errors(cm.status)

    # Return the results
    count = _ndarray_copy_from_intptr(ColCount, N, use_int32)

    # Compute height of the elimination tree
    cdef int32_t h_int32 = 0
    cdef int64_t h_int64 = 0
    cdef size_t i

    if use_int32:
        for i in range(N):
            h_int32 = max(h_int32, (<int32_t*>Level)[i])
        h = h_int32 + 1
    else:
        for i in range(N):
            h_int64 = max(h_int64, (<int64_t*>Level)[i])
        h = h_int64 + 1

    parent = _ndarray_copy_from_intptr(Parent, N, use_int32)
    post = _ndarray_copy_from_intptr(Post, N, use_int32)

    # Construct symbolic L if requested
    cdef cholmod_sparse *Ls
    cdef cholmod_sparse *Rs

    if return_factor:
        if use_int32:
            Ls = _cholesky_pattern(
                Ac, Fc, N, <int32_t*>Parent, <int32_t*>ColCount, col_etree, cm
            )
            if not lower:
                Rs = cholmod_transpose(Ls, CHOLMOD_TRANS_PATTERN, cm)
                cholmod_free_sparse(&Ls, cm)
                Ls = Rs
        else:
            Ls = _cholesky_l_pattern(
                Ac, Fc, N, <int64_t*>Parent, <int64_t*>ColCount, col_etree, cm
            )
            if not lower:
                Rs = cholmod_l_transpose(Ls, CHOLMOD_TRANS_PATTERN, cm)
                cholmod_l_free_sparse(&Ls, cm)
                Ls = Rs

        # Convert the symbolic L to a CSC array
        L = _csc_from_cholmod_sparse(Ls, cm)

        # Fill the L matrix data with boolean ones (for python)
        L.data = np.ones(L.nnz, dtype=np.bool_)

    # Free memory (arrays are copied to numpy)
    if use_int32:
        cholmod_free(N, sizeof(int32_t), Parent, cm)
        cholmod_free(N, sizeof(int32_t), Post, cm)
        cholmod_free(N, sizeof(int32_t), ColCount, cm)
        cholmod_free(N, sizeof(int32_t), First, cm)
        cholmod_free(N, sizeof(int32_t), Level, cm)
        cholmod_free_sparse(&Fc, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free(N, sizeof(int64_t), Parent, cm)
        cholmod_l_free(N, sizeof(int64_t), Post, cm)
        cholmod_l_free(N, sizeof(int64_t), ColCount, cm)
        cholmod_l_free(N, sizeof(int64_t), First, cm)
        cholmod_l_free(N, sizeof(int64_t), Level, cm)
        cholmod_l_free_sparse(&Fc, cm)
        cholmod_l_finish(cm)

    if return_factor:
        return count, h, parent, post, L
    else:
        return count, h, parent, post


def etree(A, *, kind=None, bint return_post=False):
    """Symbolic factorization of a sparse matrix for Cholesky or LDL.

    This function determines the elimination tree of a sparse matrix ``A``, and
    optionally postorders the tree [#etree_c]_.

    Parameters
    ----------
    A : (N, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square and symmetric. No check is made for symmetry, so the upper (or
        lower) triangular part of the matrix is used for the factorization, depending
        on the ``lower`` parameter.
    kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization. Only the upper triangular part of
          ``A`` is used, and no check is made for symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.
        * ``lo``: Lower triangular factorization. Same as ``symbfact(A.T)``.
          Only the lower triangular part of ``A`` is used, and no check is made
          for symmetry.

        If ``kind`` is None, it defaults to ``sym``.
    return_post : bool, optional
        If True, the function returns the postorder of the elimination tree.
        Default is False.

    Returns
    -------
    parent : (N,) ndarray of int
        The parent of each node in the elimination tree. The root has no parent
        (parent[0] = -1).
    post : (N,) ndarray of int, optional
        The postorder of the elimination tree. The first node in the postorder
        is the root of the tree.

    .. versionadded:: 0.5.0

    See Also
    --------
    symbfact

    References
    ----------
    .. [#etree_c] ``etree2.c`` - CHOLMOD MATLAB symbolic factorization function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/etree2.c

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> from sksparse.cholmod import etree
    >>> # Create a symmetric positive definite matrix from (Davis, Eqn 2.1)
    >>> N = 11
    >>> rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    >>> cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    >>> rng = np.random.default_rng(56)
    >>> vals = rng.random(len(rows), dtype=np.float64)
    >>> L = coo_array((vals, (rows, cols)), shape=(N, N))
    >>> A = L + L.T   # make it symmetric
    >>> A.setdiag(N)  # make it strongly positive definite
    >>> A = A.tocsc()
    >>> parent, post = etree(A, return_post=True)
    >>> parent
    array([ 5,  2,  7,  5,  7,  6,  8,  9,  9, 10, -1])
    >>> post
    array([ 1,  2,  4,  7,  0,  3,  5,  6,  8,  9, 10])
    """
    cdef bint use_int32
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col", "lo"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t N = A.shape[1]

    if kind not in ["row", "col"] and M != N:
        raise ValueError(f"Input matrix A must be square, got shape {A.shape}.")

    # Special Cases
    # sym: A = (0, 0)
    # row: AA.T = (0, N) * (N, 0) = (0, 0)
    # col: A.TA = (0, M) * (M, 0) = (0, 0)
    if kind == "row" and M == 0 or N == 0:
        parent = np.array([], dtype=out_itype)
        if return_post:
            return parent, parent.copy()
        else:
            return parent

    if A.nnz == 0:
        D = N if kind == "col" else M
        parent = np.full(D, -1, dtype=out_itype)
        if return_post:
            post = np.arange(D, dtype=out_itype)
            return parent, post
        else:
            return parent

    # -------------------------------------------------------------------------
    #         Start the Analysis
    # -------------------------------------------------------------------------
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix

    cdef int stype = 1  # default kind="sym" uses triu(A) only
    N = A.shape[0]
    cdef bint col_etree = False

    if kind == "row":
        stype = 0  # use A * A.T
    elif kind == "col":
        N = A.shape[1]
        stype = 0  # use A.T * A
        col_etree = True
    elif kind == "lo":
        stype = -1  # use tril(A) only

    # Get sparse *pattern*
    _cholmod_sparse_from_csc(
        A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
    )
    Ac.xtype = CHOLMOD_PATTERN
    Ac.x = NULL

    # -------------------------------------------------------------------------
    #         Compute the Outputs
    # -------------------------------------------------------------------------
    cdef void *Parent
    cdef void *Post

    if use_int32:
        Parent = cholmod_malloc(N, sizeof(int32_t), cm)
    else:
        Parent = cholmod_l_malloc(N, sizeof(int64_t), cm)

    cdef cholmod_sparse *Rc

    if Ac.stype == 1 or col_etree:
        # symmetric case: etree(A), using triu(A)
        # column case: column etree of A, which is etree(A.T @ A)
        if use_int32:
            cholmod_etree(Ac, <int32_t*>Parent, cm)
        else:
            cholmod_l_etree(Ac, <int64_t*>Parent, cm)
    else:
        # symmetric case: etree(A), using tril(A)
        # row case: row etree of A, which is etree(A @ A.T)
        # R = A.T
        if use_int32:
            Rc = cholmod_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            cholmod_etree(Rc, <int32_t*>Parent, cm)
            cholmod_free_sparse(&Rc, cm)
        else:
            Rc = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            cholmod_l_etree(Rc, <int64_t*>Parent, cm)
            cholmod_l_free_sparse(&Rc, cm)

    _handle_errors(cm.status)

    # Get the ndarray to return
    parent = _ndarray_copy_from_intptr(Parent, N, use_int32)

    if return_post:
        if use_int32:
            Post = cholmod_malloc(N, sizeof(int32_t), cm)
            if cholmod_postorder(<int32_t*>Parent, N, NULL, <int32_t*>Post, cm) != N:
                raise CholmodError("Postordering failed.")
        else:
            Post = cholmod_l_malloc(N, sizeof(int64_t), cm)
            if cholmod_l_postorder(<int64_t*>Parent, N, NULL, <int64_t*>Post, cm) != N:
                raise CholmodError("Postordering failed.")

        post = _ndarray_copy_from_intptr(Post, N, use_int32)

    # Free memory (arrays are copied to numpy)
    if use_int32:
        cholmod_free(N, sizeof(int32_t), Parent, cm)
        if return_post:
            cholmod_free(N, sizeof(int32_t), Post, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free(N, sizeof(int64_t), Parent, cm)
        if return_post:
            cholmod_l_free(N, sizeof(int64_t), Post, cm)
        cholmod_l_finish(cm)

    if return_post:
        return parent, post
    else:
        return parent


# -----------------------------------------------------------------------------
#         Partition Functions
# -----------------------------------------------------------------------------
def bisect(A, *, kind=None):
    """Compute a node separator for a sparse matrix graph.

    Parameters
    ----------
    A : (M, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square and symmetric if ``kind`` is None or ``"sym"``. No check is made
        for symmetry.
    kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization. Only the upper triangular part of
          ``A`` is used, and no check is made for symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.

        If ``kind`` is None, it defaults to ``sym``.

    Returns
    -------
    s : (K,) ndarray of int
        The dimension ``K`` is either ``M`` or ``N``, depending on the
        ``kind`` parameter. The output can take 3 values:

        * ``0``: The node is in the left subgraph.
        * ``1``: The node is in the right subgraph.
        * ``2``: The node is in the separator.

    See Also
    --------
    nesdis, metis

    Notes
    -----
    This function is based on the SuiteSparse CHOLMOD MATLAB interface
    [#bisect_c]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#bisect_c] ``bisect.c`` - CHOLMOD MATLAB bisect function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/bisect.c

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> from sksparse.cholmod import bisect
    >>> # Create a symmetric positive definite matrix from (Davis, Eqn 2.1)
    >>> N = 11
    >>> rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    >>> cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    >>> rng = np.random.default_rng(56)
    >>> vals = rng.random(len(rows), dtype=np.float64)
    >>> L = coo_array((vals, (rows, cols)), shape=(N, N))
    >>> A = L + L.T   # make it symmetric
    >>> A.setdiag(N)  # make it strongly positive definite
    >>> A = A.tocsc()
    >>> s = bisect(A)
    >>> s
    array([0, 1, 1, 0, 1, 0, 0, 1, 0, 2, 2])
    """
    cdef bint use_int32
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t N = A.shape[1]

    if kind not in ["row", "col"] and M != N:
        raise ValueError(f"Input matrix A must be square, got shape {A.shape}.")

    # Special Cases
    # sym: A = (0, 0)
    # row: AA.T = (0, N) * (N, 0) = (0, 0)
    # col: A.TA = (0, M) * (M, 0) = (0, 0)
    if kind == "row" and M == 0 or N == 0:
        return np.array([], dtype=out_itype)

    if A.nnz == 0:
        D = N if kind == "col" else M
        s = np.empty(D, dtype=out_itype)
        k = D // 2
        s[:k] = 0  # left subgraph
        s[k:] = 1  # right subgraph
        s[-1] = 2  # separator
        return s

    # -------------------------------------------------------------------------
    #         Start the Analysis
    # -------------------------------------------------------------------------
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix

    cdef int stype = -1  # default kind="sym" uses tril(A) only
    cdef bint transpose = False

    if kind == "row":
        stype = 0  # use A * A.T
    elif kind == "col":
        stype = 0  # use A.T * A
        transpose = True
    elif kind == "lo":
        stype = -1  # use tril(A) only

    # Get sparse *pattern*
    _cholmod_sparse_from_csc(
        A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
    )
    Ac.xtype = CHOLMOD_PATTERN
    Ac.x = NULL

    # -------------------------------------------------------------------------
    #         Compute the Outputs
    # -------------------------------------------------------------------------
    cdef void *Partition
    cdef cholmod_sparse *C
    cdef int64_t ok

    if transpose:
        # C = A.T, then bisect C @ C.T
        if use_int32:
            C = cholmod_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            N = C.nrow
            Partition = cholmod_malloc(N, sizeof(int32_t), cm)
            ok = (cholmod_bisect(C, NULL, 0, True, <int32_t*>Partition, cm) >= 0)
            cholmod_free_sparse(&C, cm)
        else:
            C = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            N = C.nrow
            Partition = cholmod_l_malloc(N, sizeof(int64_t), cm)
            ok = (cholmod_l_bisect(C, NULL, 0, True, <int64_t*>Partition, cm) >= 0)
            cholmod_l_free_sparse(&C, cm)
    else:
        N = Ac.nrow
        if use_int32:
            Partition = cholmod_malloc(N, sizeof(int32_t), cm)
            ok = (cholmod_bisect(Ac, NULL, 0, True, <int32_t*>Partition, cm) >= 0)
        else:
            Partition = cholmod_l_malloc(N, sizeof(int64_t), cm)
            ok = (cholmod_l_bisect(Ac, NULL, 0, True, <int64_t*>Partition, cm) >= 0)

    if not ok:
        raise CholmodError("Bisecting failed.")

    # Get the ndarray to return
    s = _ndarray_copy_from_intptr(Partition, N, use_int32)

    # Free memory (arrays are copied to numpy)
    if use_int32:
        cholmod_free(N, sizeof(int32_t), Partition, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free(N, sizeof(int64_t), Partition, cm)
        cholmod_l_finish(cm)

    return s


class SeparatorTree():
    """The separator tree of a sparse matrix graph.

    This object is typically created by :func:`.nesdis`.

    Attributes
    ----------
    cp : *(C,)* numpy.ndarray of int, optional
        The separator tree, where ``C`` is the number of components found. The
        value ``cp[c]`` is the parent of the component ``c`` in the separator
        tree, or ``-1`` if ``c`` is the root of the tree. There is a maximum of
        ``N`` components, where ``N`` is the dimension of the input matrix.
    cmember : *(N,)* numpy.ndarray of int, optional
        The component membership vector, where ``cmember[i]`` is the component
        to which node ``i`` belongs.


    .. versionadded:: 0.5.0
    """
    def __init__(self, cp, cmember):
        self._cp = np.ascontiguousarray(cp)
        self._cmember = np.ascontiguousarray(cmember)

    @property
    def cp(self):
        return self._cp

    @property
    def cmember(self):
        return self._cmember

    def __repr__(self):
        return f"SeparatorTree(components={len(self._cp)}, nodes={len(self._cmember)})"

    def prune(self, *, nd_oksep=None, nd_small=None):
        """Prune the separator tree.

        Parameters
        ----------------
        nd_oksep : double in [0, 1], optional
            Controls when a separator is kept. A separator is kept if
            ``nsep < nd_oksep * n``, where ``nsep`` is the number of nodes in the
            separator and ``n`` is the number of nodes in the graph being cut
            (default is 1.0).
        nd_small : int >= 0, optional
            The smallest subgraph that should not be partitioned (default is 200).

        Returns
        -------
        pruned_septree : SeparatorTree
            The pruned separator tree. ``cp`` will be of length ``C'``, where
            ``C' <= C`` is the number of components remaining after pruning.

        Notes
        -----
        This function is based on the SuiteSparse CHOLMOD MATLAB interface
        [#septree_c]_.

        References
        ----------
        .. [#septree_c] ``septree.c`` - CHOLMOD MATLAB septree function
            https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/septree.c
        """
        if nd_oksep is None:
            nd_oksep = 1.0  # see CHOLMOD/MATLAB/nesdis.c

        if nd_small is None:
            nd_small = 200  # see CHOLMOD/MATLAB/nesdis.c

        return self._prune(self._cp, self._cmember, nd_oksep, nd_small)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _prune(
        self,
        index_t[::1] cp not None,
        index_t[::1] cmember not None,
        double nd_oksep,
        Py_ssize_t nd_small
    ):
        cdef bint use_int32 = index_t is int32_t

        cdef cholmod_common Common
        cdef cholmod_common *cm = &Common

        if use_int32:
            cholmod_start(cm)
        else:
            cholmod_l_start(cm)

        cdef Py_ssize_t Nc = cp.shape[0]
        cdef Py_ssize_t N = cmember.shape[0]

        # Copy input arrays into new cholmod arrays (modified for output)
        cdef index_t *CParent
        cdef index_t *CMember

        ch_malloc = cholmod_malloc if use_int32 else cholmod_l_malloc
        CParent = <index_t*>ch_malloc(Nc, sizeof(index_t), cm)
        CMember = <index_t*>ch_malloc(N, sizeof(index_t), cm)
        memcpy(CParent, &cp[0], Nc * sizeof(index_t))
        memcpy(CMember, &cmember[0], N * sizeof(index_t))

        cdef int64_t nc_new

        if use_int32:
            nc_new = cholmod_collapse_septree(
                N, Nc, nd_oksep, nd_small, <int32_t*>CParent, <int32_t*>CMember, cm
            )
        else:
            nc_new = cholmod_l_collapse_septree(
                N, Nc, nd_oksep, nd_small, <int64_t*>CParent, <int64_t*>CMember, cm
            )

        if nc_new < 0:
            raise CholmodError("Pruning the separator tree failed.")

        # Get the ndarrays to return
        cp_out = _ndarray_copy_from_intptr(CParent, nc_new, use_int32)
        cmember_out = _ndarray_copy_from_intptr(CMember, N, use_int32)

        # Free memory (arrays are copied to numpy)
        if use_int32:
            cholmod_free(Nc, sizeof(int32_t), CParent, cm)
            cholmod_free(N, sizeof(int32_t), CMember, cm)
            cholmod_finish(cm)
        else:
            cholmod_l_free(Nc, sizeof(int64_t), CParent, cm)
            cholmod_l_free(N, sizeof(int64_t), CMember, cm)
            cholmod_l_finish(cm)

        return SeparatorTree(cp_out, cmember_out)


def nesdis(
    A,
    *,
    kind=None,
    bint return_separator=False,
    nd_small=None,
    nd_components=None,
    nd_oksep=None,
    nd_camd=None,
):
    """Nested dissection ordering of a sparse matrix.

    Parameters
    ----------
    A : (M, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square and symmetric if ``kind`` is None or ``"sym"``. No check is made
        for symmetry.
    kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization. Only the upper triangular part of
          ``A`` is used, and no check is made for symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.

        If ``kind`` is None, it defaults to ``sym``.
    return_separator : bool, optional
        If True, the function returns the separator tree and component
        membership vector. Default is False.

    Returns
    -------
    p : (M or N,) ndarray of int
        The permutation vector that gives the nested dissection ordering of the
        nodes in the graph represented by the sparse matrix ``A``.
    septree : SeparatorTree, optional
        The separator tree and component membership vector, returned if
        ``return_separator`` is True.

    Other Parameters
    ----------------
    nd_small : int, optional
        The smallest subgraph that should not be partitioned (default is 200).
    nd_components : bool, optional
        True if connected components should be split independently (default is
        False).
    nd_oksep : double, optional
        Controls when a separator is kept. A separator is kept if
        ``nsep < nd_oksep * n``, where ``nsep`` is the number of nodes in the
        separator and ``n`` is the number of nodes in the graph being cut
        (default is 1).
    nd_camd : int, optional
        Controls whether the smallest subgraphs should be ordered. If 0, they
        are not ordered. For the "sym" case, 1 to order by ``camd``, 2 to order
        by ``csymamd`` (default 1). For other cases: 0 to order naturally, or
        1 to order by ``colamd``.

    See Also
    --------
    bisect, metis

    Notes
    -----
    This function is based on the SuiteSparse CHOLMOD MATLAB interface
    [#nesdis_c]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#nesdis_c] ``nesdis.c`` - CHOLMOD MATLAB nesdis function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/nesdis.c

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> from sksparse.cholmod import nesdis
    >>> # Create a symmetric positive definite matrix from (Davis, Eqn 2.1)
    >>> N = 11
    >>> rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    >>> cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    >>> rng = np.random.default_rng(56)
    >>> vals = rng.random(len(rows), dtype=np.float64)
    >>> L = coo_array((vals, (rows, cols)), shape=(N, N))
    >>> A = L + L.T   # make it symmetric
    >>> A.setdiag(N)  # make it strongly positive definite
    >>> A = A.tocsc()
    >>> p, s = nesdis(A, return_separator=True)
    >>> p
    array([ 1,  4,  6,  8,  0,  3,  5,  2,  9, 10,  7])
    >>> s
    SeparatorTree(components=1, nodes=11)
    """
    cdef bint use_int32
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t N = A.shape[1]

    if kind not in ["row", "col"] and M != N:
        raise ValueError(f"Input matrix A must be square, got shape {A.shape}.")

    # Special Cases
    # sym: A = (0, 0)
    # row: AA.T = (0, N) * (N, 0) = (0, 0)
    # col: A.TA = (0, M) * (M, 0) = (0, 0)
    if kind == "row" and M == 0 or N == 0:
        p = np.array([], dtype=out_itype)
        if return_separator:
            cp = np.array([-1], dtype=out_itype)  # only one component
            cmember = np.array([], dtype=out_itype)
            return p, SeparatorTree(cp, cmember)
        else:
            return p

    if A.nnz == 0:
        D = N if kind == "col" else M
        p = np.arange(D, dtype=out_itype)
        if return_separator:
            cp = np.array([-1], dtype=out_itype)  # only one component
            cmember = np.zeros(D, dtype=out_itype)
            return p, SeparatorTree(cp, cmember)
        else:
            return p

    # -------------------------------------------------------------------------
    #         Start the Analysis
    # -------------------------------------------------------------------------
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    # Set the options for nested dissection
    if nd_small is not None:
        cm.method[0].nd_small = nd_small

    if nd_components is not None:
        cm.method[0].nd_components = nd_components

    if nd_oksep is not None:
        cm.method[0].nd_oksep = nd_oksep

    if nd_camd is not None:
        cm.method[0].nd_camd = nd_camd

    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix

    cdef int stype = -1  # default kind="sym" uses tril(A) only
    cdef bint transpose = False

    if kind == "row":
        stype = 0  # use A * A.T
    elif kind == "col":
        stype = 0  # use A.T * A
        transpose = True
    elif kind == "lo":
        stype = -1  # use tril(A) only

    # Get sparse *pattern*
    _cholmod_sparse_from_csc(
        A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
    )
    Ac.xtype = CHOLMOD_PATTERN
    Ac.x = NULL

    # -------------------------------------------------------------------------
    #         Compute the Outputs
    # -------------------------------------------------------------------------
    cdef void *Perm
    cdef void *CParent
    cdef void *CMember
    cdef cholmod_sparse *C
    cdef int64_t ncomp

    if transpose:
        # C = A.T, then order C @ C.T
        if use_int32:
            C = cholmod_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            N = C.nrow
            Perm = cholmod_malloc(N, sizeof(int32_t), cm)
            CParent = cholmod_malloc(N, sizeof(int32_t), cm)
            CMember = cholmod_malloc(N, sizeof(int32_t), cm)
            ncomp = cholmod_nested_dissection(
                C, NULL, 0, <int32_t*>Perm, <int32_t*>CParent, <int32_t*>CMember, cm
            )
            cholmod_free_sparse(&C, cm)
        else:
            C = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            N = C.nrow
            Perm = cholmod_l_malloc(N, sizeof(int64_t), cm)
            CParent = cholmod_l_malloc(N, sizeof(int64_t), cm)
            CMember = cholmod_l_malloc(N, sizeof(int64_t), cm)
            ncomp = cholmod_l_nested_dissection(
                C, NULL, 0, <int64_t*>Perm, <int64_t*>CParent, <int64_t*>CMember, cm
            )
            cholmod_l_free_sparse(&C, cm)
    else:
        N = Ac.nrow
        if use_int32:
            Perm = cholmod_malloc(N, sizeof(int32_t), cm)
            CParent = cholmod_malloc(N, sizeof(int32_t), cm)
            CMember = cholmod_malloc(N, sizeof(int32_t), cm)
            ncomp = cholmod_nested_dissection(
                Ac, NULL, 0, <int32_t*>Perm, <int32_t*>CParent, <int32_t*>CMember, cm
            )
        else:
            Perm = cholmod_l_malloc(N, sizeof(int64_t), cm)
            CParent = cholmod_l_malloc(N, sizeof(int64_t), cm)
            CMember = cholmod_l_malloc(N, sizeof(int64_t), cm)
            ncomp = cholmod_l_nested_dissection(
                Ac, NULL, 0, <int64_t*>Perm, <int64_t*>CParent, <int64_t*>CMember, cm
            )

    if ncomp < 0:
        raise CholmodError("Nested dissection failed.")

    # Get the ndarrays to return
    p = _ndarray_copy_from_intptr(Perm, N, use_int32)
    cp = _ndarray_copy_from_intptr(CParent, ncomp, use_int32)
    cmember = _ndarray_copy_from_intptr(CMember, N, use_int32)

    # Free memory (arrays are copied to numpy)
    if use_int32:
        cholmod_free(N, sizeof(int32_t), Perm, cm)
        cholmod_free(N, sizeof(int32_t), CParent, cm)
        cholmod_free(N, sizeof(int32_t), CMember, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free(N, sizeof(int64_t), Perm, cm)
        cholmod_free(N, sizeof(int64_t), CParent, cm)
        cholmod_free(N, sizeof(int64_t), CMember, cm)
        cholmod_l_finish(cm)

    if return_separator:
        return p, SeparatorTree(cp, cmember)
    else:
        return p


def metis(A, *, kind=None):
    """Nested dissection ordering of a sparse matrix using METIS.

    Parameters
    ----------
    A : (M, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square and symmetric if ``kind`` is None or ``"sym"``. No check is made
        for symmetry.
    kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization. Only the upper triangular part of
          ``A`` is used, and no check is made for symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.

        If ``kind`` is None, it defaults to ``sym``.

    Returns
    -------
    p : (M or N,) ndarray of int
        The permutation vector that gives the nested dissection ordering of the
        nodes in the graph represented by the sparse matrix ``A``.

    See Also
    --------
    bisect, nesdis

    Notes
    -----
    This function is based on the SuiteSparse CHOLMOD MATLAB interface
    [#metis_c]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#metis_c] ``metis.c`` - CHOLMOD MATLAB metis function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/metis.c

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> from sksparse.cholmod import metis
    >>> # Create a symmetric positive definite matrix from (Davis, Eqn 2.1)
    >>> N = 11
    >>> rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    >>> cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    >>> rng = np.random.default_rng(56)
    >>> vals = rng.random(len(rows), dtype=np.float64)
    >>> L = coo_array((vals, (rows, cols)), shape=(N, N))
    >>> A = L + L.T   # make it symmetric
    >>> A.setdiag(N)  # make it strongly positive definite
    >>> A = A.tocsc()
    >>> p = metis(A)
    >>> p
    array([ 8,  3,  6,  0,  5,  2,  4,  7,  1,  9, 10])
    """
    cdef bint use_int32
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t N = A.shape[1]

    if kind not in ["row", "col"] and M != N:
        raise ValueError(f"Input matrix A must be square, got shape {A.shape}.")

    # Special Cases
    # sym: A = (0, 0)
    # row: AA.T = (0, N) * (N, 0) = (0, 0)
    # col: A.TA = (0, M) * (M, 0) = (0, 0)
    if kind == "row" and M == 0 or N == 0:
        return np.array([], dtype=out_itype)

    if A.nnz == 0:
        D = N if kind == "col" else M
        return np.arange(D, dtype=out_itype)

    # -------------------------------------------------------------------------
    #         Start the Analysis
    # -------------------------------------------------------------------------
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix

    cdef int stype = -1  # default kind="sym" uses tril(A) only
    cdef bint transpose = False

    if kind == "row":
        stype = 0  # use A * A.T
    elif kind == "col":
        stype = 0  # use A.T * A
        transpose = True
    elif kind == "lo":
        stype = -1  # use tril(A) only

    # Get sparse *pattern*
    _cholmod_sparse_from_csc(
        A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
    )
    Ac.xtype = CHOLMOD_PATTERN
    Ac.x = NULL

    # -------------------------------------------------------------------------
    #         Compute the Outputs
    # -------------------------------------------------------------------------
    cdef void *Perm
    cdef cholmod_sparse *C
    cdef bint postorder = True
    cdef int64_t ok

    if transpose:
        # C = A.T, then metis C @ C.T
        if use_int32:
            C = cholmod_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            N = C.nrow
            Perm = cholmod_malloc(N, sizeof(int32_t), cm)
            ok = cholmod_metis(C, NULL, 0, postorder, <int32_t*>Perm, cm)
            cholmod_free_sparse(&C, cm)
        else:
            C = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            N = C.nrow
            Perm = cholmod_l_malloc(N, sizeof(int64_t), cm)
            ok = cholmod_l_metis(C, NULL, 0, postorder, <int64_t*>Perm, cm)
            cholmod_l_free_sparse(&C, cm)
    else:
        N = Ac.nrow
        if use_int32:
            Perm = cholmod_malloc(N, sizeof(int32_t), cm)
            ok = cholmod_metis(Ac, NULL, 0, postorder, <int32_t*>Perm, cm)
        else:
            Perm = cholmod_l_malloc(N, sizeof(int64_t), cm)
            ok = cholmod_l_metis(Ac, NULL, 0, postorder, <int64_t*>Perm, cm)

    if not ok:
        raise CholmodError("metis failed.")

    # Get the ndarray to return
    p = _ndarray_copy_from_intptr(Perm, N, use_int32)

    # Free memory (arrays are copied to numpy)
    if use_int32:
        cholmod_free(N, sizeof(int32_t), Perm, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free(N, sizeof(int64_t), Perm, cm)
        cholmod_l_finish(cm)

    return p
