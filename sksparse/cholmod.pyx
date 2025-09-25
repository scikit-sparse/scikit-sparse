# Part of the scikit-sparse project.
# Copyright (C) 2008-2025 The scikit-sparse developers. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: cholmod.pyx
#  Created: 2025-08-11 14:49
# =============================================================================

"""sksparse.cholmod: Python interface to the CHOLMOD library.

This module provides a Python interface to the CHOLMOD library, which is part
of the SuiteSparse collection by Timothy A. Davis. The main algorithm computes
the Cholesky factorization of a sparse matrix, and solves linear systems.

Interfaces
----------
* `cholesky`: Computes the Cholesky factorization of a sparse matrix.
* `ldl`: Computes the LDL.T factorization of a sparse matrix.
* `CholeskyFactor`: Class representing a Cholesky factorization.
* `cho_factor`: Computes the Cholesky factorization of a sparse matrix.
* `ldl_factor`: Computes the LDL.T factorization of a sparse matrix.
* `symbfact`: Computes the symbolic factorization of a sparse matrix.
* `etree`: Computes the elimination tree of a sparse matrix.
* `bisect`: Bisects a graph using nested dissection.
* `metis`: Computes a fill-reducing ordering using METIS.
* `nesdis`: Computes a fill-reducing ordering using NESDIS.

This wrapper handles both 32-bit and 64-bit integer types, depending on the
input matrix format.

.. versionadded:: 0.5.0

References
----------
* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse CHOLMOD:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD
"""

import numpy as np
cimport numpy as np

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
    pass


class CholmodNotPositiveDefiniteError(CholmodError):
    pass


class CholmodNotInstalledError(CholmodError):
    pass


class CholmodOutOfMemoryError(CholmodError):
    pass


class CholmodOverflowError(CholmodError):
    pass


class CholmodInvalidInputError(CholmodError):
    pass


class CholmodGpuProblemError(CholmodError):
    pass


class CholmodWarning(Warning):
    pass


class CholmodSmallDiagonalWarning(CholmodWarning):
    pass


cdef _handle_errors(int status) except * with gil:
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
        return

    status_msg = f"(code {status:d})"

    # Known Errors
    cdef dict error_map = {
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

    # Fallback to generic error for unknown codes
    exc_class, msg = error_map.get(status, CholmodError)
    full_msg = msg + " " + status_msg

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class)
    else:
        raise exc_class(full_msg)


# -----------------------------------------------------------------------------
#         CSC <==> CHOLMOD Sparse
# -----------------------------------------------------------------------------
cdef _supported_dtypes = (
    np.bool_,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128
)


cdef int _single_or_double(np.dtype dtype):
    """Return the CHOLMOD dtype number for a given NumPy dtype."""
    return CHOLMOD_SINGLE if dtype in [np.float32, np.complex64] else CHOLMOD_DOUBLE


cdef int _real_or_complex(np.dtype dtype):
    """Return the CHOLMOD xtype number for a given NumPy dtype."""
    return CHOLMOD_COMPLEX if np.issubdtype(dtype, np.complexfloating) else CHOLMOD_REAL


cdef object _cholmod_sparse_from_csc(
    object A_py,
    int stype,
    bint use_int32,
    cholmod_sparse *A_static,
):
    """Create a CHOLMOD sparse matrix from a scipy.sparse.csc_array.

    See the CHOLMOD MATLAB interface for details [#sputil_get_sparse]_.

    Parameters
    ----------
    A_py : (N, N) csc_array
        The input sparse matrix in Compressed Sparse Column (CSC) format.
    stype : int
        The assumed symmetry type of ``A_py``:
        * -1: lower triangular,
        *  0: unsymmetric (not used here),
        *  1: upper triangular.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for indices and indptr.
    A_static : cholmod_sparse*
        Pointer to a preallocated CHOLMOD sparse matrix structure. Contents
        need not be initialized. Contains the CHOLMOD sparse matrix on output.

    Returns
    -------
    res : tuple
        A tuple containing a reference to ``A_py`` and the three arrays that
        make it up: ``A.indptr``, ``A.indices``, and ``A.data``. There is no
        use for the output of this function, except to keep the underlying data
        from being garbage collected until the cholmod_sparse object is freed.

    References
    ----------
    .. [#sputil_get_sparse] ``sputil2.c`` - CHOLMOD MATLAB utilities
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/sputil2.c
    """
    if not isinstance(A_py, csc_array):
        raise ValueError("Input must be a csc_array.")

    dtype = A_py.dtype

    if dtype not in _supported_dtypes:
        raise ValueError(f"Unsupported data type for CHOLMOD: {dtype}")

    # Initialize the CHOLMOD sparse matrix
    cdef cholmod_sparse* A = A_static
    memset(A, 0, sizeof(cholmod_sparse))

    A.nrow, A.ncol = A_py.shape
    A.nzmax = A_py.nnz
    A.packed = True
    A.sorted = True  # NOTE requires input indices to be sorted
    A.itype = CHOLMOD_INT if use_int32 else CHOLMOD_LONG
    A.stype = -1 if stype < 0 else (0 if stype == 0 else 1)
    A.dtype = _single_or_double(dtype)
    A.z = NULL

    cdef np.ndarray indptr = A_py.indptr
    cdef np.ndarray indices = A_py.indices
    cdef np.ndarray data = A_py.data

    # Create the index arrays
    if use_int32:
        A.p = <int32_t*>indptr.data
        A.i = <int32_t*>indices.data
    else:
        A.p = <int64_t*>indptr.data
        A.i = <int64_t*>indices.data

    # Get the numerical values of A
    if dtype == np.bool_:
        A.xtype = CHOLMOD_PATTERN
        A.x = NULL
    else:
        A.xtype = _real_or_complex(dtype)

        if dtype == np.float32:
            A.x = <float32_t*>data.data
        elif dtype == np.float64:
            A.x = <float64_t*>data.data
        elif dtype == np.complex64:
            A.x = <complex64_t*>data.data
        elif dtype == np.complex128:
            A.x = <complex128_t*>data.data

    return A_py


cdef class _CholmodSparseDestructor:
    """A destructor for CHOLMOD sparse matrices.

    This class is used as a base for NumPy arrays that are views on CHOLMOD
    sparse matrices. It ensures that the CHOLMOD sparse matrix is properly
    freed when the NumPy array is no longer in use.

    Attributes
    ----------
    _sparse : cholmod_sparse*
        The CHOLMOD sparse matrix to be freed.
    _common : cholmod_common*
        The CHOLMOD common structure used for memory management.
    """

    cdef cholmod_sparse* _sparse
    cdef cholmod_common* _common

    cdef void init(self, cholmod_sparse* A, cholmod_common* common):
        assert A is not NULL
        assert common is not NULL
        self._sparse = A
        self._common = common

    def __dealloc__(self):
        if self._sparse.itype == CHOLMOD_INT:
            cholmod_free_sparse(&self._sparse, self._common)
        else:
            cholmod_l_free_sparse(&self._sparse, self._common)


# dict[xtype, dtype] -> numpy typenum
cdef dict _np_dtypenum_from_cholmod = {
    (CHOLMOD_REAL, CHOLMOD_SINGLE): np.NPY_FLOAT32,
    (CHOLMOD_REAL, CHOLMOD_DOUBLE): np.NPY_FLOAT64,
    (CHOLMOD_COMPLEX, CHOLMOD_SINGLE): np.NPY_COMPLEX64,
    (CHOLMOD_COMPLEX, CHOLMOD_DOUBLE): np.NPY_COMPLEX128,
    (CHOLMOD_PATTERN, CHOLMOD_SINGLE): np.NPY_BOOL,
    (CHOLMOD_PATTERN, CHOLMOD_DOUBLE): np.NPY_BOOL,
}


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
    cdef int np_itypenum = np.NPY_INT32 if A.itype == CHOLMOD_INT else np.NPY_INT64
    cdef int np_dtypenum = _np_dtypenum_from_cholmod.get(
        (A.xtype, A.dtype), np.NPY_OBJECT
    )

    # convert to NumPy arrays
    cdef np.ndarray indptr = np.PyArray_SimpleNewFromData(
        1, [A.ncol + 1], np_itypenum, A.p
    )
    cdef np.ndarray indices = np.PyArray_SimpleNewFromData(
        1, [A.nzmax], np_itypenum, A.i
    )
    cdef np.ndarray data = np.PyArray_SimpleNewFromData(
        1, [A.nzmax], np_dtypenum, A.x
    )

    # Take ownership of the data
    cdef _CholmodSparseDestructor base = _CholmodSparseDestructor()
    base.init(A, common)

    for array in (indptr, indices, data):
        np.set_array_base(array, base)
        assert np.PyArray_ISWRITEABLE(array)

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
    cdef bint use_int32 = L.itype == CHOLMOD_INT

    cdef int to_ll = L.is_ll if ldl is None else not ldl
    cdef int to_super = False  # simplicial format
    cdef int to_packed = True
    cdef int to_monotonic = True

    change_factor = cholmod_change_factor if use_int32 else cholmod_l_change_factor
    change_factor(
        L.xtype, to_ll, to_super, to_packed, to_monotonic, L, common
    )
    _handle_errors(common.status)

    # Create numpy arrays
    cdef int np_itypenum = np.NPY_INT32 if use_int32 else np.NPY_INT64
    cdef int np_dtypenum = _np_dtypenum_from_cholmod.get(
        (L.xtype, L.dtype), np.NPY_OBJECT
    )

    cdef np.ndarray indptr = np.PyArray_SimpleNewFromData(
        1, [L.n + 1], np_itypenum, L.p
    )
    cdef np.ndarray indices = np.PyArray_SimpleNewFromData(
        1, [L.nzmax], np_itypenum, L.i
    )
    cdef np.ndarray data = np.PyArray_SimpleNewFromData(
        1, [L.nzmax], np_dtypenum, L.x
    )

    # Take ownership of the data
    for array in (indptr, indices, data):
        np.set_array_base(array, py_factor)
        np.PyArray_CLEARFLAGS(array, np.NPY_ARRAY_WRITEABLE)  # make read-only

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
cdef object _cholmod_dense_from_ndarray(np.ndarray X_py, cholmod_dense *X_static):
    """Create a CHOLMOD dense matrix from a numpy.ndarray.

    See the CHOLMOD MATLAB interface for details [#sputil_get_dense]_.

    Parameters
    ----------
    X_py : (M, N) ndarray
        The input sparse matrix. Boolean data types are converted to float64.
    X_static : cholmod_sparse*
        Pointer to a preallocated CHOLMOD sparse matrix structure. Contents
        need not be initialized. Contains the CHOLMOD sparse matrix on output.

    Returns
    -------
    res : ndarray
        A reference to the array ``X_py``. If it has been type-converted, the
        reference will not be the original array. There is no use for the
        output of this function, except to keep the underlying data from being
        garbage collected until the cholmod_sparse object is freed.

    References
    ----------
    .. [#sputil_get_dense] ``sputil2.c`` - CHOLMOD MATLAB utilities
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/sputil2.c
    """
    # NOTE cholmod_dense objects are stored in column-major order.
    cdef np.ndarray Xd = np.asfortranarray(X_py)

    if Xd.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    dtype = Xd.dtype

    if dtype not in _supported_dtypes:
        raise ValueError(f"Unsupported data type for CHOLMOD: {dtype}")

    # Convert boolean to float64, as CHOLMOD does not support boolean dense
    if dtype == np.bool_:
        Xd = Xd.astype(np.float64)
        dtype = Xd.dtype

    # Initialize the CHOLMOD dense matrix
    cdef cholmod_dense* X = X_static
    memset(X, 0, sizeof(cholmod_dense))

    X.nrow = Xd.shape[0]
    X.ncol = Xd.shape[1]
    X.d = X.nrow
    X.nzmax = X.nrow * X.ncol
    X.dtype = _single_or_double(dtype)
    X.z = NULL

    # Get the numerical values of X
    X.xtype = _real_or_complex(dtype)

    if dtype == np.float32:
        X.x = <float32_t*>Xd.data
    elif dtype == np.float64:
        X.x = <float64_t*>Xd.data
    elif dtype == np.complex64:
        X.x = <complex64_t*>Xd.data
    elif dtype == np.complex128:
        X.x = <complex128_t*>Xd.data

    return Xd


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
    _common : cholmod_common*
        The CHOLMOD common structure used for memory management.
    """

    cdef cholmod_dense* _dense
    cdef cholmod_common* _common
    cdef bint _use_int32

    cdef void init(self, cholmod_dense* A, bint use_int32, cholmod_common* common):
        assert A is not NULL
        assert common is not NULL
        self._dense = A
        self._common = common
        self._use_int32 = use_int32

    def __dealloc__(self):
        if self._use_int32:
            cholmod_free_dense(&self._dense, self._common)
        else:
            cholmod_l_free_dense(&self._dense, self._common)


cdef np.ndarray _ndarray_from_cholmod_dense(
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
    cdef int np_dtypenum = _np_dtypenum_from_cholmod.get(
        (X.xtype, X.dtype), np.NPY_OBJECT
    )

    # convert to NumPy array
    cdef np.ndarray arr = np.PyArray_SimpleNewFromData(
        1, [X.nrow * X.ncol], np_dtypenum, X.x
    )

    # set destructor and check if writeable
    cdef _CholmodDenseDestructor base = _CholmodDenseDestructor()
    base.init(X, use_int32, common)
    np.set_array_base(arr, base)
    assert np.PyArray_ISWRITEABLE(arr)

    # Cholmod dense matrices are stored in column-major order, so reshape
    arr = arr.reshape((X.nrow, X.ncol), order="F")

    return arr


cdef np.ndarray _ndarray_copy_from_intptr(void* ptr, size_t N, bint use_int32):
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

    cdef int np_itypenum = np.NPY_INT32 if use_int32 else np.NPY_INT64
    cdef np.ndarray p = np.PyArray_SimpleNewFromData(1, [N], np_itypenum, ptr)
    return p.copy()  # return a copy in case ptr is freed


cdef np.ndarray _ndarray_int_view_from_factor(
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

    cdef int np_itypenum = np.NPY_INT32 if L.itype == CHOLMOD_INT else np.NPY_INT64
    cdef np.ndarray p = np.PyArray_SimpleNewFromData(1, [N], np_itypenum, ptr)
    np.set_array_base(p, py_factor)                   # keep object alive
    np.PyArray_CLEARFLAGS(p, np.NPY_ARRAY_WRITEABLE)  # set to be read-only

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


cdef dict _npdtype_class_from_xdtype = {
    (CHOLMOD_REAL, CHOLMOD_SINGLE): np.float32,
    (CHOLMOD_REAL, CHOLMOD_DOUBLE): np.float64,
    (CHOLMOD_COMPLEX, CHOLMOD_SINGLE): np.complex64,
    (CHOLMOD_COMPLEX, CHOLMOD_DOUBLE): np.complex128,
    (CHOLMOD_PATTERN, CHOLMOD_SINGLE): np.bool_,
    (CHOLMOD_PATTERN, CHOLMOD_DOUBLE): np.bool_,
}


# -----------------------------------------------------------------------------
#         CholeskyFactor Object
# -----------------------------------------------------------------------------
cdef void _copy_cholmod_common(cholmod_common* dest, cholmod_common* src):
    """Copy the contents of one cholmod_common struct to another."""
    assert dest is not NULL
    assert src is not NULL

    # Copy known input fields, ignore others
    dest.supernodal = src.supernodal
    dest.quick_return_if_not_posdef = src.quick_return_if_not_posdef

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
    dest.itype = src.itype

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
    dest.nsbounds_hit = src.nsbounds_hit
    dest.rowfacfl = src.rowfacfl
    dest.aatfl = src.aatfl
    dest.called_nd = src.called_nd
    dest.blas_ok = src.blas_ok

    # Skip SPQR related fields and GPU related fields


cdef void _cleanup_factor(CholeskyFactor cf):
    """Deallocate memory used by a CholeskyFactor."""
    if cf._cm is not NULL:
        if cf._use_int32:
            if cf._factor is not NULL:
                cholmod_free_factor(&cf._factor, cf._cm)
            cholmod_finish(cf._cm)
        else:
            if cf._factor is not NULL:
                cholmod_l_free_factor(&cf._factor, cf._cm)
            cholmod_l_finish(cf._cm)


# Define a special internal class for copying CholeskyFactor only
cdef class _CopySentinel:
    pass


cdef class CholeskyFactor:
    """The main object used for creating and manipulating a Cholesky factor.

    The constructor computes the symbolic analysis of the matrix and
    determines a fill-reducing ordering (if ``order`` is not ``None`` or
    ``"natural"``) such that:

    .. math ::

        L L^{\\top} = P A P^{\\top}.

    The numeric factorization is not computed until :meth:`.factorize` is
    called.

    Attributes
    ----------
    N : int
        The number of rows and columns in the factor.
    is_ll : bool
        Whether the factor is in ``LL.T`` form (True) or ``LDL.T`` form (False).
    is_super : bool
        Whether the factor is in supernodal (True) or simplicial (False) format.
    itype : np.dtype in {np.int32, np.int64}
        The integer type used for indices and indptr in the factor.
    dtype : np.dtype
        The data type used for numerical values in the factor.
    colcount : (N,) ndarray of int
        The number of nonzeros in each column of the factor.
    nnz : int
        The number of nonzeros in the factor.
    order : str or int
        The ordering method used for the factorization. If an unknown ordering
        was used, returns the integer value.
    perm : (N,) ndarray of int
        A read-only view of the permutation vector used for the factorization.

    Parameters
    ----------
    A : (N, N) {array_like, sparse array}
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format. The matrix must be square and symmetric positive
        definite. Only the upper or lower triangular part of the matrix is
        used, and no check is made for symmetry.
    sym_kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization. No check is made for symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.

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
        The permutation algorithm to use for the factorization. By default,
        the natural ordering of the input matrix is used. The other options
        are:

        * ``default``: Use the default method, which first tries AMD, then
            METIS.
        * ``best``: Automatically select the best ordering based on the
            input.
        * ``metis``: Use the METIS library for graph partitioning.
        * ``nesdis``: Use the NESDIS library for nested dissection.
        * ``amd``: Use the Approximate Minimum Degree (AMD) algorithm.
        * ``colamd``: Use the Approximate Minimum Degree (AMD) algorithm
            for the symmetric case, or the COLAMD algorithm for the
            unsymmetric case (:math:`A A^{{\\top}}` or :math:`A^{{\\top}} A`).
        * ``postordered``: Use natural ordering followed by postordering.

        By default, methods other than ``natural`` will also be
        postordered.

        .. warning::

            The ordering method ``best`` may be quite slow for large
            matrices, but if the factorization is reused many times, it can
            be worth it.

    Raises
    ------
    CholmodNotPositiveDefiniteError
        If the input matrix is structurally singular (*e.g.*, if it is the zero
        matrix). The input *may* be numerically indefinite, but this property
        is not checked until :meth:`.factorize` is called.

    See Also
    --------
    * :func:`.cholesky` : Factorize a matrix using Cholesky decomposition.
    * :func:`.ldl` : Factorize a matrix using LDL decomposition.
    * :func:`.cho_factor` : Factorize a matrix using Cholesky decomposition.
    * :func:`.ldl_factor` : Factorize a matrix using LDL decomposition.

    .. versionadded:: 0.5.0

    Notes
    -----
    The symbolic analysis follows that of the SuiteSparse CHOLMOD ``analyze``
    MATLAB function [#analyze_c]_.

    References
    ----------
    .. [#analyze_c] ``analyze.c`` - CHOLMOD MATLAB analyze function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/analyze.c
    """

    cdef cholmod_common _Common
    cdef cholmod_common *_cm
    cdef cholmod_factor *_factor
    cdef bint _use_int32
    cdef bint _is_lower
    cdef int _stype

    def __cinit__(
        self,
        object A,
        *,
        bint lower=True,
        object order=None,
        object sym_kind=None,
        object supernodal_mode=None,
    ):
        # Internal value to create an empty class during a copy
        if A is _CopySentinel:
            self._cm = NULL
            self._factor = NULL
            return

        A, use_int32, _ = validate_csc_input(A, require_square=True)

        if sym_kind is None:
            sym_kind = "sym"

        if supernodal_mode is None:
            supernodal_mode = "auto"

        if sym_kind not in {"sym", "row", "col"}:
            raise ValueError(
                f"Unknown symmetry kind: {sym_kind}. "
                "Must be one of 'sym', 'row', 'col'."
            )

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

        cdef size_t N = A.shape[0]
        self._use_int32 = use_int32

        # Matrix of all zeros
        if N > 0 and A.nnz == 0:
            raise CholmodNotPositiveDefiniteError("Input matrix not positive definite.")

        # Get the input matrix into CHOLMOD format
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix
        cdef cholmod_sparse *C

        # Use lower or upper triangular part of A
        self._is_lower = lower
        cdef int stype = -1 if self._is_lower else 1
        cdef bint transpose = False

        if sym_kind in ["row", "col"]:
            stype = 0                        # unsymmetric A @ A.T or A.T @ A
            transpose = (sym_kind == "col")  # A.T @ A

        # keep a reference to the input matrix
        cdef object _ref = _cholmod_sparse_from_csc(A, stype, self._use_int32, Ac)

        self._stype = Ac.stype

        try:
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
                    self._factor = cholmod_analyze(Ac, self._cm)
                    cholmod_free_sparse(&C, self._cm)
                else:
                    C = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, self._cm)
                    self._factor = cholmod_l_analyze(Ac, self._cm)
                    cholmod_l_free_sparse(&C, self._cm)
            else:
                if self._use_int32:
                    self._factor = cholmod_analyze(Ac, self._cm)
                else:
                    self._factor = cholmod_l_analyze(Ac, self._cm)

            # Check for errors
            _handle_errors(self._cm.status)

        except Exception as e:
            _cleanup_factor(self)
            raise e

    def __dealloc__(self):
        """Deallocate memory used by the CholeskyFactor."""
        _cleanup_factor(self)

    def _require_factorized(self):
        """Raise an error if the factor is symbolic only."""
        if self._factor.xtype == CHOLMOD_PATTERN:
            raise CholmodError("Factor is symbolic. Call `factorize` before updating.")

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
    def is_lower(self):
        return bool(self._is_lower)

    @property
    def is_super(self):
        return bool(self._factor.is_super)

    @property
    def itype(self):
        return np.dtype(np.int32 if self._factor.itype == CHOLMOD_INT else np.int64)

    @property
    def dtype(self):
        # "np.int32" etc. are dtype classes, not actual dtypes. numpy handles
        # both well, but be explicit and return a dtype object.
        return np.dtype(
            _npdtype_class_from_xdtype.get(
                (self._factor.xtype, self._factor.dtype), None
            )
        )

    @property
    def N(self):
        return self._factor.n

    @property
    def colcount(self):
        return _ndarray_int_view_from_factor(self._factor.ColCount, self._factor.n, self)

    @property
    def nnz(self):
        return np.sum(self.colcount)

    @property
    def order(self):
        cdef int iorder = self._factor.ordering
        return _ordering_methods_inv.get(iorder, iorder)

    @property
    def perm(self):
        return _ndarray_int_view_from_factor(self._factor.Perm, self._factor.n, self)

    @property
    def factor(self):
        """Return a view of the Cholesky factor.

        .. warning::

            The returned matrix is a view on the internal data of the CHOLMOD
            factor. It will be modified if the factor is modified (*e.g.*, by
            calling :meth:`.factorize`). To get a copy, use
            :meth:`.get_factor`.

        Returns
        -------
        L : csc_array
            The Cholesky factor in Compressed Sparse Column (CSC) format. If
            ``self.is_ll``, the returned matrix is lower triangular. Otherwise,
            the matrix view contains the lower triangular and the diagonal
            factors combined.

        .. note :: The view is always in lower triangular form, even if the
            factor was created using ``lower=False``. To get the upper
            triangular factor, use :obj:`get_factor` with ``lower=False``. To
            get the split `L` and `D` factors, use :obj:`get_factor` with
            ``kind="LDL"``.
        """
        return _csc_view_from_cholmod_factor(self)

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
        cdef CholeskyFactor cf = CholeskyFactor.__new__(CholeskyFactor, _CopySentinel)

        cf._cm = &cf._Common

        # Allocate and copy the CHOLMOD common struct
        if self._use_int32:
            cholmod_start(cf._cm)
        else:
            cholmod_l_start(cf._cm)

        _copy_cholmod_common(self._cm, cf._cm)

        # Copy the factor
        if self._use_int32:
            cf._factor = cholmod_copy_factor(self._factor, cf._cm)
        else:
            cf._factor = cholmod_l_copy_factor(self._factor, cf._cm)

        _handle_errors(cf._cm.status)

        cf._use_int32 = self._use_int32
        cf._is_lower = self._is_lower
        cf._stype = self._stype

        return cf

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
            lower = self._is_lower

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

    def factorize(self, object A, object ldl=None, float beta=0.0, object lower=None):
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
        lower : bool, optional
            If True, only use the lower triangular part of `A`. Otherwise, use
            the upper triangular part. Default is None, which uses the same
            triangular part used for the object initialization or the
            previous call to :meth:`.factorize`.

        Notes
        -----
        If ``ldl=False``, this function computes the Cholesky factorization of
        a symmetric positive definite matrix `A`:

        .. math::

            R^{\\top} R = P A P^{\\top},

        where `R` is an upper triangular matrix. Only the upper triangular part
        of `A` is used. If ``lower`` is True, the lower triangular factor `L`
        is computed instead, such that:

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

        A, _, _ = validate_csc_input(A, require_square=True)

        if ldl is None:
            try:
                ldl = not self.is_ll  # use the existing factor type
            except ValueError:
                ldl = False  # default to LL if no factor exists yet

        if not isinstance(ldl, bool):
            raise ValueError("ldl must be a boolean value.")

        if lower is None:
            lower = self._is_lower  # use the existing triangle
        elif isinstance(lower, bool):
            self._is_lower = lower
        else:
            raise ValueError("lower must be a boolean value.")

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

        stype = -1 if lower else 1  # use lower or upper triangular part
        # Keep a reference to the input matrix to keep it alive
        cdef object _ref = _cholmod_sparse_from_csc(A, stype, self._use_int32, Ac)

        # Set stype and beta
        cdef double betac[2]

        if not np.isscalar(beta):
            raise ValueError("beta must be a scalar value.")

        betac[0] = beta
        betac[1] = 0.0

        Ac.stype = self._stype  # set in __cinit__ with sym_kind

        # Factorize the matrix
        if self._use_int32:
            cholmod_factorize_p(Ac, betac, NULL, 0, self._factor, self._cm)
        else:
            cholmod_l_factorize_p(Ac, betac, NULL, 0, self._factor, self._cm)

        # Check for errors
        _handle_errors(self._cm.status)

        return self  # for method chaining

    def solve(self, b):
        """Solve the linear system :math:`A x = b` for `x`, using the
        factorization.

        Parameters
        ----------
        b : (N,) or (N, K) ndarray or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) ndarray or sparse matrix
            The solution vector or matrix, returned in the same format as `b`.

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
        cdef size_t K = b.shape[1] if b.ndim == 2 else 0

        if b.shape[0] != N:
            raise ValueError(
                "Right-hand side b must have the same number of rows as L."
            )

        # Special case: empty matrix
        if N == 0:
            return type(b)(b.shape, dtype=b.dtype)

        if issparse(b):
            X = self._solve_sparse(b)
        else:
            X = self._solve_dense(b)

        # For LDL, unpermute the solution
        if not self.is_ll:
            X = X[np.argsort(self.perm)]

        # Convert to 1D array if input b is 1D
        if K == 0:
            X = X[:, 0]

        return X

    cdef object _solve_sparse(self, object b):
        """Solve the system A x = b with a sparse right-hand side."""
        # Get the b vector or matrix into CHOLMOD format
        cdef cholmod_sparse Bspmatrix
        cdef cholmod_sparse* Bs = &Bspmatrix

        # CHOLMOD expects at least a column vector for the RHS
        if b.ndim == 1:
            b = b.reshape((-1, 1)).tocsc()  # (N, 1)

        # For LDL, permute the RHS
        if not self.is_ll:
            b = b[self.perm]

        cdef int stype = 0
        cdef bint b_use_int32

        b, b_use_int32, _ = validate_csc_input(b)

        # keep a reference to b so it is not garbage collected
        cdef object _b_ref = _cholmod_sparse_from_csc(b, stype, b_use_int32, &Bspmatrix)

        # Check the condition number before solving
        self._check_rcond()

        # Solve the system
        cdef cholmod_sparse* Xs

        cdef int system = CHOLMOD_A if self.is_ll else CHOLMOD_LDLt

        if self._use_int32:
            Xs = cholmod_spsolve(system, self._factor, Bs, self._cm)
        else:
            Xs = cholmod_l_spsolve(system, self._factor, Bs, self._cm)

        _handle_errors(self._cm.status)

        return _csc_from_cholmod_sparse(Xs, self._cm)

    cdef np.ndarray _solve_dense(self, np.ndarray b):
        """Solve the system A x = b with a dense right-hand side."""
        # Get the b vector or matrix into CHOLMOD format
        cdef cholmod_dense Bmatrix
        cdef cholmod_dense* Bd = &Bmatrix

        # CHOLMOD expects at least a column vector for the RHS
        if b.ndim == 1:
            b = b[:, np.newaxis]  # (N, 1)

        # For LDL, permute the RHS
        if not self.is_ll:
            b = b[self.perm]

        # keep a reference to b so it is not garbage collected
        cdef object _b_ref = _cholmod_dense_from_ndarray(b, &Bmatrix)

        # Check the condition number before solving
        self._check_rcond()

        # Solve the system
        cdef cholmod_dense* Xd

        cdef int system = CHOLMOD_A if self.is_ll else CHOLMOD_LDLt

        if self._use_int32:
            Xd = cholmod_solve(system, self._factor, Bd, self._cm)
        else:
            Xd = cholmod_l_solve(system, self._factor, Bd, self._cm)

        _handle_errors(self._cm.status)

        return _ndarray_from_cholmod_dense(Xd, self._use_int32, self._cm)

    cdef void _check_rcond(self) except *:
        """Check the condition number."""
        cdef double rcond
        cdef double eps = np.finfo(np.float64).eps

        if self._use_int32:
            rcond = cholmod_rcond(self._factor, self._cm)
        else:
            rcond = cholmod_l_rcond(self._factor, self._cm)

        _handle_errors(self._cm.status)

        if rcond == 0:
            raise CholmodNotPositiveDefiniteError(
                "Matrix is indefinite or singular to working precision."
            )
        elif rcond < eps:
            warnings.warn(
                "Matrix is nearly singular."
                f"  Results may be inaccurate (rcond={rcond:.2e}).",
                CholmodWarning,
            )

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

        # Ensure C is in CSC format
        if C.ndim == 1:
            C = C.reshape((-1, 1)).tocsc()  # (N, 1)

        cdef int stype = 0  # use all of C
        cdef bint C_use_int32
        C, C_use_int32, _ = validate_csc_input(C)

        cdef cholmod_sparse Cmatrix
        cdef cholmod_sparse* Cc = &Cmatrix

        # Keep a reference to C so it is not garbage collected
        cdef object _C_ref = _cholmod_sparse_from_csc(C, stype, C_use_int32, &Cmatrix)

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

        # Get C Matrix
        cdef cholmod_sparse Cmatrix
        cdef cholmod_sparse* Cc = &Cmatrix
        cdef int stype = 0  # use all of C

        # Ensure C is in CSC format
        if C.ndim == 1:
            C = C.reshape((-1, 1)).tocsc()  # (N, 1)

        C, C_use_int32, _ = validate_csc_input(C)
        # keep a reference to C so it is not garbage collected
        cdef object _C_ref = _cholmod_sparse_from_csc(C, stype, C_use_int32, &Cmatrix)

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
        CholeskyFactor
            The current object, for method chaining.

        See Also
        --------
        :func:`.cholesky`, :func:`.ldl`, :meth:`.update`, :meth:`.rowadd`,
        :meth:`.rowdel`

        .. versionadded:: 0.5.0

        References
        ----------
        .. [#resymbol_c] ``resymbol.c`` - CHOLMOD MATLAB resymbolization function
            https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/resymbol.c
        """
        self._require_factorized()

        cdef bint A_use_int32
        A, A_use_int32, _ = validate_csc_input(A, require_square=True)

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

        # Get sparse *pattern*
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse* Ac = &Amatrix
        cdef int stype = -1  # use tril(A) only

        cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, self._use_int32, Ac)
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

        See Also
        --------
        :meth:`.slogdet`, :meth:`.det`, :func:`numpy.linalg.slogdet`,
        :func:`numpy.linalg.det`, :func:`scipy.linalg.det`
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

        See Also
        --------
        :meth:`.logdet`, :meth:`.det`, :func:`numpy.linalg.slogdet`,
        :func:`numpy.linalg.det`, :func:`scipy.linalg.det`
        """
        return (self.dtype.type(1.0), self.logdet())

    def det(self):
        """Compute the determinant of the matrix from its Cholesky
        factorization.

        .. warning::

            This function may overflow or underflow for large matrices. Use
            :meth:`.logdet` or :meth:`.slogdet` instead.

        Returns
        -------
        det : float
            The determinant of the matrix `A` that was factorized.

        .. versionadded:: 0.2

        See Also
        --------
        :meth:`.logdet`, :meth:`.slogdet`, :func:`numpy.linalg.det`,
        :func:`numpy.linalg.slogdet`, :func:`scipy.linalg.det`
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

        See Also
        --------
        :func:`numpy.linalg.inv`, :func:`scipy.linalg.inv`
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
    A, beta=0.0, *, lower=False, order=None, sym_kind=None, supernodal_mode=None
):
    return CholeskyFactor(
        A, lower=lower, order=order, sym_kind=sym_kind, supernodal_mode=supernodal_mode
    ).factorize(A, ldl=False, beta=beta, lower=lower)


def ldl_factor(
    A, beta=0.0, *, lower=True, order=None, sym_kind=None, supernodal_mode=None
):
    return CholeskyFactor(
        A, lower=lower, order=order, sym_kind=sym_kind, supernodal_mode=supernodal_mode
    ).factorize(A, ldl=True, beta=beta, lower=lower)


# csc_arrays from the factorization, and optionally the permutation
def cholesky(
    A, beta=0.0, *, lower=False, order=None, sym_kind=None, supernodal_mode=None
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


def ldl(A, beta=0.0, *, lower=True, order=None, sym_kind=None, supernodal_mode=None):
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
    The permutation algorithm to use for the factorization. By default, the
    natural ordering of the input matrix is used. The other options are:

    * ``default``: Use the default method, which first tries AMD, then METIS.
    * ``best``: Automatically select the best ordering based on the input.
    * ``metis``: Use the METIS library for graph partitioning.
    * ``nesdis``: Use the NESDIS library for nested dissection.
    * ``amd``: Use the Approximate Minimum Degree (AMD) algorithm.
    * ``colamd``: Use the Approximate Minimum Degree (AMD) algorithm for the
        symmetric case, or the COLAMD algorithm for the unsymmetric case
        (:math:`A A^{{\\top}}` or :math:`A^{{\\top}} A`).
    * ``postordered``: Use natural ordering followed by postordering.

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

.. versionadded:: 0.5.0

References
----------
.. [{doc_tag}] ``cholmod.h`` - SuiteSparse CHOLMOD header file.
    https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Include/cholmod.h
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


_cholesky_see_also = """
  * :func:`.ldl` : Factorize a matrix using LDL decomposition.
  * :func:`.ldl_factor` : Factorize a matrix using LDL decomposition."""


_cho_factor_returns = """CholeskyFactor
    The factorization object. Use its methods to solve linear systems
    and manipulate the factorization.
"""

cho_factor.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_cholesky_intro,
    returns=_cho_factor_returns,
    see_also=_cholesky_see_also,
    doc_tag="#cho_factor_h",
)


cholesky.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_cholesky_intro,
    returns=_CHOLESKY_RETURNS.format(ldl_D_output=""),
    see_also=_cholesky_see_also,
    doc_tag="#cholesky_h",
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


_ldl_see_also = """
    * :func:`.cholesky` : Factorize a matrix using Cholesky decomposition.
    * :func:`.cho_factor` : Factorize a matrix using Cholesky decomposition."""


ldl_factor.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_ldl_intro,
    returns=_cho_factor_returns,
    see_also=_ldl_see_also,
    doc_tag="#ldl_factor_h",
)


ldl.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_ldl_intro,
    returns=_CHOLESKY_RETURNS.format(ldl_D_output=_ldl_D_output),
    see_also=_ldl_see_also,
    doc_tag="#ldl_h",
)


# -----------------------------------------------------------------------------
#         Symbolic Functions
# -----------------------------------------------------------------------------
def symbfact(A, *, kind=None, lower=False, return_factor=False):
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

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#symbfact_c] ``symbfact2.c`` - CHOLMOD MATLAB symbolic factorization function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/symbfact2.c
    """
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col", "lo"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]

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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, Ac)
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


def etree(A, *, kind=None, return_post=False):
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

    References
    ----------
    .. [#etree_c] ``etree2.c`` - CHOLMOD MATLAB symbolic factorization function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/etree2.c
    """
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col", "lo"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]

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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, Ac)
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
    :func:`.nesdis`, :func:`.metis`

    Notes
    -----
    This function is based on the SuiteSparse CHOLMOD MATLAB interface
    [#bisect_c]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#bisect_c] ``bisect.c`` - CHOLMOD MATLAB bisect function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/bisect.c
    """
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]

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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, Ac)
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
    cp : (C,) ndarray of int, optional
        The separator tree, where ``C`` is the number of components found. The
        value ``cp[c]`` is the parent of the component ``c`` in the separator
        tree, or ``-1`` if ``c`` is the root of the tree. There is a maximum of
        ``N`` components, where ``N`` is the dimension of the input matrix.
    cmember : (N,) ndarray of int, optional
        The component membership vector, where ``cmember[i]`` is the component
        to which node ``i`` belongs.
    """
    def __init__(self, cp, cmember):
        self._cp = cp
        self._cmember = cmember

    @property
    def cp(self):
        """(C,) ndarray of int: The component parent array."""
        return self._cp

    @property
    def cmember(self):
        """(N,) ndarray of int: The component membership array."""
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

        .. versionadded:: 0.5.0

        References
        ----------
        .. [#septree_c] ``septree.c`` - CHOLMOD MATLAB septree function
            https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/septree.c
        """
        if nd_oksep is None:
            nd_oksep = 1.0  # see CHOLMOD/MATLAB/nesdis.c

        if nd_small is None:
            nd_small = 200  # see CHOLMOD/MATLAB/nesdis.c

        cdef np.ndarray cp = self._cp
        cdef np.ndarray cmember = self._cmember

        cdef bint use_int32 = cp.dtype == np.int32 and cmember.dtype == np.int32

        cdef cholmod_common Common
        cdef cholmod_common *cm = &Common

        if use_int32:
            cholmod_start(cm)
        else:
            cholmod_l_start(cm)

        cdef size_t Nc = cp.size
        cdef size_t N = cmember.size

        # Copy input arrays into new cholmod arrays (modified for output)
        cdef void *CParent
        cdef void *CMember

        if use_int32:
            CParent = cholmod_malloc(Nc, sizeof(int32_t), cm)
            CMember = cholmod_malloc(N, sizeof(int32_t), cm)
            memcpy(<int32_t*>CParent, <int32_t*>cp.data, Nc * sizeof(int32_t))
            memcpy(<int32_t*>CMember, <int32_t*>cmember.data, N * sizeof(int32_t))
        else:
            CParent = cholmod_l_malloc(Nc, sizeof(int64_t), cm)
            CMember = cholmod_l_malloc(N, sizeof(int64_t), cm)
            memcpy(<int64_t*>CParent, <int64_t*>cp.data, Nc * sizeof(int64_t))
            memcpy(<int64_t*>CMember, <int64_t*>cmember.data, N * sizeof(int64_t))

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
    return_separator=False,
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
    :func:`.bisect`, :func:`.metis`

    Notes
    -----
    This function is based on the SuiteSparse CHOLMOD MATLAB interface
    [#nesdis_c]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#nesdis_c] ``nesdis.c`` - CHOLMOD MATLAB nesdis function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/nesdis.c
    """
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]

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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, Ac)
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
    :func:`.bisect`, :func:`.nesdis`

    Notes
    -----
    This function is based on the SuiteSparse CHOLMOD MATLAB interface
    [#metis_c]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#metis_c] ``metis.c`` - CHOLMOD MATLAB metis function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/metis.c
    """
    A, use_int32, out_itype = validate_csc_input(A)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    cdef size_t M = A.shape[0]
    cdef size_t N = A.shape[1]

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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, Ac)
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
