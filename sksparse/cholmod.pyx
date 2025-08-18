# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
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
* `solve`: Solves a linear system using the Cholesky factorization.

This wrapper handles both 32-bit and 64-bit integer types, depending on the
input matrix format.

References
----------
* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse CHOLMOD:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD
"""

import numpy as np
cimport numpy as np

from scipy.sparse import csc_array, diags_array, issparse
import warnings

from .utils import validate_csc_input


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


cdef _error_handler(int status) except * with gil:
    """Handle CHOLMOD errors by raising Python exceptions or warnings.

    This function should be set as the error handler in the CHOLMOD common
    struct before passing to any CHOLMOD function.

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

    # TODO include informative error messages.
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

    A.nrow = A_py.shape[0]
    A.ncol = A_py.shape[1]
    A.nzmax = A_py.nnz
    A.packed = True
    A.sorted = True
    A.itype = CHOLMOD_INT if use_int32 else CHOLMOD_LONG
    A.stype = -1 if stype < 0 else (0 if stype == 0 else 1)
    A.dtype = _single_or_double(dtype)
    A.z = NULL

    # Declare memoryviews for the index and data arrays
    cdef int32_t[::1] Ap_mv_int32, Ai_mv_int32
    cdef int64_t[::1] Ap_mv_int64, Ai_mv_int64

    cdef float32_t[::1] Ax_mv_float32
    cdef float64_t[::1] Ax_mv_float64
    cdef complex64_t[::1] Ax_mv_complex64
    cdef complex128_t[::1] Ax_mv_complex128

    # Declare array references to keep memory alive
    cdef np.ndarray Ap, Ai, Ax

    # Create the index arrays
    itype = np.int32 if use_int32 else np.int64
    Ap = np.ascontiguousarray(A_py.indptr, dtype=itype)
    Ai = np.ascontiguousarray(A_py.indices, dtype=itype)

    if use_int32:
        Ap_mv_int32 = Ap
        Ai_mv_int32 = Ai
        A.p = &Ap_mv_int32[0]
        A.i = &Ai_mv_int32[0]
    else:
        Ap_mv_int64 = Ap
        Ai_mv_int64 = Ai
        A.p = &Ap_mv_int64[0]
        A.i = &Ai_mv_int64[0]

    # Get the numerical values of A
    if dtype == np.bool_:
        A.xtype = CHOLMOD_PATTERN
        A.x = NULL
    else:
        A.xtype = _real_or_complex(dtype)
        Ax = np.ascontiguousarray(A_py.data, dtype=dtype)

        if dtype == np.float32:
            Ax_mv_float32 = Ax
            A.x = &Ax_mv_float32[0]
        elif dtype == np.float64:
            Ax_mv_float64 = Ax
            A.x = &Ax_mv_float64[0]
        elif dtype == np.complex64:
            Ax_mv_complex64 = Ax
            A.x = &Ax_mv_complex64[0]
        elif dtype == np.complex128:
            Ax_mv_complex128 = Ax
            A.x = &Ax_mv_complex128[0]

    return (A_py, Ap, Ai, Ax)


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
    """Build a scipy.sparse.csc_array that's a view onto A, with a 'base' with
    appropriate destructor. 'A' must have been allocated by cholmod."""

    # This is a little tricky: We build 3 arrays, views on each part of the
    # cholmod_dense object. They all have the same _CholmodSparseDestructor
    # object as base. So none of them will be deallocated until they have all
    # become unused. Then those are built into a csc_array.

    # init destructor for cholmod data
    cdef _CholmodSparseDestructor base = _CholmodSparseDestructor()
    base.init(A, common)

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

    # set destructor and check if writeable
    for array in (indptr, indices, data):
        np.set_array_base(array, base)
        assert np.PyArray_ISWRITEABLE(array)

    return csc_array((data, indices, indptr), shape=(A.nrow, A.ncol))


cdef object _cholmod_factor_from_csc(
    object LD_py,
    bint use_int32,
    cholmod_factor *L_static,
    cholmod_common *cm
):
    """Create a CHOLMOD factor from a scipy.sparse.csc_array.

    See the ``ldlsolve.m`` function in [#ldlsolve_ref]_.

    Parameters
    ----------
    LD_py : csc_array
        The input sparse matrix in Compressed Sparse Column (CSC) format. This
        should be a combination of the ``L`` and ``D`` factors computed from
        :func:`ldl`.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for indices and indptr.
    L_static : cholmod_factor*
        Pointer to a preallocated CHOLMOD factor structure. Contents need not
        be initialized. Contains the CHOLMOD factor on output.
    cm : cholmod_common*
        Pointer to a CHOLMOD common structure for configuration and status.

    Returns
    -------
    res : tuple
        A reference to the underlying arrays whose data the CHOLMOD factor
        references. There is no use for this object other than to prevent it
        from being garbarge collected.

    References
    ----------
    .. [#ldlsolve_ref] ``ldlsolve.c`` - CHOLMOD MATLAB utilities
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/ldlsolve.c
    """
    if not isinstance(LD_py, csc_array):
        raise ValueError("Input must be a csc_array.")

    cdef np.dtype dtype = LD_py.dtype

    if dtype not in _supported_dtypes:
        raise ValueError(f"Unsupported data type for CHOLMOD: {dtype}")

    # Initialize the CHOLMOD factor
    cdef cholmod_factor* L = L_static
    assert L is not NULL

    cdef size_t N = LD_py.shape[0]

    L.ordering = CHOLMOD_NATURAL  # LD is already ordered

    # Get views on the data
    cdef int32_t[::1] LDp_mv_int32, LDi_mv_int32
    cdef int64_t[::1] LDp_mv_int64, LDi_mv_int64

    cdef float32_t[::1] LDx_mv_float32
    cdef float64_t[::1] LDx_mv_float64
    cdef complex64_t[::1] LDx_mv_complex64
    cdef complex128_t[::1] LDx_mv_complex128

    # Declare array references to keep memory alive
    cdef np.ndarray LDp, LDi, LDx

    itype = np.int32 if use_int32 else np.int64
    LDp = np.ascontiguousarray(LD_py.indptr, dtype=itype)
    LDi = np.ascontiguousarray(LD_py.indices, dtype=itype)

    if use_int32:
        LDp_mv_int32 = LDp
        LDi_mv_int32 = LDi
        L.p = &LDp_mv_int32[0]
        L.i = &LDi_mv_int32[0]
    else:
        LDp_mv_int64 = LDp
        LDi_mv_int64 = LDi
        L.p = &LDp_mv_int64[0]
        L.i = &LDi_mv_int64[0]

    # Get the data values
    L.itype = CHOLMOD_INT if use_int32 else CHOLMOD_LONG
    L.dtype = _single_or_double(dtype)
    L.xtype = _real_or_complex(dtype)

    LDx = np.ascontiguousarray(LD_py.data, dtype=dtype)

    if dtype == np.float32:
        LDx_mv_float32 = LDx
        L.x = &LDx_mv_float32[0]
    elif dtype == np.float64:
        LDx_mv_float64 = LDx
        L.x = &LDx_mv_float64[0]
    elif dtype == np.complex64:
        LDx_mv_complex64 = LDx
        L.x = &LDx_mv_complex64[0]
    elif dtype == np.complex128:
        LDx_mv_complex128 = LDx
        L.x = &LDx_mv_complex128[0]

    L.z = NULL

    # Allocate and initialize the rest of L
    if use_int32:
        L.nz = cholmod_malloc(N, sizeof(int32_t), cm)
        L.prev = cholmod_malloc(N + 2, sizeof(int32_t), cm)
        L.next = cholmod_malloc(N + 2, sizeof(int32_t), cm)
        _initialize_factor(L, N)
    else:
        L.nz = cholmod_l_malloc(N, sizeof(int64_t), cm)
        L.prev = cholmod_l_malloc(N + 2, sizeof(int64_t), cm)
        L.next = cholmod_l_malloc(N + 2, sizeof(int64_t), cm)
        _initialize_l_factor(L, N)

    return (LD_py, LDp, LDi, LDx)


cdef void _initialize_factor(cholmod_factor* L, size_t N):
    """Initialize the additional fields of a CHOLMOD factor (32-bit)."""
    cdef int32_t *Lp = <int32_t*>L.p
    cdef int32_t *Lnz = <int32_t*>L.nz
    cdef int32_t *Lnext = <int32_t*>L.next
    cdef int32_t *Lprev = <int32_t*>L.prev

    cdef size_t j

    for j in range(N):
        Lnz[j] = Lp[j + 1] - Lp[j]

    cdef int head = N + 1
    cdef int tail = N

    Lnext[head] = 0
    Lprev[head] = -1
    Lnext[tail] = -1
    Lprev[tail] = N - 1

    for j in range(N):
        Lnext[j] = j + 1
        Lprev[j] = j - 1

    Lprev[0] = head

    L.nzmax = Lp[N]


cdef void _initialize_l_factor(cholmod_factor* L, size_t N):
    """Initialize the additional fields of a CHOLMOD factor (64-bit)."""
    cdef int64_t *Lp = <int64_t*>L.p
    cdef int64_t *Lnz = <int64_t*>L.nz
    cdef int64_t *Lnext = <int64_t*>L.next
    cdef int64_t *Lprev = <int64_t*>L.prev

    cdef size_t j

    for j in range(N):
        Lnz[j] = Lp[j + 1] - Lp[j]

    cdef int head = N + 1
    cdef int tail = N

    Lnext[head] = 0
    Lprev[head] = -1
    Lnext[tail] = -1
    Lprev[tail] = N - 1

    for j in range(N):
        Lnext[j] = j + 1
        Lprev[j] = j - 1

    Lprev[0] = head


# -----------------------------------------------------------------------------
#         CSC <==> CHOLMOD Dense
# -----------------------------------------------------------------------------
cdef object _cholmod_dense_from_ndarray(
    object X_py,
    cholmod_dense *X_static,
):
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
    X_py = np.asfortranarray(X_py)

    if X_py.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    dtype = X_py.dtype

    if dtype not in _supported_dtypes:
        raise ValueError(f"Unsupported data type for CHOLMOD: {dtype}")

    # Convert boolean to float64, as CHOLMOD does not support boolean dense
    if dtype == np.bool_:
        X_py = X_py.astype(np.float64)
        dtype = X_py.dtype

    # Initialize the CHOLMOD dense matrix
    cdef cholmod_dense* X = X_static
    memset(X, 0, sizeof(cholmod_dense))

    X.nrow = X_py.shape[0]
    X.ncol = X_py.shape[1]
    X.d = X.nrow
    X.nzmax = X.nrow * X.ncol
    X.dtype = _single_or_double(dtype)
    X.z = NULL

    # Declare memoryviews for the index and data arrays
    cdef float32_t[::1] X_mv_float32
    cdef float64_t[::1] X_mv_float64
    cdef complex64_t[::1] X_mv_complex64
    cdef complex128_t[::1] X_mv_complex128

    # Get the numerical values of X
    X.xtype = _real_or_complex(dtype)

    # Flatten the array to a 1D array for CHOLMOD
    X_py = X_py.ravel(order="F")

    if dtype == np.float32:
        X_mv_float32 = X_py
        X.x = &X_mv_float32[0]
    elif dtype == np.float64:
        X_mv_float64 = X_py
        X.x = &X_mv_float64[0]
    elif dtype == np.complex64:
        X_mv_complex64 = X_py
        X.x = &X_mv_complex64[0]
    elif dtype == np.complex128:
        X_mv_complex128 = X_py
        X.x = &X_mv_complex128[0]

    return X_py


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



cdef object _ndarray_from_cholmod_dense(
    cholmod_dense* X, bint use_int32, cholmod_common* common
):
    """Build a numpy.ndarray that is a view onto a cholmod_dense object.

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
    cdef _CholmodDenseDestructor base = _CholmodDenseDestructor()
    base.init(X, use_int32, common)

    cdef int np_dtypenum = _np_dtypenum_from_cholmod.get(
        (X.xtype, X.dtype), np.NPY_OBJECT
    )

    # convert to NumPy array
    cdef np.ndarray arr = np.PyArray_SimpleNewFromData(
        1, [X.nrow * X.ncol], np_dtypenum, X.x
    )

    # set destructor and check if writeable
    np.set_array_base(arr, base)
    assert np.PyArray_ISWRITEABLE(arr)

    # Cholmod dense matrices are stored in column-major order, so reshape
    arr = arr.reshape((X.nrow, X.ncol), order="F")

    return arr


cdef np.ndarray _array_from_cholmod_permutation(
    cholmod_factor* L, size_t N, bint use_int32
):
    """Create a NumPy array from the permutation vector in a CHOLMOD factor.

    Parameters
    ----------
    L : cholmod_factor*
        The CHOLMOD factor containing the permutation vector.
    N : size_t
        The size of the permutation vector.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for the permutation indices.

    Returns
    -------
    p : ndarray
        The permutation vector as a NumPy array.
    """
    if L is NULL or L.minor != N:
        raise ValueError("CHOLMOD factorization failed, cannot get permutation.")

    cdef int np_itypenum = np.NPY_INT32 if use_int32 else np.NPY_INT64
    cdef void* data_ptr = L.Perm
    cdef np.ndarray p = np.PyArray_SimpleNewFromData(1, [N], np_itypenum, data_ptr)
    # TODO Instead of returning a copy here, create a view and
    # a _CholmodFactorDestructor object to retain a reference?
    # Return a copy in case L is freed
    return p.copy()


# -----------------------------------------------------------------------------
#         Utilities
# -----------------------------------------------------------------------------
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


cdef void _set_ordering_method(object order, cholmod_common* cm):
    """Set the ordering method in the CHOLMOD common struct."""
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


cdef bint _check_perm(np.ndarray p, bint use_int32, cholmod_common *cm):
    """Check if the permutation array is valid."""
    p = np.ascontiguousarray(p)
    cdef size_t N = p.shape[0]

    cdef bint ok
    cdef int32_t[::1] p_mv_int32
    cdef int64_t[::1] p_mv_int64

    if use_int32:
        p_mv_int32 = p
        ok = cholmod_check_perm(&p_mv_int32[0], N, N, cm)
    else:
        p_mv_int64 = p
        ok = cholmod_l_check_perm(&p_mv_int64[0], N, N, cm)

    return ok

# -----------------------------------------------------------------------------
#         Cholesky and LDL Factorizations
# -----------------------------------------------------------------------------
def _cholesky_base(
    A, *, ldl=False, beta=None, lower=False, order=None, remove_zeros=True
):
    """Base function for Cholesky factorization."""
    A, use_int32, out_itype = validate_csc_input(A, require_square=True)

    N = A.shape[0]

    # Check the input ordering method
    if order is not None and order not in _ordering_methods:
        raise ValueError(f"Unknown ordering method: {order}")

    # Empty matrix
    if N == 0:
        R = csc_array((0, 0), dtype=A.dtype)
        p = np.array([], dtype=out_itype)
        if ldl:
            D = diags_array((0,), shape=(0, 0), dtype=A.dtype)
            return (R, D) if order is None else (R, D, p)
        else:
            return R if order is None else (R, p)

    # Matrix of all zeros
    if A.nnz == 0:
        raise CholmodNotPositiveDefiniteError("Input matrix not positive definite.")

    # -------------------------------------------------------------------------
    #         Set up Data Structures
    # -------------------------------------------------------------------------
    # Create the CHOLMOD common object
    cdef cholmod_common cm

    if use_int32:
        cholmod_start(&cm)
    else:
        cholmod_l_start(&cm)

    # Convert to packed LL.T when done
    cm.final_asis = False
    cm.final_super = False
    cm.final_ll = not ldl  # LL.T for Cholesky, LDL.T for LDL
    cm.final_pack = True
    cm.final_monotonic = True

    # TODO test if this is needed when we implement chol_update (see ldlchol.c)
    # If we do *not* drop numerically zero entries from the symbolic pattern,
    # we *do* need to drop entries that result from supernodal amalgamation.
    # Otherwise, all zeros are dropped in cholmod_drop, so save the extra step.
    cm.final_resymbol = not remove_zeros

    cm.quick_return_if_not_posdef = True

    _set_ordering_method(order, &cm)

    # Get the input matrix into CHOLMOD format
    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse *Ac = &Amatrix

    stype = -1 if lower else 1  # use lower or upper triangular part
    # Keep a reference to the input matrix to keep it alive
    cdef object ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)

    # Set stype and beta for LDL
    cdef double betac[2]

    if ldl:
        if beta is None:
            Amatrix.stype = -1  # lower triangular
            betac[0] = 0.0
            betac[1] = 0.0
        else:
            if not np.isscalar(beta):
                raise ValueError("beta must be a scalar value.")
            Amatrix.stype = 0  # symmetric, not triangular
            betac[0] = beta
            betac[1] = 0.0

    # -------------------------------------------------------------------------
    #         Analyze and Factorize
    # -------------------------------------------------------------------------
    cdef cholmod_factor* Lc

    if use_int32:
        Lc = cholmod_analyze(Ac, &cm)

        if ldl:
            cholmod_factorize_p(Ac, betac, NULL, 0, Lc, &cm)
        else:
            cholmod_factorize(Ac, Lc, &cm)
    else:
        Lc = cholmod_l_analyze(Ac, &cm)

        if ldl:
            cholmod_l_factorize_p(Ac, betac, NULL, 0, Lc, &cm)
        else:
            cholmod_l_factorize(Ac, Lc, &cm)

    # Check for errors
    _error_handler(cm.status)

    # -------------------------------------------------------------------------
    #         Convert to scipy csc_array
    # -------------------------------------------------------------------------
    # NOTE there is no need to keep "minor" here, since we just raise an error
    # if the matrix is not positive definite.
    cdef cholmod_sparse* Lsparse
    cdef cholmod_sparse* Rc

    if use_int32:
        Lsparse = cholmod_factor_to_sparse(Lc, &cm)
    else:
        Lsparse = cholmod_l_factor_to_sparse(Lc, &cm)

    if remove_zeros:
        # drop explicit zeros from Lsparse
        if use_int32:
            cholmod_drop(0, Lsparse, &cm)
        else:
            cholmod_l_drop(0, Lsparse, &cm)

    if lower:
        Rc = Lsparse
    else:
        # Convert to upper triangular (conjugate transpose)
        if use_int32:
            Rc = cholmod_transpose(Lsparse, CHOLMOD_TRANS_CONJ, &cm)
            cholmod_free_sparse(&Lsparse, &cm)
        else:
            Rc = cholmod_l_transpose(Lsparse, CHOLMOD_TRANS_CONJ, &cm)
            cholmod_l_free_sparse(&Lsparse, &cm)

    # -------------------------------------------------------------------------
    #         Create outputs
    # -------------------------------------------------------------------------
    R = _csc_from_cholmod_sparse(Rc, &cm)
    p = _array_from_cholmod_permutation(Lc, N, use_int32)

    # For LDL, we need to extract the diagonal matrix D
    if ldl:
        D = diags_array(R.diagonal())
        R.setdiag(1.0)  # set unit diagonal

    # Free everything else
    # NOTE there is no need to free Ac here, since it is just a pointer to the
    # original input matrix A. The MATLAB interface creates a *new*
    # cholmod_sparse object, so it needs to be freed.
    if use_int32:
        cholmod_free_factor(&Lc, &cm)
        cholmod_finish(&cm)
    else:
        cholmod_l_free_factor(&Lc, &cm)
        cholmod_l_finish(&cm)

    if ldl:
        return (R, D) if order is None else (R, D, p)
    else:
        return R if order is None else (R, p)


def cholesky(A, *, lower=False, order=None, remove_zeros=True):
    return _cholesky_base(
        A, ldl=False, lower=lower, order=order, remove_zeros=remove_zeros
    )


def ldl(A, beta=None, *, lower=True, order=None, remove_zeros=True):
    return _cholesky_base(
        A, ldl=True, beta=beta, lower=lower, order=order, remove_zeros=remove_zeros
    )


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
{beta_param}
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

lower : bool, optional
    If True, return the lower triangular factor `L`.
remove_zeros : bool, optional
    If False, do not remove explicit zeros from the factor ``L`` or ``R``.
    This flag allows use of the ``chol_update`` function afterwards.
    Default is True, so that the output is in canonical form.

Returns
-------
R : csc_array
    The triangular factor of the Cholesky decomposition. The data type will
    match that of ``A``.
{ldl_D_output}
p : ndarray of int, optional
    The permutation vector used in the factorization. Only returned if the
    ordering is not ``None``.

Raises
------
:exc:`CholmodNotPositiveDefiniteError`
    If the input matrix is not positive definite.

See Also
--------
{see_also}
* :func:`.cholmod` : Solve a linear system using the Cholesky factorization.
* :func:`.ldlsolve` : Solve a linear system using the LDL factorization.

Notes
-----
This function is an interface to the CHOLMOD library, which is part of
the SuiteSparse collection by Timothy A. Davis. For more details, see the
documentation in the header file [{doc_tag}]_.

References
----------
.. [{doc_tag}] ``cholmod.h`` - SuiteSparse CHOLMOD header file.
    https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Include/cholmod.h
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
"""


_cholesky_see_also = """* :func:`.ldl` : Factorize a matrix using LDL decomposition."""


cholesky.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_cholesky_intro,
    beta_param="",
    ldl_D_output="",
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

    L D L^{\\top} = P A A^{\\top} P^{\\top} + \\beta I,

where `I` is the identity matrix.
"""

_beta_param = """beta : float, optional
    The scalar value to add to the diagonal of the symmetrized matrix
    :math:`A A^{\\top}` before factorization. Default is None, which
    computes the factorization of :math:`A` itself."""


_ldl_D_output = """D : dia_array
    The diagonal matrix `D` of the factorization, in sparse DIA format.
    The data type will match that of ``A``."""


_ldl_see_also = """* :func:`.cholesky` : Factorize a matrix using Cholesky decomposition."""


ldl.__doc__ = _CHOLMOD_DOC_TEMPLATE.format(
    intro=_ldl_intro,
    beta_param=_beta_param,
    ldl_D_output=_ldl_D_output,
    see_also=_ldl_see_also,
    doc_tag="#ldl_h",
)


# -----------------------------------------------------------------------------
#         Solve Functions
# -----------------------------------------------------------------------------
def cholmod(A, b, *, order=None, p=None):
    """Solve a linear system using the Cholesky factorization.

    This function solves the linear system:

    .. math::

        R^{\\top} R x = b,

    where `R` is the upper triangular factor from the Cholesky factorization
    of `A`. The input `b` is either dense or sparse, vector or matrix.

    If ``order`` or ``p`` is provided, it is used as a permutation vector to
    solve the system:

    .. math::

        P^{\\top} R^{\\top} R P x = b

    where `P` is the permutation matrix corresponding to the permutation
    vector. ``order`` should be one of the methods supported by
    :func:`.cholesky`. If ``p`` is provided, it should be the permutation
    vector returned by the :func:`.cholesky` function with ``order != None``.
    Only one of ``order`` or ``p`` should be provided.

    Parameters
    ----------
    A : (N, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square, symmetric positive definite. Only the upper triangular part is
        used, and no check is made for symmetry.
    b : (N, K) sparray or ndarray
        The right-hand side vector or matrix.
    order : None or str in {"default", "best", "natural", "metis", "nesdis", \
            "amd", "colamd", "postordered"}, optional
        The permutation algorithm to use for the factorization. By default, the
        natural ordering of the input matrix is used. The other options are:

        * ``default``: Use the default method, which first tries AMD, then METIS.
        * ``best``: Automatically select the best ordering based on the input.
        * ``metis``: Use the METIS library for graph partitioning.
        * ``nesdis``: Use the NESDIS library for nested dissection.
        * ``amd``: Use the Approximate Minimum Degree (AMD) algorithm.
        * ``colamd``: Use the Approximate Minimum Degree (AMD) algorithm for the
            symmetric case, or the COLAMD algorithm for the unsymmetric case
            (:math:`A A^{\\top}` or :math:`A^{\\top} A`).
        * ``postordered``: Use natural ordering followed by postordering.

        By default, methods other than ``natural`` will also be postordered.

        .. warning::

            The ordering method ``best`` may be quite slow for large matrices.

    p : ndarray of int, optional
        The permutation vector used in the factorization. This may be the
        output of :func:`.cholesky` with ``order != None``. Only one of 
        ``order`` or ``p`` should be provided.

    Returns
    -------
    x : (N, K) coo_array, csc_array or ndarray
        The solution vector or matrix. The shape and type of the output matches
        that of the right-hand side ``b``.

    See Also
    --------
    * :func:`.cholesky` : Factorize a matrix using Cholesky decomposition.
    * :func:`.ldl` : Factorize a matrix using LDL decomposition.
    * :func:`.ldlsolve` : Solve a linear system using the LDL factorization.

    Notes
    -----
    This function uses the CHOLMOD library to solve the linear system. It is
    intended to replicate the MATLAB interface ``cholmod2.m`` [#cholmod_c]_.

    References
    ----------
    .. [#cholmod_c] ``cholmod2.c`` - CHOLMOD MATLAB interface
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/cholmod2.c
    """
    A, use_int32, _ = validate_csc_input(A, require_square=True)

    if b.ndim not in {1, 2}:
        raise ValueError("Right-hand side b must be a vector or matrix.")

    N = A.shape[0]
    K = b.shape[1] if b.ndim == 2 else 0
    
    if b.shape[0] != N:
        raise ValueError("Right-hand side b must have the same number of rows as A.")

    if order is not None:
        if p is not None:
            raise ValueError("Only one of 'order' or 'p' should be provided.")

        if order not in _ordering_methods:
            raise ValueError(f"Unknown ordering method: {order}")

    # Initialize the CHOLMOD common object
    cdef cholmod_common cm
    
    if use_int32:
        cholmod_start(&cm)
    else:
        cholmod_l_start(&cm)

    cm.final_ll = True
    cm.quick_return_if_not_posdef = True

    if p is not None:
        if not isinstance(p, np.ndarray) or p.shape != (N,):
            raise ValueError("Permutation vector p must be a 1D array of length N.")

        if not _check_perm(p, use_int32, &cm):
            raise ValueError("Permutation vector p is not valid.")

        p = np.ascontiguousarray(p)
        order = "given"

    _set_ordering_method(order, &cm)

    # -------------------------------------------------------------------------
    #         Special Cases
    # -------------------------------------------------------------------------
    # Empty matrix
    if N == 0:
        if issparse(b):
            return type(b)(b.shape, dtype=b.dtype)
        else:
            return np.empty_like(b)

    if A.nnz == 0:
        raise CholmodError("Input matrix A is empty.")

    # -------------------------------------------------------------------------
    #         Get the A matrix
    # -------------------------------------------------------------------------
    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix
    cdef int stype = 1  # use triu(A) only

    cdef object A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)

    # -------------------------------------------------------------------------
    #         Get the b vector or matrix into CHOLMOD format
    # -------------------------------------------------------------------------
    cdef cholmod_sparse Bspmatrix
    cdef cholmod_sparse* Bs = &Bspmatrix
    cdef cholmod_dense Bmatrix
    cdef cholmod_dense* Bd = &Bmatrix

    # CHOLMOD expects at least a column vector for the RHS
    if b.ndim == 1:
        if issparse(b):
            b = b.reshape((-1, 1)).tocsc()  # (N, 1)
        else:
            b = b[:, np.newaxis]  # (N, 1)

    cdef object b_ref  # keep a reference to b so it is not garbage collected

    if issparse(b):
        b, b_use_int32, _ = validate_csc_input(b)
        b_ref = _cholmod_sparse_from_csc(b, 0, b_use_int32, &Bspmatrix)
    else:
        b_ref = _cholmod_dense_from_ndarray(b, &Bmatrix)

    # -------------------------------------------------------------------------
    #         Analyze and Factorize the Matrix
    # -------------------------------------------------------------------------
    cdef int32_t[::1] Perm_mv_int32
    cdef int32_t* Perm_int32_ptr = NULL

    cdef int64_t[::1] Perm_mv_int64
    cdef int64_t* Perm_int64_ptr = NULL

    cdef cholmod_factor* Lc
    if use_int32:
        if p is not None:
            Perm_mv_int32 = p
            Perm_int32_ptr = &Perm_mv_int32[0]
        Lc = cholmod_analyze_p(Ac, Perm_int32_ptr, NULL, 0, &cm)
        cholmod_factorize(Ac, Lc, &cm)
    else:
        if p is not None:
            Perm_mv_int64 = p
            Perm_int64_ptr = &Perm_mv_int64[0]
        Lc = cholmod_l_analyze_p(Ac, Perm_int64_ptr, NULL, 0, &cm)
        cholmod_l_factorize(Ac, Lc, &cm)

    # Check the condition number
    cdef double rcond

    if use_int32:
        rcond = cholmod_rcond(Lc, &cm)
    else:
        rcond = cholmod_l_rcond(Lc, &cm)

    if rcond == 0:
        raise CholmodNotPositiveDefiniteError(
            "Matrix is indefinite or singular to working precision."
        )
    elif rcond < np.finfo(np.float64).eps:
        raise CholmodNotPositiveDefiniteError(
            "Matrix is nearly singular."
            f"  Results may be inaccurate (rcond={rcond:.2e})."
        )

    # -------------------------------------------------------------------------
    #         Solve the System
    # -------------------------------------------------------------------------
    cdef cholmod_sparse* Xs
    cdef cholmod_dense* Xd 

    if issparse(b):
        # Solve the sparse system
        if use_int32:
            Xs = cholmod_spsolve(CHOLMOD_A, Lc, Bs, &cm)
        else:
            Xs = cholmod_l_spsolve(CHOLMOD_A, Lc, Bs, &cm)

        X = _csc_from_cholmod_sparse(Xs, &cm)
    else:
        # Solve the dense system
        if use_int32:
            Xd = cholmod_solve(CHOLMOD_A, Lc, Bd, &cm)
        else:
            Xd = cholmod_l_solve(CHOLMOD_A, Lc, Bd, &cm)

        X = _ndarray_from_cholmod_dense(Xd, use_int32, &cm)

    # Convert to 1D array if input b is 1D
    if K == 0:
        X = X[:, 0]

    # TODO stats data structure

    # Free data
    if use_int32:
        cholmod_free_factor(&Lc, &cm)
        cholmod_finish(&cm)
    else:
        cholmod_l_free_factor(&Lc, &cm)
        cholmod_l_finish(&cm)

    return X


def ldlsolve(L, D, b, p=None):
    """Solve a linear system using the LDL factorization.

    This function solves the linear system:

    .. math::

        L D L^{\\top} x = b,

    where `L` is a lower triangular matrix with unit diagonal, and `D` is
    a diagonal matrix. The input `b` is either dense or sparse, vector or
    matrix.

    If ``p`` is provided, it is used as a permutation vector to solve the
    system:

    .. math::

        P^{\\top} L D L^{\\top} P x = b

    where `P` is the permutation matrix corresponding to the permutation
    vector. ``p`` should be the permutation vector returned by the :func:`ldl`
    function with ``order != None``.

    Parameters
    ----------
    L : (N, N) csc_array
        The lower triangular factor ``L`` from the LDL factorization, as
        computed by :func:`.ldl`.
    D : (N, N) dia_array
        The diagonal matrix ``D`` from the LDL factorization, as computed by
        :func:`.ldl`.
    b : (N, K) sparray or ndarray
        The right-hand side vector or matrix.
    p : (N,) ndarray of int, optional
        The permutation vector used in the factorization. If provided, it
        should be the same as the one returned by :func:`.ldl` with
        ``order != None``.

    Returns
    -------
    x : (N, K) coo_array, csc_array or ndarray
        The solution vector or matrix. The type of the output matches that
        of the right-hand side ``b``.

    See Also
    --------
    * :func:`.cholesky` : Factorize a matrix using Cholesky decomposition.
    * :func:`.cholmod` : Solve a linear system using the Cholesky factorization.
    * :func:`.ldl` : Factorize a matrix using LDL decomposition.

    Notes
    -----
    This function uses the CHOLMOD library to solve the linear system. It is
    intended to replicate the MATLAB interface ``ldlsolve.m`` [#ldlsolve_c]_.

    References
    ----------
    .. [#ldlsolve_c] ``ldlsolve.c`` - CHOLMOD MATLAB interface
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/ldlsolve.c
    """
    L, use_int32, _ = validate_csc_input(L, require_square=True)

    if not issparse(D):
        raise ValueError(f"Diagonal matrix D is type {type(D)}. "
                         "Expected a scipy.sparse.dia_array or similar.")

    if L.dtype != D.dtype:
        raise ValueError(
            f"Data types of L and D must match. Got {L.dtype} and {D.dtype}."
        )

    if L.dtype != b.dtype:
        raise ValueError(
            f"Data types of L and b must match. Got {L.dtype} and {b.dtype}."
        )

    if D.shape != L.shape:
        raise ValueError("Diagonal matrix D must match the size of L.")

    if b.ndim not in {1, 2}:
        raise ValueError("Right-hand side b must be a vector or matrix.")

    N = L.shape[0]
    K = b.shape[1] if b.ndim == 2 else 0
    
    if b.shape[0] != N:
        raise ValueError("Right-hand side b must have the same number of rows as L.")

    # Initialize the CHOLMOD common object
    cdef cholmod_common cm
    
    if use_int32:
        cholmod_start(&cm)
    else:
        cholmod_l_start(&cm)

    if p is not None:
        if not isinstance(p, np.ndarray) or p.shape != (N,):
            raise ValueError("Permutation vector p must be a 1D array of length N.")

        if not _check_perm(p, use_int32, &cm):
            raise ValueError("Permutation vector p is not valid.")

    # -------------------------------------------------------------------------
    #         Special Cases
    # -------------------------------------------------------------------------
    # Empty matrix
    if N == 0:
        if issparse(b):
            return type(b)(b.shape, dtype=b.dtype)
        else:
            return np.empty_like(b)

    if L.nnz == 0 or D.nnz == 0:
        raise CholmodError("Input matrix L or diagonal matrix D is empty.")

    # -------------------------------------------------------------------------
    #         Get the b vector or matrix into CHOLMOD format
    # -------------------------------------------------------------------------
    cdef cholmod_sparse Bspmatrix
    cdef cholmod_sparse* Bs = &Bspmatrix
    cdef cholmod_dense Bmatrix
    cdef cholmod_dense* Bd = &Bmatrix

    # CHOLMOD expects at least a column vector for the RHS
    if b.ndim == 1:
        if issparse(b):
            b = b.reshape((-1, 1)).tocsc()  # (N, 1)
        else:
            b = b[:, np.newaxis]  # (N, 1)

    if p is not None:
        b = b[p]  # apply the permutation to b

    cdef object b_ref  # keep a reference to b so it is not garbage collected

    if issparse(b):
        b, b_use_int32, _ = validate_csc_input(b)
        b_ref = _cholmod_sparse_from_csc(b, 0, b_use_int32, &Bspmatrix)
    else:
        b_ref = _cholmod_dense_from_ndarray(b, &Bmatrix)

    # -------------------------------------------------------------------------
    #         Create the CHOLMOD Factor from L and D
    # -------------------------------------------------------------------------
    cdef cholmod_factor* Lc

    if use_int32:
        Lc = cholmod_allocate_factor(N, &cm)
    else:
        Lc = cholmod_l_allocate_factor(N, &cm)

    # Combine the input L and D into a CHOLMOD factor
    LD = L.copy()
    LD.setdiag(D.diagonal())

    cdef object L_ref = _cholmod_factor_from_csc(LD, use_int32, Lc, &cm)

    # -------------------------------------------------------------------------
    #         Solve the System
    # -------------------------------------------------------------------------
    cdef cholmod_sparse* Xs
    cdef cholmod_dense* Xd 

    if issparse(b):
        # Solve the sparse system
        if use_int32:
            Xs = cholmod_spsolve(CHOLMOD_LDLt, Lc, Bs, &cm)
        else:
            Xs = cholmod_l_spsolve(CHOLMOD_LDLt, Lc, Bs, &cm)

        X = _csc_from_cholmod_sparse(Xs, &cm)
    else:
        # Solve the dense system
        if use_int32:
            Xd = cholmod_solve(CHOLMOD_LDLt, Lc, Bd, &cm)
        else:
            Xd = cholmod_l_solve(CHOLMOD_LDLt, Lc, Bd, &cm)

        X = _ndarray_from_cholmod_dense(Xd, use_int32, &cm)

    if p is not None:
        # Apply the inverse permutation to the solution before (possibly)
        # converting to a 1D COO array (not subscriptable)
        X = X[np.argsort(p)]

    # Convert to 1D array if input b is 1D
    if K == 0:
        X = X[:, 0]

    # Check the condition number of the solution
    cdef double rcond

    if use_int32:
        rcond = cholmod_rcond(Lc, &cm)
    else:
        rcond = cholmod_l_rcond(Lc, &cm)

    if rcond == 0:
        raise CholmodNotPositiveDefiniteError(
            "Matrix is indefinite or singular to working precision."
        )
    elif rcond < np.finfo(np.float64).eps:
        raise CholmodNotPositiveDefiniteError(
            "Matrix is nearly singular."
            f"  Results may be inaccurate (rcond={rcond:.2e})."
        )

    # Free memory
    Lc.p = NULL
    Lc.i = NULL
    Lc.x = NULL

    # NOTE there is no need to free Bspmatrix or Bmatrix here, since they
    # are just pointers to the original input b.
    if use_int32:
        cholmod_free_factor(&Lc, &cm)
        cholmod_finish(&cm)
    else:
        cholmod_l_free_factor(&Lc, &cm)
        cholmod_l_finish(&cm)

    return X
