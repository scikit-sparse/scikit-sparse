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
    "analyze",
    "bisect",
    "cholesky",
    "cholmod",
    "etree",
    "ldl",
    "ldlrowmod",
    "ldlsolve",
    "ldlupdate",
    "metis",
    "nesdis",
    "resymbol",
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

    # Create the index arrays
    if use_int32:
        Ap_mv_int32 = A_py.indptr
        Ai_mv_int32 = A_py.indices
        A.p = &Ap_mv_int32[0]
        # Handle empty matrices
        if Ai_mv_int32.shape[0] == 0 and (A.nrow == 0 or A.ncol == 0):
            A.i = <int32_t*>malloc(0)  # TODO needs to be freed
        else:
            A.i = &Ai_mv_int32[0]
    else:
        Ap_mv_int64 = A_py.indptr
        Ai_mv_int64 = A_py.indices
        A.p = &Ap_mv_int64[0]
        if Ai_mv_int64.shape[0] == 0:
            A.i = <int64_t*>malloc(0)
        else:
            A.i = &Ai_mv_int64[0]

    # Get the numerical values of A
    if dtype == np.bool_:
        A.xtype = CHOLMOD_PATTERN
        A.x = NULL
    else:
        A.xtype = _real_or_complex(dtype)

        if dtype == np.float32:
            Ax_mv_float32 = A_py.data
            if Ax_mv_float32.shape[0] == 0 and (A.nrow == 0 or A.ncol == 0):
                A.x = <float32_t*>malloc(0)
            else:
                A.x = &Ax_mv_float32[0]
        elif dtype == np.float64:
            Ax_mv_float64 = A_py.data
            if Ax_mv_float64.shape[0] == 0 and (A.nrow == 0 or A.ncol == 0):
                A.x = <float64_t*>malloc(0)
            else:
                A.x = &Ax_mv_float64[0]
        elif dtype == np.complex64:
            Ax_mv_complex64 = A_py.data
            if Ax_mv_complex64.shape[0] == 0 and (A.nrow == 0 or A.ncol == 0):
                A.x = <complex64_t*>malloc(0)
            else:
                A.x = &Ax_mv_complex64[0]
        elif dtype == np.complex128:
            Ax_mv_complex128 = A_py.data
            if Ax_mv_complex128.shape[0] == 0 and (A.nrow == 0 or A.ncol == 0):
                A.x = <complex128_t*>malloc(0)
            else:
                A.x = &Ax_mv_complex128[0]

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


cdef object _csc_from_cholmod_factor(object py_factor):
    """Build a sparse matrix from a CHOLMOD factor.

    This function is similar to _csc_from_cholmod_sparse, but builds the matrix
    directly from the factor, without the intermediate cholmod_factor_to_sparse
    call.

    Parameters
    ----------
    py_factor : CholeskyFactor
        The input cholmod_factor and cholmod_common objects, wrapped in
        a Python object.

    Returns
    -------
    res : csc_array
        L scipy.sparse.csc_array that is a view onto the CHOLMOD factor.

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
    cdef CholeskyFactor factor_obj = py_factor
    cdef cholmod_factor *L = factor_obj.factor
    cdef cholmod_common *common = factor_obj.cm

    if L is NULL:
        raise ValueError("The factor pointer is NULL.")

    # TODO handle below to return a symbolic factor
    if L.xtype == CHOLMOD_PATTERN:
        raise ValueError("The factor has no numerical values.")

    # Ensure the factor is in simplicial, packed, monotonic format
    cdef bint use_int32 = L.itype == CHOLMOD_INT

    is_super = False  # simplicial format
    is_packed = True
    is_monotonic = True

    change_factor = cholmod_change_factor if use_int32 else cholmod_l_change_factor
    change_factor(
        L.xtype, L.is_ll, is_super, is_packed, is_monotonic, L, common
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
        np.set_array_base(array, factor_obj)

    return csc_array((data, indices, indptr), shape=(L.n, L.n))


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

    if use_int32:
        LDp_mv_int32 = LD_py.indptr
        LDi_mv_int32 = LD_py.indices
        L.p = &LDp_mv_int32[0]
        L.i = &LDi_mv_int32[0]
    else:
        LDp_mv_int64 = LD_py.indptr
        LDi_mv_int64 = LD_py.indices
        L.p = &LDp_mv_int64[0]
        L.i = &LDi_mv_int64[0]

    # Get the data values
    L.itype = CHOLMOD_INT if use_int32 else CHOLMOD_LONG
    L.dtype = _single_or_double(dtype)
    L.xtype = _real_or_complex(dtype)

    if dtype == np.float32:
        LDx_mv_float32 = LD_py.data
        L.x = &LDx_mv_float32[0]
    elif dtype == np.float64:
        LDx_mv_float64 = LD_py.data
        L.x = &LDx_mv_float64[0]
    elif dtype == np.complex64:
        LDx_mv_complex64 = LD_py.data
        L.x = &LDx_mv_complex64[0]
    elif dtype == np.complex128:
        LDx_mv_complex128 = LD_py.data
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

    return LD_py


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


cdef object _ldlupdate_factor_from_csc(
    object LD_py,
    bint use_int32,
    int xtype,
    cholmod_factor *L_static,
    cholmod_common *cm
):
    """Create a CHOLMOD factor from a scipy.sparse.csc_array for ldlupdate.

    See the ``ldlupdate.c`` function in [#ldlupdate_ref]_.

    Parameters
    ----------
    LD_py : csc_array
        The input sparse matrix in Compressed Sparse Column (CSC) format. This
        should be a combination of the ``L`` and ``D`` factors computed from
        :func:`ldl`.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for indices and indptr.
    xtype : int in {CHOLMOD_REAL, CHOLMOD_COMPLEX}
        The functions :func:`.ldlupdate` and :func:`.ldlrowmod` only use
        CHOLMOD_REAL, whereas :func:`.resymbol` allows for real or complex.
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
    .. [#ldlupdate_ref] ``ldlupdate.c`` - CHOLMOD MATLAB utilities
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/ldlupdate.c
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

    # Cast pointers for arithmetic and memcpy operations
    cdef int32_t* Lp_int32
    cdef int32_t* Li_int32

    cdef int64_t* Lp_int64
    cdef int64_t* Li_int64

    cdef int32_t* ColCount_int32
    cdef int64_t* ColCount_int64

    cdef size_t j

    # Set the ColCount array
    if use_int32:
        LDp_mv_int32 = LD_py.indptr
        Lp_int32 = <int32_t*>&LDp_mv_int32[0]
        ColCount_int32 = <int32_t*>L.ColCount
        for j in range(N):
            ColCount_int32[j] = Lp_int32[j + 1] - Lp_int32[j]
    else:
        LDp_mv_int64 = LD_py.indptr
        Lp_int64 = <int64_t*>&LDp_mv_int64[0]
        ColCount_int64 = <int64_t*>L.ColCount
        for j in range(N):
            ColCount_int64[j] = Lp_int64[j + 1] - Lp_int64[j]

    # Allocate space for a CHOLMOD LDL.T packed factor
    cdef int to_xtype = xtype
    cdef int to_ll = False  # LDL.T
    cdef int to_super = False
    cdef int to_packed = True
    cdef int to_monotonic = True

    if use_int32:
        cholmod_change_factor(
            to_xtype, to_ll, to_super, to_packed, to_monotonic, L, cm
        )
    else:
        cholmod_l_change_factor(
            to_xtype, to_ll, to_super, to_packed, to_monotonic, L, cm
        )

    cdef size_t lnz = L.nzmax

    # Copy the data from LD_py to the CHOLMOD factor
    if use_int32:
        LDi_mv_int32 = LD_py.indices
        Lp_int32 = <int32_t*>L.p
        Li_int32 = <int32_t*>L.i
        memcpy(Lp_int32, &LDp_mv_int32[0], (N + 1) * sizeof(int32_t))
        memcpy(Li_int32, &LDi_mv_int32[0], lnz * sizeof(int32_t))
    else:
        LDi_mv_int64 = LD_py.indices
        Lp_int64 = <int64_t*>L.p
        Li_int64 = <int64_t*>L.i
        memcpy(Lp_int64, &LDp_mv_int64[0], (N + 1) * sizeof(int64_t))
        memcpy(Li_int64, &LDi_mv_int64[0], lnz * sizeof(int64_t))

    # Get the numerical values of LD
    cdef float32_t* Lx_float32
    cdef float64_t* Lx_float64
    cdef complex64_t* Lx_complex64
    cdef complex128_t* Lx_complex128

    if dtype == np.float32:
        LDx_mv_float32 = LD_py.data
        Lx_float32 = &LDx_mv_float32[0]
        memcpy(L.x, Lx_float32, lnz * sizeof(float32_t))
    elif dtype == np.float64:
        LDx_mv_float64 = LD_py.data
        Lx_float64 = &LDx_mv_float64[0]
        memcpy(L.x, Lx_float64, lnz * sizeof(float64_t))
    elif dtype == np.complex64:
        LDx_mv_complex64 = LD_py.data
        Lx_complex64 = &LDx_mv_complex64[0]
        memcpy(L.x, Lx_complex64, lnz * sizeof(complex64_t))
    elif dtype == np.complex128:
        LDx_mv_complex128 = LD_py.data
        Lx_complex128 = &LDx_mv_complex128[0]
        memcpy(L.x, Lx_complex128, lnz * sizeof(complex128_t))

    cdef int32_t* Lnz_int32
    cdef int64_t* Lnz_int64

    if use_int32:
        Lnz_int32 = <int32_t*>L.nz
        for j in range(N):
            Lnz_int32[j] = Lp_int32[j + 1] - Lp_int32[j]
    else:
        Lnz_int64 = <int64_t*>L.nz
        for j in range(N):
            Lnz_int64[j] = Lp_int64[j + 1] - Lp_int64[j]


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


cdef np.ndarray _ndarray_from_cholmod_intarray(void* ptr, size_t N, bint use_int32):
    """Create a NumPy array from the permutation vector in a CHOLMOD factor.

    Parameters
    ----------
    ptr : void*
        A pointer to the C array.
    N : size_t
        The size of the permutation vector.
    use_int32 : bool
        Whether to use 32-bit or 64-bit integers for the permutation indices.

    Returns
    -------
    p : ndarray
        The permutation vector as a NumPy array.
    """
    if ptr is NULL:
        raise ValueError("ptr is NULL, cannot get array")

    cdef int np_itypenum = np.NPY_INT32 if use_int32 else np.NPY_INT64
    cdef np.ndarray p = np.PyArray_SimpleNewFromData(1, [N], np_itypenum, ptr)
    # TODO set destructor base object
    # Return a copy in case ptr is freed
    return p.copy()


cdef np.ndarray _perm_from_cholmod_factor(object py_factor):
    """Create a NumPy array from the permutation vector in a CHOLMOD factor.

    Parameters
    ----------
    py_factor : CholeskyFactor
        The CholeskyFactor object from which to extract the permutation.

    Returns
    -------
    p : ndarray
        The permutation vector as a NumPy array.
    """
    cdef CholeskyFactor factor_obj = py_factor
    cdef cholmod_factor *L = factor_obj.factor

    if L is NULL:
        raise ValueError("The factor pointer is NULL.")

    if L.Perm is NULL:
        raise ValueError("The factor does not have a permutation.")

    cdef int np_itypenum = np.NPY_INT32 if L.itype == CHOLMOD_INT else np.NPY_INT64
    cdef np.ndarray p = np.PyArray_SimpleNewFromData(1, [L.n], np_itypenum, L.Perm)
    np.set_array_base(p, factor_obj)  # keep factor_obj alive

    return p


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
#         CholeskyFactor Object
# -----------------------------------------------------------------------------
cdef void _cleanup_factor(CholeskyFactor cf):
    """Deallocate memory used by a CholeskyFactor."""
    if cf.cm is not NULL:
        if cf.use_int32:
            if cf.factor is not NULL:
                cholmod_free_factor(&cf.factor, cf.cm)
            cholmod_finish(cf.cm)
        else:
            if cf.factor is not NULL:
                cholmod_l_free_factor(&cf.factor, cf.cm)
            cholmod_l_finish(cf.cm)


cdef class CholeskyFactor:
    """The main object used for creating and manipulating a Cholesky factor.

    Attributes
    ----------
    Common : cholmod_common
        CHOLMOD common structure for configuration and status.
    cm : cholmod_common*
        A pointer to ``Common``.
    factor : cholmod_factor*
        The underlying C data structure.
    use_int32 : bint
        Whether to use 32-bit or 64-bit integers.
    N : size_t
        The number of rows and columns in the factor.
    _beta : float or None
        The value added to the diagonal of :math:`A A^{\\top}` before
        factorization, or None if no value was added.
    is_lower : bool
        Whether the factor is lower triangular (True) or upper triangular
        (False).
    """

    cdef cholmod_common Common
    cdef cholmod_common *cm
    cdef cholmod_factor *factor
    cdef bint use_int32
    cdef size_t N
    cdef object _beta
    cdef bint is_lower

    def __cinit__(self, object A, object beta=None, int lower=False, object order=None):
        """Construct a CholeskyFactor from a sparse matrix.

        Parameters
        ----------
        A : (N, N) {{array_like, sparse array}}
            An array convertible to a sparse matrix in Compressed Sparse Column
            (CSC) format. The matrix must be square and symmetric positive
            definite. Only the upper or lower triangular part of the matrix is
            used, and no check is made for symmetry.
        beta : float, optional
            The scalar value to add to the diagonal of the symmetrized matrix
            :math:`A A^{\\top}` before factorization. Default is None, which
            computes the factorization of :math:`A` itself.
        order : None or str in {{"default", "best", "natural", "metis", \
                "nesdis", "amd", "colamd", "postordered"}}, optional
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
                unsymmetric case
                (:math:`A A^{{\\top}}` or :math:`A^{{\\top}} A`).
            * ``postordered``: Use natural ordering followed by postordering.

            By default, methods other than ``natural`` will also be
            postordered.

            .. warning::

                The ordering method ``best`` may be quite slow for large
                matrices, but if the factorization is reused many times, it can
                be worth it.

        lower : bool, optional
            If True, return the lower triangular factor `L`.
        """
        A, use_int32, _ = validate_csc_input(A, require_square=True)

        # Check the input ordering method
        if order is not None and order not in _ordering_methods:
            raise ValueError(f"Unknown ordering method: {order}")

        self.N = A.shape[0]
        self.use_int32 = use_int32

        # Matrix of all zeros
        if self.N > 0 and A.nnz == 0:
            raise CholmodNotPositiveDefiniteError("Input matrix not positive definite.")

        # Get the input matrix into CHOLMOD format
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix

        # Use lower or upper triangular part of A
        self.is_lower = lower
        cdef int stype = -1 if self.is_lower else 1

        # keep a reference to the input matrix
        cdef object _ref = _cholmod_sparse_from_csc(A, stype, self.use_int32, &Amatrix)

        # Analyze A @ A.T + beta*I if requested
        if beta is None:
            self._beta = None
        else:
            if not np.isscalar(beta):
                raise ValueError("beta must be a scalar value.")
            self._beta = float(beta)

        if self._beta is None:
            Ac.stype = -1  # use lower triangular part of A
        else:
            Ac.stype = 0   # use all of A, factorizing A @ A.T

        try:
            self.cm = &self.Common

            if self.use_int32:
                cholmod_start(self.cm)
            else:
                cholmod_l_start(self.cm)

            _set_ordering_method(order, self.cm)

            # Analyze the matrix, but do not factorize yet
            if self.use_int32:
                self.factor = cholmod_analyze(Ac, self.cm)
            else:
                self.factor = cholmod_l_analyze(Ac, self.cm)

            # Check for errors
            _handle_errors(self.cm.status)

        except Exception as e:
            _cleanup_factor(self)
            raise e

    def __dealloc__(self):
        """Deallocate memory used by the CholeskyFactor."""
        _cleanup_factor(self)

    # -------------------------------------------------------------------------
    #         Properties
    # -------------------------------------------------------------------------
    @property
    def is_ll(self):
        """Whether the factor is in LL.T form (True) or LDL.T form (False)."""
        if self.factor is NULL:
            raise ValueError("The factor pointer is NULL. Run `factorize` first.")
        return self.factor.is_ll

    @property
    def N(self):
        """The number of rows and columns in the factor."""
        if self.factor is NULL:
            raise ValueError("The factor pointer is NULL. Run `factorize` first.")
        return self.factor.n

    # -------------------------------------------------------------------------
    #         Public API
    # -------------------------------------------------------------------------
    def view_factor(self, kind=None):
        """Return a view of the Cholesky factor in the specified format.

        .. warning::

            The returned matrix or matrices are views on the internal data of
            the CHOLMOD factor. They will become invalid if the factor is
            modified (*e.g.*, by calling ``factorize``).

        Parameters
        ----------
        kind : None or str in {'LL', 'LDL'}, optional
            The type of factor to return. If ``LL``, return the Cholesky
            factor `L` such that :math:`L L^{\\top} = P A P^{\\top}`. If
            ``LDL``, return the combined `LD` factor such that
            :math:`L D L^{\\top} = P A P^{\\top}`. Default is None, which
            uses the kind with which ``factorize`` was called.

        Returns
        -------
        L : csc_array
            The Cholesky factor in Compressed Sparse Column (CSC) format. If
            ``kind="LL"``, the returned matrix is lower triangular. If
            ``kind="LDL"``, the returned matrix contains the lower triangular
            and the diagonal factors. The unit diagonal of `L` is not stored.
        """
        if kind is None:
            kind = "LL" if self.is_ll else "LDL"

        if kind not in ("LL", "LDL"):
            raise ValueError("kind must be 'LL' or 'LDL'.")

        self._convert_factor(kind)

        return _csc_from_cholmod_factor(self)

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

        Lv = self.view_factor(kind)

        # Drop explicit zeros from returned copies
        if kind == "LL":
            L = Lv.copy()
            L.eliminate_zeros()
            if not lower:
                L = L.T.conj()
            return L
        else:
            # Extract L and D from combined LD factor
            L = Lv.copy()
            D = diags_array(L.diagonal())
            L.setdiag(1.0)
            L.eliminate_zeros()
            if not lower:
                L = L.T.conj()
            return L, D

    def view_perm(self):
        """Return a view of permutation vector used in the factorization.

        Returns
        -------
        p : ndarray
            The permutation vector `p` such that :math:`P A P^{\\top}` is the
            matrix that was factorized, where `P` is the permutation matrix
            corresponding to `p`, *i.e.*, ``P = I[p]``.
        """
        return _perm_from_cholmod_factor(self)

    def get_perm(self):
        """Return a copy of the permutation vector used in the factorization.

        Returns
        -------
        p : ndarray
            The permutation vector `p` such that :math:`P A P^{\\top}` is the
            matrix that was factorized, where `P` is the permutation matrix
            corresponding to `p`, *i.e.*, ``P = I[p]``.
        """
        return self.view_perm().copy()

    def factorize(self, object A, object ldl=None, object beta=None, bint lower=False):
        """Compute the Cholesky factorization of a sparse matrix.

        This function computes the Cholesky factorization of a symmetric
        positive definite matrix `A`:

        .. math::

            R^{\\top} R = P A P^{\\top},

        where `R` is an upper triangular matrix. Only the upper triangular part
        of `A` is used. If ``lower`` is True, the lower triangular factor `L`
        is returned instead, such that:

        .. math::

            L L^{\\top} = P A P^{\\top}.

        In this case, only the lower triangular part of `A` is used.

        Parameters
        ----------
        A : (N, N) {{array_like, sparse array}}
            An array convertible to a sparse matrix in Compressed Sparse Column
            (CSC) format. The matrix must be square and symmetric positive
            definite. Only the upper or lower triangular part of the matrix is
            used, and no check is made for symmetry. This matrix can be
            numericaly different from the matrix used to initialize the
            :obj:`CholeskyFactor` object, but it must have the same sparsity
            pattern.
        ldl : None or bool, optional
            If True, compute the LDL factorization instead of the
            Cholesky factorization. Default is None, which uses the same type of
            factorization as the previous call to ``factorize``, or ``LL`` if
            this is the first call.
        beta : float, optional
            The scalar value to add to the diagonal of the symmetrized matrix
            :math:`A A^{\\top}` before factorization. Default is None, which
            computes the factorization of :math:`A` itself.
        lower : bool, optional
            If True, only use the lower triangular part of `A`. Default is
            False.
        """
        A, _, _ = validate_csc_input(A, require_square=True)

        if ldl is None:
            try:
                ldl = not self.is_ll  # use the existing factor type
            except ValueError:
                ldl = False  # default to LL if no factor exists yet

        if not isinstance(ldl, bool):
            raise ValueError("ldl must be a boolean value.")

        self.is_lower = lower

        # Convert to packed LL.T when done
        self.cm.final_asis = False
        self.cm.final_super = False
        self.cm.final_ll = not ldl  # LL.T for Cholesky, LDL.T for LDL
        self.cm.final_pack = True
        self.cm.final_monotonic = True

        # If we do *not* drop numerically zero entries from the symbolic
        # pattern, we *do* need to drop entries that result from supernodal
        # amalgamation. Otherwise, all zeros are dropped in cholmod_drop, so
        # save the extra step.
        self.cm.final_resymbol = True

        self.cm.quick_return_if_not_posdef = True

        # Get the input matrix into CHOLMOD format
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix

        stype = -1 if lower else 1  # use lower or upper triangular part
        # Keep a reference to the input matrix to keep it alive
        cdef object _ref = _cholmod_sparse_from_csc(A, stype, self.use_int32, &Amatrix)

        # Set stype and beta for LDL
        cdef double betac[2]

        if beta is None:
            b = self._beta
        else:
            if not np.isscalar(beta):
                raise ValueError("beta must be a scalar value.")
            b = float(beta)
            self._beta = b  # cache value

        if ldl:
            if b is None:
                Ac.stype = -1    # use lower triangular part of A
                betac[0] = 0.0
                betac[1] = 0.0
            else:
                Ac.stype = 0     # use all of A, factorizing A @ A.T
                betac[0] = b
                betac[1] = 0.0

        # Factorize the matrix
        if self.use_int32:
            if ldl:
                cholmod_factorize_p(Ac, betac, NULL, 0, self.factor, self.cm)
            else:
                cholmod_factorize(Ac, self.factor, self.cm)
        else:
            if ldl:
                cholmod_l_factorize_p(Ac, betac, NULL, 0, self.factor, self.cm)
            else:
                cholmod_l_factorize(Ac, self.factor, self.cm)

        # Check for errors
        _handle_errors(self.cm.status)

        return self  # for method chaining

    def solve(self, b):
        """Solve the linear system A x = b using the factorization.

        Parameters
        ----------
        b : (N,) or (N, K) ndarray or sparse matrix
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) ndarray
            The solution vector or matrix.
        """
        if self.factor is NULL:
            raise ValueError("The factor pointer is NULL. Run `factorize` first.")

        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")

        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        cdef size_t N = self.factor.n
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
            p = self.view_perm()
            X = X[np.argsort(p)]

        # Convert to 1D array if input b is 1D
        if K == 0:
            X = X[:, 0]

        return X

    # TODO docs from Modify/cholmod_updown.c on permutation of C.
    def update(self, C, updown="up"):
        """Multiple-rank update or downdate of a sparse LDL factorization.

        Update the Cholesky factorization of a sparse matrix `A`:

        .. math::

            L' D' L'^{\\top} = P A P^{\\top} \\pm C C^{\\top}

        where `L` is a lower triangular matrix with unit diagonal, and `D` is
        a diagonal matrix. The input ``C`` is a sparse matrix representing the
        update or downdate to the factorization. If ``updown == "up"``, the
        factorization is updated (+ sign), otherwise it is downdated (- sign).

        Parameters
        ----------
        C : (N, K) csc_array
            The sparse matrix representing the rank-`k` update or downdate to
            the factorization.
        update : str in {"up", "down"}, optional
            If ``up``, perform an update to the factorization. If ``down``,
            perform a downdate. Default is ``up``.

        Returns
        -------
        CholeskyFactor
            The current object, for method chaining.

        .. versionadded:: 0.5.0
        """
        if self.factor is NULL:
            raise ValueError("The factor pointer is NULL. Run `factorize` first.")

        if updown not in ("up", "down"):
            raise ValueError("updown must be 'up' or 'down'.")

        if not issparse(C) or C.ndim not in {1, 2}:
            raise ValueError(f"Update matrix C is type {type(C)}. "
                             "Expected a 1D or 2D sparse array.")

        cdef size_t N = self.factor.n

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

        # Permute C so it is accepted in "matrix" space 
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

        if self.use_int32:
            C_perm = cholmod_submatrix(
                Cc, <int32_t*>self.factor.Perm, N, NULL, -1, True, True, self.cm
            )
        else:
            C_perm = cholmod_l_submatrix(
                Cc, <int64_t*>self.factor.Perm, N, NULL, -1, True, True, self.cm
            )

        # Ensure the factor is in LDL form
        self._convert_factor("LDL")

        # Compute the update or downdate
        cdef int update = updown == "up"
        cdef int ok

        if self.use_int32:
            ok = cholmod_updown(update, C_perm, self.factor, self.cm)
        else:
            ok = cholmod_l_updown(update, C_perm, self.factor, self.cm)

        if self.use_int32:
            cholmod_free_sparse(&C_perm, self.cm)
        else:
            cholmod_l_free_sparse(&C_perm, self.cm)

        if not ok:
            raise CholmodError("Update or downdate failed.")

        return self

    # -------------------------------------------------------------------------
    #         Private API
    # -------------------------------------------------------------------------
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
            p = self.view_perm()
            b = b[p]

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

        if self.use_int32:
            Xs = cholmod_spsolve(system, self.factor, Bs, self.cm)
        else:
            Xs = cholmod_l_spsolve(system, self.factor, Bs, self.cm)

        return _csc_from_cholmod_sparse(Xs, self.cm)

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
            p = self.view_perm()
            b = b[p]

        # keep a reference to b so it is not garbage collected
        cdef object _b_ref = _cholmod_dense_from_ndarray(b, &Bmatrix)

        # Check the condition number before solving
        self._check_rcond()

        # Solve the system
        cdef cholmod_dense* Xd

        cdef int system = CHOLMOD_A if self.is_ll else CHOLMOD_LDLt

        if self.use_int32:
            Xd = cholmod_solve(system, self.factor, Bd, self.cm)
        else:
            Xd = cholmod_l_solve(system, self.factor, Bd, self.cm)

        return _ndarray_from_cholmod_dense(Xd, self.use_int32, self.cm)

    cdef object _convert_factor(self, object kind):
        """Convert the factor to the desired form in-place.

        Parameters
        ----------
        kind : str in {'LL', 'LDL'}
            The desired form of the factor. If ``'LL'``, convert to LL.T form.
            If ``'LDL'``, convert to LDL.T form.

        Returns
        -------
        CholeskyFactor
            The current object, for method chaining.
        """
        if kind not in ("LL", "LDL"):
            raise ValueError("kind must be 'LL' or 'LDL'.")

        to_xtype = self.factor.xtype
        to_ll = kind == "LL"

        # NOTE In CHOLMOD, supernodal factorizations are always LL.T. If we
        # request to change to a supernodal LDL.T factorization,
        # cholmod_change_factor will silently do nothing! So we can only stay
        # supernodal when LL.T is requested.
        to_super = self.factor.is_super and kind == "LL"

        to_packed = True
        to_monotonic = self.factor.is_monotonic

        if (kind == "LL" and not self.factor.is_ll) or (
            kind == "LDL" and self.factor.is_ll
        ):
            # Convert LDL to LL
            if self.use_int32:
                change_factor = cholmod_change_factor
            else:
                change_factor = cholmod_l_change_factor

            change_factor(
                to_xtype,
                to_ll,
                to_super,
                to_packed,
                to_monotonic,
                self.factor,
                self.cm
            )

        return self

    cdef void _check_rcond(self):
        """Check the condition number."""
        cdef double rcond
        cdef double eps = np.finfo(np.float64).eps

        if self.use_int32:
            rcond = cholmod_rcond(self.factor, self.cm)
        else:
            rcond = cholmod_l_rcond(self.factor, self.cm)

        if rcond == 0:
            raise CholmodNotPositiveDefiniteError(
                "Matrix is indefinite or singular to working precision."
            )
        elif rcond < eps:
            raise CholmodNotPositiveDefiniteError(
                "Matrix is nearly singular."
                f"  Results may be inaccurate (rcond={rcond:.2e})."
            )


# Interface functions
def cho_factor(A, *, lower=False, order=None):
    return CholeskyFactor(A, lower=lower, order=order).factorize(
        A, ldl=False, lower=lower
    )


def ldl_factor(A, beta=None, *, lower=True, order=None):
    return CholeskyFactor(A, beta=beta, lower=lower, order=order).factorize(
        A, ldl=True, beta=beta, lower=lower
    )


# -----------------------------------------------------------------------------
#         Cholesky and LDL Factorizations
# -----------------------------------------------------------------------------
def _cholesky_base(A, *, ldl=False, beta=None, lower=False, order=None):
    """Base function for Cholesky factorization."""
    f = CholeskyFactor(A, beta=beta, lower=lower, order=order).factorize(
        A, ldl=ldl, beta=beta, lower=lower
    )
    kind = "LDL" if ldl else "LL"
    R = f.get_factor(kind=kind, lower=lower)
    p = f.get_perm()

    if ldl:
        R, D = R
        return (R, D) if order is None else (R, D, p)
    else:
        return R if order is None else (R, p)


def cholesky(A, *, lower=False, order=None):
    return _cholesky_base(A, ldl=False, lower=lower, order=order)


def ldl(A, beta=None, *, lower=True, order=None):
    return _cholesky_base(A, ldl=True, beta=beta, lower=lower, order=order)


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

.. versionadded:: 0.5.0

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


_ldl_see_also = "* :func:`.cholesky` : Factorize a matrix using Cholesky decomposition."


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

    .. versionadded:: 0.5.0

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

    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)

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

    .. versionadded:: 0.5.0

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

    if (not (isinstance(b, np.ndarray) or issparse(b)) or b.ndim not in {1, 2}):
        raise ValueError("Right-hand side b must be a vector or matrix.")

    N = L.shape[0]
    K = b.shape[1] if b.ndim == 2 else 0

    if b.shape[0] != N:
        raise ValueError("Right-hand side b must have the same number of rows as L.")

    # Initialize the CHOLMOD common object
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    if p is not None:
        if not isinstance(p, np.ndarray) or p.shape != (N,):
            raise ValueError("Permutation vector p must be a 1D array of length N.")

        if not _check_perm(p, use_int32, cm):
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
    cdef int stype = 0  # symmetric

    if issparse(b):
        b, b_use_int32, _ = validate_csc_input(b)
        b_ref = _cholmod_sparse_from_csc(b, stype, b_use_int32, &Bspmatrix)
    else:
        b_ref = _cholmod_dense_from_ndarray(b, &Bmatrix)

    # -------------------------------------------------------------------------
    #         Create the CHOLMOD Factor from L and D
    # -------------------------------------------------------------------------
    cdef cholmod_factor* Lc

    if use_int32:
        Lc = cholmod_allocate_factor(N, cm)
    else:
        Lc = cholmod_l_allocate_factor(N, cm)

    # Combine the input L and D into a CHOLMOD factor
    LD = L.copy()
    LD.setdiag(D.diagonal())

    cdef object _LD_ref = _cholmod_factor_from_csc(LD, use_int32, Lc, cm)

    # -------------------------------------------------------------------------
    #         Solve the System
    # -------------------------------------------------------------------------
    cdef cholmod_sparse* Xs
    cdef cholmod_dense* Xd

    if issparse(b):
        # Solve the sparse system
        if use_int32:
            Xs = cholmod_spsolve(CHOLMOD_LDLt, Lc, Bs, cm)
        else:
            Xs = cholmod_l_spsolve(CHOLMOD_LDLt, Lc, Bs, cm)

        X = _csc_from_cholmod_sparse(Xs, cm)
    else:
        # Solve the dense system
        if use_int32:
            Xd = cholmod_solve(CHOLMOD_LDLt, Lc, Bd, cm)
        else:
            Xd = cholmod_l_solve(CHOLMOD_LDLt, Lc, Bd, cm)

        X = _ndarray_from_cholmod_dense(Xd, use_int32, cm)

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
        rcond = cholmod_rcond(Lc, cm)
    else:
        rcond = cholmod_l_rcond(Lc, cm)

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
        cholmod_free_factor(&Lc, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free_factor(&Lc, cm)
        cholmod_l_finish(cm)

    return X


def ldlupdate(L, D, C, *, update=True):
    """Multiple-rank update or downdate of a sparse LDL factorization.

    Update the Cholesky factorization of a sparse matrix `A`:

    .. math::

        L' D' L'^{\\top} = P A P^{\\top} \\pm C C^{\\top}

    where `L` is a lower triangular matrix with unit diagonal, and `D` is
    a diagonal matrix. The input ``C`` is a sparse matrix representing the
    update or downdate to the factorization. If ``update`` is True, the
    factorization is updated (+ sign), otherwise it is downdated (- sign).

    Parameters
    ----------
    L : (N, N) csc_array
        The lower triangular factor `L` from the LDL factorization, as
        computed by :func:`.ldl`.
    D : (N, N) dia_array
        The diagonal matrix `D` from the LDL factorization, as computed by
        :func:`.ldl`.
    C : (N, K) csc_array
        The sparse matrix representing the rank-`k` update or downdate to the
        factorization. The number of rows must match that of ``L`` and ``D``.
    update : bool, optional
        If True, perform an update to the factorization. If False, perform a
        downdate. Default is True.

    Returns
    -------
    L' : (N, N) csc_array
        The updated lower triangular factor `L'` of the LDL factorization.
    D' : (N, N) dia_array
        The updated diagonal matrix `D'` of the LDL factorization.

    .. versionadded:: 0.5.0
    """
    L, use_int32, _ = validate_csc_input(L, require_square=True)

    if not issparse(D):
        raise ValueError(f"Diagonal matrix D is type {type(D)}. "
                         "Expected a scipy.sparse.dia_array or similar.")

    if L.dtype != D.dtype:
        raise ValueError(
            f"Data types of L and D must match. Got {L.dtype} and {D.dtype}."
        )

    if D.shape != L.shape:
        raise ValueError("Diagonal matrix D must match the size of L.")

    if not issparse(C) or C.ndim not in {1, 2}:
        raise ValueError(f"Update matrix C is type {type(C)}. "
                         "Expected a 1D or 2D sparse array.")

    N = L.shape[0]

    if C.shape[0] != N:
        raise ValueError("Update matrix C must have the same number of rows as L.")

    # -------------------------------------------------------------------------
    #         Special Cases
    # -------------------------------------------------------------------------
    # Empty matrix
    if N == 0:
        return L.copy(), D.copy()

    if L.nnz == 0 or D.nnz == 0:
        raise CholmodError("Input matrix L or diagonal matrix D is empty.")

    # -------------------------------------------------------------------------
    #         Get the Inputs
    # -------------------------------------------------------------------------
    # Initialize the CHOLMOD common object
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    # Ensure C is in CSC format
    if C.ndim == 1:
        C = C.reshape((-1, 1)).tocsc()  # (N, 1)

    cdef cholmod_sparse Cmatrix
    cdef cholmod_sparse* Cc = &Cmatrix

    cdef object _C_ref  # keep a reference to C so it is not garbage collected
    C, C_use_int32, _ = validate_csc_input(C)

    cdef int stype = 0  # use all of C
    _C_ref = _cholmod_sparse_from_csc(C, stype, C_use_int32, &Cmatrix)

    # Get a factor from the L and D matrices
    LD = L.copy()
    LD.setdiag(D.diagonal())

    cdef cholmod_factor* Lc

    if use_int32:
        Lc = cholmod_allocate_factor(N, cm)
    else:
        Lc = cholmod_l_allocate_factor(N, cm)

    _ldlupdate_factor_from_csc(LD, use_int32, CHOLMOD_REAL, Lc, cm)

    # -------------------------------------------------------------------------
    #         Compute the Update
    # -------------------------------------------------------------------------
    cdef int ok

    if use_int32:
        ok = cholmod_updown(update, Cc, Lc, cm)
    else:
        ok = cholmod_l_updown(update, Cc, Lc, cm)

    if not ok:
        raise CholmodError("Update or downdate failed.")

    # Re-separate the updated L and D from the factorization
    cdef cholmod_sparse* LDsparse

    # Get the packed LD sparse matrix
    if use_int32:
        LDsparse = cholmod_factor_to_sparse(Lc, cm)
    else:
        LDsparse = cholmod_l_factor_to_sparse(Lc, cm)

    # -------------------------------------------------------------------------
    #         Return the Updated Factors
    # -------------------------------------------------------------------------
    L = _csc_from_cholmod_sparse(LDsparse, cm)
    D = diags_array(L.diagonal())
    L.setdiag(1.0)

    # Free the CHOLMOD factor
    if use_int32:
        cholmod_free_factor(&Lc, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free_factor(&Lc, cm)
        cholmod_l_finish(cm)

    return L, D


def ldlrowmod(L, D, k, *, C=None):
    """Add or delete a row from a sparse LDL factorization.

    This function computes a rank-one update of a sparse LDL factorization. It
    either "adds" a row by setting the :math:`k^{th}` row and column of the
    original matrix to ``C``, or "deletes" a row by setting the :math:`k^{th}`
    row and column of the original matrix to the identity.

    Parameters
    ----------
    L : (N, N) csc_array
        The lower triangular factor `L` from the LDL factorization, as
        computed by :func:`.ldl`.
    D : (N, N) dia_array
        The diagonal matrix `D` from the LDL factorization, as computed by
        :func:`.ldl`.
    k : int
        The row/column index to modify. Must be in the range ``0 <= k < N``.
    C : (N, 1) csc_array, optional
        If given, change the factorization such that row and column ``k`` of
        the original matrix equal ``C``. The number of rows must match that of
        ``L`` and ``D``.

    Returns
    -------
    L' : (N, N) csc_array
        The updated lower triangular factor `L'` of the LDL factorization.
    D' : (N, N) dia_array
        The updated diagonal matrix `D'` of the LDL factorization.

    .. versionadded:: 0.5.0
    """
    L, use_int32, _ = validate_csc_input(L, require_square=True)

    if not issparse(D):
        raise ValueError(f"Diagonal matrix D is type {type(D)}. "
                         "Expected a scipy.sparse.dia_array or similar.")

    if L.dtype != D.dtype:
        raise ValueError(
            f"Data types of L and D must match. Got {L.dtype} and {D.dtype}."
        )

    if D.shape != L.shape:
        raise ValueError("Diagonal matrix D must match the size of L.")

    N = L.shape[0]

    if not (0 <= k < N):
        raise ValueError(
            f"Row index k={k} is out of bounds for matrix of size {N}."
        )

    if C is not None:
        if not issparse(C) or C.ndim not in {1, 2}:
            raise ValueError(
                f"Update matrix C is type {type(C)}. Expected a 1D or 2D sparse array."
            )

        if C.shape[0] != N:
            raise ValueError("Update matrix C must have the same number of rows as L.")

    # -------------------------------------------------------------------------
    #         Special Cases
    # -------------------------------------------------------------------------
    # Empty matrix
    if N == 0:
        return L.copy(), D.copy()

    if L.nnz == 0 or D.nnz == 0:
        raise CholmodError("Input matrix L or diagonal matrix D is empty.")

    # -------------------------------------------------------------------------
    #         Get the Inputs
    # -------------------------------------------------------------------------
    # Initialize the CHOLMOD common object
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    rowadd = C is not None

    cdef cholmod_sparse Cmatrix
    cdef cholmod_sparse* Cc = &Cmatrix
    cdef object _C_ref  # keep a reference to C so it is not garbage collected
    cdef int stype = 0  # use all of C

    if rowadd:
        # Ensure C is in CSC format
        if C.ndim == 1:
            C = C.reshape((-1, 1)).tocsc()  # (N, 1)

        C, C_use_int32, _ = validate_csc_input(C)
        _C_ref = _cholmod_sparse_from_csc(C, stype, C_use_int32, &Cmatrix)

    # Get a factor from the L and D matrices
    LD = L
    LD.setdiag(D.diagonal())

    cdef cholmod_factor* Lc

    if use_int32:
        Lc = cholmod_allocate_factor(N, cm)
    else:
        Lc = cholmod_l_allocate_factor(N, cm)

    _ldlupdate_factor_from_csc(LD, use_int32, CHOLMOD_REAL, Lc, cm)

    # -------------------------------------------------------------------------
    #         Compute the Update
    # -------------------------------------------------------------------------
    cdef int ok

    if rowadd:
        if use_int32:
            ok = cholmod_rowadd(k, Cc, Lc, cm)
        else:
            ok = cholmod_l_rowadd(k, Cc, Lc, cm)
    else:
        if use_int32:
            ok = cholmod_rowdel(k, NULL, Lc, cm)
        else:
            ok = cholmod_l_rowdel(k, NULL, Lc, cm)

    if not ok:
        raise CholmodError("ldlrowmod failed.")

    # Re-separate the updated L and D from the factorization
    cdef cholmod_sparse* LDsparse

    # Get the packed LD sparse matrix
    if use_int32:
        LDsparse = cholmod_factor_to_sparse(Lc, cm)
    else:
        LDsparse = cholmod_l_factor_to_sparse(Lc, cm)

    # -------------------------------------------------------------------------
    #         Return the Updated Factors
    # -------------------------------------------------------------------------
    L = _csc_from_cholmod_sparse(LDsparse, cm)
    D = diags_array(L.diagonal())
    L.setdiag(1.0)

    # Free the CHOLMOD factor
    if use_int32:
        cholmod_free_factor(&Lc, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free_factor(&Lc, cm)
        cholmod_l_finish(cm)

    return L, D


# -----------------------------------------------------------------------------
#         Symbolic Functions
# -----------------------------------------------------------------------------
def analyze(A, *, kind=None, order=None):
    """Order and analyze a sparse matrix for Cholesky or LDL factorization.

    This function performs the symbolic analysis of a sparse matrix `A` for
    either Cholesky or LDL factorization. It computes a fill-reducing
    permutation and analyzes the sparsity pattern of the matrix [#analyze_c]_.

    Parameters
    ----------
    A : (N, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square and symmetric.
    kind : str in {"sym", "row", "col"}, optional
        The type of factorization for which to analyze the matrix:

        * ``sym``: Symmetric factorization. Only the lower triangular part of
          ``A`` is used, and no check is made for symmetry.
        * ``row``: Unsymmetric factorization of :math:`A A^{\\top}`.
        * ``col``: Unsymmetric factorization of :math:`A^{\\top} A`.

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

    Returns
    -------
    p : (N,) ndarray of int
        The permutation vector such that the factorization is performed on
        :math:`P A P^{\\top}` (or the corresponding symmetric version), where
        `P` is the permutation matrix corresponding to the permutation vector.
    count : (N,) ndarray of int
        The count of nonzeros in each column of the Cholesky factor.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#analyze_c] ``analyze.c`` - CHOLMOD MATLA analyze function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/analyze.c
    """
    A, use_int32, out_itype = validate_csc_input(A, require_square=True)

    if kind is None:
        kind = "sym"

    if kind not in {"sym", "row", "col"}:
        raise ValueError(f"Unknown factorization kind: {kind}")

    if order is None:
        order = "default"

    if order not in _ordering_methods:
        raise ValueError(f"Unknown ordering method: {order}")

    N = A.shape[0]

    # Special Cases
    if N == 0:
        empty = np.array([], dtype=out_itype)
        return empty, empty

    if A.nnz == 0:
        return np.arange(N, dtype=out_itype), np.zeros(N, dtype=out_itype)

    # -------------------------------------------------------------------------
    #         Get the CHOLMOD Inputs
    # -------------------------------------------------------------------------
    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    cm.supernodal = CHOLMOD_SIMPLICIAL

    # TODO support all ordering methods -- see analyze.m
    _set_ordering_method(order, cm)

    cdef int stype = -1
    cdef bint transpose = False

    if kind in ["row", "col"]:
        stype = 0
        transpose = (kind == "col")

    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix
    cdef cholmod_sparse* C
    cdef cholmod_factor* Lc

    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)

    # -------------------------------------------------------------------------
    #         Analyze and Order
    # -------------------------------------------------------------------------
    if transpose:
        if use_int32:
            C = cholmod_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            Lc = cholmod_analyze(C, cm)
            cholmod_free_sparse(&C, cm)
        else:
            C = cholmod_l_transpose(Ac, CHOLMOD_TRANS_PATTERN, cm)
            Lc = cholmod_l_analyze(C, cm)
            cholmod_l_free_sparse(&C, cm)
    else:
        if use_int32:
            Lc = cholmod_analyze(Ac, cm)
        else:
            Lc = cholmod_l_analyze(Ac, cm)

    # -------------------------------------------------------------------------
    #         Return Results
    # -------------------------------------------------------------------------
    p = _ndarray_from_cholmod_intarray(Lc.Perm, N, use_int32)
    count = _ndarray_from_cholmod_intarray(Lc.ColCount, N, use_int32)

    # Free memory
    if use_int32:
        cholmod_free_factor(&Lc, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free_factor(&Lc, cm)
        cholmod_l_finish(cm)

    return p, count


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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)
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
    count = _ndarray_from_cholmod_intarray(ColCount, N, use_int32)

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

    parent = _ndarray_from_cholmod_intarray(Parent, N, use_int32)
    post = _ndarray_from_cholmod_intarray(Post, N, use_int32)

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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)
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
    parent = _ndarray_from_cholmod_intarray(Parent, N, use_int32)

    if return_post:
        if use_int32:
            Post = cholmod_malloc(N, sizeof(int32_t), cm)
            if cholmod_postorder(<int32_t*>Parent, N, NULL, <int32_t*>Post, cm) != N:
                raise CholmodError("Postordering failed.")
        else:
            Post = cholmod_l_malloc(N, sizeof(int64_t), cm)
            if cholmod_l_postorder(<int64_t*>Parent, N, NULL, <int64_t*>Post, cm) != N:
                raise CholmodError("Postordering failed.")

        post = _ndarray_from_cholmod_intarray(Post, N, use_int32)

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


def resymbol(L, A):
    """Recompute the symbolic Cholesky factorization of a sparse matrix.

    This function is useful after a series of downdates via :func:`.ldlupdate`
    or :func:`.ldlrowmod`, since downdates do not remove any entries in ``L``
    [#resymbol_c]_.

    Parameters
    ----------
    L : (N, N) csc_array
        The lower triangular factor ``L`` from the LDL factorization, as
        computed by :func:`.cholesky` or :func:`.ldl`.
    A : (N, N) csc_array
        The input matrix in Compressed Sparse Column (CSC) format. Must be
        square and symmetric. Only the lower triangular part of ``A`` is used,
        and no check is made for symmetry. The numerical values of ``A`` are
        ignored. Only its non-zero pattern is used.

    Returns
    -------
    L : (N, N) csc_array
        The updated lower triangular factor.

    See Also
    --------
    :func:`.cholesky`, :func:`.ldl`, :func:`.ldlupdate`, :func:`.ldlrowmod`

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#resymbol_c] ``resymbol.c`` - CHOLMOD MATLAB resymbolization function
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/MATLAB/resymbol.c
    """
    A, use_int32, _ = validate_csc_input(A, require_square=True)
    L, _, _ = validate_csc_input(L, require_square=True)

    N = A.shape[0]

    if L.shape != (N, N):
        raise ValueError(
            "Input matrix L must be square and match the size of A. "
            f"Got shape {L.shape}."
        )

    if use_int32 and L.indptr.dtype != np.int32:
        raise ValueError(
            "A and L must have the same integer type. "
            f"Got {A.indptr.dtype=}, and {L.indptr.dtype=}."
        )

    # Special Cases
    if N == 0:
        return L.copy()

    if A.nnz == 0:
        raise CholmodNotPositiveDefiniteError("Input matrix not positive definite.")

    cdef cholmod_common Common
    cdef cholmod_common *cm = &Common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    # Get sparse *pattern*
    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse* Ac = &Amatrix
    cdef int stype = -1  # use tril(A) only

    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)
    Ac.xtype = CHOLMOD_PATTERN
    Ac.x = NULL

    # Get a factor from the L matrix
    cdef cholmod_factor* Lc

    if use_int32:
        Lc = cholmod_allocate_factor(N, cm)
    else:
        Lc = cholmod_l_allocate_factor(N, cm)

    cdef int xtype = _real_or_complex(A.dtype)

    _ldlupdate_factor_from_csc(L, use_int32, xtype, Lc, cm)

    # -------------------------------------------------------------------------
    #         Resymbolic Factorization
    # -------------------------------------------------------------------------
    if use_int32:
        cholmod_resymbol(Ac, NULL, 0, True, Lc, cm)
    else:
        cholmod_l_resymbol(Ac, NULL, 0, True, Lc, cm)

    # Convert back to a CSC array
    cdef cholmod_sparse* Lsparse

    if use_int32:
        Lsparse = cholmod_factor_to_sparse(Lc, cm)
    else:
        Lsparse = cholmod_l_factor_to_sparse(Lc, cm)

    L = _csc_from_cholmod_sparse(Lsparse, cm)

    if use_int32:
        cholmod_free_factor(&Lc, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free_factor(&Lc, cm)
        cholmod_l_finish(cm)

    return L


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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)
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
    s = _ndarray_from_cholmod_intarray(Partition, N, use_int32)

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

        cdef bint use_int32 = (
            self._cp.dtype == np.int32 and self._cmember.dtype == np.int32
        )

        cdef cholmod_common Common
        cdef cholmod_common *cm = &Common

        if use_int32:
            cholmod_start(cm)
        else:
            cholmod_l_start(cm)

        cdef size_t Nc = self._cp.size
        cdef size_t N = self._cmember.size

        # Copy input arrays into new cholmod arrays (modified for output)
        cdef void *CParent
        cdef void *CMember

        cdef int32_t[::1] cp_mv_int32, cmember_mv_int32
        cdef int64_t[::1] cp_mv_int64, cmember_mv_int64

        # TODO could do checks of each value in a for-loop here
        if use_int32:
            cp_mv_int32 = self._cp
            cmember_mv_int32 = self._cmember
            CParent = cholmod_malloc(Nc, sizeof(int32_t), cm)
            CMember = cholmod_malloc(N, sizeof(int32_t), cm)
            memcpy(<int32_t*>CParent, &cp_mv_int32[0], Nc * sizeof(int32_t))
            memcpy(<int32_t*>CMember, &cmember_mv_int32[0], N * sizeof(int32_t))
        else:
            cp_mv_int64 = self._cp
            cmember_mv_int64 = self._cmember
            CParent = cholmod_l_malloc(Nc, sizeof(int64_t), cm)
            CMember = cholmod_l_malloc(N, sizeof(int64_t), cm)
            memcpy(<int64_t*>CParent, &cp_mv_int64[0], Nc * sizeof(int64_t))
            memcpy(<int64_t*>CMember, &cmember_mv_int64[0], N * sizeof(int64_t))

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
        cp_out = _ndarray_from_cholmod_intarray(CParent, nc_new, use_int32)
        cmember_out = _ndarray_from_cholmod_intarray(CMember, N, use_int32)

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


# TODO get defaults?
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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)
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
    p = _ndarray_from_cholmod_intarray(Perm, N, use_int32)
    cp = _ndarray_from_cholmod_intarray(CParent, ncomp, use_int32)
    cmember = _ndarray_from_cholmod_intarray(CMember, N, use_int32)

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
    cdef object _A_ref = _cholmod_sparse_from_csc(A, stype, use_int32, &Amatrix)
    Ac.xtype = CHOLMOD_PATTERN
    Ac.x = NULL

    # -------------------------------------------------------------------------
    #         Compute the Outputs
    # -------------------------------------------------------------------------
    cdef void *Perm
    cdef cholmod_sparse *C
    cdef bint postorder = True  # TODO accept options inputs
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
    p = _ndarray_from_cholmod_intarray(Perm, N, use_int32)

    # Free memory (arrays are copied to numpy)
    if use_int32:
        cholmod_free(N, sizeof(int32_t), Perm, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free(N, sizeof(int64_t), Perm, cm)
        cholmod_l_finish(cm)

    return p
