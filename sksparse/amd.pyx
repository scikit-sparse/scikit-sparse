# cython: language_level=3

import numpy as np
cimport numpy as np

from scipy.sparse import issparse, csc_array

DEF CONTROL_SIZE = 5
DEF INFO_SIZE = 20

# TODO create python objects for control and info with more readable attributes
# than just an array


def amd(A, control=None, return_info=False):
    """Compute the approximate minimum degree ordering of a sparse matrix.

    Parameters
    ----------
    A : (N, N) array_like or sparse matrix
        A square matrix in CSC format or convertible to CSC.
    control : array_like, optional
        Control parameters for AMD. If not provided, default control is used.
    return_info : bool, optional
        If True, returns additional information about the ordering process.
        Default is False.

    Returns
    -------
    p : ndarray
        The permutation vector such that the Cholesky factor of ``A[p][:, p]``
        has fewer nonzeros than the Cholesky factor of ``A``.
    info : ndarray, optional
        Additional information about the ordering process, returned if
        ``return_info`` is True. Contains various statistics and status codes.
    """
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.atleast_2d(np.asarray(A))

    try: 
        A = csc_array(A)
    except ValueError:
        raise ValueError("amd: input must be convertible to CSC format")

    if A.shape[0] != A.shape[1]:
        raise ValueError("amd: A must be square")

    N = A.shape[0]

    # Choose index width: int32 or int64
    use_int32 = A.indptr.dtype == np.int32 and A.indices.dtype == np.int32

    if N == 0:
        return np.empty(0, dtype=np.int32 if use_int32 else np.int64)

    # Declare typed memory views for Cython
    cdef int[::1] Ap_mv_int32
    cdef int[::1] Ai_mv_int32
    cdef int[::1] p_mv_int32
    cdef long long[::1] Ap_mv_int64
    cdef long long[::1] Ai_mv_int64
    cdef long long[::1] p_mv_int64

    # Always ensure arrays are contiguous and correct dtype
    if use_int32:
        Ap_mv_int32 = np.ascontiguousarray(A.indptr, dtype=np.int32)
        Ai_mv_int32 = np.ascontiguousarray(A.indices, dtype=np.int32)
        p = p_mv_int32 = np.empty(N, dtype=np.int32)
    else:
        Ap_mv_int64 = np.ascontiguousarray(A.indptr, dtype=np.int64)
        Ai_mv_int64 = np.ascontiguousarray(A.indices, dtype=np.int64)
        p = p_mv_int64 = np.empty(N, dtype=np.int64)

    # Prepare control parameters
    ctrl = np.empty(CONTROL_SIZE, dtype=np.float64)
    cdef double[::1] ctrl_mv = ctrl

    if use_int32:
        amd_defaults(<double*>&ctrl_mv[0])
    else:
        amd_l_defaults(<double*>&ctrl_mv[0])

    # Update the defaults with user control parameters
    if control is not None:
        user_ctrl = np.ascontiguousarray(control, dtype=np.float64)

        for i in range(CONTROL_SIZE):
            if i < user_ctrl.shape[0]:
                ctrl[i] = user_ctrl[i]

    info = np.zeros(INFO_SIZE, dtype=np.float64)
    cdef double[::1] info_mv = info

    # AMD ordering
    if use_int32:
        status = amd_order(
            N,
            <int*>&Ap_mv_int32[0],
            <int*>&Ai_mv_int32[0],
            <int*>&p_mv_int32[0],
            <double*>&ctrl_mv[0],
            <double*>&info_mv[0]
        )
    else:
        status = amd_l_order(
            N,
            <long long*>&Ap_mv_int64[0],
            <long long*>&Ai_mv_int64[0],
            <long long*>&p_mv_int64[0],
            <double*>&ctrl_mv[0],
            <double*>&info_mv[0]
        )

    if status == AMD_OUT_OF_MEMORY:
        raise MemoryError("amd: out of memory")
    elif status == AMD_INVALID:
        raise ValueError("amd: input matrix A is corrupted")

    if return_info:
        return p, info
    else:
        return p


def print_amd_default_control():
    """Print the default control parameters for AMD."""
    cdef double[::1] ctrl_mv = np.empty(CONTROL_SIZE, dtype=np.float64)
    amd_l_defaults(<double*>&ctrl_mv[0])
    amd_l_control(<double*>&ctrl_mv[0])
