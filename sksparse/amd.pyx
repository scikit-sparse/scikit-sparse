# cython: language_level=3

import numpy as np
cimport numpy as np

from scipy.sparse import issparse, csc_array

DEF CONTROL_SIZE = 5
DEF INFO_SIZE = 20

# TODO create python objects for control and info with more readable attributes
# than just an array


def amd(A, dense_thresh=None, aggressive=None, return_info=False):
    """Compute the approximate minimum degree ordering of a sparse matrix.

    Parameters
    ----------
    A : (N, N) array_like or sparse matrix
        A square matrix in CSC format or convertible to CSC.
    dense_thresh : float, optional
        Threshold number of entries for considering a row/column dense. If
        None, use the default value from AMD. The default value is 10.

        From the SuiteSparse `amd.h` documentation [0]_:

            A dense row/column in ``A + A.T`` can cause AMD to spend a lot of
            time in ordering the matrix. If ``dense_thresh >= 0``, rows/columns
            with more than ``dense_thresh * sqrt(N)`` entries are ignored
            during the ordering, and placed last in the output order. The
            default value of ``dense_thresh`` is 10. If negative, no
            rows/columns are treated as "dense". Rows/columns with 16 or fewer
            off-diagonal entries are never considered "dense".

        For more details, see the SuiteSparse homepage [1]_.
    aggressive : bool, optional
        If True, use aggressive absorption. If None, uses the default value
        from AMD. The default value is True.

        From the SuiteSparse `amd.h` documentation [0]_:

            Controls whether or not to use aggressive absorption, in which
            a prior element is absorbed into the current element if is a subset
            of the current element, even if it is not adjacent to the current
            pivot element (refer to Amestoy, Davis, & Duff, 1996, for more
            details). The default value is ``True``, which means to perform
            aggressive absorption. This nearly always leads to a better
            ordering (because the approximate degrees are more accurate) and
            a lower execution time. There are cases where it can lead to
            a slightly worse ordering, however.

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

    Notes
    -----
    This function wraps the AMD (Approximate Minimum Degree) algorithm from
    the SuiteSparse by Timothy A. Davis. For details, see the SuiteSparse
    repository [2]_.

    References
    ----------
    .. [0] `amd.h` - Source header file from SuiteSparse.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/AMD/Include/amd.h
    .. [1] SuiteSparse homepage.
        https://people.engr.tamu.edu/davis/suitesparse.html
    .. [2] SuiteSparse GitHub repository.
        https://github.com/DrTimothyAldenDavis/SuiteSparse
    """
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.atleast_2d(np.asarray(A))

    try: 
        A = csc_array(A)
    except ValueError:
        raise ValueError("Input must be convertible to CSC format.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("Input must be square.")

    N = A.shape[0]

    # Choose index width: int32 or int64
    use_int32 = A.indptr.dtype == np.int32 and A.indices.dtype == np.int32

    if N == 0:
        return np.empty(0, dtype=np.int32 if use_int32 else np.int64)

    if A.nnz == 0:
        return np.arange(N, dtype=np.int32 if use_int32 else np.int64)

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
    if dense_thresh is not None:
        ctrl[AMD_DENSE] = float(dense_thresh)

    if aggressive is not None:
        ctrl[AMD_AGGRESSIVE] = 1.0 if aggressive else 0.0

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
