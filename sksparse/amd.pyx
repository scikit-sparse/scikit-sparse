# cython: language_level=3

import numpy as np
cimport numpy as np

from dataclasses import dataclass
from scipy.sparse import issparse, csc_array

DEF CONTROL_SIZE = 5
DEF INFO_SIZE = 20


@dataclass(frozen=True)
class AMDInfo:
    """Information statistics returned by the AMD algorithm.

    This class wraps the contents of the `Info` array output by `amd_order()`
    into a Python dataclass.

    Attributes
    ----------
    status : int
        Return status: 
          - 0 = OK,
          - 1 = OK but jumbled,
          - -1 = out of memory,
          - -2 = invalid matrix.
    N : int
        Number of rows and columns of the input matrix ``A``.
    nz : int
        Number of nonzeros in the input matrix ``A``.
    symmetry : float in [0, 1]
        Symmetry of pattern of ``A``. The symmetry is the number of "matched"
        off-diagonal entries divided by the total number of off-diagonal
        entries. An entry ``A[i, j]`` is matched if ``A[j, i]`` is also an
        entry, for any pair ``[i, j]`` where ``i != j``. In python code:

        .. code::
            S = A.astype(bool)
            B = sparse.tril(S, -1) + sparse.triu(S, 1)
            symmetry = (B * B.T).nnz / B.nnz

    nzdiag : int
        Number of entries on the diagonal of ``A``.
    nz_A_plus_AT : int
        Number of nonzeros in ``A + A.T`` (excluding diagonal).
        If ``A`` is perfectly symmetric (``symmetry = 1``), with a fully
        non-zero diagonal, then ``nz_A_plus_AT = nz - N`` (the smallest
        possible value).
        If ``A`` is perfectly unsymmetric (``symmetry = 0``, for an upper
        triangular matrix, *e.g.*) with no diagonal, 
        then ``nz_A_plus_AT = 2 * nz`` (the largest possible value).
    Ndense : int
        Number of dense rows/columns ignored during ordering. These
        rows/columns are placed last in the output order ``p``.
    memory : float
        Memory used, in bytes. This is equal to:
        ``(1.2 * nz_A_plus_AT + 9 * N) * sizeof(int)``. This coefficient is at
        most ``2.4 * nz + 9 * N``. This accounting excludes the size of the
        input arguments ``Ap``, ``Ai``, and ``p``, which have a total size of
        ``nz + 2 * N + 1`` integers.
    Ncmpa : int
        Number of components in the matrix (excluding dense rows/columns).
    Lnz : int
        Number of nonzeros in the Cholesky factor ``L`` of ``A``, excluding
        the diagonal. This is a slight upper bound because of the approximate
        degree algorithm. It is a rough upper bound if there are many dense
        rows/columns. The remaining statistics are also slight or rough upper
        bounds for the same reason.
    Ndiv : int
        Number of division operations for LU or Cholesky factorization of the
        permuted matrix ``A[p][:, p]``.
    Nmultsubs_LDL : int
        Number of multiply-subtract pairs for ``LDL.T`` factorization.
    Nmultsubs_LU : int
        Number of multiply-subtract pairs for LU factorization, assuming that
        no numerical pivoting is required.
    dmax : int
        Maximum number of nonzeros in any column of ``L``, including the
        diagonal.

    Notes
    -----
    Field descriptions are adapted from SuiteSparse `amd.h` [0]_.

    References
    ----------
    .. [0]: `amd.h` - SuiteSparse AMD header file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/AMD/Include/amd.h
    """
    status: int
    N: int
    nz: int
    symmetry: float
    nzdiag: int
    nz_A_plus_AT: int
    Ndense: int
    memory: float
    Ncmpa: int
    Lnz: int
    Ndiv: int
    Nmultsubs_LDL: int
    Nmultsubs_LU: int
    dmax: int

    @classmethod
    def from_array(cls, info: "np.ndarray") -> "AMDInfo":
        return cls(
            status=int(info[AMD_STATUS]),
            N=int(info[AMD_N]),
            nz=int(info[AMD_NZ]),
            symmetry=float(info[AMD_SYMMETRY]),
            nzdiag=int(info[AMD_NZDIAG]),
            nz_A_plus_AT=int(info[AMD_NZ_A_PLUS_AT]),
            Ndense=int(info[AMD_NDENSE]),
            memory=float(info[AMD_MEMORY]),
            Ncmpa=int(info[AMD_NCMPA]),
            Lnz=int(info[AMD_LNZ]),
            Ndiv=int(info[AMD_NDIV]),
            Nmultsubs_LDL=int(info[AMD_NMULTSUBS_LDL]),
            Nmultsubs_LU=int(info[AMD_NMULTSUBS_LU]),
            dmax=int(info[AMD_DMAX]),
        )


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
        dump_info = AMDInfo.from_array(info)
        raise ValueError(f"amd: input matrix A is invalid:\n{dump_info}")

    if return_info:
        return p, info
    else:
        return p


def print_amd_default_control():
    """Print the default control parameters for AMD."""
    cdef double[::1] ctrl_mv = np.empty(CONTROL_SIZE, dtype=np.float64)
    amd_l_defaults(<double*>&ctrl_mv[0])
    amd_l_control(<double*>&ctrl_mv[0])
