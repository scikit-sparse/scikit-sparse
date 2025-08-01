# Cython COLAMD python interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: colamd.pyx
#  Created: 2025-07-31 10:13
# =============================================================================

"""sksparse.colamd: Cython interface to COLAMD, a column approximate minimum
degree ordering algorithm.

This module provides a Cython interface to the COLAMD algorithm from the
SuiteSparse library by Timothy A. Davis. The algorithm computes a column
ordering for sparse matrices that is suitable for various numerical
factorizations, such as LU and QR.

Interfaces
----------
* `colamd`: Function to compute the column ordering of any shape sparse matrix.

This wrapper handles both 32-bit and 64-bit integer indices, depending on the
input matrix format.

References
----------
* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse COLAMD:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/COLAMD
* COLAMD Algorithm Publications:
  -	T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, An approximate column
	minimum degree ordering algorithm, ACM Transactions on Mathematical
	Software, vol. 30, no. 3., pp. 353-376, 2004.
  -	T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836: COLAMD,
	an approximate column minimum degree ordering algorithm, ACM
	Transactions on Mathematical Software, vol. 30, no. 3., pp. 377-380,
	2004.
"""

import numpy as np
cimport numpy as np

import warnings

from dataclasses import dataclass
from scipy.sparse import csc_array, issparse, SparseEfficiencyWarning


class COLAMDError(Exception):
    """Base class for COLAMD errors."""
    pass


class COLAMDValueError(COLAMDError, ValueError):
    """Raised when COLAMD encounters a value error."""
    pass


class COLAMDMemoryError(COLAMDError, MemoryError):
    """Raised when COLAMD runs out of memory."""
    pass


class COLAMDInternalError(COLAMDError, RuntimeError):
    """Raised when COLAMD encounters an internal error."""
    pass


# Define COLAMD error codes
_COLAMD_ERROR_CODES = dict({
    COLAMD_OK: "ok",
    COLAMD_OK_BUT_JUMBLED: "ok but A has unsorted columns or duplicate entries",
    COLAMD_ERROR_A_not_present: "A is a null pointer",
    COLAMD_ERROR_p_not_present: "p is a null pointer",
    COLAMD_ERROR_nrow_negative: "nrow is negative",
    COLAMD_ERROR_ncol_negative: "ncol is negative",
    COLAMD_ERROR_nnz_negative: "nnz is negative",
    COLAMD_ERROR_p0_nonzero: "p[0] is nonzero",
    COLAMD_ERROR_A_too_small: "A is too small",
    COLAMD_ERROR_col_length_negative: "column has a negative number of entries",
    COLAMD_ERROR_row_index_out_of_bounds: "row index out of bounds",
    COLAMD_ERROR_out_of_memory: "out of memory",
    COLAMD_ERROR_internal_error: "internal error"
})


@dataclass(frozen=True)
class COLAMDStats:
    """Information statistics returned by the COLAMD algorithm.

    This class wraps the contents of the ``stats`` array returned by
    C ``colamd()`` into a Python dataclass.

    Attributes
    ----------
    N_rows_ignored : int
        The number of dense or empty rows ignored in the ordering.
    N_cols_ignored : int
        The number of dense or empty columns ignored in the ordering.
    Ncmpa : int
        The number of garbage collections performed.
    status : int
        Status code indicating the result of the COLAMD operation. If non-zero,
        ``colamd`` will throw an appropriate exception that interprets this
        status code.

    The following fields take on different meanings depending on the value of
    ``status``:

    info1 : int
        Value of ``status``:

        * 0: the highest numbered column that is unsorted or has
          duplicate entries.
        * -3: the value of ``n_row``.
        * -4: the value of ``n_col``.
        * -5: the value of ``nnz == p[n_col]``.
        * -6: the value of ``p[0]``.
        * -7: the required ``Alen`` value.
        * -8: the column with negative entries.
        * -9: the column with a row index out of bounds.
    info2 : int
        Value of ``status``:

        * 0: the last seen duplicate or unsorted row index.
        * -7: the actual ``Alen`` value.
        * -9: the bad row index.
    info3 : int
        Value of ``status``:

        * 0: the number of duplicates or unsorted row indices.
        * -9: ``n_row``.

    Notes
    -----
    Field descriptions are adapted from SuiteSparse ``colamd.c``
    [#colamd_fields]_.

    References
    ----------
    .. [#colamd_fields] ``colamd.c`` - SuiteSparse AMD source file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/COLAMD/Source/colamd.c
    """
    N_rows_ignored : int
    N_cols_ignored : int
    Ncmpa : int
    status : int
    info1 : int
    info2 : int
    info3 : int

    @classmethod
    def from_array(cls, stats: "np.ndarray") -> "COLAMDStats":
        """Create a COLAMDStats instance from an array."""
        return cls(
            N_rows_ignored=int(stats[COLAMD_DENSE_ROW]),
            N_cols_ignored=int(stats[COLAMD_DENSE_COL]),
            Ncmpa=int(stats[COLAMD_DEFRAG_COUNT]),
            status=int(stats[COLAMD_STATUS]),
            info1=int(stats[COLAMD_INFO1]),
            info2=int(stats[COLAMD_INFO2]),
            info3=int(stats[COLAMD_INFO3]),
        )


def colamd(
    A, 
    dense_row_thresh=None, 
    dense_col_thresh=None, 
    aggressive=None, 
    return_info=False
):
    """Compute the column approximate minimum degree ordering of a sparse matrix.

    Adapted from the COLAMD documentation [#colamd]_:

        This function computes a column ordering for a sparse matrix `A` that
        is appropriate for LU factorization of symmetric or unsymmetric
        matrices, QR factorization, least squares, interior point methods for
        linear programming problems, and other related problems.

        COLAMD computes a permutation `Q` such that the Cholesky factorization
        of :math:`(AQ)^{\\top}(AQ)` has less fill-in and requires fewer floating
        point operations than :math:`A^{\\top}A`.  This also provides a good
        ordering for sparse partial pivoting methods, :math:`P(AQ) = LU`, where
        `Q` is computed prior to numerical factorization, and `P` is computed
        during numerical factorization via conventional partial pivoting with
        row interchanges.

    Parameters
    ----------
    A : {array_like, sparse matrix}
        The input matrix for which to compute the column ordering.
        Must be 2D and convertible to CSC format. Need not be square.
    dense_row_thresh, dense_col_thresh : float, optional
        Threshold for considering a row/column dense. If
        None, use the default value from COLAMD. The default value is 10.
        The actual number of entries in a row/column is to be considered
        "dense" is ``max(dense_row_thresh * sqrt(M), 16)`` where ``M`` is the
        number of rows (or ``N`` for columns). Dense rows/columns are ignored
        during ordering and moved to the end of the matrix.
    aggressive : bool, optional
        If True, use aggressive absorption. If None, uses the default value
        from COLAMD. The default value is True. 

        See the :func:`sksparse.amd.amd` documentation for more details on
        aggressive absorption.
    return_info : bool, optional
        If True, also return the COLAMD statistics.

    Returns
    -------
    q : ndarray
        The permutation array such that ``A[:, q]`` is the column ordered matrix.
    stats : ndarray, optional
        If ``return_info`` is True, returns an array containing COLAMD statistics.
        The contents of this array depend on the COLAMD implementation and may
        include information such as the number of nonzeros, memory usage, etc.

    References
    ----------
    .. [#colamd] ``colamd.c`` - SuiteSparse AMD source file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/COLAMD/Source/colamd.c
    """
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be 2D.")

    try:
        if not isinstance(A, csc_array):
            warnings.warn(
                "Input matrix is not in CSC format. Converting to CSC.",
                SparseEfficiencyWarning,
                stacklevel=2
            )
            A = csc_array(A)
    except ValueError:
        raise ValueError("Input must be convertible to CSC format.")

    M, N = A.shape

    # Choose index width: int32 or int64
    use_int32 = A.indptr.dtype == np.int32 and A.indices.dtype == np.int32
    out_dtype = np.int32 if use_int32 else np.int64

    if M == 0 or N == 0:
        return np.empty(0, dtype=out_dtype)

    if A.nnz == 0 or M == 1:
        return np.arange(N, dtype=out_dtype)

    if N == 1:
        return np.zeros(N, dtype=out_dtype)

    # Get the recommended size for the Alen array
    if use_int32:
        Alen = colamd_recommended(A.nnz, M, N)
    else:
        Alen = colamd_l_recommended(A.nnz, M, N)

    if Alen == 0:
        raise ValueError("Recommended Alen is zero: one of {A.nnz, M, N} is erroneous.")

    # Set the default knobs
    knobs = np.zeros(COLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_mv = knobs
    colamd_set_defaults(&knobs_mv[0])

    # Override with user knobs if provided
    if dense_row_thresh is not None:
        knobs[COLAMD_DENSE_ROW] = float(dense_row_thresh)

    if dense_col_thresh is not None:
        knobs[COLAMD_DENSE_COL] = float(dense_col_thresh)

    if aggressive is not None:
        knobs[COLAMD_AGGRESSIVE] = 1.0 if aggressive else 0.0

    # Declare typed memory views for Cython
    cdef int32_t[::1] Ai_mv_int32
    cdef int64_t[::1] Ai_mv_int64

    cdef int32_t[::1] p_mv_int32
    cdef int64_t[::1] p_mv_int64

    cdef int32_t[::1] stats_mv_int32
    cdef int64_t[::1] stats_mv_int64

    # Compute the ordering
    if use_int32:
        # Copy the arrays, since they are altered in the C function
        workspace = np.zeros(Alen, dtype=np.int32, order='C')
        workspace[:A.nnz] = A.indices.copy()
        Ai_mv_int32 = workspace
        p_mv_int32 = np.array(A.indptr, dtype=np.int32, copy=True, order='C')
        stats = stats_mv_int32 = np.zeros(COLAMD_STATS, dtype=np.int32)
        ok = c_colamd(
            M, 
			N, 
			Alen, 
			&Ai_mv_int32[0], 
			&p_mv_int32[0], 
			&knobs_mv[0], 
			&stats_mv_int32[0]
        )
    else:
        # Copy the arrays, since they are altered in the C function
        workspace = np.zeros(Alen, dtype=np.int64, order='C')
        workspace[:A.nnz] = A.indices.copy()
        Ai_mv_int64 = workspace
        p_mv_int64 = np.array(A.indptr, dtype=np.int64, copy=True, order='C')
        stats = stats_mv_int64 = np.zeros(COLAMD_STATS, dtype=np.int64)
        ok = c_colamd_l(
            M, 
			N, 
			Alen, 
			&Ai_mv_int64[0], 
			&p_mv_int64[0], 
			&knobs_mv[0], 
			&stats_mv_int64[0]
        )

    # Check the return status
    if ok:
        assert stats[COLAMD_STATUS] == COLAMD_OK, \
            "COLAMD returned OK but status is not COLAMD_OK."
    else:
        if stats[COLAMD_STATUS] == COLAMD_ERROR_out_of_memory:
            raise COLAMDMemoryError("COLAMD ran out of memory.")
        elif stats[COLAMD_STATUS] == COLAMD_ERROR_internal_error:
            raise COLAMDInternalError("COLAMD encountered an internal error.")
        else:
            raise COLAMDValueError(
                f"COLAMD returned an error:{_COLAMD_ERROR_CODES[stats[COLAMD_STATUS]]}."
            )

    # Only take the first N entries of the permutation array
    if use_int32:
        q_slice = p_mv_int32[:N]
    else:
        q_slice = p_mv_int64[:N]

    q = np.asarray(q_slice)

    if return_info:
        return q, COLAMDStats.from_array(stats)
    else:
        return q


# TODO implement, but then refactor the generic code
def symamd(
    A, 
    dense_row_thresh=None, 
    dense_col_thresh=None, 
    aggressive=None, 
    return_info=False
):
    """Compute the column approximate minimum degree ordering of a sparse matrix.

    Adapted from the COLAMD documentation [#symamd]_:

        This function computes an approximate minimum degree ordering for
        Cholesky factorization of symmetric matrices.

        Symamd computes a permutation `P` of a symmetric matrix `A` such that
        the Cholesky factorization of :math:`PAP^{\\top}` has less fill-in and
        requires fewer floating point operations than `A`.  Symamd constructs
        a matrix `M` such that :math:`M^{\\top}M` has the same nonzero pattern
        of `A`, and then orders the columns of `M` using colamd.  The column
        ordering of `M` is then returned as the row and column ordering `P` of
        `A`. 

    Parameters
    ----------
    A : {array_like, sparse matrix}
        The input matrix for which to compute the column ordering.
        Must be 2D and convertible to CSC format. Need not be square.
    dense_row_thresh, dense_col_thresh : float, optional
        Threshold for considering a row/column dense. If
        None, use the default value from COLAMD. The default value is 10.
        The actual number of entries in a row/column is to be considered
        "dense" is ``max(dense_row_thresh * sqrt(M), 16)`` where ``M`` is the
        number of rows (or ``N`` for columns). Dense rows/columns are ignored
        during ordering and moved to the end of the matrix.
    aggressive : bool, optional
        If True, use aggressive absorption. If None, uses the default value
        from COLAMD. The default value is True. 

        See the :func:`sksparse.amd.amd` documentation for more details on
        aggressive absorption.
    return_info : bool, optional
        If True, also return the COLAMD statistics.

    Returns
    -------
    p : ndarray
        The permutation array such that ``A[p][:, p]`` is the ordered matrix.
    stats : ndarray, optional
        If ``return_info`` is True, returns an array containing COLAMD statistics.
        The contents of this array depend on the COLAMD implementation and may
        include information such as the number of nonzeros, memory usage, etc.

    References
    ----------
    .. [#symamd] ``colamd.c`` - SuiteSparse AMD source file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/COLAMD/Source/colamd.c
    """
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be 2D.")

    M, N = A.shape

    if M != N:
        raise ValueError("Input matrix must be square.")

    try:
        if not isinstance(A, csc_array):
            warnings.warn(
                "Input matrix is not in CSC format. Converting to CSC.",
                SparseEfficiencyWarning,
                stacklevel=2
            )
            A = csc_array(A)
    except ValueError:
        raise ValueError("Input must be convertible to CSC format.")

    # Choose index width: int32 or int64
    use_int32 = A.indptr.dtype == np.int32 and A.indices.dtype == np.int32
    out_dtype = np.int32 if use_int32 else np.int64

    if N == 0:
        return np.empty(0, dtype=out_dtype)

    if A.nnz == 0:
        return np.arange(N, dtype=out_dtype)

    if N == 1:
        return np.zeros(N, dtype=out_dtype)

    # Get the recommended size for the Alen array
    if use_int32:
        Alen = colamd_recommended(A.nnz, M, N)
    else:
        Alen = colamd_l_recommended(A.nnz, M, N)

    if Alen == 0:
        raise ValueError("Recommended Alen is zero: one of {A.nnz, M, N} is erroneous.")

    # Set the default knobs
    knobs = np.zeros(COLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_mv = knobs
    colamd_set_defaults(&knobs_mv[0])

    # Override with user knobs if provided
    if dense_row_thresh is not None:
        knobs[COLAMD_DENSE_ROW] = float(dense_row_thresh)

    if dense_col_thresh is not None:
        knobs[COLAMD_DENSE_COL] = float(dense_col_thresh)

    if aggressive is not None:
        knobs[COLAMD_AGGRESSIVE] = 1.0 if aggressive else 0.0

    # Declare typed memory views for Cython
    cdef int32_t[::1] Ai_mv_int32
    cdef int64_t[::1] Ai_mv_int64

    cdef int32_t[::1] p_mv_int32
    cdef int64_t[::1] p_mv_int64

    cdef int32_t[::1] perm_mv_int32
    cdef int64_t[::1] perm_mv_int64

    cdef int32_t[::1] stats_mv_int32
    cdef int64_t[::1] stats_mv_int64

    # Compute the ordering
    if use_int32:
        # Copy the arrays, since they are altered in the C function
        Ai_mv_int32  = np.array(A.indices, dtype=np.int32, copy=True, order='C')
        p_mv_int32 = np.array(A.indptr, dtype=np.int32, copy=True, order='C')
        perm_mv_int32 = np.zeros(N + 1, dtype=np.int32, order='C')
        stats = stats_mv_int32 = np.zeros(COLAMD_STATS, dtype=np.int32)
        ok = c_symamd(
			N, 
			&Ai_mv_int32[0], 
			&p_mv_int32[0], 
            &perm_mv_int32[0],
			&knobs_mv[0], 
			&stats_mv_int32[0],
            calloc,
            free
        )
    else:
        # Copy the arrays, since they are altered in the C function
        Ai_mv_int64  = np.array(A.indices, dtype=np.int64, copy=True, order='C')
        p_mv_int64 = np.array(A.indptr, dtype=np.int64, copy=True, order='C')
        perm_mv_int64 = np.zeros(N + 1, dtype=np.int64, order='C')
        stats = stats_mv_int64 = np.zeros(COLAMD_STATS, dtype=np.int64)
        ok = c_symamd_l(
			N, 
			&Ai_mv_int64[0], 
			&p_mv_int64[0], 
            &perm_mv_int64[0],
			&knobs_mv[0], 
			&stats_mv_int64[0],
            calloc,
            free
        )

    # Check the return status
    if ok:
        assert stats[COLAMD_STATUS] == COLAMD_OK, \
            "COLAMD returned OK but status is not COLAMD_OK."
    else:
        if stats[COLAMD_STATUS] == COLAMD_ERROR_out_of_memory:
            raise COLAMDMemoryError("COLAMD ran out of memory.")
        elif stats[COLAMD_STATUS] == COLAMD_ERROR_internal_error:
            raise COLAMDInternalError("COLAMD encountered an internal error.")
        else:
            raise COLAMDValueError(
                f"COLAMD returned an error:{_COLAMD_ERROR_CODES[stats[COLAMD_STATUS]]}."
            )

    # Only take the first N entries of the permutation array
    if use_int32:
        q_slice = perm_mv_int32[:N]
    else:
        q_slice = perm_mv_int64[:N]

    q = np.asarray(q_slice)

    if return_info:
        return q, COLAMDStats.from_array(stats)
    else:
        return q


def colamd_get_defaults():
    """Get the default knobs for COLAMD.

    Returns
    -------
    knobs : dict
        A dictionary containing the default knobs for COLAMD.

        The keys are:

        * 'dense_row_thresh': Threshold for considering a row/column dense.
          Rows with more than ``max(dense_row_thresh * sqrt(M), 16)`` entries
          are permuted to the end of the matrix.
        * 'dense_col_thresh': Like `dense_row_thresh`, but for columns.
        * 'aggressive': Default value for the aggressive knob.

    """
    knobs = np.zeros(COLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_mv = knobs
    colamd_set_defaults(&knobs_mv[0])
    return dict(
        dense_row_thresh=knobs[COLAMD_DENSE_ROW],
        dense_col_thresh=knobs[COLAMD_DENSE_COL],
        aggressive=knobs[COLAMD_AGGRESSIVE]
    )
