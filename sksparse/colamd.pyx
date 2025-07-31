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

"""
sksparse.colamd: Cython interface to COLAMD, a column approximate minimum
degree ordering algorithm.
"""

import numpy as np
cimport numpy as np

import warnings

from scipy.sparse import csc_array, issparse, SparseEfficiencyWarning


# TODO docstring
def colamd(A, return_info=False):
    """Compute the column approximate minimum degree ordering of a sparse matrix."""
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
    knobs = np.empty(COLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_mv = knobs
    colamd_set_defaults(&knobs_mv[0])

    # TODO override with user knobs if provided

    # Declare typed memory views for Cython
    cdef int32_t[::1] Ap_mv_int32
    cdef int64_t[::1] Ap_mv_int64

    cdef int32_t[::1] Ai_mv_int32
    cdef int64_t[::1] Ai_mv_int64

    cdef int32_t[::1] stats_mv_int32
    cdef int64_t[::1] stats_mv_int64

    # Compute the ordering
    if use_int32:
        # Copy the arrays, since they are altered in the C function
        Ai_mv_int32 = np.array(A.indices, dtype=np.int32, copy=True, order='C')
        p = Ap_mv_int32 = np.array(A.indptr, dtype=np.int32, copy=True, order='C')
        stats = stats_mv_int32 = np.empty(COLAMD_STATS, dtype=np.int32)
        ok = c_colamd(
            M, 
			N, 
			Alen, 
			&Ai_mv_int32[0], 
			&Ap_mv_int32[0], 
			&knobs_mv[0], 
			&stats_mv_int32[0]
        )
    else:
        # Copy the arrays, since they are altered in the C function
        Ai_mv_int64 = np.array(A.indices, dtype=np.int64, copy=True, order='C')
        p = Ap_mv_int64 = np.array(A.indptr, dtype=np.int64, copy=True, order='C')
        stats = stats_mv_int64 = np.empty(COLAMD_STATS, dtype=np.int64)
        ok = c_colamd_l(
            M, 
			N, 
			Alen, 
			&Ai_mv_int64[0], 
			&Ap_mv_int64[0], 
			&knobs_mv[0], 
			&stats_mv_int64[0]
        )

    if not ok:
        # TODO raise appropriate error
        raise ValueError(f"COLAMD failed with error code: {stats[COLAMD_STATUS]}")

    if return_info:
        # TODO create a COLAMDStats dataclass
        # return p, COLAMDStats.from_array(stats)
        return p, stats
    else:
        return p
