# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: btf.pyx
#  Created: 2025-08-04 20:22
# =============================================================================

"""sksparse.btf: Python interface to the Block Triangular Format (BTF) library.

This module provides a Cython interface to the BTF module of the SuiteSparse
library by Timothy A. Davis. The main algorithm computes a permutation of a
sparse matrix into Block Triangular Form (BTF).

Interfaces
----------
* `maxtrans`: Maximum transversal of a sparse matrix.
* `strongcomp`: Strongly connected components of a directed graph.
* `btf`: Permutation into Block Triangular Form (BTF).

This wrapper handles both 32-bit and 64-bit integer types, depending on the
input matrix format.

References
----------
* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse BTF:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF
* Duff, Iain. "On Algorithms for Obtaining a Maximum Transversal", *ACM Trans.
  Mathematical Software*, vol 7, no. 1, pp. 315-330.
* "Algorithm 575: Permutations for a Zero-Free Diagonal", *ACM Trans.
  Mathematical Software*, vol 7, no. 1, pp. 387-390. Algorithm 575 is MC21A in
  the Harwell Subroutine Library.
"""

import numpy as np
cimport numpy as np

import warnings

from scipy.sparse import csc_array, issparse, SparseEfficiencyWarning


def maxtrans(A):
    """Compute the maximum transversal of a sparse matrix.

    This function finds a permutation of the columns of a sparse matrix
    so that it has a zero-free diagonal, if possible [#maxtrans]_.

    Parameters
    ----------
    A : (M, N) {array-like, sparse array}
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format.

    Returns
    -------
    jmatch : (M,) ndarray
        Array containing the maximum transversal.

        Adapted from the BTF maxtrans documentation [#maxtrans]_:

            The output is an array ``jmatch`` of size ``N``.  If row ``i`` is
            matched with column ``j``, then ``A[i, j]`` is nonzero, and then
            ``jmatch[i] = j``.  If the matrix is structurally nonsingular, all
            entries in the ``jmatch`` array are unique, and ``jmatch`` can be
            viewed as a column permutation if `A` is square.  That is, column
            `k` of the original matrix becomes column ``jmatch[k]`` of the
            permuted matrix.

            If row ``i`` is not matched with any column,
            then ``jmatch[i] = -1``.

    References
    ----------
    .. [#maxtrans] BTF maxtrans header file:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/Include/btf.h
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

    if M == 0 or N == 0:
        return np.empty(0, dtype=np.int32 if use_int32 else np.int64)

    if A.nnz == 0:
        return np.full(M, -1, dtype=np.int32 if use_int32 else np.int64)

    # Declare typed memory views for Cython
    cdef int32_t[::1] Ap_mv_int32
    cdef int32_t[::1] Ai_mv_int32
    cdef int32_t[::1] Match_mv_int32
    cdef int32_t[::1] Work_mv_int32

    cdef int64_t[::1] Ap_mv_int64
    cdef int64_t[::1] Ai_mv_int64
    cdef int64_t[::1] Match_mv_int64
    cdef int64_t[::1] Work_mv_int64

    if use_int32:
        Ap_mv_int32 = np.ascontiguousarray(A.indptr, dtype=np.int32)
        Ai_mv_int32 = np.ascontiguousarray(A.indices, dtype=np.int32)
        jmatch = Match_mv_int32 = np.zeros(M, dtype=np.int32)
        Work_mv_int32 = np.zeros(5 * N, dtype=np.int32)
    else:
        Ap_mv_int64 = np.ascontiguousarray(A.indptr, dtype=np.int64)
        Ai_mv_int64 = np.ascontiguousarray(A.indices, dtype=np.int64)
        jmatch = Match_mv_int64 = np.zeros(M, dtype=np.int64)
        Work_mv_int64 = np.zeros(5 * N, dtype=np.int64)

    cdef double work
    maxwork = 0  # TODO default value?

    if use_int32:
        nnz_diag = btf_maxtrans(
            M,
            N,
            &Ap_mv_int32[0],
            &Ai_mv_int32[0],
            maxwork,
            &work,
            &Match_mv_int32[0],
            &Work_mv_int32[0]
        )
    else:
        nnz_diag = btf_l_maxtrans(
            M,
            N,
            &Ap_mv_int64[0],
            &Ai_mv_int64[0],
            maxwork,
            &work,
            &Match_mv_int64[0],
            &Work_mv_int64[0]
        )

    if nnz_diag < 0:
        raise ValueError(f"BTF maxtrans failed with error code: {nnz_diag}")

    return jmatch
