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
    so that it has a zero-free diagonal, if possible [#maxtrans_h]_.

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
    .. [#maxtrans_h] BTF maxtrans header file:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/Include/btf.h
    .. [#maxtrans_mex] BTF maxtrans MATLAB interface:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/MATLAB/maxtrans.m
    """
    # TODO refactor this check to a separate function for all modules
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

    # Initialize output variable
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


def strongcomp(A, qin=None):
    """Compute the strongly connected components of a directed graph.

    This function finds a symmetric permutation of a sparse matrix so that 
    ``P @ A @ P.T`` is block upper triangular form.

    Parameters
    ----------
    A : (N, N) {array-like, sparse array}
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format. Must be square.
    qin : (N,) ndarray, optional
        A permutation vector. If provided, find the strongly connected
        components of ``A[:, qin]``.

    Returns
    -------
    p : (N,) ndarray
        The permutation vector such that ``A[p][:, p]`` is in block upper
        triangular form, unless ``q`` is provided (see below).
    q : (N,) ndarray, optional
        If ``q`` is provided on input, ``A[p][:, q]`` is in block upper
        triangular form.
    r : (N+1,) ndarray
        The array of pointers to the start of each block in the permuted matrix.
        Block ``b`` is in rows/columns ``r[b]`` to ``r[b+1] - 1``.
        The number of blocks is ``r[-1]``.

    References
    ----------
    .. [#strongcomp_h] BTF strongcomp header file:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/Include/btf.h
    .. [#strongcomp_mex] BTF strongcomp MATLAB interface:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/MATLAB/strongcomp.m
    """
    # TODO refactor this check to a separate function for all modules
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be 2D.")

    M, N = A.shape

    if M != N:
        raise ValueError("Input must be square.")

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
        p = np.empty(0, dtype=out_dtype)
        r = np.zeros(1, dtype=out_dtype)  # no blocks
        if qin is not None:
            q = np.empty(0, dtype=out_dtype)
            return p, q, r
        else:
            return p, r

    if A.nnz == 0:
        p = np.arange(N, dtype=out_dtype)
        r = np.zeros(N + 1, dtype=out_dtype)
        r[-1] = N  # N blocks of size 1
        if qin is not None:
            q = np.arange(N, dtype=out_dtype)
            return p, q, r
        else:
            return p, r

    # Declare typed memory views for Cython
    cdef int32_t[::1] Ap_mv_int32
    cdef int32_t[::1] Ai_mv_int32
    cdef int32_t[::1] P_mv_int32
    cdef int32_t[::1] Q_mv_int32
    cdef int32_t[::1] R_mv_int32
    cdef int32_t[::1] Work_mv_int32

    cdef int64_t[::1] Ap_mv_int64
    cdef int64_t[::1] Ai_mv_int64
    cdef int64_t[::1] P_mv_int64
    cdef int64_t[::1] Q_mv_int64
    cdef int64_t[::1] R_mv_int64
    cdef int64_t[::1] Work_mv_int64

    # Use a NULL pointer for Q if qin is not provided
    cdef int32_t* Q_ptr_int32 = NULL
    cdef int64_t* Q_ptr_int64 = NULL

    if qin is not None:
        try:
            q = np.ascontiguousarray(qin, dtype=np.int32 if use_int32 else np.int64)
        except ValueError:
            raise ValueError("qin must be an integer array.")

        if len(q) != N:
            raise ValueError("qin must have the same length"
                             "as the number of columns in A.")

        if use_int32:
            Q_mv_int32 = q
            Q_ptr_int32 = &Q_mv_int32[0]
        else:
            Q_mv_int64 = q
            Q_ptr_int64 = &Q_mv_int64[0]

    # Assign memory for the input/output arrays
    if use_int32:
        Ap_mv_int32 = np.ascontiguousarray(A.indptr, dtype=np.int32)
        Ai_mv_int32 = np.ascontiguousarray(A.indices, dtype=np.int32)
        p = P_mv_int32 = np.zeros(N, dtype=np.int32)
        R_mv_int32 = np.zeros(N + 1, dtype=np.int32)
        Work_mv_int32 = np.zeros(4 * N, dtype=np.int32)
    else:
        Ap_mv_int64 = np.ascontiguousarray(A.indptr, dtype=np.int64)
        Ai_mv_int64 = np.ascontiguousarray(A.indices, dtype=np.int64)
        p = P_mv_int64 = np.zeros(N, dtype=np.int64)
        R_mv_int64 = np.zeros(N + 1, dtype=np.int64)
        Work_mv_int64 = np.zeros(4 * N, dtype=np.int64)

    if use_int32:
        nblocks = btf_strongcomp(
            N,
            &Ap_mv_int32[0],
            &Ai_mv_int32[0],
            Q_ptr_int32,
            &P_mv_int32[0],
            &R_mv_int32[0],
            &Work_mv_int32[0]
        )
    else:
        nblocks = btf_l_strongcomp(
            N,
            &Ap_mv_int64[0],
            &Ai_mv_int64[0],
            Q_ptr_int64,
            &P_mv_int64[0],
            &R_mv_int64[0],
            &Work_mv_int64[0]
        )

    if nblocks < 0:
        raise ValueError(f"BTF strongcomp failed with error code: {nblocks}")

    # Take only the first nblocks of r
    if use_int32:
        r_slice = R_mv_int32[:nblocks + 1]
    else:
        r_slice = R_mv_int64[:nblocks + 1]

    r = np.asarray(r_slice)

    if qin is not None:
        return p, q, r
    else:
        return p, r


def btf(A):
    """Permute the square sparse matrix into Block Triangular Form (BTF).

    This function finds a permutation of a sparse matrix so that 
    `PAQ` (``A[p][:, q]``) is block upper triangular form with a zero-free
    diagonal, or with a maximum number of nonzeros on the diagonal if
    a zero-free permutation does not exist [#btf_h]_.

    Parameters
    ----------
    A : (N, N) {array-like, sparse array}
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format. Must be square.

    Returns
    -------
    p : (N,) ndarray of int
        The row permutation vector such that ``A[p][:, q]`` is in block upper
        triangular form.
    q : (N,) ndarray of int
        The column permutation vector. If ``A`` is structurally nonsingular,
        ``A[p][:, q]`` has a zero-free diagonal. If ``A`` is structurally
        singular, ``q`` will contain negative entries. The permuted matrix
        is ``A[p][:, abs(q)]``. If ``q[k] < 0``, then ``PAQ[k, k]`` is zero.
    r : (N+1,) ndarray of int
        The array of indices of the start of each block in the permuted matrix.
        Block ``b`` is in rows/columns ``r[b]`` to ``r[b+1] - 1``.
        The number of blocks is ``r[-1]``.

    Notes
    -----
    Adapted from the BTF documentation [#btf_h]_:

        The function finds a maximum matching (or perhaps a limited matching if
        the work is limited), via the :func:`.maxtrans` function. If a complete
        matching is not found, :func:`.btf` completes the permutation, but
        flags the columns of ``A[p][:, q]`` to denote which columns are not
        matched. If the matrix is structurally rank deficient, some of the
        entries on the diagonal of the permuted matrix will be zero.

    References
    ----------
    .. [#btf_h] BTF header file:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/Include/btf.h
    .. [#btf_mex] BTF MATLAB interface:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/MATLAB/btf.m
    """
    # TODO refactor this check to a separate function for all modules
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be 2D.")

    M, N = A.shape

    if M != N:
        raise ValueError("Input must be square.")

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
        p = np.empty(0, dtype=out_dtype)
        q = np.empty(0, dtype=out_dtype)
        r = np.zeros(1, dtype=out_dtype)  # no blocks
        return p, q, r

    if A.nnz == 0:
        p = np.arange(N, dtype=out_dtype)
        # FIXME all -1?
        q = np.arange(N, dtype=out_dtype)
        r = np.zeros(N + 1, dtype=out_dtype)
        r[-1] = N  # N blocks of size 1
        return p, q, r

    # Declare typed memory views for Cython
    cdef int32_t[::1] Ap_mv_int32
    cdef int32_t[::1] Ai_mv_int32
    cdef int32_t[::1] P_mv_int32
    cdef int32_t[::1] Q_mv_int32
    cdef int32_t[::1] R_mv_int32
    cdef int32_t[::1] Work_mv_int32

    cdef int64_t[::1] Ap_mv_int64
    cdef int64_t[::1] Ai_mv_int64
    cdef int64_t[::1] P_mv_int64
    cdef int64_t[::1] Q_mv_int64
    cdef int64_t[::1] R_mv_int64
    cdef int64_t[::1] Work_mv_int64

    # Assign memory for the input/output arrays
    if use_int32:
        Ap_mv_int32 = np.ascontiguousarray(A.indptr, dtype=np.int32)
        Ai_mv_int32 = np.ascontiguousarray(A.indices, dtype=np.int32)
        p = P_mv_int32 = np.zeros(N, dtype=np.int32)
        q = Q_mv_int32 = np.zeros(N, dtype=np.int32)
        r = R_mv_int32 = np.zeros(N + 1, dtype=np.int32)
        Work_mv_int32 = np.zeros(5 * N, dtype=np.int32)
    else:
        Ap_mv_int64 = np.ascontiguousarray(A.indptr, dtype=np.int64)
        Ai_mv_int64 = np.ascontiguousarray(A.indices, dtype=np.int64)
        p = P_mv_int64 = np.zeros(N, dtype=np.int64)
        q = Q_mv_int64 = np.zeros(N, dtype=np.int64)
        r = R_mv_int64 = np.zeros(N + 1, dtype=np.int64)
        Work_mv_int64 = np.zeros(5 * N, dtype=np.int64)

    maxwork = 0  # TODO default value?
    cdef double work
    cdef int32_t nmatch_int32
    cdef int64_t nmatch_int64

    if use_int32:
        nblocks = btf_order(
            N,
            &Ap_mv_int32[0],
            &Ai_mv_int32[0],
            maxwork,
            &work,
            &P_mv_int32[0],
            &Q_mv_int32[0],
            &R_mv_int32[0],
            &nmatch_int32,
            &Work_mv_int32[0]
        )
    else:
        nblocks = btf_l_order(
            N,
            &Ap_mv_int64[0],
            &Ai_mv_int64[0],
            maxwork,
            &work,
            &P_mv_int64[0],
            &Q_mv_int64[0],
            &R_mv_int64[0],
            &nmatch_int64,
            &Work_mv_int64[0]
        )

    if nblocks < 0:
        raise ValueError(f"BTF failed with error code: {nblocks}")

    return p, q, r
