# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: btf.pyx
#  Created: 2025-08-04 20:22
# =============================================================================

"""
=================================================
Block Triangular Form (BTF) (:mod:`sksparse.btf`)
=================================================

.. currentmodule:: sksparse.btf

.. versionadded:: 0.5.0

Python interface to the `Block Triangular Format (BTF)
<https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF>`_ library.


Interface
---------

.. autosummary::
   :toctree: generated/

   maxtrans - Maximum transversal of a sparse matrix.
   strongcomp - Strongly connected components of a directed graph.
   btf - Permutation into Block Triangular Form (BTF).
   btf_q_permutation - Convert raw BTF column permutation to valid permutation.


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse BTF <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF>`_
* Duff, Iain. "On Algorithms for Obtaining a Maximum Transversal", *ACM Trans.
  Mathematical Software*, vol 7, no. 1, pp. 315-330.
* "Algorithm 575: Permutations for a Zero-Free Diagonal", *ACM Trans.
  Mathematical Software*, vol 7, no. 1, pp. 387-390. Algorithm 575 is MC21A in
  the Harwell Subroutine Library.
"""

cimport cython

import numpy as np

from .utils import validate_csc_input

__all__ = ['maxtrans', 'strongcomp', 'btf', 'btf_q_permutation']


ctypedef fused index_t:
    int32_t
    int64_t


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

        Adapted from the BTF maxtrans documentation [#maxtrans_h]_:

            The output is an array ``jmatch`` of size ``N``.  If row ``i`` is
            matched with column ``j``, then ``A[i, j]`` is nonzero, and then
            ``jmatch[i] = j``.  If the matrix is structurally nonsingular, all
            entries in the ``jmatch`` array are unique, and ``jmatch`` can be
            viewed as a column permutation if `A` is square.  That is, column
            `k` of the original matrix becomes column ``jmatch[k]`` of the
            permuted matrix.

            If row ``i`` is not matched with any column,
            then ``jmatch[i] = -1``.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#maxtrans_h] BTF maxtrans header file:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/Include/btf.h
    """
    A, _, out_dtype = validate_csc_input(A)

    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t N = A.shape[1]

    if M == 0 or N == 0:
        return np.empty(0, dtype=out_dtype)

    if A.nnz == 0:
        return np.full(M, -1, dtype=out_dtype)

    # Allocate output array
    jmatch = np.zeros(M, dtype=out_dtype)

    cdef double maxwork = 0  # TODO default value?

    _maxtrans(M, N, A.indptr, A.indices, maxwork, jmatch)

    return jmatch


@cython.boundscheck(False)
@cython.wraparound(False)
def _maxtrans(
    Py_ssize_t M,
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    double maxwork,
    index_t[::1] jmatch,
):
    cdef index_t nnz_diag
    cdef double work

    # Allocate workspace
    itype = np.int32 if index_t is int32_t else np.int64
    cdef index_t[::1] workspace = np.zeros(5 * N, dtype=itype)

    if index_t is int32_t:
        nnz_diag = btf_maxtrans(
            M,
            N,
            &Ap[0],
            &Ai[0],
            maxwork,
            &work,
            &jmatch[0],
            &workspace[0]
        )
    else:
        nnz_diag = btf_l_maxtrans(
            M,
            N,
            &Ap[0],
            &Ai[0],
            maxwork,
            &work,
            &jmatch[0],
            &workspace[0]
        )

    if nnz_diag < 0:
        raise ValueError(f"BTF maxtrans failed with error code: {nnz_diag}")


def strongcomp(A, q=None):
    """Compute the strongly connected components of a directed graph.

    This function finds a symmetric permutation of a sparse matrix so that
    ``A[p][:, p]`` is block upper triangular form [#strongcomp_h]_.

    Parameters
    ----------
    A : (N, N) {array-like, sparse array}
        An array convertible to a sparse matrix in Compressed Sparse Column
        (CSC) format. Must be square.
    q : (N,) ndarray of int, optional
        A permutation vector. If provided, find the strongly connected
        components of ``A[:, qin]``.

    Returns
    -------
    p : (N,) ndarray of int
        The permutation vector such that ``A[p][:, p]`` is in block upper
        triangular form, unless ``q`` is provided (see below).
    q : (N,) ndarray of int, optional
        If ``q`` is provided on input, ``A[p][:, q]`` is in block upper
        triangular form.
    r : (Nb+1,) ndarray of int
        The array of indices of the start of each block in the permuted matrix.
        Block ``b`` is in rows/columns ``r[b]`` to ``r[b+1] - 1``.
        The number of blocks is ``len(r) - 1``.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#strongcomp_h] BTF strongcomp header file:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/Include/btf.h
    """
    A, _, out_dtype = validate_csc_input(A, require_square=True)

    cdef Py_ssize_t N = A.shape[0]

    if N == 0:
        p = np.empty(0, dtype=out_dtype)
        r = np.zeros(1, dtype=out_dtype)  # no blocks
        if q is not None:
            q = np.empty(0, dtype=out_dtype)
            return p, q, r
        else:
            return p, r

    if A.nnz == 0:
        p = np.arange(N, dtype=out_dtype)
        r = np.zeros(N + 1, dtype=out_dtype)
        r[-1] = N  # N blocks of size 1
        if q is not None:
            q = np.arange(N, dtype=out_dtype)
            return p, q, r
        else:
            return p, r

    if q is not None:
        try:
            q = np.ascontiguousarray(q, dtype=out_dtype)
        except TypeError:
            raise TypeError("qin must be an integer array.")

    # Allocate output arrays
    p = np.zeros(N, dtype=out_dtype)
    r = np.zeros(N + 1, dtype=out_dtype)

    nblocks = _strongcomp(N, A.indptr, A.indices, q, p, r)

    # Take only the first nblocks of r
    r = np.asarray(r[:nblocks + 1])

    if q is not None:
        return p, q, r
    else:
        return p, r


@cython.boundscheck(False)
@cython.wraparound(False)
def _strongcomp(
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    index_t[::1] q,
    index_t[::1] p,
    index_t[::1] r,
):
    cdef index_t nblocks
    cdef index_t* q_ptr = NULL

    if q is not None:
        if len(q) != N:
            raise ValueError("qin must have the same length"
                             "as the number of columns in A.")

        q_ptr = &q[0]

    # Assign memory for the input/output arrays
    itype = np.int32 if index_t is int32_t else np.int64
    cdef index_t[::1] workspace = np.zeros(4 * N, dtype=itype)

    if index_t is int32_t:
        nblocks = btf_strongcomp(N, &Ap[0], &Ai[0], q_ptr, &p[0], &r[0], &workspace[0])
    else:
        nblocks = btf_l_strongcomp(N, &Ap[0], &Ai[0], q_ptr, &p[0], &r[0], &workspace[0])

    if nblocks < 0:
        raise ValueError(f"BTF strongcomp failed with error code: {nblocks}")

    return nblocks


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
    r : (Nb+1,) ndarray of int
        The array of indices of the start of each block in the permuted matrix.
        Block ``b`` is in rows/columns ``r[b]`` to ``r[b+1] - 1``.
        The number of blocks is ``len(r) - 1``.

    Notes
    -----
    Adapted from the BTF documentation [#btf_h]_:

        The function finds a maximum matching (or perhaps a limited matching if
        the work is limited), via the :func:`.maxtrans` function. If a complete
        matching is not found, :func:`.btf` completes the permutation, but
        flags the columns of ``A[p][:, q]`` to denote which columns are not
        matched. If the matrix is structurally rank deficient, some of the
        entries on the diagonal of the permuted matrix will be zero.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#btf_h] BTF header file:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/Include/btf.h
    """
    A, _, out_dtype = validate_csc_input(A, require_square=True)

    cdef Py_ssize_t N = A.shape[0]
    cdef Py_ssize_t M = A.shape[1]

    if N == 0:
        p = np.empty(0, dtype=out_dtype)
        q = np.empty(0, dtype=out_dtype)
        r = np.zeros(1, dtype=out_dtype)  # no blocks
        return p, q, r

    if A.nnz == 0:
        # p = [0, 1, ..., N - 1]
        p = np.arange(N, dtype=out_dtype)
        # q = [-2, -3, ..., -(N + 1)]
        q = -np.arange(N, dtype=out_dtype) - 2  # flag all columns
        r = np.arange(N + 1, dtype=out_dtype)      # N blocks of size 1
        return p, q, r

    # Assign memory for the input/output arrays
    p = np.zeros(N, dtype=out_dtype)
    q = np.zeros(N, dtype=out_dtype)
    r = np.zeros(N + 1, dtype=out_dtype)

    cdef double maxwork = 0  # TODO default value?

    _btf(N, A.indptr, A.indices, maxwork, p, q, r)

    return p, q, r


@cython.boundscheck(False)
@cython.wraparound(False)
def _btf(
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    double maxwork,
    index_t[::1] p,
    index_t[::1] q,
    index_t[::1] r,
):
    cdef:
        index_t nblocks
        double work
        index_t nmatch

    # Define workspace
    itype = np.int32 if index_t is int32_t else np.int64
    cdef index_t[::1] workspace = np.zeros(5 * N, dtype=itype)

    if index_t is int32_t:
        nblocks = btf_order(
            N,
            &Ap[0],
            &Ai[0],
            maxwork,
            &work,
            &p[0],
            &q[0],
            &r[0],
            &nmatch,
            &workspace[0]
        )
    else:
        nblocks = btf_l_order(
            N,
            &Ap[0],
            &Ai[0],
            maxwork,
            &work,
            &p[0],
            &q[0],
            &r[0],
            &nmatch,
            &workspace[0]
        )

    if nblocks < 0:
        raise ValueError(f"BTF failed with error code: {nblocks}")


def btf_q_permutation(q):
    """Convert a raw BTF column permutation vector to a valid permutation.

    Parameters
    ----------
    q : (N,) ndarray of int
        The raw BTF column permutation vector. Contains negative entries for
        unmatched columns.

    Returns
    -------
    q_perm : (N,) ndarray of int
        The valid BTF column permutation vector. Contains only non-negative
        entries, where unmatched columns are replaced with their shifted
        absolute values.

    Notes
    -----
    In C, the values of ``q`` are converted using ``j = BTF_UNFLIP(Q[k])``,
    which is a macro for:

    .. code:: C

        j = (Q[k] < 0) ? -Q[k] - 2 : Q[k]

    This function is a Python equivalent of that macro.

    .. versionadded:: 0.5.0
    """
    q = np.asarray(q)

    if q.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    idx = q < 0
    q[idx] = -q[idx] - 2  # flip negative values
    return q
