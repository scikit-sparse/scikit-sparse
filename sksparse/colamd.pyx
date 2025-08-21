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

.. versionadded:: 0.5.0

References
----------
* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse COLAMD:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/COLAMD
* COLAMD Algorithm Publications:
  - T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, An approximate column
    minimum degree ordering algorithm, ACM Transactions on Mathematical
    Software, vol. 30, no. 3., pp. 353-376, 2004.
  - T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836: COLAMD,
    an approximate column minimum degree ordering algorithm, ACM
    Transactions on Mathematical Software, vol. 30, no. 3., pp. 377-380,
    2004.
"""

cimport cython

import numpy as np
from dataclasses import dataclass

from .utils import validate_csc_input


ctypedef fused index_t:
    int32_t
    int64_t


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
cdef dict _COLAMD_ERROR_CODES = {
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
}


cdef int _handle_errors(int ok, index_t[::1] stats) except -1 with gil:
    """Handle errors from COLAMD based on the return status and stats array."""
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

    .. versionadded:: 0.5.0

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


def _colamd_base(
    object A,
    *,
    bint is_symmetric=False,
    object dense_row_thresh=None,
    object dense_col_thresh=None,
    object aggressive=None,
    bint return_info=False
):
    """A common base function for colamd and symamd."""
    A, _, out_dtype = validate_csc_input(A, is_symmetric)

    cdef Py_ssize_t M = A.shape[0]
    cdef Py_ssize_t N = A.shape[1]

    if M == 0 or N == 0:
        return np.empty(0, dtype=out_dtype)

    if A.nnz == 0 or (not is_symmetric and M == 1):
        return np.arange(N, dtype=out_dtype)

    if N == 1:
        return np.zeros(N, dtype=out_dtype)

    # Set the default knobs
    knobs = np.zeros(COLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_view = knobs
    colamd_set_defaults(&knobs_view[0])

    # Override with user knobs if provided
    if dense_row_thresh is not None:
        knobs_view[COLAMD_DENSE_ROW] = <float>dense_row_thresh

    if dense_col_thresh is not None:
        knobs_view[COLAMD_DENSE_COL] = <float>dense_col_thresh

    if aggressive is not None:
        knobs_view[COLAMD_AGGRESSIVE] = 1.0 if aggressive else 0.0

    # Allocate output arrays
    perm = np.zeros(N + 1, dtype=out_dtype)
    stats = np.zeros(COLAMD_STATS, dtype=out_dtype)

    # Compute the ordering
    if is_symmetric:
        _symamd(M, N, A.indptr, A.indices, perm, knobs_view, stats)
    else:
        _colamd(M, N, A.indptr, A.indices, perm, knobs_view, stats)

    # Return the permutation array
    q = np.asarray(perm[:N])

    if return_info:
        return q, COLAMDStats.from_array(stats)
    else:
        return q


@cython.boundscheck(False)
@cython.wraparound(False)
def _colamd(
    Py_ssize_t M,
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    index_t[::1] perm,
    double[::1] knobs,
    index_t[::1] stats
):
    """Internal Cython wrapper for COLAMD.

    Parameters
    ----------
    M : int
        Number of rows in the matrix.
    N : int
        Number of columns in the matrix.
    Ap : array_like
        Column pointer array of size (N + 1,).
    Ai : array_like
        Row index array of size (nnz,).
    perm : array_like
        Output permutation array of size (N + 1,).
    knobs : array_like
        Knobs array of size (COLAMD_KNOBS,).
    stats : array_like
        Stats array of size (COLAMD_STATS,).
    """
    cdef int ok

    # Get the recommended size for the Alen array
    cdef index_t Alen = 0
    cdef Py_ssize_t nnz = Ai.shape[0]

    if index_t is int32_t:
        Alen = colamd_recommended(nnz, M, N)
    else:
        Alen = colamd_l_recommended(nnz, M, N)

    if Alen == 0:
        raise ValueError("Recommended Alen is zero: one of {A.nnz, M, N} is erroneous.")

    assert Alen >= nnz, "Recommended Alen is less than nnz."

    # Copy the input arrays, since they are altered in the C function
    itype = np.int32 if index_t is int32_t else np.int64
    cdef index_t[::1] Ai_work = np.zeros(Alen, dtype=itype)
    Ai_work[:nnz] = Ai
    perm[:] = Ap

    # Compute the ordering
    if index_t is int32_t:
        ok = c_colamd(M, N, Alen, &Ai_work[0], &perm[0], &knobs[0], &stats[0])
    else:
        ok = c_colamd_l(M, N, Alen, &Ai_work[0], &perm[0], &knobs[0], &stats[0])

    _handle_errors(ok, stats)


@cython.boundscheck(False)
@cython.wraparound(False)
def _symamd(
    Py_ssize_t M,
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    index_t[::1] perm,
    double[::1] knobs,
    index_t[::1] stats
):
    """Internal Cython wrapper for SYMAMD.

    Parameters
    ----------
    M : int
        Number of rows in the matrix.
    N : int
        Number of columns in the matrix.
    Ap : array_like
        Column pointer array of size (N + 1,).
    Ai : array_like
        Row index array of size (nnz,).
    perm : array_like
        Output permutation array of size (N + 1,).
    knobs : array_like
        Knobs array of size (COLAMD_KNOBS,).
    stats : array_like
        Stats array of size (COLAMD_STATS,).
    """
    cdef int ok

    # Compute the ordering
    if index_t is int32_t:
        ok = c_symamd(N, &Ai[0], &Ap[0], &perm[0], &knobs[0], &stats[0], calloc, free)
    else:
        ok = c_symamd_l(N, &Ai[0], &Ap[0], &perm[0], &knobs[0], &stats[0], calloc, free)

    _handle_errors(ok, stats)


def colamd(
    A,
    dense_row_thresh=None,
    dense_col_thresh=None,
    aggressive=None,
    return_info=False
):
    return _colamd_base(
        A,
        is_symmetric=False,
        dense_row_thresh=dense_row_thresh,
        dense_col_thresh=dense_col_thresh,
        aggressive=aggressive,
        return_info=return_info
    )


def symamd(
    A,
    dense_row_thresh=None,
    dense_col_thresh=None,
    aggressive=None,
    return_info=False
):
    return _colamd_base(
        A,
        is_symmetric=True,
        dense_row_thresh=dense_row_thresh,
        dense_col_thresh=dense_col_thresh,
        aggressive=aggressive,
        return_info=return_info
    )


_COLAMD_DOC_TEMPLATE = """
{intro}
Parameters
----------
{A_param}
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

Returns
-------
q : (N,) ndarray
    The permutation vector.
stats : COLAMDStats, optional
    If ``return_info`` is True, returns an object containing statistics
    about the ordering.

.. versionadded:: 0.5.0

References
----------
.. {reftag} ``colamd.c`` - SuiteSparse AMD source file.
    https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/COLAMD/Source/colamd.c
"""

# Define the docstrings
colamd_reftag = "[#colamd_c]"

colamd_intro = f"""Compute the column approximate minimum degree ordering of
a sparse matrix.

Adapted from the COLAMD documentation {colamd_reftag}_:

    This function computes a column ordering for a sparse matrix `A` that
    is appropriate for LU factorization of symmetric or unsymmetric
    matrices, QR factorization, least squares, interior point methods for
    linear programming problems, and other related problems.

    COLAMD computes a permutation `Q` such that the Cholesky factorization
    of :math:`(AQ)^{{\\top}}(AQ)` has less fill-in and requires fewer floating
    point operations than :math:`A^{{\\top}}A`.  This also provides a good
    ordering for sparse partial pivoting methods, :math:`P(AQ) = LU`, where
    `Q` is computed prior to numerical factorization, and `P` is computed
    during numerical factorization via conventional partial pivoting with
    row interchanges.
"""

colamd_A_param = """A : (M, N) {array_like, sparse matrix}
    The input matrix for which to compute the column ordering.
    Must be 2D and convertible to CSC format. Need not be square."""

colamd.__doc__ = _COLAMD_DOC_TEMPLATE.format(
    intro=colamd_intro, A_param=colamd_A_param, reftag=colamd_reftag,
)


# Define the docstring for symamd
symamd_reftag = "[#symamd_c]"

symamd_intro = f"""Compute the column approximate minimum degree ordering of
a sparse symmetric matrix.

Adapted from the COLAMD documentation {symamd_reftag}_:

    This function computes an approximate minimum degree ordering for
    Cholesky factorization of symmetric matrices.

    Symamd computes a permutation `P` of a symmetric matrix `A` such that
    the Cholesky factorization of :math:`PAP^{{\\top}}` has less fill-in and
    requires fewer floating point operations than `A`.  Symamd constructs
    a matrix `M` such that :math:`M^{{\\top}}M` has the same nonzero pattern
    of `A`, and then orders the columns of `M` using colamd.  The column
    ordering of `M` is then returned as the row and column ordering `P` of
    `A`.
"""

symamd_A_param = """A : (N, N) {array_like, sparse matrix}
    The input matrix for which to compute the column ordering.
    Must be 2D, square, and convertible to CSC format.

    .. note::
        This routine only accesses the lower triangular part of ``A``,
        which is *assumed* to be symmetric. If it is not, the results may
        be incorrect or undefined.
"""

symamd.__doc__ = _COLAMD_DOC_TEMPLATE.format(
    intro=symamd_intro, A_param=symamd_A_param, reftag=symamd_reftag,
)


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

    .. versionadded:: 0.5.0
    """
    knobs = np.zeros(COLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_view = knobs
    colamd_set_defaults(&knobs_view[0])
    return dict(
        dense_row_thresh=knobs[COLAMD_DENSE_ROW],
        dense_col_thresh=knobs[COLAMD_DENSE_COL],
        aggressive=knobs[COLAMD_AGGRESSIVE]
    )
