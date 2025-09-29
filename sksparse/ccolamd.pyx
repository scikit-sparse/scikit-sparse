# Cython CCOLAMD python interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: ccolamd.pyx
#  Created: 2025-07-31 10:13
# =============================================================================

"""
==========================================================================================
Constrained Column Approximate Minimum Degree (CCOLAMD) Ordering (:mod:`sksparse.ccolamd`)
==========================================================================================

.. currentmodule:: sksparse.ccolamd

.. versionadded:: 0.5.0

Python interface to the `Constrained Column Approximate Minimum Degree
(CCOLAMD)
<https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CCOLAMD>`_
ordering algorithm.


Interface
---------

.. autosummary::
   :toctree: generated/

   ccolamd - Function to compute the column ordering of any shape sparse matrix.
   csymamd - Function to compute the column ordering of a symmetric sparse matrix.
   ccolamd_get_defaults - Function to get the default knobs for CCOLAMD.


Exceptions and Warnings
-----------------------

.. autosummary::
   :toctree: generated/

   CCOLAMDError - Base class for CCOLAMD errors.
   CCOLAMDValueError - Raised when CCOLAMD encounters a value error.
   CCOLAMDMemoryError - Raised when CCOLAMD runs out of memory.
   CCOLAMDInternalError - Raised when CCOLAMD encounters an internal error.
   CCOLAMDStats - Dataclass containing statistics about the ordering.


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse CCOLAMD <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CCOLAMD>`_
* CCOLAMD Algorithm Publications:

  * T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, An approximate column
    minimum degree ordering algorithm, *ACM Transactions on Mathematical
    Software*, vol. 30, no. 3., pp. 353-376, 2004.

  * T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836: CCOLAMD,
    an approximate column minimum degree ordering algorithm, *ACM
    Transactions on Mathematical Software*, vol. 30, no. 3., pp. 377-380,
    2004.
"""

cimport cython

import numpy as np

from dataclasses import dataclass

from .utils import validate_csc_input

__all__ = [
    "CCOLAMDError",
    "CCOLAMDValueError",
    "CCOLAMDMemoryError",
    "CCOLAMDInternalError",
    "CCOLAMDStats",
    "ccolamd",
    "csymamd",
    "ccolamd_get_defaults"
]


ctypedef fused index_t:
    int32_t
    int64_t


class CCOLAMDError(Exception):
    """Base class for CCOLAMD errors."""
    pass


class CCOLAMDValueError(CCOLAMDError, ValueError):
    """Raised when CCOLAMD encounters a value error."""
    pass


class CCOLAMDMemoryError(CCOLAMDError, MemoryError):
    """Raised when CCOLAMD runs out of memory."""
    pass


class CCOLAMDInternalError(CCOLAMDError, RuntimeError):
    """Raised when CCOLAMD encounters an internal error."""
    pass


# Define CCOLAMD error codes
cdef dict _CCOLAMD_ERROR_CODES = {
    CCOLAMD_OK: "ok",
    CCOLAMD_OK_BUT_JUMBLED: "ok but A has unsorted columns or duplicate entries",
    CCOLAMD_ERROR_A_not_present: "A is a null pointer",
    CCOLAMD_ERROR_p_not_present: "p is a null pointer",
    CCOLAMD_ERROR_nrow_negative: "nrow is negative",
    CCOLAMD_ERROR_ncol_negative: "ncol is negative",
    CCOLAMD_ERROR_nnz_negative: "nnz is negative",
    CCOLAMD_ERROR_p0_nonzero: "p[0] is nonzero",
    CCOLAMD_ERROR_A_too_small: "A is too small",
    CCOLAMD_ERROR_col_length_negative: "column has a negative number of entries",
    CCOLAMD_ERROR_row_index_out_of_bounds: "row index out of bounds",
    CCOLAMD_ERROR_out_of_memory: "out of memory",
    CCOLAMD_ERROR_internal_error: "internal error"
}


cdef int _handle_errors(ok, stats) except -1 with gil:
    """Handle errors from CCOLAMD."""
    # Check the return status
    if ok:
        assert stats[CCOLAMD_STATUS] == CCOLAMD_OK, \
            "CCOLAMD returned OK but status is not CCOLAMD_OK."
    else:
        if stats[CCOLAMD_STATUS] == CCOLAMD_ERROR_out_of_memory:
            raise CCOLAMDMemoryError("CCOLAMD ran out of memory.")
        elif stats[CCOLAMD_STATUS] == CCOLAMD_ERROR_internal_error:
            raise CCOLAMDInternalError("CCOLAMD encountered an internal error.")
        else:
            raise CCOLAMDValueError(
                f"CCOLAMD returned an error:{_CCOLAMD_ERROR_CODES[stats[CCOLAMD_STATUS]]}."
            )


@dataclass(frozen=True)
class CCOLAMDStats:
    """Information statistics returned by the CCOLAMD algorithm.

    This class wraps the contents of the ``stats`` array returned by
    C ``ccolamd()`` into a Python dataclass.

    Attributes
    ----------
    N_rows_ignored : int
        The number of dense or empty rows ignored in the ordering.
    N_cols_ignored : int
        The number of dense or empty columns ignored in the ordering.
    Ncmpa : int
        The number of garbage collections performed.
    status : int
        Status code indicating the result of the CCOLAMD operation. If non-zero,
        ``ccolamd`` will throw an appropriate exception that interprets this
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
    Field descriptions are adapted from SuiteSparse ``ccolamd.c``
    [#ccolamd_fields]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#ccolamd_fields] ``ccolamd.c`` - SuiteSparse AMD source file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CCOLAMD/Source/ccolamd.c
    """
    N_rows_ignored : int
    N_cols_ignored : int
    Ncmpa : int
    status : int
    info1 : int
    info2 : int
    info3 : int

    @classmethod
    def from_array(cls, stats: "np.ndarray") -> "CCOLAMDStats":
        """Create a CCOLAMDStats instance from an array."""
        return cls(
            N_rows_ignored=int(stats[CCOLAMD_DENSE_ROW]),
            N_cols_ignored=int(stats[CCOLAMD_DENSE_COL]),
            Ncmpa=int(stats[CCOLAMD_DEFRAG_COUNT]),
            status=int(stats[CCOLAMD_STATUS]),
            info1=int(stats[CCOLAMD_INFO1]),
            info2=int(stats[CCOLAMD_INFO2]),
            info3=int(stats[CCOLAMD_INFO3]),
        )


def _ccolamd_base(
    object A,
    *,
    object constraints=None,
    bint is_symmetric=False,
    object dense_row_thresh=None,
    object dense_col_thresh=None,
    object aggressive=None,
    object opt_lu=None,
    bint return_info=False,
):
    """A common base function for ccolamd and csymamd."""
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
    knobs = np.zeros(CCOLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_view = knobs
    ccolamd_set_defaults(&knobs_view[0])

    # Override with user knobs if provided
    if dense_row_thresh is not None:
        knobs_view[CCOLAMD_DENSE_ROW] = <float>dense_row_thresh

    if dense_col_thresh is not None:
        knobs_view[CCOLAMD_DENSE_COL] = <float>dense_col_thresh

    if aggressive is not None:
        knobs_view[CCOLAMD_AGGRESSIVE] = 1.0 if aggressive else 0.0

    if opt_lu is not None:
        if opt_lu not in ('lu', 'cholesky'):
            raise ValueError("opt_lu must be either 'lu' or 'cholesky'.")
        knobs_view[CCOLAMD_LU] = 1.0 if opt_lu == 'lu' else 0.0

    if constraints is not None:
        try:
            constraints = np.asarray(constraints, dtype=out_dtype, order='C')
        except TypeError:
            raise TypeError("Constraints must be an array of integers.")

    # Allocate output rrays
    perm = np.zeros(N + 1, dtype=out_dtype)
    stats = np.zeros(CCOLAMD_STATS, dtype=out_dtype)

    if is_symmetric:
        _csymamd(N, A.indptr, A.indices, perm, knobs_view, stats, constraints)
    else:
        _ccolamd(M, N, A.indptr, A.indices, perm, knobs_view, stats, constraints)

    # Return the permutation array
    q = np.asarray(perm[:N])

    if return_info:
        return q, CCOLAMDStats.from_array(stats)
    else:
        return q


@cython.boundscheck(False)
@cython.wraparound(False)
def _ccolamd(
    Py_ssize_t M,
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    index_t[::1] perm,
    double[::1] knobs,
    index_t[::1] stats,
    index_t[::1] constraints=None,
):
    """Internal Cython wrapper for CCOLAMD.

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
        Knobs array of size (CCOLAMD_KNOBS,).
    stats : array_like
        Stats array of size (CCOLAMD_STATS,).
    """
    cdef int ok

    # Get the recommended size for the Alen array
    cdef index_t Alen = 0
    cdef Py_ssize_t nnz = Ai.shape[0]

    if index_t is int32_t:
        Alen = ccolamd_recommended(nnz, M, N)
    else:
        Alen = ccolamd_l_recommended(nnz, M, N)

    if Alen == 0:
        raise ValueError("Recommended Alen is zero: one of {A.nnz, M, N} is erroneous.")

    assert Alen >= nnz, "Recommended Alen is less than nnz."

    cdef index_t *constraints_ptr = NULL

    if constraints is not None:
        if len(constraints) != N:
            raise ValueError("Constraints must have the same length as the matrix size.")

        constraints_ptr = &constraints[0]

    # Copy the input arrays, since they are altered in the C function
    itype = np.int32 if index_t is int32_t else np.int64
    cdef index_t[::1] Ai_work = np.zeros(Alen, dtype=itype)
    Ai_work[:nnz] = Ai
    perm[:] = Ap

    if index_t is int32_t:
        ok = c_ccolamd(M, N, Alen, &Ai_work[0], &perm[0], &knobs[0], &stats[0], constraints_ptr)
    else:
        ok = c_ccolamd_l(M, N, Alen, &Ai_work[0], &perm[0], &knobs[0], &stats[0], constraints_ptr)

    _handle_errors(ok, stats)


@cython.boundscheck(False)
@cython.wraparound(False)
def _csymamd(
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    index_t[::1] perm,
    double[::1] knobs,
    index_t[::1] stats,
    index_t[::1] constraints=None,
):
    """Internal Cython wrapper for CSYMAMD.

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
    cdef index_t stype = -1  # only lower triangular part is used in csymamd
    cdef index_t *constraints_ptr = NULL

    if constraints is not None:
        if len(constraints) != N:
            raise ValueError("Constraints must have the same length as the matrix size.")

        constraints_ptr = &constraints[0]

    # Compute the ordering
    if index_t is int32_t:
        ok = c_csymamd(
            N,
            &Ai[0],
            &Ap[0],
            &perm[0],
            &knobs[0],
            &stats[0],
            calloc,
            free,
            constraints_ptr,
            stype
        )
    else:
        ok = c_csymamd_l(
            N,
            &Ai[0],
            &Ap[0],
            &perm[0],
            &knobs[0],
            &stats[0],
            calloc,
            free,
            constraints_ptr,
            stype
        )

    _handle_errors(ok, stats)


def ccolamd(
    A,
    constraints=None,
    dense_row_thresh=None,
    dense_col_thresh=None,
    aggressive=None,
    opt_lu=None,
    return_info=False
):
    return _ccolamd_base(
        A,
        constraints=constraints,
        is_symmetric=False,
        dense_row_thresh=dense_row_thresh,
        dense_col_thresh=dense_col_thresh,
        aggressive=aggressive,
        opt_lu=None,
        return_info=return_info
    )


def csymamd(
    A,
    constraints=None,
    dense_row_thresh=None,
    dense_col_thresh=None,
    aggressive=None,
    return_info=False
):
    return _ccolamd_base(
        A,
        constraints=constraints,
        is_symmetric=True,
        dense_row_thresh=dense_row_thresh,
        dense_col_thresh=dense_col_thresh,
        aggressive=aggressive,
        opt_lu=None,
        return_info=return_info
    )


_CCOLAMD_DOC_TEMPLATE = """
{intro}
Parameters
----------
{A_param}
contraints : (N,) array_like, optional
    A 1D array of constraints for the ordering. Each column `i` in
    `A` has a constraint, ``constraints[i]``, in the range [0, N-1]. All
    columns with ``constraints[i] = 0`` are ordered first, followed by nodes
    with `C(i) = 1`, and so on. Thus, ``constraints[p]`` is monotonically
    non-decreasing. If None, no constraints are applied, and the ordering
    will be similar to :func:`~sksparse.colamd.colamd`, except that the default
    values of ``dense_row_thresh``, ``dense_col_thresh``, and ``aggressive``
    may differ.
dense_row_thresh, dense_col_thresh : float, optional
    Threshold for considering a row/column dense. If
    None, use the default value from CCOLAMD. The default value is 10.
    The actual number of entries in a row/column is to be considered
    "dense" is ``max(dense_row_thresh * sqrt(M), 16)`` where ``M`` is the
    number of rows (or ``N`` for columns). Dense rows/columns are ignored
    during ordering and moved to the end of the matrix.
aggressive : bool, optional
    If True, use aggressive absorption. If None, uses the default value
    from CCOLAMD. The default value is True.
{opt_lu_param}

Returns
-------
q : (N,) :class:`~numpy.ndarray`
    The permutation vector.
stats : :class:`CCOLAMDStats`, optional
    If ``return_info`` is True, returns an object containing statistics
    about the ordering.


.. versionadded:: 0.5.0

References
----------
.. {reftag} ``ccolamd.c`` - SuiteSparse AMD source file.
    https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CCOLAMD/Source/ccolamd.c
"""

# Define the docstrings
ccolamd_reftag = "[#ccolamd_c]"

ccolamd_intro = f"""Compute the column approximate minimum degree ordering of
a sparse matrix.

Adapted from the CCOLAMD documentation {ccolamd_reftag}_:

    This function computes a column ordering for a sparse matrix `A` that
    is appropriate for LU factorization of symmetric or unsymmetric
    matrices, QR factorization, least squares, interior point methods for
    linear programming problems, and other related problems.

    CCOLAMD computes a permutation `Q` such that the Cholesky factorization
    of :math:`(AQ)^{{\\top}}(AQ)` has less fill-in and requires fewer floating
    point operations than :math:`A^{{\\top}}A`.  This also provides a good
    ordering for sparse partial pivoting methods, :math:`P(AQ) = LU`, where
    `Q` is computed prior to numerical factorization, and `P` is computed
    during numerical factorization via conventional partial pivoting with
    row interchanges.
"""

ccolamd_A_param = """A : (M, N) {array_like, sparse matrix}
    The input matrix for which to compute the column ordering.
    Must be 2D and convertible to CSC format. Need not be square."""

ccolamd_opt_lu_param = """opt_lu : {'lu', 'cholesky'}, optional
    If 'lu', the ordering is optimized for LU factorization of `A`. If 'cholesky',
    the ordering is optimized for Cholesky factorization of :math:`A^{\\top}
    A`. If None, uses the default value from CCOLAMD, which is 'cholesky'."""

ccolamd.__doc__ = _CCOLAMD_DOC_TEMPLATE.format(
    intro=ccolamd_intro,
    A_param=ccolamd_A_param,
    opt_lu_param=ccolamd_opt_lu_param,
    reftag=ccolamd_reftag,
)


# Define the docstring for csymamd
csymamd_reftag = "[#csymamd_c]"

csymamd_intro = f"""Compute the column approximate minimum degree ordering of
a sparse symmetric matrix.

Adapted from the CCOLAMD documentation {csymamd_reftag}_:

    This function computes an approximate minimum degree ordering for
    Cholesky factorization of symmetric matrices.

    Symamd computes a permutation `P` of a symmetric matrix `A` such that
    the Cholesky factorization of :math:`PAP^{{\\top}}` has less fill-in and
    requires fewer floating point operations than `A`.  Symamd constructs
    a matrix `M` such that :math:`M^{{\\top}}M` has the same nonzero pattern
    of `A`, and then orders the columns of `M` using ccolamd.  The column
    ordering of `M` is then returned as the row and column ordering `P` of
    `A`.
"""

csymamd_A_param = """A : (N, N) array_like or sparse matrix
    The input matrix for which to compute the column ordering.
    Must be 2D, square, and convertible to CSC format.

    .. note::

        This routine only accesses the lower triangular part of ``A``,
        which is *assumed* to be symmetric. If it is not, the results may
        be incorrect or undefined.

"""

csymamd.__doc__ = _CCOLAMD_DOC_TEMPLATE.format(
    intro=csymamd_intro,
    A_param=csymamd_A_param,
    opt_lu_param='',
    reftag=csymamd_reftag,
)


def ccolamd_get_defaults():
    """Get the default knobs for CCOLAMD.

    Returns
    -------
    knobs : dict
        A dictionary containing the default knobs for CCOLAMD.

        The keys are:

        * 'dense_row_thresh': Threshold for considering a row/column dense.
          Rows with more than ``max(dense_row_thresh * sqrt(M), 16)`` entries
          are permuted to the end of the matrix.
        * 'dense_col_thresh': Like `dense_row_thresh`, but for columns.
        * 'aggressive': Default value for the aggressive knob.


    .. versionadded:: 0.5.0
    """
    knobs = np.zeros(CCOLAMD_KNOBS, dtype=np.double)
    cdef double[::1] knobs_view = knobs
    ccolamd_set_defaults(&knobs_view[0])
    return dict(
        dense_row_thresh=knobs[CCOLAMD_DENSE_ROW],
        dense_col_thresh=knobs[CCOLAMD_DENSE_COL],
        aggressive=knobs[CCOLAMD_AGGRESSIVE],
        opt_lu='lu' if knobs[CCOLAMD_LU] else 'cholesky',
    )
