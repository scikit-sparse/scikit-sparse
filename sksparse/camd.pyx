# Cython CAMD public Python interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: camd.pyx
#  Created: 2025-08-01 13:04
# =============================================================================
# cython: language_level=3

"""sksparse.camd: Python interface to the Approximate Minimum Degree (CAMD)
ordering algorithm.

This module provides a Cython interface to the CAMD algorithm from the
SuiteSparse library by Timothy A. Davis. The algorithm computes a fill-reducing
ordering of a sparse matrix, which is useful for improving the performance of
Cholesky or LU factorization and subsequent linear algebra operations.

Interfaces
----------
* `camd`: Main function to compute the CAMD ordering.
* `CAMDInfo`: Dataclass to hold information statistics returned by the CAMD
  algorithm.
* `camd_default_control`: Get the default control parameters for CAMD.

This wrapper handles both 32-bit and 64-bit integer indices, depending on the
input matrix format.

.. versionadded:: 0.5.0

References
----------
* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse CAMD:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CAMD
* AMD Algorithm Publication:
  Amestoy, P. R., Davis, T. A., & Duff, I. S. (1996). An approximate
    minimum degree ordering algorithm. SIAM Journal on Matrix Analysis and
    Applications, 17(4), 886-905.
"""

cimport cython

import numpy as np

from dataclasses import dataclass

from .utils import validate_csc_input

ctypedef fused index_t:
    int32_t
    int64_t


class CAMDError(Exception):
    """Base class for CAMD-related errors."""
    pass


class CAMDInvalidMatrixError(CAMDError, ValueError):
    """Raised when the input matrix is invalid for CAMD."""
    pass


class CAMDMemoryError(CAMDError, MemoryError):
    """Raised when CAMD runs out of memory."""
    pass


@dataclass(frozen=True)
class CAMDInfo:
    """Information statistics returned by the CAMD algorithm.

    This class wraps the contents of the ``Info`` array output by
    ``camd_order()`` into a Python dataclass.

    Attributes
    ----------
    status : int
        Return status:
          * 0 = OK,
          * 1 = OK but jumbled,
          * -1 = out of memory,
          * -2 = invalid matrix.
    N : int
        Number of rows and columns of the input matrix ``A``.
    nz : int
        Number of nonzeros in the input matrix ``A``.
    symmetry : float in [0, 1]
        Symmetry of pattern of ``A``. The symmetry is the number of "matched"
        off-diagonal entries divided by the total number of off-diagonal
        entries. An entry ``A[i, j]`` is matched if ``A[j, i]`` is also an
        entry, for any pair ``[i, j]`` where ``i != j``. In python code:

        .. code:: python

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
    Field descriptions are adapted from SuiteSparse ``camd.h`` [#camd_h]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#camd_h] ``camd.h`` - SuiteSparse CAMD header file.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CAMD/Include/camd.h
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
    def from_array(cls, info: "np.ndarray") -> "CAMDInfo":
        return cls(
            status=int(info[CAMD_STATUS]),
            N=int(info[CAMD_N]),
            nz=int(info[CAMD_NZ]),
            symmetry=float(info[CAMD_SYMMETRY]),
            nzdiag=int(info[CAMD_NZDIAG]),
            nz_A_plus_AT=int(info[CAMD_NZ_A_PLUS_AT]),
            Ndense=int(info[CAMD_NDENSE]),
            memory=float(info[CAMD_MEMORY]),
            Ncmpa=int(info[CAMD_NCMPA]),
            Lnz=int(info[CAMD_LNZ]),
            Ndiv=int(info[CAMD_NDIV]),
            Nmultsubs_LDL=int(info[CAMD_NMULTSUBS_LDL]),
            Nmultsubs_LU=int(info[CAMD_NMULTSUBS_LU]),
            dmax=int(info[CAMD_DMAX]),
        )


def camd(A, constraints=None, dense_thresh=None, aggressive=None, return_info=False):
    """Compute the approximate minimum degree ordering of a sparse matrix.

    Adapted from the SuiteSparse `camd.h` documentation [0]_:

        CAMD finds a fill-reducing ordering of a sparse matrix ``A``,
        using the approximate minimum degree algorithm. The output is
        a permutation vector ``p`` such that the Cholesky factor of
        ``A[p][:, p]`` has fewer nonzeros than the Cholesky factor of ``A``.
        If ``A`` is not symmetric, the algorithm computes an ordering of
        ``A + A.T``.

    For more details on the entire package, see the SuiteSparse homepage [1]_
    and Github repository [2]_.

    Parameters
    ----------
    A : (N, N) array_like or sparse matrix
        A square matrix in CSC format or convertible to CSC.
    constraints : (N,) array_like, optional
        A 1D array of constraints for the ordering. Each node `i` in the graph
        of `A` has a constraint, ``constraints[i]``, in the range [0, N-1]. All
        nodes with ``constraints[i] = 0`` are ordered first, followed by nodes
        with `C(i) = 1`, and so on. Thus, ``constraints[p]`` is monotonically
        non-decreasing. If None, no constraints are applied, and the ordering
        will be similar to :func:`~sksparse.amd.amd`, except that the
        post-ordering is different.
    dense_thresh : float, optional
        Threshold number of entries for considering a row/column dense. If
        None, use the default value from CAMD. The default value is 10.

        Adapted from the SuiteSparse `camd.h` documentation [0]_:

            A dense row/column in ``A + A.T`` can cause CAMD to spend a lot of
            time in ordering the matrix. If ``dense_thresh >= 0``, rows/columns
            with more than ``max(dense_thresh * sqrt(N), 16)`` entries are
            ignored during the ordering, and placed last in the output order.
            The default value of ``dense_thresh`` is 10. If negative, no
            rows/columns are treated as "dense". Rows/columns with 16 or fewer
            off-diagonal entries are never considered "dense".

    aggressive : bool, optional
        If True, use aggressive absorption. If None, uses the default value
        from CAMD. The default value is True.

        Adapted from the SuiteSparse `camd.h` documentation [0]_:

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

    Raises
    ------
    SparseEfficiencyWarning
        If the input matrix is not in CSC format, a warning is raised and the
        matrix is converted to CSC format.
    ValueError
        If the input matrix is not square or cannot be converted to CSC format.
    CAMDInvalidMatrixError
        If the input matrix is invalid for CAMD, such as having unsupported
        data types or formats.
    CAMDMemoryError
        If the CAMD algorithm runs out of memory during execution.

    Notes
    -----
    This function wraps the CAMD (Approximate Minimum Degree) algorithm from
    the SuiteSparse by Timothy A. Davis. For details, see the SuiteSparse
    repository [2]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [0] `camd.h` - Source header file from SuiteSparse.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CAMD/Include/camd.h
    .. [1] SuiteSparse homepage.
        https://people.engr.tamu.edu/davis/suitesparse.html
    .. [2] SuiteSparse GitHub repository.
        https://github.com/DrTimothyAldenDavis/SuiteSparse
    """
    A, _, out_itype = validate_csc_input(A, require_square=True)

    cdef Py_ssize_t N = A.shape[0]

    if N == 0:
        return np.empty(0, dtype=out_itype)

    if A.nnz == 0:
        return np.arange(N, dtype=out_itype)

    # Prepare control parameters
    ctrl = np.empty(CAMD_CONTROL, dtype=np.double)
    cdef double[::1] ctrl_view = ctrl

    camd_defaults(&ctrl_view[0])

    # Update the defaults with user control parameters
    if dense_thresh is not None:
        ctrl_view[CAMD_DENSE] = <float>dense_thresh

    if aggressive is not None:
        ctrl_view[CAMD_AGGRESSIVE] = 1.0 if aggressive else 0.0

    info = np.zeros(CAMD_INFO, dtype=np.double)

    # Prepare output permutation array
    p = np.empty(N, dtype=out_itype)

    if constraints is not None:
        # Convert the dtype so the user doesn't have to
        try:
            constraints = np.ascontiguousarray(constraints, dtype=out_itype)
        except TypeError:
            raise TypeError("Constraints must be an array of integers.")

    _camd_order(N, A.indptr, A.indices, p, ctrl_view, info, constraints)

    if return_info:
        return p, CAMDInfo.from_array(info)
    else:
        return p


@cython.boundscheck(False)
@cython.wraparound(False)
def _camd_order(
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    index_t[::1] p,
    double[::1] ctrl,
    double[::1] info,
    index_t[::1] constraints=None,
):
    """Internal Cython wrapper for amd_order and amd_l_order.

    Parameters
    ----------
    Ap : array_like
        Column pointer array of the CSC matrix.
    Ai : array_like
        Row indices array of the CSC matrix.
    p : array_like
        Output permutation array.
    ctrl : array_like
        Control parameters array.
    info : array_like
        Output information array.
    constraints : array_like, optional
        Constraints array.
    """
    cdef int status

    # Prepare constraints
    # Use a raw pointer to pass NULL if no constraints are given
    cdef index_t* constraints_ptr = NULL

    if constraints is not None:
        if constraints.shape[0] != N:
            raise ValueError("Constraints must have the same length as the matrix size.")

        constraints_ptr = &constraints[0]

    # CAMD ordering
    if index_t is int32_t:
        status = camd_order(
            N,
            &Ap[0],
            &Ai[0],
            &p[0],
            &ctrl[0],
            &info[0],
            constraints_ptr
        )
    else:
        status = camd_l_order(
            N,
            &Ap[0],
            &Ai[0],
            &p[0],
            &ctrl[0],
            &info[0],
            constraints_ptr
        )

    if status == CAMD_OUT_OF_MEMORY:
        raise CAMDMemoryError("camd: out of memory")
    elif status == CAMD_INVALID:
        dump_info = CAMDInfo.from_array(info)
        raise CAMDInvalidMatrixError(f"camd: input matrix A is invalid:\n{dump_info}")


def camd_default_control():
    """Get the default control parameters for CAMD.

    Returns
    -------
    control : dict
        A dictionary containing the default control parameters for CAMD.

        The keys are:

        * 'dense_thresh': Threshold for considering a row/column dense. Rows or
          columns with more than ``max(dense_thresh * sqrt(N), 16)`` entries
          are permuted to the end of the matrix.
        * 'aggressive': Whether to use aggressive absorption.

    .. versionadded:: 0.5.0
    """
    cdef double[::1] ctrl_view = np.empty(CAMD_CONTROL, dtype=np.float64)
    camd_defaults(&ctrl_view[0])
    return dict(
        dense_thresh=ctrl_view[CAMD_DENSE],
        aggressive=bool(ctrl_view[CAMD_AGGRESSIVE]),
    )
