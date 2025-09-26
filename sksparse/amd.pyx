# Cython AMD public Python interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: amd.pyx
#  Created: 2025-07-28 11:12
# =============================================================================

"""
===============================================================
Approximate Minimum Degree (AMD) Ordering (:mod:`sksparse.amd`)
===============================================================

.. currentmodule:: sksparse.amd

.. versionadded:: 0.5.0

Python interface to the `Approximate Minimum Degree (AMD)
<https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/AMD>`_ ordering
algorithm.


Interface
---------

.. autosummary::
   :toctree: generated/

   AMDInfo - Dataclass to hold information statistics returned by the AMD algorithm.
   amd - Main function to compute the AMD ordering.
   amd_default_control - Get the default control parameters for AMD.


Exceptions and Warnings
-----------------------

.. autosummary::
   :toctree: generated/

   AMDError - Base class for AMD-related errors.
   AMDInvalidMatrixError - Raised when the input matrix is invalid for AMD.
   AMDMemoryError - Raised when AMD runs out of memory.


References
----------

* SuiteSparse homepage:
  https://people.engr.tamu.edu/davis/suitesparse.html
* SuiteSparse AMD:
  https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/AMD
* AMD Algorithm Publication:
  Amestoy, P. R., Davis, T. A., & Duff, I. S. (1996). An approximate minimum
  degree ordering algorithm. *SIAM Journal on Matrix Analysis and
  Applications*, 17(4), 886-905.
"""

cimport cython

import numpy as np

from dataclasses import dataclass

from .utils import validate_csc_input

__all__ = [
    "AMDError",
    "AMDInvalidMatrixError",
    "AMDMemoryError",
    "AMDInfo",
    "amd",
    "amd_default_control"
]


ctypedef fused index_t:
    int32_t
    int64_t


class AMDError(Exception):
    """Base class for AMD-related errors."""
    pass


class AMDInvalidMatrixError(AMDError, ValueError):
    """Raised when the input matrix is invalid for AMD."""
    pass


class AMDMemoryError(AMDError, MemoryError):
    """Raised when AMD runs out of memory."""
    pass


@dataclass(frozen=True)
class AMDInfo:
    """Information statistics returned by the AMD algorithm.

    This class wraps the contents of the ``Info`` array output by
    ``amd_order()`` into a Python dataclass.

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
    Field descriptions are adapted from SuiteSparse ``amd.h`` [#amd_h]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [#amd_h] ``amd.h`` - SuiteSparse AMD header file.
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

    Adapted from the SuiteSparse `amd.h` documentation [0]_:

        AMD finds a fill-reducing ordering of a sparse matrix ``A``,
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
    dense_thresh : float, optional
        Threshold number of entries for considering a row/column dense. If
        None, use the default value from AMD. The default value is 10.

        Adapted from the SuiteSparse `amd.h` documentation [0]_:

            A dense row/column in ``A + A.T`` can cause AMD to spend a lot of
            time in ordering the matrix. If ``dense_thresh >= 0``, rows/columns
            with more than ``max(dense_thresh * sqrt(N), 16)`` entries are
            ignored during the ordering, and placed last in the output order.
            The default value of ``dense_thresh`` is 10. If negative, no
            rows/columns are treated as "dense". Rows/columns with 16 or fewer
            off-diagonal entries are never considered "dense".

    aggressive : bool, optional
        If True, use aggressive absorption. If None, uses the default value
        from AMD. The default value is True.

        Adapted from the SuiteSparse `amd.h` documentation [0]_:

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
    AMDInvalidMatrixError
        If the input matrix is invalid for AMD, such as having unsupported
        data types or formats.
    AMDMemoryError
        If the AMD algorithm runs out of memory during execution.

    Notes
    -----
    This function wraps the AMD (Approximate Minimum Degree) algorithm from
    the SuiteSparse by Timothy A. Davis. For details, see the SuiteSparse
    repository [2]_.

    .. versionadded:: 0.5.0

    References
    ----------
    .. [0] `amd.h` - Source header file from SuiteSparse.
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/AMD/Include/amd.h
    .. [1] SuiteSparse homepage.
        https://people.engr.tamu.edu/davis/suitesparse.html
    .. [2] SuiteSparse GitHub repository.
        https://github.com/DrTimothyAldenDavis/SuiteSparse
    """

    A, _, out_itype = validate_csc_input(A, require_square=True)

    N = A.shape[0]

    if N == 0:
        return np.empty(0, dtype=out_itype)

    if A.nnz == 0:
        return np.arange(N, dtype=out_itype)

    # Prepare control parameters
    ctrl = np.empty(AMD_CONTROL, dtype=np.double)
    cdef double[::1] ctrl_view = ctrl

    amd_defaults(&ctrl_view[0])

    # Update the defaults with user control parameters
    if dense_thresh is not None:
        ctrl_view[AMD_DENSE] = <double>dense_thresh

    if aggressive is not None:
        ctrl_view[AMD_AGGRESSIVE] = 1.0 if aggressive else 0.0

    info = np.zeros(AMD_INFO, dtype=np.double)

    # Prepare output permutation array
    p = np.empty(N, dtype=out_itype)

    # Compute the ordering
    _amd_order(N, A.indptr, A.indices, p, ctrl_view, info)

    if return_info:
        return p, AMDInfo.from_array(info)
    else:
        return p


@cython.boundscheck(False)
@cython.wraparound(False)
def _amd_order(
    Py_ssize_t N,
    index_t[::1] Ap,
    index_t[::1] Ai,
    index_t[::1] p,
    double[::1] ctrl,
    double[::1] info
):
    """Internal Cython wrapper for amd_order and amd_l_order.

    Parameters
    ----------
    N : int
        Number of rows and columns of the input matrix.
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
    """
    cdef int status

    # AMD ordering
    if index_t is int32_t:
        status = amd_order(N, &Ap[0], &Ai[0], &p[0], &ctrl[0], &info[0])
    else:
        status = amd_l_order(N, &Ap[0], &Ai[0], &p[0], &ctrl[0], &info[0])

    if status == AMD_OUT_OF_MEMORY:
        raise AMDMemoryError("amd: out of memory")
    elif status == AMD_INVALID:
        dump_info = AMDInfo.from_array(info)
        raise AMDInvalidMatrixError(f"amd: input matrix A is invalid:\n{dump_info}")


def amd_default_control():
    """Get the default control parameters for AMD.

    Returns
    -------
    control : dict
        A dictionary containing the default control parameters for AMD.

        The keys are:

        * 'dense_thresh': Threshold for considering a row/column dense. Rows or
          columns with more than ``max(dense_thresh * sqrt(N), 16)`` entries
          are permuted to the end of the matrix.
        * 'aggressive': Whether to use aggressive absorption.

    .. versionadded:: 0.5.0
    """
    cdef double[::1] ctrl_view = np.empty(AMD_CONTROL, dtype=np.float64)
    amd_defaults(&ctrl_view[0])
    return dict(
        dense_thresh=ctrl_view[AMD_DENSE],
        aggressive=bool(ctrl_view[AMD_AGGRESSIVE]),
    )
