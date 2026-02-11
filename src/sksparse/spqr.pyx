# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: spqr.pyx
#  Created: 2025-11-06 20:19
# =============================================================================

"""
==============================================
Sparse QR Decomposition (:mod:`sksparse.spqr`)
==============================================

.. currentmodule:: sksparse.spqr

.. versionadded:: 0.5.0


An interface to the SuiteSparse `SPQR
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR>`_
package, which computes the QR factorization and solves systems of equations
for sparse, possibly non-square, non-symmetric, indefinite matrices.


Function Interface
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    spqr - Compute the SPQR factorization of a sparse matrix.
    spqr_qmult - Multiply by Q from the SPQR factorization.
    spqr_solve - Solve a linear system using the SPQR factorization.
    SPQRHouseholder - A class representing the Householder vectors.


Object Interface
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    spqr_factor - Compute the QR factorization of a sparse matrix.
    SPQRFactor - An object-oriented interface to SPQR.
    SPQRInfo - A dataclass to return SPQR info.


.. _spqr-exceptions:

Warnings and Exceptions
-----------------------

.. autosummary::
    :toctree: generated/

    SPQRWarning
    SPQRRankDeficiencyWarning

    SPQRError
    SPQRNotInstalledError
    SPQROutOfMemoryError
    SPQROverflowError
    SPQRInvalidInputError
    SPQRGpuProblemError


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse SPQR <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/SPQR>`_
"""

cimport cython
from cython cimport doublecomplex as cdouble
cimport numpy as cnp

from sksparse.cholmod cimport (
    CHOLMOD_OK,
    CHOLMOD_GPU_PROBLEM,
    CHOLMOD_INVALID,
    CHOLMOD_NOT_INSTALLED,
    CHOLMOD_OUT_OF_MEMORY,
    CHOLMOD_TOO_LARGE,
    CHOLMOD_INT,
    CHOLMOD_REAL,
    cholmod_start,
    cholmod_l_start,
    cholmod_finish,
    cholmod_l_finish,
    _ndarray_copy_from_intptr,
    _cholmod_dense_from_ndarray,
    _ndarray_from_cholmod_dense,
    _copy_cholmod_common,
    _csc_from_cholmod_sparse,
    cholmod_free,
    cholmod_l_free,
)

import numpy as np
from scipy.sparse import csc_array, issparse, hstack
from typing import NamedTuple
import warnings

from sksparse.cholmod import _cholmod_sparse_from_csc

from .utils import validate_csc_input


__all__ = [
    "SPQRError",
    "SPQRNotInstalledError",
    "SPQROutOfMemoryError",
    "SPQROverflowError",
    "SPQRInvalidInputError",
    "SPQRGpuProblemError",
    "SPQRFactor",
    "spqr_factor",
    "spqr_solve",
]


ctypedef fused index_t:
    int32_t
    int64_t


ctypedef fused value_t:
    double
    double complex


# Define specific instantiations
ctypedef spqr_symbolic[int32_t] spqr_symbolic_i
ctypedef spqr_symbolic[int64_t] spqr_symbolic_l

ctypedef fused symbolic_t:
    spqr_symbolic_i
    spqr_symbolic_l


ctypedef spqr_numeric[double, int32_t] spqr_numeric_di
ctypedef spqr_numeric[double, int64_t] spqr_numeric_dl
ctypedef spqr_numeric[cdouble, int32_t] spqr_numeric_zi
ctypedef spqr_numeric[cdouble, int64_t] spqr_numeric_zl

ctypedef fused numeric_t:
    spqr_numeric_di
    spqr_numeric_dl
    spqr_numeric_zi
    spqr_numeric_zl


ctypedef SuiteSparseQR_factorization[double, int32_t] spqr_factor_di
ctypedef SuiteSparseQR_factorization[double, int64_t] spqr_factor_dl
ctypedef SuiteSparseQR_factorization[cdouble, int32_t] spqr_factor_zi
ctypedef SuiteSparseQR_factorization[cdouble, int64_t] spqr_factor_zl

ctypedef fused factor_t:
    spqr_factor_di
    spqr_factor_dl
    spqr_factor_zi
    spqr_factor_zl


# NOTE These are not defined in the header files, so we define them here
cdef:
    int SPQR_ISTAT_NNZR_UPPER = 0
    int SPQR_ISTAT_NNZH_UPPER = 1
    int SPQR_ISTAT_NFRONTAL = 2
    int SPQR_ISTAT_NTASKS = 3
    int SPQR_ISTAT_EST_RANKA = 4
    int SPQR_ISTAT_COL_SINGLETONS = 5
    int SPQR_ISTAT_ROW_SINGLETONS = 6
    int SPQR_ISTAT_ORDERING = 7


# -------------------------------------------------------------------------------------
#         Error Handling
# -------------------------------------------------------------------------------------
class SPQRWarning(Warning):
    """Base class for SPQR warnings."""
    pass


class SPQRRankDeficiencyWarning(SPQRWarning):
    """Raised when SPQR detects a rank-deficient matrix."""
    pass


class SPQRError(Exception):
    """Base class for SPQR exceptions."""
    pass


class SPQRNotInstalledError(SPQRError):
    """Raised when the SPQR library is not installed."""
    pass


class SPQROutOfMemoryError(MemoryError, SPQRError):
    """Raised when SPQR runs out of memory."""
    pass


class SPQROverflowError(SPQRError):
    """Raised when SPQR encounters an integer overflow."""
    pass


class SPQRInvalidInputError(SPQRError):
    """Raised when SPQR receives invalid input."""
    pass


class SPQRGpuProblemError(SPQRError):
    """Raised when SPQR encounters a problem with CUDA."""
    pass


# Known Errors -- SPQR uses CHOLMOD error codes
cdef dict _ERROR_INDEX = {
    CHOLMOD_NOT_INSTALLED: (
        SPQRNotInstalledError, "SPQR library is not installed or not found."
    ),
    CHOLMOD_OUT_OF_MEMORY: (SPQROutOfMemoryError, "SPQR ran out of memory."),
    CHOLMOD_TOO_LARGE: (SPQROverflowError, "SPQR encountered an integer overflow."),
    CHOLMOD_INVALID: (SPQRInvalidInputError, "SPQR received invalid input."),
    CHOLMOD_GPU_PROBLEM: (SPQRGpuProblemError, "SPQR encountered a problem with CUDA."),
}


cdef int _handle_errors(int status) except -1 with gil:
    """Handle SPQR errors by raising Python exceptions or warnings.

    This function should be called with the return ``status`` after any SPQR
    C function that may fail.

    Parameters
    ----------
    status : int
        The SPQR exit status code.

    Returns
    -------
    None

    Raises
    ------
    :exc:`SPQRWarning` or subclass
        Raises a warning for non-critical issues.
    :exc:`SPQRError` or subclass
        Raises an appropriate Python exception based on the SPQR status code.
    """
    if status == CHOLMOD_OK:
        return 0

    # Fallback to generic error for unknown codes
    exc_class, msg = _ERROR_INDEX.get(
        status,
        (SPQRError, "An unknown SPQR error occurred.")
    )
    full_msg = f"{msg} (code {status:d})"

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class, stacklevel=2)
    else:
        raise exc_class(full_msg)


# -------------------------------------------------------------------------------------
#         Info and Control
# -------------------------------------------------------------------------------------
cdef dict _ordering_methods = {
    "default": SPQR_ORDERING_DEFAULT,
    "fixed": SPQR_ORDERING_FIXED,
    "natural": SPQR_ORDERING_NATURAL,
    "colamd": SPQR_ORDERING_COLAMD,
    "cholmod": SPQR_ORDERING_CHOLMOD,
    "amd": SPQR_ORDERING_AMD,
    "metis": SPQR_ORDERING_METIS,
    "best": SPQR_ORDERING_BEST,
    "bestamd": SPQR_ORDERING_BESTAMD,
}


cdef dict _ordering_methods_inv = {v: k for k, v in _ordering_methods.items()}


cdef inline int _qmult_int_from_str(str method) except -1:
    """Return the SPQR qmult method constant from string."""
    if method == 'QX':
        return SPQR_QX
    elif method == 'QTX':
        return SPQR_QTX
    elif method == 'XQ':
        return SPQR_XQ
    elif method == 'XQT':
        return SPQR_XQT
    else:
        raise ValueError(
            f"Invalid method '{method}'. "
            "Expected one of ['QX', 'QTX', 'XQ', 'XQT']."
        )


@cython.dataclasses.dataclass(frozen=True)
cdef class SPQRInfo:
    """A dataclass to hold SPQR info statistics.

    Attributes
    ----------
    nnzR_upper_bound : int
        Bound on the number of nonzeros in ``R``.
    nnzH_upper_bound : int
        Bound on the number of nonzeros in ``H``.
    nf : int
        Number of frontal matrices.
    rank_A_estimate : int
        Estimated rank of ``A``.
    n1cols : int
        Number of singleton columns.
    n1rows : int
        Number of singleton rows.
    ordering : str
        Ordering method used.
    memory : int
        Memory usage in bytes.
    flops_upper_bound : int
        Upper bound on flop count (excluding backsolve).
    tol : float
        Column norm tolerance used.
    norm_E_fro : float
        Norm of dropped diagonal of R.
    analyze_time : float
        Time taken for the symbolic analysis in seconds.
    factorize_time : int
        Time take for the numeric factorization (including applying ``Q.T``)
    solve_time : int
        Time taken for the backsolve only :math:`R x = Q^T b` in seconds.
    total_time : int
        Total time in seconds.
    flops : int
        Actual flops for the factorization and solve (including backsolve).
    """
    nnzR_upper_bound : int | None = None
    nnzH_upper_bound : int | None = None
    nf : int | None = None
    rank_A_estimate : int | None = None
    n1cols : int | None = None
    n1rows : int | None = None
    ordering : str | None = None
    memory : int | None = None
    flops_upper_bound : int | None = None
    tol : float | None = None
    norm_E_fro : float | None = None
    analyze_time : float | None = None
    factorize_time : float | None = None
    solve_time : float | None = None
    total_time : float | None = None
    flops : int | None = None

    # __init__ can't take a C pointer, so we use a separate method
    cdef int _init_from_common(self, cholmod_common* cm) except -1:
        """Initialize SPQRInfo from a cholmod_common object."""
        assert cm is not NULL
        self.nnzR_upper_bound = cm.SPQR_istat[SPQR_ISTAT_NNZR_UPPER]
        self.nnzH_upper_bound = cm.SPQR_istat[SPQR_ISTAT_NNZH_UPPER]
        self.nf = cm.SPQR_istat[SPQR_ISTAT_NFRONTAL]
        self.rank_A_estimate = cm.SPQR_istat[SPQR_ISTAT_EST_RANKA]
        self.n1cols = cm.SPQR_istat[SPQR_ISTAT_COL_SINGLETONS]
        self.n1rows = cm.SPQR_istat[SPQR_ISTAT_ROW_SINGLETONS]
        cdef int order = cm.SPQR_istat[SPQR_ISTAT_ORDERING]
        self.ordering = _ordering_methods_inv.get(order, f"unknown {order}")
        self.memory = cm.memory_usage
        self.flops_upper_bound = cm.SPQR_flopcount_bound
        self.tol = cm.SPQR_tol_used
        self.norm_E_fro = cm.SPQR_norm_E_fro
        self.analyze_time = cm.SPQR_analyze_time
        self.factorize_time = cm.SPQR_factorize_time
        self.solve_time = cm.SPQR_solve_time
        self.total_time = self.analyze_time + self.factorize_time + self.solve_time
        self.flops = cm.SPQR_flopcount


# -------------------------------------------------------------------------------------
#         Copy Functions
# -------------------------------------------------------------------------------------
cdef inline void* _malloc_copy(
    const void* src,
    size_t n,
    size_t size
):
    """Allocate memory and copy data from src to the new memory."""
    if src is NULL:
        return NULL

    cdef void* dest = malloc(n * size)

    if dest is NULL:
        return NULL

    if n > 0:
        memcpy(dest, src, n * size)

    return dest


cdef int _copy_spqr_symbolic_base(
    symbolic_t* dest,
    const symbolic_t* src,
    index_t _dummy_idx=0,
) except -1:
    """Deep copy a SuiteSparseQR_symbolic object."""
    assert dest is not NULL
    assert src is not NULL

    # Prune invalid type combinations
    if not (
        (symbolic_t is spqr_symbolic_i and index_t is int32_t)
        or (symbolic_t is spqr_symbolic_l and index_t is int64_t)
    ):
        assert False
        return 0

    dest.m = src.m
    dest.n = src.n
    dest.anz = src.anz

    dest.Sp = <index_t*>_malloc_copy(src.Sp, src.m + 1, sizeof(index_t))
    dest.Sj = <index_t*>_malloc_copy(src.Sj, src.anz, sizeof(index_t))

    dest.Qfill = <index_t*>_malloc_copy(src.Qfill, src.n, sizeof(index_t))
    dest.PLinv = <index_t*>_malloc_copy(src.PLinv, src.m, sizeof(index_t))
    dest.Sleft = <index_t*>_malloc_copy(src.Sleft, src.n + 2, sizeof(index_t))

    dest.nf = src.nf
    dest.maxfn = src.maxfn

    dest.Parent = <index_t*>_malloc_copy(src.Parent, src.nf + 1, sizeof(index_t))
    dest.Child = <index_t*>_malloc_copy(src.Child, src.nf + 1, sizeof(index_t))
    dest.Childp = <index_t*>_malloc_copy(src.Childp, src.nf + 2, sizeof(index_t))

    dest.Super = <index_t*>_malloc_copy(src.Super, src.nf + 1, sizeof(index_t))

    dest.Rp = <index_t*>_malloc_copy(src.Rp, src.nf + 1, sizeof(index_t))
    dest.Rj = <index_t*>_malloc_copy(src.Rj, src.rjsize, sizeof(index_t))
    dest.Post = <index_t*>_malloc_copy(src.Post, src.nf + 1, sizeof(index_t))

    dest.rjsize = src.rjsize
    dest.do_rank_detection = src.do_rank_detection
    dest.maxstack = src.maxstack
    dest.hisize = src.hisize
    dest.keepH = src.keepH

    dest.Hip = <index_t*>_malloc_copy(src.Hip, src.nf + 1, sizeof(index_t))

    dest.ntasks = src.ntasks
    dest.ns = src.ns

    if dest.ntasks > 1:
        raise NotImplementedError("SPQR task parallelism and GPU not yet supported.")

    dest.TaskChildp = NULL
    dest.TaskChild = NULL

    dest.TaskStack = NULL

    dest.TaskFront = NULL
    dest.TaskFrontp = NULL

    dest.On_stack = NULL

    dest.Stack_maxstack = NULL
    dest.Fm = NULL
    dest.Cm = NULL

    # Values used in GPU factorization
    dest.maxcsize = src.maxcsize
    dest.maxesize = src.maxesize
    dest.ColCount = NULL

    # Not yet supported
    dest.QRgpu = NULL

    return 0


cdef inline int _copy_spqr_symbolic(
    symbolic_t* dest,
    const symbolic_t* src
) except -1:
    """Deep copy a spqr_symbolic struct."""
    if symbolic_t is spqr_symbolic_i:
        return _copy_spqr_symbolic_base[spqr_symbolic_i, int32_t](dest, src)
    else:  # symbolic_t is spqr_symbolic_l
        return _copy_spqr_symbolic_base[spqr_symbolic_l, int64_t](dest, src)


cdef int _copy_spqr_numeric_base(
    numeric_t* dest,
    const numeric_t* src,
    value_t _dummy_val=0,
    index_t _dummy_idx=0,
) except -1:
    """Deep copy a SuiteSparseQR_numeric object."""
    assert dest is not NULL
    assert src is not NULL

    # Prune invalid type combinations
    if not (
        (numeric_t is spqr_numeric_di and index_t is int32_t and value_t is double)
        or (numeric_t is spqr_numeric_dl and index_t is int64_t and value_t is double)
        or (numeric_t is spqr_numeric_zi and index_t is int32_t and value_t is cdouble)
        or (numeric_t is spqr_numeric_zl and index_t is int64_t and value_t is cdouble)
    ):
        assert False
        return 0

    dest.Stacks = <value_t**>malloc(src.ns * sizeof(value_t*))
    dest.Stack_size = <index_t*>_malloc_copy(
        src.Stack_size, src.ns, sizeof(index_t)
    )

    # Deep copy each stack
    cdef size_t k

    if (
        dest.Stacks is not NULL
        and src.Stacks is not NULL
        and src.Stack_size is not NULL
    ):
        for k in range(src.ns):
            dest.Stacks[k] = <value_t*>_malloc_copy(
                src.Stacks[k], src.Stack_size[k], sizeof(value_t)
            )

    # Point each Rblock to the copied stacks
    # See: SPQR/Source/spqr_kernel.cpp:186 for Rblock assignment logic
    dest.Rblock = <value_t**>malloc(src.nf * sizeof(value_t*))

    cdef:
        size_t s              # index for stacks
        size_t stack_size     # size of stack in bytes
        size_t offset         # byte offset within stack
        int f                 # stack index that Rblock points to
        value_t *stack_start  # pointer to start of stack
        value_t *rblock_ptr   # pointer to Rblock[k]

    if dest.Rblock is not NULL and src.Rblock is not NULL:
        for k in range(src.nf):
            rblock_ptr = src.Rblock[k]
            if rblock_ptr is NULL:
                continue

            # Find which stack this Rblock points to
            f = -1
            for s in range(src.ns):
                # Check if the pointer is in this stack
                stack_start = src.Stacks[s]
                if stack_start is NULL:
                    continue

                stack_size = src.Stack_size[s] * sizeof(value_t)

                # Check if the Rblock poitner falls within the memory range of stack s
                if (<char*>rblock_ptr >= <char*>stack_start) and (
                    <char*>rblock_ptr < <char*>stack_start + stack_size
                ):
                    f = s
                    break

            # Compute the offset within the stack
            if f == -1:
                raise SPQRError("Failed to copy Rblock pointers.")
            else:
                # Compute the byte offset to Rblock[k] within the stack
                offset = <char*>rblock_ptr - <char*>src.Stacks[f]
                # Apply the same offset to the copied stack
                dest.Rblock[k] = <value_t*>(<char*>dest.Stacks[f] + offset)

    dest.hisize = src.hisize
    dest.m = src.m
    dest.n = src.n
    dest.nf = src.nf
    dest.ntasks = src.ntasks
    dest.ns = src.ns
    dest.maxstack = src.maxstack

    dest.Rdead = <char*>_malloc_copy(src.Rdead, src.n, sizeof(char))

    dest.rank = src.rank
    dest.rank1 = src.rank1
    dest.maxfrank = src.maxfrank
    dest.norm_E_fro = src.norm_E_fro

    dest.keepH = src.keepH
    dest.rjsize = src.rjsize

    dest.HStair = <index_t*>_malloc_copy(src.HStair, src.rjsize, sizeof(index_t))
    dest.HTau = <value_t*>_malloc_copy(src.HTau, src.rjsize, sizeof(value_t))

    dest.Hii = <index_t*>_malloc_copy(src.Hii, src.hisize, sizeof(index_t))
    dest.HPinv = <index_t*>_malloc_copy(src.HPinv, src.m, sizeof(index_t))

    dest.Hm = <index_t*>_malloc_copy(src.Hm, src.nf, sizeof(index_t))
    dest.Hr = <index_t*>_malloc_copy(src.Hr, src.nf, sizeof(index_t))

    dest.maxfm = src.maxfm

    return 0


cdef inline int _copy_spqr_numeric(
    numeric_t* dest,
    const numeric_t* src,
) except -1:
    """Deep copy a spqr_numeric struct."""
    if numeric_t is spqr_numeric_di:
        return _copy_spqr_numeric_base[spqr_numeric_di, double, int32_t](dest, src)
    elif numeric_t is spqr_numeric_dl:
        return _copy_spqr_numeric_base[spqr_numeric_dl, double, int64_t](dest, src)
    elif numeric_t is spqr_numeric_zi:
        return _copy_spqr_numeric_base[spqr_numeric_zi, cdouble, int32_t](dest, src)
    else:  # numeric_t is spqr_numeric_zl
        return _copy_spqr_numeric_base[spqr_numeric_zl, cdouble, int64_t](dest, src)


cdef int _copy_spqr_factor_base(
    factor_t* dest,
    const factor_t* src,
    value_t _dummy_val=0,
    index_t _dummy_idx=0,
) except -1:
    """Deep copy a SuiteSparseQR_factorization object."""
    assert dest is not NULL
    assert src is not NULL

    # Validate types to ensure correct instantiation. Invalid combos will be pruned.
    if not (
        (factor_t is spqr_factor_di and index_t is int32_t and value_t is double)
        or (factor_t is spqr_factor_dl and index_t is int64_t and value_t is double)
        or (factor_t is spqr_factor_zi and index_t is int32_t and value_t is cdouble)
        or (factor_t is spqr_factor_zl and index_t is int64_t and value_t is cdouble)
    ):
        assert False
        return 0

    dest.tol = src.tol

    # Deep copy symbolic factorization (if present)
    dest.QRsym = NULL

    if src.QRsym is not NULL:
        if index_t is int32_t:
            dest.QRsym = <spqr_symbolic_i*>malloc(sizeof(spqr_symbolic_i))
        else:
            dest.QRsym = <spqr_symbolic_l*>malloc(sizeof(spqr_symbolic_l))

        _copy_spqr_symbolic(dest.QRsym, src.QRsym)

    # Deep copy numeric factorization (if present)
    dest.QRnum = NULL

    if src.QRnum is not NULL:
        if factor_t is spqr_factor_di:
            dest.QRnum = <spqr_numeric_di*>malloc(sizeof(spqr_numeric_di))
        elif factor_t is spqr_factor_dl:
            dest.QRnum = <spqr_numeric_dl*>malloc(sizeof(spqr_numeric_dl))
        elif factor_t is spqr_factor_zi:
            dest.QRnum = <spqr_numeric_zi*>malloc(sizeof(spqr_numeric_zi))
        else:  # factor_t is spqr_factor_zl
            dest.QRnum = <spqr_numeric_zl*>malloc(sizeof(spqr_numeric_zl))

        _copy_spqr_numeric(dest.QRnum, src.QRnum)

    dest.R1p = <index_t*>_malloc_copy(src.R1p, src.n1rows + 1, sizeof(index_t))
    dest.R1j = <index_t*>_malloc_copy(src.R1j, src.r1nz, sizeof(index_t))
    dest.R1x = <value_t*>_malloc_copy(src.R1x, src.r1nz, sizeof(value_t))
    dest.r1nz = src.r1nz

    cdef size_t m = src.narows
    cdef size_t n = src.nacols

    dest.Q1fill = <index_t*>_malloc_copy(src.Q1fill, n + src.bncols, sizeof(index_t))
    dest.P1inv = <index_t*>_malloc_copy(src.P1inv, m, sizeof(index_t))
    dest.HP1inv = <index_t*>_malloc_copy(src.HP1inv, m, sizeof(index_t))

    dest.Rmap = <index_t*>_malloc_copy(src.Rmap, n, sizeof(index_t))
    dest.RmapInv = <index_t*>_malloc_copy(src.RmapInv, n, sizeof(index_t))

    dest.n1rows = src.n1rows
    dest.n1cols = src.n1cols
    dest.narows = src.narows
    dest.nacols = src.nacols
    dest.bncols = src.bncols
    dest.rank = src.rank
    dest.allow_tol = src.allow_tol

    return 0


cdef inline int _copy_spqr_factor(
    factor_t* dest,
    const factor_t* src,
) except -1:
    """Deep copy a SuiteSparseQR_factorization struct."""
    if factor_t is spqr_factor_di:
        return _copy_spqr_factor_base[spqr_factor_di, double, int32_t](dest, src)
    elif factor_t is spqr_factor_dl:
        return _copy_spqr_factor_base[spqr_factor_dl, double, int64_t](dest, src)
    elif factor_t is spqr_factor_zi:
        return _copy_spqr_factor_base[spqr_factor_zi, cdouble, int32_t](dest, src)
    else:  # factor_t is spqr_factor_zl
        return _copy_spqr_factor_base[spqr_factor_zl, cdouble, int64_t](dest, src)


# -------------------------------------------------------------------------------------
#         SPQR Factor Class
# -------------------------------------------------------------------------------------
cdef class SPQRFactor:
    r"""The main object used for creating and manipulating SPQR factorizations.

    The constructor computes the sybolic analysis of the matrix and determines
    a fill-reducing ordering such that:

    .. math::

        Q R = A E

    where :math:`E` is a column permutation matrix, :math:`Q` is an orthogonal
    matrix, and :math:`R` is an upper-triangular matrix.

    The numerical factorization is computed in one of two ways:

    1. by setting ``use_singletons=True`` in the constructor, which computes
        both the symbolic and numeric factorizations at once, or
    2. by calling :meth:`SPQRFactor.factorize`, which computes the numeric
        factorization after symbolic analysis has been performed.

    The first method is useful when factoring a single matrix, but solving multiple
    right-hand sides.

    The second method is useful when factoring multiple matrices with the same sparsity
    pattern but different numerical values.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    use_singletons : bool, optional
        If True, directly compute the numeric factorization to exploit singleton rows.
        Otherwise, only perform symbolic analysis. Default is False.
    order : str, optional
        The column ordering strategy to use. Let :math:`S` be the matrix :math:`A` with
        singleton rows/columns removed, the ordering options are:

        * ``default``: COLAMD(S),
        * ``fixed``: identity permutation (*i.e.* no singletons removed),
        * ``natural``: singletons removed, but no fill-reducing ordering applied,
        * ``colamd``: COLAMD(S),
        * ``amd``: AMD(:math:`S^{\top} S`),
        * ``metis``: METIS(:math:`S^{\top} S`),
        * ``best``: try all of ``amd``, ``colamd``, ``metis`` and pick the best,
        * ``cholmod``: Same as ``best``,
        * ``bestamd``: try ``amd`` and ``colamd`` and pick the best.

    tol : float, optional
        If the 2-norm of a column in ``A`` is less than ``tol``, that column is
        considered to be a zero column. If ``tol = 0``, no columns are treated as zero.
        If ``None``, the default tolerance is used. The default is
        ``tol =`` :math:`20 \epsilon (M + N) \sqrt{\max{\mathrm{diag}(A^{\top} A)}}`,
        where :math:`\epsilon` is the machine precision.


    Attributes
    ----------
    is_numeric : bool
        Whether the numeric factorization has been computed.
    shape : tuple
        The shape of the input matrix (M, N).
    itype : ~numpy.dtype
        The integer type used for indices (``int32`` or ``int64``).
    dtype : ~numpy.dtype
        The data type of the matrix (``float64`` or ``complex128``).
    rank : int
        The rank of the matrix as determined by SPQR.
    perm : ~numpy.ndarray of int
        The combined singleton and fill-reducing column permutation vector.
    info : SPQRInfo
        An object containing various SPQR statistics.

    See Also
    --------
    spqr_factor, spqr, spqr_qmult, spqr_solve

    Notes
    -----
    This object is an interface to the SuiteSparse SPQR library [#spqr_url]_.


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#spqr_url] SuiteSparse SPQR
        https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR
    """

    cdef:
        cholmod_common _common
        cholmod_common *_cm
        spqr_factor_di *_factor_di
        spqr_factor_dl *_factor_dl
        spqr_factor_zi *_factor_zi
        spqr_factor_zl *_factor_zl
        bint _use_int32
        bint _is_real
        int _M
        int _N
        readonly object itype
        readonly object dtype
        double _tol

    def __init__(
        self,
        object A,
        *,
        bint use_singletons=False,
        object order=None,
        object tol=None,
    ):
        """Initialize the SPQRFactor object and perform symbolic analysis."""
        A, _, _ = validate_csc_input(A)

        # Validate inputs
        cdef int ordering

        if order is None:
            ordering = SPQR_ORDERING_DEFAULT
        else:
            try:
                ordering = _ordering_methods[order]
            except KeyError:
                raise ValueError(
                    "Unknown ordering method: {ordering}. "
                    f"Must be one of {set(_ordering_methods.keys())}."
                )

        self._tol = <double>tol if tol is not None else SPQR_DEFAULT_TOL

        # Get the input matrix into CHOLMOD format
        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix
        cdef int stype = 0  # assume matrix is not symmetric

        _cholmod_sparse_from_csc(
            A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
        )

        self._use_int32 = (Ac.itype == CHOLMOD_INT)
        self._is_real = (Ac.xtype == CHOLMOD_REAL)

        # Initialize the common object
        self._cm = &self._common

        if self._use_int32:
            cholmod_start(self._cm)
        else:
            cholmod_l_start(self._cm)

        cdef bint allow_tol = True  # if False, do not perform rank detection

        if use_singletons:
            # Perform both symbolic and numeric factorization
            if self._is_real:
                if self._use_int32:
                    self._factor_di = SuiteSparseQR_factorize[double, int32_t](
                        ordering, self._tol, Ac, self._cm
                    )
                else:
                    self._factor_dl = SuiteSparseQR_factorize[double, int64_t](
                        ordering, self._tol, Ac, self._cm
                    )
            else:
                if self._use_int32:
                    self._factor_zi = SuiteSparseQR_factorize[cdouble, int32_t](
                        ordering, self._tol, Ac, self._cm
                    )
                else:
                    self._factor_zl = SuiteSparseQR_factorize[cdouble, int64_t](
                        ordering, self._tol, Ac, self._cm
                    )
        else:
            # Perform symbolic analysis only
            if self._is_real:
                if self._use_int32:
                    self._factor_di = SuiteSparseQR_symbolic[double, int32_t](
                        ordering, allow_tol, Ac, self._cm
                    )
                else:
                    self._factor_dl = SuiteSparseQR_symbolic[double, int64_t](
                        ordering, allow_tol, Ac, self._cm
                    )
            else:
                if self._use_int32:
                    self._factor_zi = SuiteSparseQR_symbolic[cdouble, int32_t](
                        ordering, allow_tol, Ac, self._cm
                    )
                else:
                    self._factor_zl = SuiteSparseQR_symbolic[cdouble, int64_t](
                        ordering, allow_tol, Ac, self._cm
                    )

        _handle_errors(self._cm.status)

        self.itype = np.dtype(np.int32 if self._use_int32 else np.int64)
        self.dtype = np.dtype(np.float64 if self._is_real else np.complex128)
        self._M = Ac.nrow
        self._N = Ac.ncol

    def __dealloc__(self):
        """Free the SPQR factorization and common objects."""
        if self._cm is NULL:
            return

        if self._is_real:
            if self._use_int32:
                assert SuiteSparseQR_free[double, int32_t](&self._factor_di, self._cm)
            else:
                assert SuiteSparseQR_free[double, int64_t](&self._factor_dl, self._cm)
        else:
            if self._use_int32:
                assert SuiteSparseQR_free[cdouble, int32_t](&self._factor_zi, self._cm)
            else:
                assert SuiteSparseQR_free[cdouble, int64_t](&self._factor_zl, self._cm)

        if self._use_int32:
            cholmod_finish(self._cm)
        else:
            cholmod_l_finish(self._cm)

    def __repr__(self):
        cls_name = self.__class__.__name__
        factor_type = 'numeric' if self.is_numeric else 'symbolic'
        return (
            f"<{cls_name} {factor_type} factor of dtype '{self.dtype}' "
            f"with '{self.itype}' indices\n"
            f"    A: shape={self.shape}, rank={self.rank}>"
        )

    def __str__(self):
        return self.__repr__()

    # ---------------------------------------------------------------------------------
    #         Properties
    # ---------------------------------------------------------------------------------
    @property
    def is_numeric(self):
        try:
            self._require_numeric()
            return True
        except AssertionError:
            return False

    @property
    def shape(self):
        return (self._M, self._N)

    @property
    def rank(self):
        if not self.is_numeric:
            return None

        if self._is_real:
            if self._use_int32:
                return self._factor_di.rank
            else:
                return self._factor_dl.rank
        else:
            if self._use_int32:
                return self._factor_zi.rank
            else:
                return self._factor_zl.rank

    @property
    def perm(self):
        self._require_symbolic()

        cdef void* ptr
        if self._is_real:
            if self._use_int32:
                ptr = <void*>self._factor_di.Q1fill
            else:
                ptr = <void*>self._factor_dl.Q1fill
        else:
            if self._use_int32:
                ptr = <void*>self._factor_zi.Q1fill
            else:
                ptr = <void*>self._factor_zl.Q1fill

        return _ndarray_copy_from_intptr(ptr, self._N, self._use_int32)

    @property
    def info(self):
        cdef SPQRInfo info = SPQRInfo()
        info._init_from_common(self._cm)
        return info

    # ---------------------------------------------------------------------------------
    #         Public API
    # ---------------------------------------------------------------------------------
    def copy(self):
        """Return a deep copy of the SPQRFactor object."""
        cdef SPQRFactor dest = SPQRFactor.__new__(SPQRFactor)

        dest._cm = &dest._common

        if self._use_int32:
            cholmod_start(dest._cm)
        else:
            cholmod_l_start(dest._cm)

        _copy_cholmod_common(dest._cm, self._cm)

        dest._use_int32 = self._use_int32
        dest._is_real = self._is_real
        dest._M = self._M
        dest._N = self._N
        dest.itype = self.itype
        dest.dtype = self.dtype
        dest._tol = self._tol

        # Deep copy the factorization
        if self._is_real:
            if self._use_int32:
                dest._factor_di = <spqr_factor_di*>malloc(sizeof(spqr_factor_di))
                _copy_spqr_factor(dest._factor_di, self._factor_di)
            else:
                dest._factor_dl = <spqr_factor_dl*>malloc(sizeof(spqr_factor_dl))
                _copy_spqr_factor(dest._factor_dl, self._factor_dl)
        else:
            if self._use_int32:
                dest._factor_zi = <spqr_factor_zi*>malloc(sizeof(spqr_factor_zi))
                _copy_spqr_factor(dest._factor_zi, self._factor_zi)
            else:
                dest._factor_zl = <spqr_factor_zl*>malloc(sizeof(spqr_factor_zl))
                _copy_spqr_factor(dest._factor_zl, self._factor_zl)

        return dest

    def factorize(self, object A, *, object tol=None):
        r"""Compute the numeric factorization of the matrix.

        Parameters
        ----------
        A : (M, N) array_like or sparse array, optional
            An array convertible to a sparse matrix. If None, the numeric factorization
            is computed for the matrix used in the constructor. If ``A`` is provided,
            it must have the same sparsity pattern as the matrix used in the
            constructor.
        tol : float, optional
            If the 2-norm of a column in ``A`` is less than ``tol``, that column is
            considered to be a zero column. If ``tol = 0``, no columns are treated as
            zero. If ``None``, the default tolerance is used. The default is
            ``tol =``
            :math:`20 \epsilon (M + N) \sqrt{\max{\mathrm{diag}(A^{\top} A)}}`,
            where :math:`\epsilon` is the machine precision.

        Returns
        -------
        :class:`SPQRFactor`
            The current object with the numeric factorization computed.
        """
        A, _, itype = validate_csc_input(A)
        self._check_input_matrix(A, itype)

        cdef cholmod_sparse Amatrix
        cdef cholmod_sparse *Ac = &Amatrix
        cdef int stype = 0  # assume matrix is not symmetric

        _cholmod_sparse_from_csc(
            A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
        )

        if tol is not None:
            if self.is_numeric and <double>tol != self._tol:
                warnings.warn(
                    "The tolerance has been changed from the one used "
                    "during a previous numeric factorization. This may lead to "
                    "inconsistent rank determination.",
                    UserWarning,
                )
            self._tol = <double>tol

        # Perform numeric factorization
        if self._is_real:
            if self._use_int32:
                SuiteSparseQR_numeric[double, int32_t](
                    self._tol, Ac, self._factor_di, self._cm
                )
            else:
                SuiteSparseQR_numeric[double, int64_t](
                    self._tol, Ac, self._factor_dl, self._cm
                )
        else:
            if self._use_int32:
                SuiteSparseQR_numeric[cdouble, int32_t](
                    self._tol, Ac, self._factor_zi, self._cm
                )
            else:
                SuiteSparseQR_numeric[cdouble, int64_t](
                    self._tol, Ac, self._factor_zl, self._cm
                )

        _handle_errors(self._cm.status)

        return self

    def qmult(self, object X, method="QX"):
        self._require_numeric()

        if not (isinstance(X, np.ndarray) or issparse(X)):
            raise ValueError("X must be an ndarray or sparse matrix.")

        if X.dtype != self.dtype:
            raise ValueError(
                f"Input and factor dtypes do not match. {self.dtype=} and {X.dtype=}"
            )

        if X.ndim not in (1, 2):
            raise ValueError("X must be a 1D or 2D array.")

        cdef int c_method = _qmult_int_from_str(method)

        # Check shape compatibility with Q
        cdef Py_ssize_t X_dim = (
            X.shape[0]
            if method == "QTX" or method == "QX"
            else X.shape[1]
        )

        if self._M != X_dim:
            raise ValueError(
                "Input X must have compatible shape with Q. "
                f"Expected {self._M}, got {X_dim}."
            )

        cdef bint return_1D = X.ndim == 1
        cdef bint return_sparse = issparse(X)

        if return_sparse:
            Y = self._qmult_sparse(c_method, X)
        else:
            # cholmod_dense expects column-oriented
            X = np.asfortranarray(X)
            Y = self._qmult_dense(c_method, X)

        if return_1D:
            Y = Y[:, 0]

        return Y

    cdef object _qmult_sparse(self, int method, object X):
        """Multiply a sparse matrix by Q."""
        cdef cholmod_sparse Xsparse
        cdef cholmod_sparse *Xs = &Xsparse
        cdef int stype = 0  # assume unsymmetric
        X, _, _ = validate_csc_input(X)
        _cholmod_sparse_from_csc(
            X.shape, X.indptr, X.indices, X.data, stype, <uintptr_t>Xs
        )

        cdef cholmod_sparse *Ys

        if self._is_real:
            if self._use_int32:
                Ys = SuiteSparseQR_qmult_fs[double, int32_t](
                    method, self._factor_di, Xs, self._cm
                )
            else:
                Ys = SuiteSparseQR_qmult_fs[double, int64_t](
                    method, self._factor_dl, Xs, self._cm
                )
        else:
            if self._use_int32:
                Ys = SuiteSparseQR_qmult_fs[cdouble, int32_t](
                    method, self._factor_zi, Xs, self._cm
                )
            else:
                Ys = SuiteSparseQR_qmult_fs[cdouble, int64_t](
                    method, self._factor_zl, Xs, self._cm
                )

        _handle_errors(self._cm.status)

        return _csc_from_cholmod_sparse(Ys, self._cm)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _qmult_dense(self, int method, value_t[::1, :] X not None):
        """Multiply a dense matrix by Q."""
        cdef cholmod_dense Xdense
        cdef cholmod_dense *Xd = &Xdense
        _cholmod_dense_from_ndarray(X, Xd)

        cdef cholmod_dense *Yd

        if self._is_real:
            if self._use_int32:
                Yd = SuiteSparseQR_qmult_fd[double, int32_t](
                    method, self._factor_di, Xd, self._cm
                )
            else:
                Yd = SuiteSparseQR_qmult_fd[double, int64_t](
                    method, self._factor_dl, Xd, self._cm
                )
        else:
            if self._use_int32:
                Yd = SuiteSparseQR_qmult_fd[cdouble, int32_t](
                    method, self._factor_zi, Xd, self._cm
                )
            else:
                Yd = SuiteSparseQR_qmult_fd[cdouble, int64_t](
                    method, self._factor_zl, Xd, self._cm
                )

        _handle_errors(self._cm.status)

        return _ndarray_from_cholmod_dense(Yd, self._use_int32, self._cm)

    def solve(self, object b, *, bint transpose=False, Py_ssize_t rhs_batch_size=100):
        self._require_numeric()

        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")

        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        if (
            (not transpose and b.shape[0] != self._M)
            or (transpose and b.shape[0] != self._N)
        ):
            raise ValueError(
                "Right-hand side b must have compatible shape with A. "
                f"Got {b.shape=}, but A.shape={self.shape} ({transpose=})."
            )

        if np.can_cast(b.dtype, self.dtype):
            b = b.astype(self.dtype, copy=False)
        else:
            raise TypeError(f"Cannot safely cast {b.dtype=} to {self.dtype=}.")

        # Check the rank of A and warn if rank deficient
        if self.rank < min(self._M, self._N):
            warnings.warn(
                f"Matrix is rank deficient: rank={self.rank}, A.shape={self.shape}. "
                "The solution may not be unique.",
                SPQRRankDeficiencyWarning,
            )

        cdef bint return_1D = b.ndim == 1

        # CHOLMOD routines require a 2D array
        if b.ndim == 1:
            b = b.reshape((-1, 1))

        if issparse(b):
            x = self._solve_sparse(b, transpose, rhs_batch_size)
        else:
            x = self._solve_dense(np.asfortranarray(b), transpose)

        if return_1D:
            x = x[:, 0]

        return x

    cdef _solve_sparse(self, object b, bint transpose, Py_ssize_t rhs_batch_size):
        """Solve multiple RHS systems where b is a sparse matrix.

        Parameters
        ----------
        sys : int
            The system type (UMFPACK_A, UMFPACK_Aat, UMFPACK_At).
        b : 2D array of value_t, shape (N, K)
            The right-hand side matrix.
        rhs_batch_size : int
            The number of columsn to convert to dense simultaneously.
        """
        if b.shape[1] == 1:
            b = b.tocsc()  # do not warn for conversion of a vector

        b, _, _ = validate_csc_input(b)

        cdef:
            Py_ssize_t k
            Py_ssize_t batch_end
            Py_ssize_t width
            Py_ssize_t K = b.shape[1]
            list x_blocks = []
            cnp.ndarray b_view
            cnp.ndarray x_batch

        # Pre-allocate dense space for RHS and solution
        cdef cnp.ndarray b_batch = np.empty(
            (b.shape[0], min(rhs_batch_size, K)), dtype=b.dtype, order="F"
        )

        for k in range(0, K, rhs_batch_size):
            batch_end = min(k + rhs_batch_size, K)
            width = batch_end - k
            # Views on the correct columns of the buffers
            b_view = b_batch[:, :width]
            # Convert the sparse RHS to dense in the buffer
            b[:, k:batch_end].toarray(out=b_view)
            # Solve the systems
            x_batch = self._solve_dense(b_view, transpose)
            # Only take the relevant columns
            x_blocks.append(csc_array(x_batch, dtype=b.dtype))

        x = hstack(x_blocks)
        x.indptr = x.indptr.astype(self.itype, copy=False)
        x.indices = x.indices.astype(self.itype, copy=False)

        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _solve_dense(self, value_t[::1, :] b not None, bint transpose):
        """Solve a linear system with a dense right-hand side."""
        # Get the b vector or matrix into CHOLMOD format
        cdef cholmod_dense Bmatrix
        cdef cholmod_dense *Bd = &Bmatrix

        _cholmod_dense_from_ndarray(b, Bd)

        # System is Ax = b -> (QRE.T)x = b
        # But "solve" does not touch Q, so -> (RE.T)x = (Q.T)b
        if not transpose:
            # pre-multiply by Q.T
            if self._is_real:
                if self._use_int32:
                    Bd = SuiteSparseQR_qmult_fd[double, int32_t](
                        SPQR_QTX, self._factor_di, Bd, self._cm
                    )
                else:
                    Bd = SuiteSparseQR_qmult_fd[double, int64_t](
                        SPQR_QTX, self._factor_dl, Bd, self._cm
                    )
            else:
                if self._use_int32:
                    Bd = SuiteSparseQR_qmult_fd[cdouble, int32_t](
                        SPQR_QTX, self._factor_zi, Bd, self._cm
                    )
                else:
                    Bd = SuiteSparseQR_qmult_fd[cdouble, int64_t](
                        SPQR_QTX, self._factor_zl, Bd, self._cm
                    )

        _handle_errors(self._cm.status)

        # Solve the system
        cdef int system = SPQR_RETX_EQUALS_B if not transpose else SPQR_RTX_EQUALS_ETB
        cdef cholmod_dense *Xd

        if self._is_real:
            if self._use_int32:
                Xd = SuiteSparseQR_solve[double, int32_t](
                    system, self._factor_di, Bd, self._cm
                )
            else:
                Xd = SuiteSparseQR_solve[double, int64_t](
                    system, self._factor_dl, Bd, self._cm
                )
        else:
            if self._use_int32:
                Xd = SuiteSparseQR_solve[cdouble, int32_t](
                    system, self._factor_zi, Bd, self._cm
                )
            else:
                Xd = SuiteSparseQR_solve[cdouble, int64_t](
                    system, self._factor_zl, Bd, self._cm
                )

        _handle_errors(self._cm.status)

        # System is A.T x = b -> (QRE.T).Tx = b -> (E R.T Q.T) x = b
        # But "solve" does not touch Q, so -> (E R.T) (Q.T x) = b
        if transpose:
            # post-multiply by Q.T
            if self._is_real:
                if self._use_int32:
                    Xd = SuiteSparseQR_qmult_fd[double, int32_t](
                        SPQR_QX, self._factor_di, Xd, self._cm
                    )
                else:
                    Xd = SuiteSparseQR_qmult_fd[double, int64_t](
                        SPQR_QX, self._factor_dl, Xd, self._cm
                    )
            else:
                if self._use_int32:
                    Xd = SuiteSparseQR_qmult_fd[cdouble, int32_t](
                        SPQR_QX, self._factor_zi, Xd, self._cm
                    )
                else:
                    Xd = SuiteSparseQR_qmult_fd[cdouble, int64_t](
                        SPQR_QX, self._factor_zl, Xd, self._cm
                    )

        _handle_errors(self._cm.status)

        return _ndarray_from_cholmod_dense(Xd, self._use_int32, self._cm)

    # ---------------------------------------------------------------------------------
    #         Private API
    # ---------------------------------------------------------------------------------
    cdef inline int _require_symbolic(self) except -1:
        """Raise an error if the symbolic factorization has not been computed yet."""
        if self._is_real:
            if self._use_int32:
                assert self._factor_di is not NULL and self._factor_di.QRsym is not NULL
            else:
                assert self._factor_dl is not NULL and self._factor_dl.QRsym is not NULL
        else:
            if self._use_int32:
                assert self._factor_zi is not NULL and self._factor_zi.QRsym is not NULL
            else:
                assert self._factor_zl is not NULL and self._factor_zl.QRsym is not NULL

    cdef inline int _require_numeric(self) except -1:
        """Raise an error if the numeric factorization has not been computed yet."""
        self._require_symbolic()
        if self._is_real:
            if self._use_int32:
                assert self._factor_di.QRnum is not NULL
            else:
                assert self._factor_dl.QRnum is not NULL
        else:
            if self._use_int32:
                assert self._factor_zi.QRnum is not NULL
            else:
                assert self._factor_zl.QRnum is not NULL

    def _check_input_matrix(self, object A, object itype):
        """Check that the input matrix matches the existing factorization."""
        if A.shape != self.shape:
            raise ValueError(
                "The shape of the input matrix does not match "
                "the one used for symbolic factorization. "
                f"Expected {self.shape}, got {A.shape}."
            )

        if itype != self.itype:
            raise ValueError(
                "The integer size of the input matrix does not match "
                "the one used for symbolic factorization. "
                f"Expected '{self.itype}', got '{itype}'."
            )

        if A.dtype != self.dtype:
            raise ValueError(
                "The data type of the input matrix does not match "
                "the one used for symbolic factorization. "
                f"Expected '{self.dtype}', got '{A.dtype}'."
            )


# -------------------------------------------------------------------------------------
#         Convenience Functions
# -------------------------------------------------------------------------------------
def spqr_factor(A, *, use_singletons=False, order=None, tol=None):
    r"""Compute the SPQR factorization of a sparse matrix.

    Compute the numeric factorization of the matrix and determine a fill-reducing
    ordering such that:

    .. math::

        Q R = A E

    where :math:`E` is a column permutation matrix, :math:`Q` is an orthogonal
    matrix, and :math:`R` is an upper-triangular matrix.

    This function returns a :class:`SPQRFactor` object that contains the SPQR
    factorization of the input matrix. It is not currently possible to extract the
    individual factors :math:`Q` and :math:`R` explicitly, but the object provides
    methods to reuse the factorization to solve linear systems or multiply by
    :math:`Q`.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    use_singletons : bool, optional
        If True, directly compute the numeric factorization to exploit singleton rows.
        Otherwise, only perform symbolic analysis. Default is False, so that the factor
        can be reused efficiently for multiple numeric factorizations.
    order : str, optional
        The column ordering strategy to use. Let :math:`S` be the matrix :math:`A` with
        singleton rows/columns removed, the ordering options are:

        * ``default``: COLAMD(S),
        * ``fixed``: identity permutation (*i.e.* no singletons removed),
        * ``natural``: singletons removed, but no fill-reducing ordering applied,
        * ``colamd``: COLAMD(S),
        * ``amd``: AMD(:math:`S^{\top} S`),
        * ``metis``: METIS(:math:`S^{\top} S`),
        * ``best``: try all of ``amd``, ``colamd``, ``metis`` and pick the best,
        * ``cholmod``: Same as ``best``,
        * ``bestamd``: try ``amd`` and ``colamd`` and pick the best.

    tol : float, optional
        If the 2-norm of a column in ``A`` is less than ``tol``, that column is
        considered to be a zero column. If ``tol = 0``, no columns are treated as zero.
        If ``None``, the default tolerance is used. The default is
        ``tol =`` :math:`20 \epsilon (M + N) \sqrt{\max{\mathrm{diag}(A^{\top} A)}}`,
        where :math:`\epsilon` is the machine precision.

    Returns
    -------
    :class:`SPQRFactor`
        The SPQR factorization of the input matrix.

    See Also
    --------
    SPQRFactor, spqr, spqr_qmult, spqr_solve

    Notes
    -----
    This function is part of an interface to the SuiteSparse SPQR library [#spqr_url]_.


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#spqr_url] SuiteSparse SPQR
        https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR
    """
    if use_singletons:
        return SPQRFactor(A, use_singletons=True, order=order, tol=tol)
    else:
        return SPQRFactor(A, use_singletons=False, order=order, tol=tol).factorize(A)


def spqr_solve(A, b, *, transpose=False, min2norm=True, Py_ssize_t rhs_batch_size=100):
    A, _, _ = validate_csc_input(A)
    M, N = A.shape

    if M < N and min2norm:
        A = A.T.tocsc()
        transpose = True

    return SPQRFactor(A, use_singletons=True).solve(
        b, transpose=transpose, rhs_batch_size=rhs_batch_size
    )


# -------------------------------------------------------------------------------------
#         SPQR
# -------------------------------------------------------------------------------------
cdef inline int _spqr_noQ(
    bint is_real,
    bint use_int32,
    int ordering,
    double tol,
    size_t econ,
    cholmod_sparse *Ac,
    cholmod_sparse **Rs,
    void **Es,
    cholmod_common *cm
) except -1:
    if is_real:
        if use_int32:
            SuiteSparseQR_noQ[double, int32_t](
                ordering, tol, econ, Ac, Rs, <int32_t**>Es, cm
            )
        else:
            SuiteSparseQR_noQ[double, int64_t](
                ordering, tol, econ, Ac, Rs, <int64_t**>Es, cm
            )
    else:
        if use_int32:
            SuiteSparseQR_noQ[cdouble, int32_t](
                ordering, tol, econ, Ac, Rs, <int32_t**>Es, cm
            )
        else:
            SuiteSparseQR_noQ[cdouble, int64_t](
                ordering, tol, econ, Ac, Rs, <int64_t**>Es, cm
            )

    _handle_errors(cm.status)
    return 0


cdef inline int _spqr_full(
    bint is_real,
    bint use_int32,
    int ordering,
    double tol,
    size_t econ,
    cholmod_sparse *Ac,
    cholmod_sparse **Qs,
    cholmod_sparse **Rs,
    void **Es,
    cholmod_common *cm
) except -1:
    if is_real:
        if use_int32:
            SuiteSparseQR_full[double, int32_t](
                ordering, tol, econ, Ac, Qs, Rs, <int32_t**>Es, cm
            )
        else:
            SuiteSparseQR_full[double, int64_t](
                ordering, tol, econ, Ac, Qs, Rs, <int64_t**>Es, cm
            )
    else:
        if use_int32:
            SuiteSparseQR_full[cdouble, int32_t](
                ordering, tol, econ, Ac, Qs, Rs, <int32_t**>Es, cm
            )
        else:
            SuiteSparseQR_full[cdouble, int64_t](
                ordering, tol, econ, Ac, Qs, Rs, <int64_t**>Es, cm
            )

    _handle_errors(cm.status)
    return 0


cdef inline int _spqr_householder(
    bint is_real,
    bint use_int32,
    int ordering,
    double tol,
    size_t econ,
    cholmod_sparse *Ac,
    cholmod_sparse **Rs,
    void **Es,
    cholmod_sparse **Hs,
    void **HPinv,
    cholmod_dense **HTau,
    cholmod_common *cm
) except -1:
    if is_real:
        if use_int32:
            SuiteSparseQR_householder[double, int32_t](
                ordering, tol, econ, Ac,
                Rs, <int32_t**>Es, Hs, <int32_t**>HPinv, HTau, cm
            )
        else:
            SuiteSparseQR_householder[double, int64_t](
                ordering, tol, econ, Ac,
                Rs, <int64_t**>Es, Hs, <int64_t**>HPinv, HTau, cm
            )
    else:
        if use_int32:
            SuiteSparseQR_householder[cdouble, int32_t](
                ordering, tol, econ, Ac,
                Rs, <int32_t**>Es, Hs, <int32_t**>HPinv, HTau, cm
            )
        else:
            SuiteSparseQR_householder[cdouble, int64_t](
                ordering, tol, econ, Ac,
                Rs, <int64_t**>Es, Hs, <int64_t**>HPinv, HTau, cm
            )

    _handle_errors(cm.status)
    return 0


class SPQRHouseholder(NamedTuple):
    """A class to hold the Householder representation of Q.

    Attributes
    ----------
    H : ~scipy.sparse.csc_array
        The Householder vectors stored in a sparse matrix.
    tau : ~numpy.ndarray of float
        The Householder coefficients.
    perm : ~numpy.ndarray of int
        The column permutation vector.
    """
    H: ~scipy.sparse.csc_array
    tau: ~numpy.ndarray
    perm: ~numpy.ndarray


def spqr(A, *, mode="full", order=None, tol=None):
    r"""Compute the QR factorization.

    This function computes the QR factorization of a sparse matrix :math:`A` such that

    .. math::
        Q R = A E

    where :math:`Q` is an orthogonal matrix and :math:`R` is an upper-triangular
    matrix. :math:`E` is a column permutation matrix that reduces fill-in during
    the factorization.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    mode : {'full', 'r', 'economic', 'householder'}, optional
        The mode of the returned Q and R matrices. Options are:

        * ``full``: ``Q`` is size ``(M, M)``, ``R`` is size ``(M, N)``.
        * ``economic``: ``Q`` is size ``(M, K)``, ``R`` is size ``(K, N)``, where
          ``K = min(M, N)``.
        * ``r``: Only return the upper-triangular matrix ``R``.
        * ``householder``: Return the Householder vectors and coefficients used to
          build ``Q``. This option is similar to ``mode='raw'`` in
          :func:`scipy.linalg.qr`.

    order : str, optional
        The column ordering strategy to use. Let :math:`S` be the matrix :math:`A` with
        singleton rows/columns removed, the ordering options are:

        * ``default``: COLAMD(S),
        * ``fixed``: identity permutation (*i.e.* no singletons removed),
        * ``natural``: singletons removed, but no fill-reducing ordering applied,
        * ``colamd``: COLAMD(S),
        * ``amd``: AMD(:math:`S^{\top} S`),
        * ``metis``: METIS(:math:`S^{\top} S`),
        * ``best``: try all of ``amd``, ``colamd``, ``metis`` and pick the best,
        * ``cholmod``: Same as ``best``,
        * ``bestamd``: try ``amd`` and ``colamd`` and pick the best.

    tol : float, optional
        If the 2-norm of a column in ``A`` is less than ``tol``, that column is
        considered to be a zero column. If ``tol = 0``, no columns are treated as zero.
        If ``None``, the default tolerance is used. The default is
        ``tol =`` :math:`20 \epsilon (M + N) \sqrt{\max{\mathrm{diag}(A^{\top} A)}}`,
        where :math:`\epsilon` is the machine precision.

    Returns
    -------
    Q : csc_array
        The orthogonal matrix :math:`Q`. Shape (M, M) or (M, K) if ``mode='economic'``.
        Not returned if ``mode='r'``.
        Replaced by :class:`SPQRHouseholder` if ``mode='householder'``.
    R : csc_array
        The upper-triangular matrix :math:`R`. Shape (M, N) or (K, N) if ``mode in
        ['economic', 'householder']``, where K = min(M, N).
    P : ndarray of int
        The permutation vector of shape (N,).

    See Also
    --------
    SPQRFactor, spqr_factor, spqr_qmult, spqr_solve

    Notes
    -----
    This function is part of an interface to the SuiteSparse SPQR library [#spqr_url]_.


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#spqr_url] SuiteSparse SPQR
        https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR
    """
    A, _, _ = validate_csc_input(A)

    cdef Py_ssize_t N = A.shape[1]

    allowed_modes = ("full", "economic", "r", "householder")
    if mode not in allowed_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected one of {allowed_modes}."
        )

    cdef int ordering

    if order is None:
        ordering = SPQR_ORDERING_DEFAULT
    else:
        try:
            ordering = _ordering_methods[order]
        except KeyError:
            raise ValueError(
                f"Unknown ordering method: {ordering}. "
                f"Must be one of {set(_ordering_methods.keys())}."
            )

    cdef double c_tol = <double>tol if tol is not None else SPQR_DEFAULT_TOL

    # Get the input matrix into CHOLMOD format
    cdef cholmod_sparse Amatrix
    cdef cholmod_sparse *Ac = &Amatrix
    cdef int stype = 0  # assume matrix is not symmetric

    _cholmod_sparse_from_csc(
        A.shape, A.indptr, A.indices, A.data, stype, <uintptr_t>Ac
    )

    cdef bint is_real = (Ac.xtype == CHOLMOD_REAL)
    cdef bint use_int32 = (Ac.itype == CHOLMOD_INT)

    # Initialize the common object
    cdef cholmod_common common
    cdef cholmod_common *cm = &common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    # ---------------------------------------------------------------------------------
    #         Perform the factorization
    # ---------------------------------------------------------------------------------
    cdef:
        cholmod_sparse *Qs = NULL
        cholmod_sparse *Rs = NULL
        void *Es = NULL
        cholmod_sparse *Hs = NULL
        void *HPinv = NULL
        cholmod_dense *HTau = NULL
        size_t econ = Ac.nrow if mode != "economic" else Ac.ncol

    if mode == "r":
        _spqr_noQ(is_real, use_int32, ordering, c_tol, econ, Ac, &Rs, &Es, cm)
    elif mode in ["full", "economic"]:
        _spqr_full(is_real, use_int32, ordering, c_tol, econ, Ac, &Qs, &Rs, &Es, cm)
    elif mode == "householder":
        _spqr_householder(
            is_real, use_int32, ordering, c_tol, econ, Ac,
            &Rs, &Es, &Hs, &HPinv, &HTau, cm
        )
    else:
        raise NotImplementedError(f"{mode=} is not supported.")

    # Get Python objects from the cholmod structs
    Q = R = E = H = tau = v = None
    cdef Py_ssize_t Mh

    if Qs is not NULL:
        Q = _csc_from_cholmod_sparse(Qs, cm)

    if Rs is not NULL:
        R = _csc_from_cholmod_sparse(Rs, cm)

    if Es is not NULL:
        E = _ndarray_copy_from_intptr(Es, N, use_int32)

    if Hs is not NULL:
        H = _csc_from_cholmod_sparse(Hs, cm)

    if HPinv is not NULL and H is not None:
        Mh = H.shape[0]
        v = _ndarray_copy_from_intptr(HPinv, Mh, use_int32)

    if HTau is not NULL:
        tau = _ndarray_from_cholmod_dense(HTau, use_int32, cm).squeeze()

    if use_int32:
        cholmod_free(N, sizeof(int32_t), Es, cm)
        cholmod_free(Mh, sizeof(int32_t), HPinv, cm)
        cholmod_finish(cm)
    else:
        cholmod_l_free(N, sizeof(int64_t), Es, cm)
        cholmod_l_free(Mh, sizeof(int64_t), HPinv, cm)
        cholmod_l_finish(cm)

    if mode == "r":
        return R, E
    elif mode in ["full", "economic"]:
        return Q, R, E
    else:  # mode == "householder"
        return SPQRHouseholder(H=H, tau=tau, perm=v), R, E


# -------------------------------------------------------------------------------------
#         Qmult
# -------------------------------------------------------------------------------------
cdef object _qmult_sparse(
    int method,
    cholmod_sparse *H,
    cholmod_dense *HTau,
    index_t[::1] HPinv,
    object X,
    cholmod_common *cm,
):
    """Multiply a sparse matrix by Q."""
    cdef cholmod_sparse Xsparse
    cdef cholmod_sparse *Xs = &Xsparse
    cdef int stype = 0  # assume unsymmetric
    X, _, _ = validate_csc_input(X)
    _cholmod_sparse_from_csc(
        X.shape, X.indptr, X.indices, X.data, stype, <uintptr_t>Xs
    )

    cdef bint is_real = (Xs.xtype == CHOLMOD_REAL)
    cdef bint use_int32 = (Xs.itype == CHOLMOD_INT)

    cdef cholmod_sparse *Ys

    if is_real:
        if use_int32:
            Ys = SuiteSparseQR_qmult_Hs[double, int32_t](
                method, H, HTau, <int32_t*>&HPinv[0], Xs, cm
            )
        else:
            Ys = SuiteSparseQR_qmult_Hs[double, int64_t](
                method, H, HTau, <int64_t*>&HPinv[0], Xs, cm
            )
    else:
        if use_int32:
            Ys = SuiteSparseQR_qmult_Hs[cdouble, int32_t](
                method, H, HTau, <int32_t*>&HPinv[0], Xs, cm
            )
        else:
            Ys = SuiteSparseQR_qmult_Hs[cdouble, int64_t](
                method, H, HTau, <int64_t*>&HPinv[0], Xs, cm
            )

    _handle_errors(cm.status)

    return _csc_from_cholmod_sparse(Ys, cm)


cdef object _qmult_dense(
    int method,
    cholmod_sparse *H,
    cholmod_dense *HTau,
    index_t[::1] HPinv,
    value_t[::1, :] X,
    cholmod_common *cm,
):
    """Multiply a dense matrix by Q."""
    cdef cholmod_dense Xdense
    cdef cholmod_dense *Xd = &Xdense
    _cholmod_dense_from_ndarray(X, Xd)

    cdef bint is_real = (H.xtype == CHOLMOD_REAL)
    cdef bint use_int32 = (H.itype == CHOLMOD_INT)

    cdef cholmod_dense *Yd

    if is_real:
        if use_int32:
            Yd = SuiteSparseQR_qmult_Hd[double, int32_t](
                method, H, HTau, <int32_t*>&HPinv[0], Xd, cm
            )
        else:
            Yd = SuiteSparseQR_qmult_Hd[double, int64_t](
                method, H, HTau, <int64_t*>&HPinv[0], Xd, cm
            )
    else:
        if use_int32:
            Yd = SuiteSparseQR_qmult_Hd[cdouble, int32_t](
                method, H, HTau, <int32_t*>&HPinv[0], Xd, cm
            )
        else:
            Yd = SuiteSparseQR_qmult_Hd[cdouble, int64_t](
                method, H, HTau, <int64_t*>&HPinv[0], Xd, cm
            )

    _handle_errors(cm.status)

    return _ndarray_from_cholmod_dense(Yd, use_int32, cm)


@cython.boundscheck(False)
@cython.wraparound(False)
def _qmult(
    int method,
    object H,
    value_t[::1, :] tau not None,
    index_t[::1] v not None,
    object X,
):
    """Dispatch the correct typed qmult function."""
    cdef bint use_int32 = (index_t is int32_t)

    # Initialize the common object
    cdef cholmod_common common
    cdef cholmod_common *cm = &common

    if use_int32:
        cholmod_start(cm)
    else:
        cholmod_l_start(cm)

    # Make cholmod objects from H, tau, v
    cdef cholmod_sparse Hsparse
    cdef cholmod_sparse *Hs = &Hsparse
    cdef int stype = 0  # not symmetric
    _cholmod_sparse_from_csc(
        H.shape, H.indptr, H.indices, H.data, stype, <uintptr_t>Hs
    )

    cdef cholmod_dense HTau_dense
    cdef cholmod_dense *HTau = &HTau_dense
    _cholmod_dense_from_ndarray(tau, HTau)

    # If X is dense, get a memoryview of the same type as tau
    cdef value_t[::1, :] X_view

    # Compute the multiplication
    if issparse(X):
        Y = _qmult_sparse(method, Hs, HTau, v, X, cm)
    else:
        X_view = X  # assign the view onto X
        Y = _qmult_dense(method, Hs, HTau, v, X_view, cm)

    if use_int32:
        cholmod_finish(cm)
    else:
        cholmod_l_finish(cm)

    return Y


def spqr_qmult(house, X, method="QX"):
    try:
        H, tau, v = house
        H, _, itype = validate_csc_input(H)
        tau = np.asfortranarray(tau).reshape((1, -1))  # for cholmod_dense
        if tau.shape != (1, H.shape[1]):
            raise TypeError("tau shape mismatch")
        if tau.dtype != H.dtype:
            raise TypeError("tau dtype mismatch")
        v = np.asfortranarray(v)
        if v.shape != (H.shape[0],):
            raise TypeError("v shape mismatch")
        if v.dtype != itype:
            raise TypeError("v dtype mismatch")
    except Exception:
        raise ValueError(
            "house must be a tuple of (H, tau, v) representing the "
            "Householder vectors, coefficients, and permutation. "
            f"Got {house}."
        )

    if not issparse(X):
        try:
            X = np.asfortranarray(X)
        except Exception:
            raise ValueError("X must be an ndarray or sparse matrix.")

    if X.ndim not in (1, 2):
        raise ValueError("X must be a 1D or 2D array.")

    cdef int c_method = _qmult_int_from_str(method)

    # Check shape compatibility with Q
    cdef Py_ssize_t X_dim = (
        X.shape[0]
        if method == "QTX" or method == "QX"
        else X.shape[1]
    )

    M = H.shape[0]
    if M != X_dim:
        raise ValueError(
            "Input X must have compatible shape with Q. "
            f"Expected {M}, got {X_dim}."
        )

    cdef bint return_1D = X.ndim == 1

    # cholmod_sparse/dense expects a 2D array
    if X.ndim == 1:
        X = X.reshape((-1, 1))

    # Perform the multiplication
    Y = _qmult(c_method, H, tau, v, X)

    if return_1D:
        Y = Y[:, 0]

    return Y


# -------------------------------------------------------------------------------------
#         Docstrings
# -------------------------------------------------------------------------------------
_SOLVE_DOC_TEMPLATE = r"""
Solve a linear system using the SPQR factorization.

Solve a linear system for :math:`x` given the right-hand side
:math:`b` as either a vector or a matrix with multiple right-hand sides.

If ``transpose=False``, solve

.. math::
    A x = b

or, if ``transpose=True``, solve

.. math::
    A^{{\top}} x = b.

Parameters
----------
{A_doc}
b : (M,) or (M, K) numpy.ndarray
    The right-hand side vector or matrix. ``M`` should be the number of rows in
    ``A`` if ``transpose=False``, otherwise the number of columns.
transpose : bool, optional
    Whether to solve the transposed system. Default is False.
{min2norm}
rhs_batch_size : int, optional
    If ``b`` is a 2D sparse array, this parameter controls the number of
    columns to be solved simultaneously. A larger number will increase
    memory consumption by converting more columns at a time to dense
    arrays, but may improve runtime.

Returns
-------
x : (N,) or (N, K) numpy.ndarray or sparse array
    The solution vector or matrix. If ``b`` is a 1D array, then ``x`` is
    returned as a 1D array. If ``b`` is a 2D array with ``K`` columns,
    then ``x`` is returned as a 2D array with ``K`` columns. If ``b``
    is a sparse array, then ``x`` is also returned as a sparse array.
    ``N`` is the number of columns in ``A`` if ``transpose=False``,
    otherwise the number of rows.

See Also
--------
SPQRFactor, spqr_factor, spqr, spqr_qmult

Notes
-----
Part of an interface to the SuiteSparse SPQR library [#spqr_url]_.


.. versionadded:: 0.5.0

References
----------
.. [#spqr_url] SuiteSparse SPQR
    https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR
"""


_A_doc= """A : (M, N) array_like or sparse array
    An array convertible to a sparse matrix."""

_min2norm_doc = """min2norm : bool, optional
    If True, compute the minimum 2-norm solution when ``A`` is underdetermined.
    ``transpose`` is ignored in this case. Default is True. If False, the
    solution of an underdetermined system is not guaranteed to be the minimum
    2-norm solution."""

# Format the docstrings
SPQRFactor.solve.__doc__ = _SOLVE_DOC_TEMPLATE.format(A_doc="", min2norm="")
spqr_solve.__doc__ = _SOLVE_DOC_TEMPLATE.format(
    A_doc=_A_doc,
    min2norm=_min2norm_doc
)


_QMULT_DOC_TEMPLATE = r"""
Multiply by `Q` using the Householder representation.

Parameters
----------
{house_doc}
X : (M, N) numpy.ndarray or sparse array
    The matrix to be multiplied. Must have compatible shape with ``Q``.
method : str , optional
    The multiplication method. Options are:

    * ``QX`` : compute :math:`Q X`
    * ``QTX`` : compute :math:`Q^{{\top}} X`
    * ``XQ`` : compute :math:`X Q`
    * ``XQT`` : compute :math:`X Q^{{\top}}`

    Default is ``QX``. The transpose is the conjugate transpose for complex data.

Returns
-------
Y : (M, N) numpy.ndarray or sparse array
    The result of the multiplication. If ``X`` is a sparse array, then ``Y`` is
    also returned as a sparse array.

See Also
--------
SPQRFactor, spqr_factor, spqr, spqr_solve{see_also}

Notes
-----
This function is part of an interface to the SuiteSparse SPQR library [#spqr_url]_.


.. versionadded:: 0.5.0

References
----------
.. [#spqr_url] SuiteSparse SPQR
    https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR
"""


_qmult_house_doc = """house : SPQRHouseholder or tuple
    A tuple ``(H, tau, v)`` representing the Householder vectors ``H``,
    coefficients ``tau``, and the column permutation vector ``v``. Typically,
    these are created from ``Ht, R, p = spqr(A, mode='householder')``."""

SPQRFactor.qmult.__doc__ = _QMULT_DOC_TEMPLATE.format(
    house_doc="",
    see_also=", spqr_qmult"
)

spqr_qmult.__doc__ = _QMULT_DOC_TEMPLATE.format(
    house_doc=_qmult_house_doc,
    see_also=""
)
