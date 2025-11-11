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

    spqr_solve - Solve a linear system using the SPQR factorization.


Object Interface
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    spqr_factor - Compute the QR factorization of a sparse matrix.
    SPQRFactor - An object-oriented interface to SPQR.
    SPQRInfo - A dataclass to return SPQR info.
    SPQRControl - A dataclass to set SPQR control parameters.


.. spqr-exceptions:

Warnings and Exceptions
-----------------------

.. autosummary::
    :toctree: generated/

    SPQRWarning
    SPQRSingularMatrixWarning

    SPQRError
    SPQROutOfMemoryError
    SPQRInvalidError
    SPQROverflowError


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse SPQR <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/SPQR>`_
"""

cimport cython
from cython cimport doublecomplex

from sksparse.cholmod cimport (
    CHOLMOD_OK,
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
)

import numpy as np
from scipy.sparse import csc_array, issparse
import warnings

from sksparse.cholmod import _cholmod_sparse_from_csc

from .utils import validate_csc_input


__all = [
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
ctypedef spqr_symbolic[int32_t] spqr_symb_i
ctypedef spqr_symbolic[int64_t] spqr_symb_l

ctypedef fused symbolic_t:
    spqr_symb_i
    spqr_symb_l


ctypedef spqr_numeric[double, int32_t] spqr_num_di
ctypedef spqr_numeric[double, int64_t] spqr_num_dl
ctypedef spqr_numeric[doublecomplex, int32_t] spqr_num_zi
ctypedef spqr_numeric[doublecomplex, int64_t] spqr_num_zl

ctypedef fused numeric_t:
    spqr_num_di
    spqr_num_dl
    spqr_num_zi
    spqr_num_zl


ctypedef SuiteSparseQR_factorization[double, int32_t] spqr_fact_di
ctypedef SuiteSparseQR_factorization[double, int64_t] spqr_fact_dl
ctypedef SuiteSparseQR_factorization[doublecomplex, int32_t] spqr_fact_zi
ctypedef SuiteSparseQR_factorization[doublecomplex, int64_t] spqr_fact_zl

ctypedef fused factor_t:
    spqr_fact_di
    spqr_fact_dl
    spqr_fact_zi
    spqr_fact_zl



# -------------------------------------------------------------------------------------
#         Error Handling
# -------------------------------------------------------------------------------------
class SPQRError(Exception):
    """Base class for SPQR exceptions."""
    pass


# -------------------------------------------------------------------------------------
#         Copy Functions
# -------------------------------------------------------------------------------------
cdef inline void* _malloc_copy(
    const void* src,
    size_t n,
    size_t size,
    const cholmod_common* cm
):
    """Allocate memory and copy data from src to the new memory."""
    assert cm is not NULL
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
    const cholmod_common* cm,
    index_t _dummy_idx=0,
) except -1:
    """Deep copy a SuiteSparseQR_symbolic object."""
    assert dest is not NULL
    assert src is not NULL
    assert cm is not NULL

    # Prune invalid type combinations
    if not (
        (symbolic_t is spqr_symb_i and index_t is int32_t)
        or (symbolic_t is spqr_symb_l and index_t is int64_t)
    ):
        assert False
        return 0

    dest.m = src.m
    dest.n = src.n
    dest.anz = src.anz

    dest.Sp = <index_t*>_malloc_copy(src.Sp, src.m + 1, sizeof(index_t), cm)
    dest.Sj = <index_t*>_malloc_copy(src.Sj, src.anz, sizeof(index_t), cm)

    dest.Qfill = <index_t*>_malloc_copy(src.Qfill, src.n, sizeof(index_t), cm)
    dest.PLinv = <index_t*>_malloc_copy(src.PLinv, src.m, sizeof(index_t), cm)
    dest.Sleft = <index_t*>_malloc_copy(src.Sleft, src.n + 2, sizeof(index_t), cm)

    dest.nf = src.nf
    dest.maxfn = src.maxfn

    dest.Parent = <index_t*>_malloc_copy(src.Parent, src.nf + 1, sizeof(index_t), cm)
    dest.Child = <index_t*>_malloc_copy(src.Child, src.nf + 1, sizeof(index_t), cm)
    dest.Childp = <index_t*>_malloc_copy(src.Childp, src.nf + 2, sizeof(index_t), cm)

    dest.Super = <index_t*>_malloc_copy(src.Super, src.nf + 1, sizeof(index_t), cm)

    dest.Rp = <index_t*>_malloc_copy(src.Rp, src.nf + 1, sizeof(index_t), cm)
    dest.Rj = <index_t*>_malloc_copy(src.Rj, src.rjsize, sizeof(index_t), cm)
    dest.Post = <index_t*>_malloc_copy(src.Post, src.nf + 1, sizeof(index_t), cm)

    dest.rjsize = src.rjsize
    dest.do_rank_detection = src.do_rank_detection
    dest.maxstack = src.maxstack
    dest.hisize = src.hisize
    dest.keepH = src.keepH

    dest.Hip = <index_t*>_malloc_copy(src.Hip, src.nf + 1, sizeof(index_t), cm)

    dest.ntasks = src.ntasks
    dest.ns = src.ns

    if dest.ntasks > 1:
        raise NotImplementedError("SPQR task parallelism not yet supported.")

    return 0


cdef inline int _copy_spqr_symbolic(
    symbolic_t* dest,
    const symbolic_t* src,
    const cholmod_common* cm,
) except -1:
    """Deep copy a spqr_symbolic struct."""
    if symbolic_t is spqr_symb_i:
        return _copy_spqr_symbolic_base[spqr_symb_i, int32_t](dest, src, cm)
    else:  # symbolic_t is spqr_symb_l
        return _copy_spqr_symbolic_base[spqr_symb_l, int64_t](dest, src, cm)


cdef int _copy_spqr_numeric_base(
    numeric_t* dest,
    const numeric_t* src,
    const cholmod_common* cm,
    value_t _dummy_val=0,
    index_t _dummy_idx=0,
) except -1:
    """Deep copy a SuiteSparseQR_numeric object."""
    assert dest is not NULL
    assert src is not NULL
    assert cm is not NULL

    # Prune invalid type combinations
    if not (
        (numeric_t is spqr_num_di and index_t is int32_t and value_t is double)
        or (numeric_t is spqr_num_dl and index_t is int64_t and value_t is double)
        or (numeric_t is spqr_num_zi and index_t is int32_t and value_t is doublecomplex)
        or (numeric_t is spqr_num_zl and index_t is int64_t and value_t is doublecomplex)
    ):
        assert False
        return 0

    dest.Stacks = <value_t**>malloc(src.ns * sizeof(value_t*))
    dest.Stack_size = <index_t*>_malloc_copy(
        src.Stack_size, src.ns, sizeof(index_t), cm
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
                src.Stacks[k], src.Stack_size[k], sizeof(value_t), cm
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

    dest.Rdead = <char*>_malloc_copy(src.Rdead, src.n, sizeof(char), cm)

    dest.rank = src.rank
    dest.rank1 = src.rank1
    dest.maxfrank = src.maxfrank
    dest.norm_E_fro = src.norm_E_fro

    dest.keepH = src.keepH
    dest.rjsize = src.rjsize

    dest.HStair = <index_t*>_malloc_copy(src.HStair, src.rjsize, sizeof(index_t), cm)
    dest.HTau = <value_t*>_malloc_copy(src.HTau, src.rjsize, sizeof(value_t), cm)

    dest.Hii = <index_t*>_malloc_copy(src.Hii, src.hisize, sizeof(index_t), cm)
    dest.HPinv = <index_t*>_malloc_copy(src.HPinv, src.m, sizeof(index_t), cm)

    dest.Hm = <index_t*>_malloc_copy(src.Hm, src.nf, sizeof(index_t), cm)
    dest.Hr = <index_t*>_malloc_copy(src.Hr, src.nf, sizeof(index_t), cm)

    dest.maxfm = src.maxfm

    return 0


cdef inline int _copy_spqr_numeric(
    numeric_t* dest,
    const numeric_t* src,
    const cholmod_common* cm,
) except -1:
    """Deep copy a spqr_numeric struct."""
    if numeric_t is spqr_num_di:
        return _copy_spqr_numeric_base[spqr_num_di, double, int32_t](dest, src, cm)
    elif numeric_t is spqr_num_dl:
        return _copy_spqr_numeric_base[spqr_num_dl, double, int64_t](dest, src, cm)
    elif numeric_t is spqr_num_zi:
        return _copy_spqr_numeric_base[spqr_num_zi, doublecomplex, int32_t](dest, src, cm)
    else:  # numeric_t is spqr_num_zl
        return _copy_spqr_numeric_base[spqr_num_zl, doublecomplex, int64_t](dest, src, cm)


cdef int _copy_spqr_factor_base(
    factor_t* dest,
    const factor_t* src,
    const cholmod_common* cm,
    value_t _dummy_val=0,
    index_t _dummy_idx=0,
) except -1:
    """Deep copy a SuiteSparseQR_factorization object."""
    assert dest is not NULL
    assert src is not NULL
    assert cm is not NULL

    # Validate types to ensure correct instantiation. Invalid combos will be pruned.
    if not (
        (factor_t is spqr_fact_di and index_t is int32_t and value_t is double)
        or (factor_t is spqr_fact_dl and index_t is int64_t and value_t is double)
        or (factor_t is spqr_fact_zi and index_t is int32_t and value_t is doublecomplex)
        or (factor_t is spqr_fact_zl and index_t is int64_t and value_t is doublecomplex)
    ):
        assert False
        return 0

    dest.tol = src.tol

    # Deep copy symbolic factorization
    if factor_t is spqr_fact_di or factor_t is spqr_fact_zi:
        dest.QRsym = <spqr_symb_i*>malloc(sizeof(spqr_symb_i))
    else:
        dest.QRsym = <spqr_symb_l*>malloc(sizeof(spqr_symb_l))

    _copy_spqr_symbolic(dest.QRsym, src.QRsym, cm)

    # Deep copy numeric factorization
    if factor_t is spqr_fact_di:
        dest.QRnum = <spqr_num_di*>malloc(sizeof(spqr_num_di))
    elif factor_t is spqr_fact_dl:
        dest.QRnum = <spqr_num_dl*>malloc(sizeof(spqr_num_dl))
    elif factor_t is spqr_fact_zi:
        dest.QRnum = <spqr_num_zi*>malloc(sizeof(spqr_num_zi))
    else:  # factor_t is spqr_fact_zl
        dest.QRnum = <spqr_num_zl*>malloc(sizeof(spqr_num_zl))

    if src.QRnum is not NULL:
        _copy_spqr_numeric(dest.QRnum, src.QRnum, cm)

    dest.R1p = <index_t*>_malloc_copy(src.R1p, src.n1rows + 1, sizeof(index_t), cm)
    dest.R1j = <index_t*>_malloc_copy(src.R1j, src.n1rows, sizeof(index_t), cm)
    dest.R1x = <value_t*>_malloc_copy(src.R1x, src.r1nz, sizeof(value_t), cm)
    dest.r1nz = src.r1nz

    dest.Q1fill = <index_t*>_malloc_copy(src.Q1fill, src.nacols, sizeof(index_t), cm)
    dest.P1inv = <index_t*>_malloc_copy(src.P1inv, src.narows, sizeof(index_t), cm)
    dest.HP1inv = <index_t*>_malloc_copy(src.HP1inv, src.narows, sizeof(index_t), cm)

    dest.Rmap = <index_t*>_malloc_copy(src.Rmap, src.nacols, sizeof(index_t), cm)
    dest.RmapInv = <index_t*>_malloc_copy(src.RmapInv, src.nacols, sizeof(index_t), cm)

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
    const cholmod_common* cm,
) except -1:
    """Deep copy a SuiteSparseQR_factorization struct."""
    if factor_t is spqr_fact_di:
        return _copy_spqr_factor_base[spqr_fact_di, double, int32_t](dest, src, cm)
    elif factor_t is spqr_fact_dl:
        return _copy_spqr_factor_base[spqr_fact_dl, double, int64_t](dest, src, cm)
    elif factor_t is spqr_fact_zi:
        return _copy_spqr_factor_base[spqr_fact_zi, doublecomplex, int32_t](dest, src, cm)
    else:  # factor_t is spqr_fact_zl
        return _copy_spqr_factor_base[spqr_fact_zl, doublecomplex, int64_t](dest, src, cm)


# -------------------------------------------------------------------------------------
#         SPQR Factor Class
# -------------------------------------------------------------------------------------
cdef class SPQRFactor:
    """The main object used for creating and manipulating SPQR factorizations.

    The constructor computes the sybolic analysis of the matrix and determines
    a fill-reducing ordering such that:

    .. math::

        Q R = A E

    where :math:`E` is a column permutation matrix, :math:`Q` is an orthogonal
    matrix, and :math:`R` is an upper-triangular matrix.

    The numerical factorization is computed in one of two ways:

    1. by setting ``use_singletons=True`` in the constructor, which computes
        both the symbolic and numeric factorizations at once, or
    2. by calling :meth:`.factorize(A)`, which computes the numeric factorization
        after symbolic analysis has been performed.

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
    order : int, optional
        The ordering strategy to use.
    tol : float, optional
        If the 2-norm of a column in ``A`` is less than ``tol``, that column is
        considered to be a zero column. If ``None``, the default tolerance is used.

    Properties
    ----------
    is_numeric : bool
        Whether the numeric factorization has been computed.
    shape : tuple
        The shape of the input matrix (M, N).
    itype : dtype
        The integer type used for indices (``int32`` or ``int64``).
    dtype : dtype
        The data type of the matrix (``float64`` or ``complex128``).
    Qshape : tuple
        The shape of the orthogonal matrix Q.
    Rshape : tuple
        The shape of the upper-triangular matrix R.
    rank : int
        The rank of the matrix as determined by SPQR.
    perm : ndarray of int
        The combined singleton and fill-reducing column permutation vector.

    See Also
    --------
    spqr_factor, spqr_solve

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
        spqr_fact_di *_fact_di
        spqr_fact_dl *_fact_dl
        spqr_fact_zi *_fact_zi
        spqr_fact_zl *_fact_zl
        bint _use_int32
        bint _is_real
        bint _econ
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

        # Promote single to double precision
        if not (
            np.issubdtype(A.dtype, np.float64) or np.issubdtype(A.dtype, np.complex128)
        ):
            if np.issubdtype(A.dtype, np.floating):
                A = A.astype(np.promote_types(A.dtype, np.float64))
            elif np.issubdtype(A.dtype, np.complexfloating):
                A = A.astype(np.promote_types(A.dtype, np.complex128))

        # Validate inputs
        cdef int ordering

        if order is None:
            ordering = SPQR_ORDERING_DEFAULT
        else:
            # TODO validate order input
            raise NotImplementedError("ordering methods not yet implemented")

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
                    self._fact_di = SuiteSparseQR_factorize[double, int32_t](
                        ordering, self._tol, Ac, self._cm
                    )
                else:
                    self._fact_dl = SuiteSparseQR_factorize[double, int64_t](
                        ordering, self._tol, Ac, self._cm
                    )
            else:
                if self._use_int32:
                    self._fact_zi = SuiteSparseQR_factorize[doublecomplex, int32_t](
                        ordering, self._tol, Ac, self._cm
                    )
                else:
                    self._fact_zl = SuiteSparseQR_factorize[doublecomplex, int64_t](
                        ordering, self._tol, Ac, self._cm
                    )
        else:
            # Perform symbolic analysis only
            if self._is_real:
                if self._use_int32:
                    self._fact_di = SuiteSparseQR_symbolic[double, int32_t](
                        ordering, allow_tol, Ac, self._cm
                    )
                else:
                    self._fact_dl = SuiteSparseQR_symbolic[double, int64_t](
                        ordering, allow_tol, Ac, self._cm
                    )
            else:
                if self._use_int32:
                    self._fact_zi = SuiteSparseQR_symbolic[doublecomplex, int32_t](
                        ordering, allow_tol, Ac, self._cm
                    )
                else:
                    self._fact_zl = SuiteSparseQR_symbolic[doublecomplex, int64_t](
                        ordering, allow_tol, Ac, self._cm
                    )

        # TODO proper error handling
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"Error {self._cm.status}")

        self.itype = np.dtype(np.int32 if self._use_int32 else np.int64)
        self.dtype = np.dtype(np.float64 if self._is_real else np.complex128)
        self._M = Ac.nrow
        self._N = Ac.ncol
        self._econ = False  # TODO econ mode (default to full Q and R shapes)

    def __dealloc__(self):
        """Free the SPQR factorization and common objects."""
        if self._is_real:
            if self._use_int32:
                assert SuiteSparseQR_free[double, int32_t](&self._fact_di, self._cm)
            else:
                assert SuiteSparseQR_free[double, int64_t](&self._fact_dl, self._cm)
        else:
            if self._use_int32:
                assert SuiteSparseQR_free[doublecomplex, int32_t](&self._fact_zi, self._cm)
            else:
                assert SuiteSparseQR_free[doublecomplex, int64_t](&self._fact_zl, self._cm)

        if self._use_int32:
            cholmod_finish(self._cm)
        else:
            cholmod_l_finish(self._cm)

    def __repr__(self):
        cls_name = self.__class__.__name__
        factor_type = 'numeric' if self.is_numeric else 'symbolic'
        # TODO nnz of Q and R
        return (
            f"<{cls_name} {factor_type} factor of dtype '{self.dtype}' "
            f"with '{self.itype}' indices:\n"
            f"    Q: {self.Qshape} with XXX stored elements\n"
            f"    R: {self.Rshape} with XXX stored elements>"
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
    def Qshape(self):
        return (self._M, self._N) if self._econ else (self._M, self._M)

    @property
    def Rshape(self):
        return (self._N, self._N) if self._econ else (self._M, self._N)

    @property
    def rank(self):
        cdef int rank
        self._require_symbolic()
        if self._is_real:
            if self._use_int32:
                return self._fact_di.rank
            else:
                return self._fact_dl.rank
        else:
            if self._use_int32:
                return self._fact_zi.rank
            else:
                return self._fact_zl.rank

    @property
    def perm(self):
        self._require_symbolic()

        cdef void* ptr
        if self._is_real:
            if self._use_int32:
                ptr = <void*>self._fact_di.Q1fill
            else:
                ptr = <void*>self._fact_dl.Q1fill
        else:
            if self._use_int32:
                ptr = <void*>self._fact_zi.Q1fill
            else:
                ptr = <void*>self._fact_zl.Q1fill

        return _ndarray_copy_from_intptr(ptr, self._N, self._use_int32)

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
        dest._econ = self._econ
        dest._M = self._M
        dest._N = self._N
        dest.itype = self.itype
        dest.dtype = self.dtype
        dest._tol = self._tol

        # Deep copy the factorization
        if self._is_real:
            if self._use_int32:
                dest._fact_di = <spqr_fact_di*>malloc(sizeof(spqr_fact_di))
                _copy_spqr_factor(dest._fact_di, self._fact_di, dest._cm)
            else:
                dest._fact_dl = <spqr_fact_dl*>malloc(sizeof(spqr_fact_dl))
                _copy_spqr_factor(dest._fact_dl, self._fact_dl, dest._cm)
        else:
            if self._use_int32:
                dest._fact_zi = <spqr_fact_zi*>malloc(sizeof(spqr_fact_zi))
                _copy_spqr_factor(dest._fact_zi, self._fact_zi, dest._cm)
            else:
                dest._fact_zl = <spqr_fact_zl*>malloc(sizeof(spqr_fact_zl))
                _copy_spqr_factor(dest._fact_zl, self._fact_zl, dest._cm)

        return dest

    def factorize(self, object A, *, object tol=None):
        """Compute the numeric factorization of the matrix.

        Parameters
        ----------
        A : (M, N) array_like or sparse array, optional
            An array convertible to a sparse matrix. If None, the numeric factorization
            is computed for the matrix used in the constructor. If ``A`` is provided,
            it must have the same sparsity pattern as the matrix used in the
            constructor.
        tol : float, optional
            If the 2-norm of a column in ``A`` is less than ``tol``, that column is
            considered to be a zero column. If ``None``, tolerance used in the
            constructor is used.

        Returns
        -------
        SPQRFactor
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
                    self._tol, Ac, self._fact_di, self._cm
                )
            else:
                SuiteSparseQR_numeric[double, int64_t](
                    self._tol, Ac, self._fact_dl, self._cm
                )
        else:
            if self._use_int32:
                SuiteSparseQR_numeric[doublecomplex, int32_t](
                    self._tol, Ac, self._fact_zi, self._cm
                )
            else:
                SuiteSparseQR_numeric[doublecomplex, int64_t](
                    self._tol, Ac, self._fact_zl, self._cm
                )

        # TODO proper error handling
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"Error {self._cm.status}")

        return self

    def qmult(self, object X, method='QX'):
        """Multiply by ``Q`` or ``Q.T`` using the SPQR factorization.

        Parameters
        ----------
        X : (M, N) numpy.ndarray or sparse array
            The matrix to be multiplied. Must have compatible shape with ``Q``.
        method : str , optional
            The multiplication method. Options are:

            * ``QX`` : compute :math:`Q X`
            * ``QTX`` : compute :math:`Q^{\top} X`
            * ``XQ`` : compute :math:`X Q`
            * ``XQT`` : compute :math:`X Q^{\top}`

            Default is ``QX``.

        Returns
        -------
        Y : (M, N) numpy.ndarray or sparse array
            The result of the multiplication. If ``X`` is a sparse array, then ``Y`` is
            also returned as a sparse array.
        """
        self._require_numeric()

        if not (isinstance(X, np.ndarray) or issparse(X)):
            raise ValueError("X must be an ndarray or sparse matrix.")

        if X.dtype != self.dtype:
            raise ValueError(
                f"Input and factor dtypes do not match. {self.dtype=} and {X.dtype=}"
            )

        if X.ndim not in (1, 2):
            raise ValueError("X must be a 1D or 2D array.")

        cdef int c_method
        if method == 'QX':
            c_method = SPQR_QX
        elif method == 'QTX':
            c_method = SPQR_QTX
        elif method == 'XQ':
            c_method = SPQR_XQ
        elif method == 'XQT':
            c_method = SPQR_XQT
        else:
            raise ValueError(
                f"Invalid method '{method}'. "
                "Expected one of ['QX', 'QTX', 'XQ', 'XQT']."
            )

        # Check shape compatibility with Q
        cdef Py_ssize_t X_dim = (
            X.shape[0]
            if method == 'QTX' or method == 'QX'
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
            X = X.toarray()

        # cholmod_dense expects column-oriented
        X = np.asfortranarray(X)

        Y = self._qmult(c_method, X)

        if return_sparse:
            Y = csc_array(Y, dtype=X.dtype)
            Y.indptr = Y.indptr.astype(self.itype)
            Y.indices = Y.indices.astype(self.itype)

        if return_1D:
            Y = Y[:, 0]

        return Y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _qmult(self, int method, value_t[::1, :] X):
        cdef cholmod_dense Xdense
        cdef cholmod_dense *Xd = &Xdense
        _cholmod_dense_from_ndarray(X, Xd)

        cdef cholmod_dense *Yd

        if self._is_real:
            if self._use_int32:
                Yd = SuiteSparseQR_qmult[double, int32_t](
                    method, self._fact_di, Xd, self._cm
                )
            else:
                Yd = SuiteSparseQR_qmult[double, int64_t](
                    method, self._fact_dl, Xd, self._cm
                )
        else:
            if self._use_int32:
                Yd = SuiteSparseQR_qmult[doublecomplex, int32_t](
                    method, self._fact_zi, Xd, self._cm
                )
            else:
                Yd = SuiteSparseQR_qmult[doublecomplex, int64_t](
                    method, self._fact_zl, Xd, self._cm
                )

        # TODO handle errors
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"qmult error {self._cm.status}")

        return _ndarray_from_cholmod_dense(Yd, self._use_int32, self._cm)

    def solve(self, object b, *, bint transpose=False):
        """Solve a linear system using the SPQR factorization.

        This method solves a linear system for :math:`x` given the right-hand side
        :math:`b` as either a vector or a matrix with multiple right-hand sides.

        If ``transpose=False``, solve

        .. math::
            A x = b

        or, if ``transpose=True``, solve

        .. math::
            A^{\top} x = b

        The method uses the QR factorization of :math:`A` previously computed by
        :meth:`.factorize`.

        Parameters
        ----------
        b : (M,) or (M, K) numpy.ndarray
            The right-hand side vector or matrix. ``M`` should be the number of rows in
            ``A`` if ``transpose=False``, otherwise the number of columns.
        transpose : bool, optional
            Whether to solve the transposed system. Default is False.

        Returns
        -------
        x : (N,) or (N, K) numpy.ndarray or sparse array
            The solution vector or matrix. If ``b`` is a 1D array, then ``x`` is
            returned as a 1D array. If ``b`` is a 2D array with ``K`` columns,
            then ``x`` is returned as a 2D array with ``K`` columns. If ``b``
            is a sparse array, then ``x`` is also returned as a sparse array.
            ``N`` is the number of columns in ``A`` if ``transpose=False``,
            otherwise the number of rows.
        """
        self._require_numeric()

        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")

        if b.dtype != self.dtype:
            raise ValueError(
                f"LHS and RHS dtypes do not match. {self.dtype=} and {b.dtype=}"
            )

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

        # TODO Check the rank of A and warn if rank deficient
        # self._check_rank()

        cdef bint return_1D = b.ndim == 1
        cdef bint return_sparse = issparse(b)

        # CHOLMOD routines require a 2D array
        if b.ndim == 1:
            if not transpose:
                b = b.reshape((self._M, 1))
            else:
                b = b.reshape((self._N, 1))

        # The SuiteSparseQR_solve "sparse" routine just converts b to
        # cholmod_dense internally.
        if issparse(b):
            b = b.toarray()

        # Ensure columns are contiguous for multiple RHS
        b = np.asfortranarray(b)

        x = self._solve(b, transpose)

        if return_sparse:
            x = csc_array(x, dtype=b.dtype)
            x.indptr = x.indptr.astype(self.itype)
            x.indices = x.indices.astype(self.itype)

        if return_1D:
            x = x[:, 0]

        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _solve(self, value_t[::1, :] b, bint transpose):
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
                    Bd = SuiteSparseQR_qmult[double, int32_t](
                        SPQR_QTX, self._fact_di, Bd, self._cm
                    )
                else:
                    Bd = SuiteSparseQR_qmult[double, int64_t](
                        SPQR_QTX, self._fact_dl, Bd, self._cm
                    )
            else:
                if self._use_int32:
                    Bd = SuiteSparseQR_qmult[doublecomplex, int32_t](
                        SPQR_QTX, self._fact_zi, Bd, self._cm
                    )
                else:
                    Bd = SuiteSparseQR_qmult[doublecomplex, int64_t](
                        SPQR_QTX, self._fact_zl, Bd, self._cm
                    )

        # TODO handle errors
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"qmult error {self._cm.status}")

        # Solve the system
        cdef int system = SPQR_RETX_EQUALS_B if not transpose else SPQR_RTX_EQUALS_ETB
        cdef cholmod_dense *Xd

        if self._is_real:
            if self._use_int32:
                Xd = SuiteSparseQR_solve[double, int32_t](
                    system, self._fact_di, Bd, self._cm
                )
            else:
                Xd = SuiteSparseQR_solve[double, int64_t](
                    system, self._fact_dl, Bd, self._cm
                )
        else:
            if self._use_int32:
                Xd = SuiteSparseQR_solve[doublecomplex, int32_t](
                    system, self._fact_zi, Bd, self._cm
                )
            else:
                Xd = SuiteSparseQR_solve[doublecomplex, int64_t](
                    system, self._fact_zl, Bd, self._cm
                )

        # TODO handle errors
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"solve error {self._cm.status}")

        # System is A.T x = b -> (QRE.T).Tx = b -> (E R.T Q.T) x = b
        # But "solve" does not touch Q, so -> (E R.T) (Q.T x) = b
        if transpose:
            # post-multiply by Q.T
            if self._is_real:
                if self._use_int32:
                    Xd = SuiteSparseQR_qmult[double, int32_t](
                        SPQR_QX, self._fact_di, Xd, self._cm
                    )
                else:
                    Xd = SuiteSparseQR_qmult[double, int64_t](
                        SPQR_QX, self._fact_dl, Xd, self._cm
                    )
            else:
                if self._use_int32:
                    Xd = SuiteSparseQR_qmult[doublecomplex, int32_t](
                        SPQR_QX, self._fact_zi, Xd, self._cm
                    )
                else:
                    Xd = SuiteSparseQR_qmult[doublecomplex, int64_t](
                        SPQR_QX, self._fact_zl, Xd, self._cm
                    )

        # TODO handle errors
        if self._cm.status != CHOLMOD_OK:
            raise SPQRError(f"qmult QX error {self._cm.status}")

        return _ndarray_from_cholmod_dense(Xd, self._use_int32, self._cm)

    # ---------------------------------------------------------------------------------
    #         Private API
    # ---------------------------------------------------------------------------------
    cdef inline int _require_symbolic(self) except -1:
        """Raise an error if the symbolic factorization has not been computed yet."""
        if self._is_real:
            if self._use_int32:
                assert self._fact_di is not NULL and self._fact_di.QRsym is not NULL
            else:
                assert self._fact_dl is not NULL and self._fact_dl.QRsym is not NULL
        else:
            if self._use_int32:
                assert self._fact_zi is not NULL and self._fact_zi.QRsym is not NULL
            else:
                assert self._fact_zl is not NULL and self._fact_zl.QRsym is not NULL

    cdef inline int _require_numeric(self) except -1:
        """Raise an error if the numeric factorization has not been computed yet."""
        self._require_symbolic()
        if self._is_real:
            if self._use_int32:
                assert self._fact_di.QRnum is not NULL
            else:
                assert self._fact_dl.QRnum is not NULL
        else:
            if self._use_int32:
                assert self._fact_zi.QRnum is not NULL
            else:
                assert self._fact_zl.QRnum is not NULL

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
    """Compute the SPQR factorization of a sparse matrix.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    use_singletons : bool, optional
        If True, directly compute the numeric factorization to exploit singleton rows.
        Otherwise, only perform symbolic analysis. Default is False, so that the factor
        can be reused efficiently for multiple numeric factorizations.
    order : int, optional
        The ordering strategy to use.
    tol : float, optional
        If the 2-norm of a column in ``A`` is less than ``tol``, that column is
        considered to be a zero column. If ``None``, the default tolerance is used.

    Returns
    -------
    SPQRFactor
        The SPQR factorization of the input matrix.
    """
    if use_singletons:
        return SPQRFactor(A, use_singletons=True, order=order, tol=tol)
    else:
        return SPQRFactor(A, use_singletons=False, order=order, tol=tol).factorize(A)


# TODO rewrite using the simple SPQR interface?
def spqr_solve(A, b, *, transpose=False):
    """Solve a linear system using the SPQR factorization.

    This function solves a linear system for :math:`x` given the right-hand side
    :math:`b` as either a vector or a matrix with multiple right-hand sides.

    If ``transpose=False``, solve

    .. math::
        A x = b

    or, if ``transpose=True``, solve

    .. math::
        A^{\top} x = b

    The function uses the QR factorization of :math:`A` previously computed by
    :meth:`.factorize`.

    Parameters
    ----------
    A : (M, N) array_like or sparse array
        An array convertible to a sparse matrix.
    b : (M,) or (M, K) numpy.ndarray
        The right-hand side vector or matrix. ``M`` should be the number of rows in
        ``A`` if ``transpose=False``, otherwise the number of columns.
    transpose : bool, optional
        Whether to solve the transposed system. Default is False.

    Returns
    -------
    x : (N,) or (N, K) numpy.ndarray or sparse array
        The solution vector or matrix. If ``b`` is a 1D array, then ``x`` is
        returned as a 1D array. If ``b`` is a 2D array with ``K`` columns,
        then ``x`` is returned as a 2D array with ``K`` columns. If ``b``
        is a sparse array, then ``x`` is also returned as a sparse array.
        ``N`` is the number of columns in ``A`` if ``transpose=False``,
        otherwise the number of rows.
    """
    return SPQRFactor(A, use_singletons=True).solve(b, transpose=transpose)
