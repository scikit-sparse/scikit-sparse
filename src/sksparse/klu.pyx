# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: klu.pyx
#  Created: 2025-10-30 21:02
# =============================================================================

"""
=================================================
Clark Kent LU Decomposition (:mod:`sksparse.klu`)
=================================================

.. currentmodule:: sksparse.klu

.. versionadded:: 0.5.0


An interface to the SuiteSparse `KLU
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/KLU>`_
package, which computes the LU factorization and solves systems of equations
for sparse, possibly non-symmetric, indefinite matrices.


Function Interface
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    klu_solve - Solve a linear system using the KLU factorization.


Object Interface
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    klu_factor - Compute the LU factorization of a sparse matrix.
    KLUFactor - An object-oriented interface to KLU.
    KLUInfo - A dataclass to return KLU info.
    KLUControl - A dataclass to set KLU control parameters.


.. _klu-exceptions:

Warnings and Exceptions
-----------------------

.. autosummary::
    :toctree: generated/

    KLUWarning
    KLUSingularMatrixWarning

    KLUError
    KLUOutOfMemoryError
    KLUInvalidError
    KLUOverflowError


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse KLU <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/KLU>`_
"""

cimport cython

from copy import deepcopy
import numpy as np
from scipy.sparse import issparse, csc_array
import warnings

from .utils import validate_csc_input


__all__ = [
    "KLUWarning",
    "KLUSingularMatrixWarning",
    "KLUError",
    "KLUOutOfMemoryError",
    "KLUInvalidError",
    "KLUOverflowError",
    "KLUInfo",
    "KLUControl",
    "KLUFactor",
    "klu_factor",
    "klu_solve",
]


# -----------------------------------------------------------------------------
#         Define types
# -----------------------------------------------------------------------------
ctypedef fused index_t:
    int32_t
    int64_t


ctypedef fused value_t:
    double
    double complex


ctypedef fused common_t:
    klu_common
    klu_l_common


ctypedef fused symbolic_t:
    klu_symbolic
    klu_l_symbolic


ctypedef fused numeric_t:
    klu_numeric
    klu_l_numeric


ctypedef fused ctrl_t:
    int
    double


# -------------------------------------------------------------------------------------
#         Warnings and Errors
# -------------------------------------------------------------------------------------
class KLUWarning(Warning):
    """Base warning for KLU-related warnings."""
    pass


class KLUSingularMatrixWarning(KLUWarning):
    """Warning raised when a singular matrix is encountered."""
    pass


class KLUError(Exception):
    """Base exception for KLU-related errors."""
    pass


class KLUOutOfMemoryError(MemoryError, KLUError):
    """Exception raised when KLU runs out of memory."""
    pass


class KLUInvalidError(KLUError):
    """Exception raised for invalid inputs to KLU."""
    pass


class KLUOverflowError(OverflowError, KLUError):
    """Exception raised when KLU encounters an overflow."""
    pass


# Known Errors
cdef dict _ERROR_INDEX = {
    KLU_SINGULAR: (KLUSingularMatrixWarning, "The matrix is singular."),
    KLU_OUT_OF_MEMORY: (KLUOutOfMemoryError, "KLU ran out of memory."),
    KLU_INVALID: (KLUInvalidError, "An invalid input was provided to KLU."),
    KLU_TOO_LARGE: (KLUOverflowError, "The matrix is too large for KLU to handle."),
}


cdef int _handle_errors(int status) except -1 with gil:
    """Handle KLU errors by raising Python exceptions or warnings.

    This function should be called with the return ``status`` after any KLU
    C function that may fail.

    Parameters
    ----------
    status : int
        The KLU exit status code.

    Returns
    -------
    None

    Raises
    ------
    :exc:`KLUWarning` or subclass
        Raises a warning for non-critical issues.
    :exc:`KLUError` or subclass
        Raises an appropriate Python exception based on the KLU status code.
    """
    if status == KLU_OK:
        return 0

    # Fallback to generic error for unknown codes
    exc_class, msg = _ERROR_INDEX.get(
        status,
        (KLUError, "An unknown KLU error occurred.")
    )
    full_msg = f"{msg} (code {status:d})"

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class, stacklevel=2)
    else:
        raise exc_class(full_msg)


# -------------------------------------------------------------------------------------
#         KLU Control and Info Classes
# -------------------------------------------------------------------------------------
cdef object _get_scale_string(int scale):
    """Convert KLU row scaling integer to string."""
    if scale == -1:
        return "none_no_check"
    elif scale == 0:
        return "none"
    elif scale == 1:
        return "sum"
    elif scale == 2:
        return "max"
    else:
        return "unknown"


cdef object _get_ordering_string(int ordering):
    """Convert KLU ordering integer to string."""
    if ordering == 0:
        return "AMD"
    elif ordering == 1:
        return "COLAMD"
    elif ordering == 2:
        return "user_perm"
    elif ordering == 3:
        return "user_func"
    else:
        return ""


@cython.dataclasses.dataclass(frozen=True)
cdef class KLUInfo:
    """A dataclass to store KLU information.

    Attributes
    ----------
    noffdiag : int
        Number of off-diagonal entries in the matrix.
    nrealloc : int
        Number of memory reallocations during factorization.
    rcond : float
        Estimate of the reciprocal of the condition number.
    singular_col : int
        Index of the first singular column, if any.
    rgrowth : float
        The reciprocal pivot growth factor.
    flops : int
        Estimated number of floating-point operations.
    nblocks : int
        Number of blocks in the BTF ordering of the matrix.
    ordering : str
        The fill-reducing ordering used.
    scale : str
        The row-scaling method used.
    lnz : int
        Number of nonzeros in the L factor.
    unz : int
        Number of nonzeros in the U factor.
    nzoff : int
        Number of nonzeros in the F factor ("offset").
    tol : float
        The pivot tolerance used.
    mempeak : int
        Peak memory usage in bytes.
    """
    noffdiag : int | None = None
    nrealloc : int | None = None
    rcond : double | None = None
    singular_col : int | None = None
    rgrowth : double | None = None
    flops : int | None = None
    nblocks : int | None = None
    ordering : str | None = None
    scale : str | None = None
    lnz : int | None = None
    unz : int | None = None
    nzoff : int | None = None
    tol : double | None = None
    mempeak : int | None = None

    cdef KLUInfo update_from_klu(
        self, klu_symbolic* symbolic, klu_numeric* numeric, klu_common* cm
    ):
        """Update a KLUInfo object from KLU structs."""
        if cm is not NULL:
            self.noffdiag = cm.noffdiag
            self.nrealloc = cm.nrealloc
            self.rcond = cm.rcond
            self.singular_col = cm.singular_col
            self.rgrowth = cm.rgrowth
            self.flops = <int>cm.flops
            self.ordering = _get_ordering_string(cm.ordering)
            self.scale = _get_scale_string(cm.scale)
            self.tol = cm.tol
            self.mempeak = cm.mempeak

        if symbolic is not NULL:
            self.nblocks = symbolic.nblocks

        if numeric is not NULL:
            self.lnz = numeric.lnz
            self.unz = numeric.unz
            self.nzoff = numeric.nzoff

        return self

    cdef KLUInfo update_from_l_klu(
        self, klu_l_symbolic* symbolic, klu_l_numeric* numeric, klu_l_common* cm
    ):
        """Create a KLUInfo object from KLU structs."""
        if cm is not NULL:
            self.noffdiag = cm.noffdiag
            self.nrealloc = cm.nrealloc
            self.rcond = cm.rcond
            self.singular_col = cm.singular_col
            self.rgrowth = cm.rgrowth
            self.flops = <int>cm.flops
            self.ordering = _get_ordering_string(cm.ordering)
            self.scale = _get_scale_string(cm.scale)
            self.tol = cm.tol
            self.mempeak = cm.mempeak

        if symbolic is not NULL:
            self.nblocks = symbolic.nblocks

        if numeric is not NULL:
            self.lnz = numeric.lnz
            self.unz = numeric.unz
            self.nzoff = numeric.nzoff

        return self


cdef dict _SCALE_INDEX = {
    "none_no_check": -1,
    "none": 0,
    "sum": 1,
    "max": 2,
}


cdef dict _ORDERING_INDEX = {
    "AMD": 0,
    "COLAMD": 1,
    "user_perm": 2,
    "user_func": 3,
}


cdef list _CONTROL_KEYS = [
    "tol",
    "memgrow",
    "initmem_amd",
    "initmem",
    "maxwork",
    "btf",
    "ordering",
    "scale",
]


cdef class KLUControl:
    """A dataclass to set KLU control parameters.

    Attributes
    ----------
    tol : float
        The pivot tolerance. Default is ``None``, which uses the ``KLU`` default of
        ``0.001``.
    memgrow : float
        The memory growth factor. Default is ``None``, which uses the ``KLU``
        default of ``1.2``.
    initmem_amd : float
        The initial memory allocation factor for AMD. Default is ``None``, which
        uses the ``KLU`` default of ``1.2``.
    initmem : float
        The initial memory allocation factor for the numeric factorization.
        Default is ``None``, which uses the ``KLU`` default of ``10``.
    maxwork : float
        The maximum work done by BTF. Default is ``None``, which uses the ``KLU``
        default of ``0``, or unlimited.
    btf : bool
        Whether to use BTF pre-ordering. Default is ``None``, which uses the
        ``KLU`` default of ``True``.
    ordering : str
        The fill-reducing ordering. Accepted values are:

        * ``AMD``: Approximate Minimum Degree ordering.
        * ``COLAMD`` : Column Approximate Minimum Degree ordering.
        * ``user_perm``: User-provided ordering (not yet supported).
        * ``user_func``: User-defined ordering function (not yet supported).

        Default is ``None``, which uses the ``KLU`` default setting of ``AMD``.
    scale : str
        The row-scaling method. Accepted values are:

        * ``none_no_check``
        * ``none``
        * ``sum``
        * ``max``

        Default is ``None``, which uses the ``KLU`` default setting of ``max``.
    """
    cdef:
        double _FLOAT_NONE
        int _INT_NONE
        double _tol
        double _memgrow
        double _initmem_amd
        double _initmem
        double _maxwork
        bint _btf
        int _ordering
        int _scale

    def __cinit__(self, **kwargs):
        """Initialize the KLUControl object."""
        self._FLOAT_NONE = -999.0
        self._INT_NONE = -999

        self._tol = self._FLOAT_NONE
        self._memgrow = self._FLOAT_NONE
        self._initmem_amd = self._FLOAT_NONE
        self._initmem = self._FLOAT_NONE
        self._maxwork = self._FLOAT_NONE
        self._btf = self._INT_NONE
        self._ordering = self._INT_NONE
        self._scale = self._INT_NONE

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except KeyError:
                raise KeyError(
                    f"Invalid control parameter: {key}. "
                    f"Expected one of {self.__dict__.keys()}"
                )

    @property
    def tol(self):
        return None if self._tol == self._FLOAT_NONE else self._tol

    @tol.setter
    def tol(self, value):
        if value is None:
            self._tol = self._FLOAT_NONE
        else:
            try:
                assert 0.0 <= value <= 1.0
            except (AssertionError, TypeError):
                raise ValueError("tol must be a float in the range [0, 1].")
            self._tol = value

    @property
    def memgrow(self):
        return None if self._memgrow == self._FLOAT_NONE else self._memgrow

    @memgrow.setter
    def memgrow(self, value):
        if value is None:
            self._memgrow = self._FLOAT_NONE
        else:
            try:
                assert value > 1.0
            except (AssertionError, TypeError):
                raise ValueError("memgrow must be a float greater than 1.0.")
            self._memgrow = value

    @property
    def initmem_amd(self):
        return None if self._initmem_amd == self._FLOAT_NONE else self._initmem_amd

    @initmem_amd.setter
    def initmem_amd(self, value):
        if value is None:
            self._initmem_amd = self._FLOAT_NONE
        else:
            try:
                assert value > 1.0
            except (AssertionError, TypeError):
                raise ValueError("initmem_amd must be a float greater than 1.0.")
            self._initmem_amd = value

    @property
    def initmem(self):
        return None if self._initmem == self._FLOAT_NONE else self._initmem

    @initmem.setter
    def initmem(self, value):
        if value is None:
            self._initmem = self._FLOAT_NONE
        else:
            try:
                assert value > 0.0
            except (AssertionError, TypeError):
                raise ValueError("initmem must be a float greater than 0.0.")
            self._initmem = value

    @property
    def maxwork(self):
        return None if self._maxwork == self._FLOAT_NONE else self._maxwork

    @maxwork.setter
    def maxwork(self, value):
        self._maxwork = value if value is not None else self._FLOAT_NONE

    @property
    def btf(self):
        return None if self._btf == self._INT_NONE else self._btf

    @btf.setter
    def btf(self, value):
        self._btf = value if value is not None else self._INT_NONE

    @property
    def ordering(self):
        return _get_ordering_string(self._ordering)

    @ordering.setter
    def ordering(self, value):
        if value is None:
            self._ordering = self._INT_NONE
            return

        if value in ["user_perm", "user_func"]:
            raise NotImplementedError(
                f"The ordering method '{value}' is not yet supported."
            )

        try:
            self._ordering = _ORDERING_INDEX[value]
        except KeyError:
            raise ValueError(
                f"Invalid value for 'ordering': {value}. "
                f"Expected one of {list(_ORDERING_INDEX.keys())}"
            )

    @property
    def scale(self):
        return _get_scale_string(self._scale)

    @scale.setter
    def scale(self, value):
        if value is None:
            self._scale = self._INT_NONE
            return

        try:
            self._scale = _SCALE_INDEX[value]
        except KeyError:
            raise ValueError(
                f"Invalid value for 'scale': {value}. "
                f"Expected one of {list(_SCALE_INDEX.keys())}"
            )

    def __iter__(self):
        cdef str k
        for k in _CONTROL_KEYS:
            yield (k, getattr(self, k))

    def __repr__(self):
        attrs = ",\n    ".join(f"{k}={repr(v)}" for k, v in self)
        return f"{self.__class__.__name__}(\n    {attrs}\n)"

    def __str__(self):
        return self.__repr__()


cdef inline void _set_if_not_none(ctrl_t *dest, ctrl_t src, ctrl_t none_value) noexcept:
    """Set a pointer if the source is not equal to none_value."""
    dest[0] = dest[0] if src == none_value else src

# -------------------------------------------------------------------------------------
#         Copy Functions
# -------------------------------------------------------------------------------------
cdef inline void* _malloc_copy(
    const void* src,
    size_t n,
    size_t size,
    const common_t* cm
):
    """Allocate memory and copy data from src to the new memory."""
    assert cm is not NULL
    if src is NULL:
        return NULL

    cdef void* dest

    if common_t is klu_common:
        dest = klu_malloc(n, size, <klu_common*>cm)
    else:
        dest = klu_l_malloc(n, size, <klu_l_common*>cm)

    _handle_errors(cm.status)

    if dest is NULL:
        return NULL

    if n > 0:
        memcpy(dest, src, n * size)

    return dest


cdef inline void _copy_symbolic_values(
    symbolic_t* dest,
    const symbolic_t* src,
) noexcept:
    """Copy the top-level values of a KLU symbolic struct, excluding pointers."""
    dest.symmetry = src.symmetry
    dest.est_flops = src.est_flops
    dest.lnz = src.lnz
    dest.unz = src.unz
    dest.n = src.n
    dest.nz = src.nz
    dest.nzoff = src.nzoff
    dest.nblocks = src.nblocks
    dest.maxblock = src.maxblock
    dest.ordering = src.ordering
    dest.do_btf = src.do_btf
    dest.structural_rank = src.structural_rank


cdef int _copy_symbolic_base(
    symbolic_t* dest,
    const symbolic_t* src,
    const common_t* cm,
    index_t _dummy=0
) except -1:
    """Deep copy a KLU symbolic struct."""
    assert dest is not NULL
    assert src is not NULL
    assert cm is not NULL

    # Check for bad type combos to prune the fused types before compilation
    # The compiler will generate *all* combinations of the fused type arguments, but
    # only some are valid, so this statement will create unreachable code that then
    # gets pruned away.
    if not (
        (symbolic_t is klu_symbolic and
         common_t is klu_common and
         index_t is int32_t)
        or
        (symbolic_t is klu_l_symbolic and
         common_t is klu_l_common and
         index_t is int64_t)
    ):
        assert False
        return 0

    # Copy the top-level data, but *not* pointers
    _copy_symbolic_values(dest, src)

    cdef size_t n = src.n

    # Deep copy internal arrays
    dest.Lnz = <double*>_malloc_copy(src.Lnz, n, sizeof(double), cm)
    dest.P = <index_t*>_malloc_copy(src.P, n, sizeof(index_t), cm)
    dest.Q = <index_t*>_malloc_copy(src.Q, n, sizeof(index_t), cm)
    dest.R = <index_t*>_malloc_copy(src.R, n + 1, sizeof(index_t), cm)

    return 0


cdef int _copy_symbolic(
    symbolic_t* dest,
    const symbolic_t* src,
    const common_t* cm
) except -1:
    """Deep copy a KLU symbolic struct."""
    if symbolic_t is klu_symbolic:
        return _copy_symbolic_base(dest, src, cm, <int32_t>0)
    else:
        return _copy_symbolic_base(dest, src, cm, <int64_t>0)


cdef inline void _copy_numeric_values(
    numeric_t* dest,
    const numeric_t* src,
) noexcept:
    """Copy the top-level values of a KLU numeric struct, excluding pointers."""
    dest.n = src.n
    dest.nblocks = src.nblocks
    dest.lnz = src.lnz
    dest.unz = src.unz
    dest.max_lnz_block = src.max_lnz_block
    dest.max_unz_block = src.max_unz_block
    dest.worksize = src.worksize
    dest.nzoff = src.nzoff


cdef int _copy_numeric_base(
    numeric_t* dest,
    const numeric_t* src,
    const common_t* cm,
    index_t _dummy_int=0,
    value_t _dummy_val=0
) except -1:
    """Deep copy a KLU numeric struct."""
    assert dest is not NULL
    assert src is not NULL
    assert cm is not NULL

    # Check for bad type combos to prune the fused types before compilation
    # The compiler will generate *all* combinations of the fused type arguments, but
    # only some are valid, so this statement will create unreachable code that then
    # gets pruned away.
    if not (
        (numeric_t is klu_numeric and
         common_t is klu_common and
         index_t is int32_t)
        or
        (numeric_t is klu_l_numeric and
         common_t is klu_l_common and
         index_t is int64_t)
    ):
        assert False
        return 0

    # Copy the top-level data and pointers
    _copy_numeric_values(dest, src)

    cdef size_t n = src.n
    cdef size_t nblocks = src.nblocks

    # Deep copy internal arrays
    dest.Pnum = <index_t*>_malloc_copy(src.Pnum, n, sizeof(index_t), cm)
    dest.Pinv = <index_t*>_malloc_copy(src.Pinv, n, sizeof(index_t), cm)
    dest.Lip = <index_t*>_malloc_copy(src.Lip, n, sizeof(index_t), cm)
    dest.Uip = <index_t*>_malloc_copy(src.Uip, n, sizeof(index_t), cm)
    dest.Llen = <index_t*>_malloc_copy(src.Llen, n, sizeof(index_t), cm)
    dest.Ulen = <index_t*>_malloc_copy(src.Ulen, n, sizeof(index_t), cm)
    dest.Udiag = <value_t*>_malloc_copy(src.Udiag, n, sizeof(value_t), cm)
    dest.LUsize = <size_t*>_malloc_copy(src.LUsize, nblocks, sizeof(size_t), cm)

    if numeric_t is klu_numeric:
        dest.LUbx = <void**>klu_malloc(nblocks, sizeof(value_t*), cm)
    else:
        dest.LUbx = <void**>klu_l_malloc(nblocks, sizeof(value_t*), cm)

    _handle_errors(cm.status)

    cdef size_t k

    if dest.LUbx is not NULL and src.LUbx is not NULL and src.LUsize is not NULL:
        for k in range(nblocks):
            dest.LUbx[k] = <value_t*>_malloc_copy(
                src.LUbx[k], src.LUsize[k], sizeof(value_t), cm
            )

    cdef size_t np1 = n + 1
    cdef size_t nzoffp1 = <size_t>src.nzoff + 1

    dest.Offp = <index_t*>_malloc_copy(src.Offp, np1, sizeof(index_t), cm)
    dest.Offi = <index_t*>_malloc_copy(src.Offi, nzoffp1, sizeof(index_t), cm)
    dest.Offx = <value_t*>_malloc_copy(src.Offx, nzoffp1, sizeof(value_t), cm)

    dest.Rs = <double*>_malloc_copy(src.Rs, n, sizeof(double), cm)

    # Workspace encompasses Xwork and Iwork, so just copy Work
    dest.Work = _malloc_copy(src.Work, src.worksize, 1, cm)
    dest.Xwork = dest.Work
    if dest.Xwork is not NULL:
        dest.Iwork = <index_t*>(<value_t*>dest.Xwork + n)

    return 0


cdef int _copy_numeric(
    numeric_t* dest,
    const numeric_t* src,
    const common_t* cm,
    bint is_real
) except -1:
    """Deep copy a KLU numeric struct."""
    cdef int32_t idx = 0
    cdef int64_t l_idx = 0
    cdef double val = 0
    cdef double complex c_val = 0

    if numeric_t is klu_numeric:
        if is_real:
            return _copy_numeric_base(dest, src, cm, idx, val)
        else:
            return _copy_numeric_base(dest, src, cm, idx, c_val)
    else:
        if is_real:
            return _copy_numeric_base(dest, src, cm, l_idx, val)
        else:
            return _copy_numeric_base(dest, src, cm, l_idx, c_val)


# -------------------------------------------------------------------------------------
#         KLU Class Interface
# -------------------------------------------------------------------------------------
cdef class KLUFactor:
    r"""Class to compute and store the KLU factorization of a sparse matrix.

    The constructor computes the symbolic analysis of a sparse matrix :math:`A`
    and determines a fill-reducing ordering such that:

    .. math::
        L U + F = R P A Q.

    The numeric factorization is not computed until :meth:`.factorize` is called.

    .. note::

        Note that the use of the scale factor ``R`` differs between KLU and UMFPACK:

        .. math::
                L U &= P R_{\mathrm{umf}} A Q \quad &&\text{(UMFPACK)}, \\
            L U + F &= R_{\mathrm{klu}} P A Q \quad &&\text{(KLU)}.

        They are related by :math:`R_{\mathrm{klu}} = P R_{\mathrm{umf}} P^{\top}`.

    Attributes
    ----------
    is_numeric : bool
        Whether the numeric factorization has been computed.
    lnz : int
        The number of nonzeros in the :math:`L` factor.
    unz : int
        The number of nonzeros in the :math:`U` factor.
    nzoff : int
        The number of nonzeros in the :math:`F` factor.
    nblocks : int
        The number of blocks in the BTF ordering of the matrix.
    nnz : int
        The number of nonzeros in the original matrix.
    shape : tuple of int
        The shape of the original matrix.
    dtype : numpy.dtype
        The data type of the matrix entries (``float64`` or ``complex128``).
    itype : numpy.dtype
        The integer type used for indexing (``int32`` or ``int64``).
    L : scipy.sparse.csc_array
        The :math:`L` factor as a sparse CSC matrix.
    U : scipy.sparse.csc_array
        The :math:`U` factor as a sparse CSC matrix.
    F : scipy.sparse.csc_array
        The :math:`F` factor as a sparse CSC matrix.
    perm_r, perm_c : numpy.ndarray
        The row and column permutation arrays.
    rscale : numpy.ndarray
        The row scaling array.
    rblocks : numpy.ndarray of int
        The row blocks in the BTF ordering.
    info : KLUInfo
        An object containing information about the factorization.

    Notes
    -----
    This object is an interface to the SuiteSparse KLU library [#klu_url]_.


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#klu_url] SuiteSparse KLU
        https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/KLU
    """

    cdef:
        Py_ssize_t _N
        readonly object itype
        readonly object dtype
        bint _use_int32
        bint _is_real
        # settings + output info
        klu_common _common
        klu_common* _cm
        klu_l_common _l_common
        klu_l_common* _l_cm
        KLUInfo _info
        # Symbolic analysis
        klu_symbolic* _symbolic
        klu_l_symbolic* _l_symbolic
        # Numeric factorization
        klu_numeric* _numeric
        klu_l_numeric* _l_numeric
        # Cached factor objects
        object _L, _U, _F, _P, _Q, _Rs, _R

    def __init__(self, A, KLUControl control=None):
        """Compute the KLU factorization of a sparse matrix.

        Parameters
        ----------
        A : (N, N) numpy.ndarray or sparse array
            The input matrix. Any object that can be converted to
            a :class:`~scipy.sparse.csc_array` is accepted.
        control : :class:`KLUControl`, optional
            An optional :class:`KLUControl` object to set the factorization parameters.
            If not provided, default parameters are used.
        """
        A, self._use_int32, _ = validate_csc_input(A, require_square=True)

        self._N = A.shape[0]
        self._init_common(control)
        self._init_symbolic(self._N, A.indptr, A.indices, A.data)

    cdef int _init_common(self, KLUControl control=None) except -1:
        """Initialize the KLU common struct with default or user settings."""
        # Initialize common struct with defaults
        if self._use_int32:
            self._cm = &self._common
            assert klu_defaults(self._cm)
        else:
            self._l_cm = &self._l_common
            assert klu_l_defaults(self._l_cm)

        if control is None:
            return 0

        # Set user-defined control parameters
        cdef double _FNONE = control._FLOAT_NONE
        cdef int _INONE = control._INT_NONE

        if self._use_int32:
            _set_if_not_none(&self._cm.tol, control._tol, _FNONE)
            _set_if_not_none(&self._cm.memgrow, control._memgrow, _FNONE)
            _set_if_not_none(&self._cm.initmem_amd, control._initmem_amd, _FNONE)
            _set_if_not_none(&self._cm.initmem, control._initmem, _FNONE)
            _set_if_not_none(&self._cm.maxwork, control._maxwork, _FNONE)
            _set_if_not_none(&self._cm.btf, control._btf, _INONE)
            _set_if_not_none(&self._cm.ordering, control._ordering, _INONE)
            _set_if_not_none(&self._cm.scale, control._scale, _INONE)
        else:
            _set_if_not_none(&self._l_cm.tol, control._tol, _FNONE)
            _set_if_not_none(&self._l_cm.memgrow, control._memgrow, _FNONE)
            _set_if_not_none(&self._l_cm.initmem_amd, control._initmem_amd, _FNONE)
            _set_if_not_none(&self._l_cm.initmem, control._initmem, _FNONE)
            _set_if_not_none(&self._l_cm.maxwork, control._maxwork, _FNONE)
            _set_if_not_none(&self._l_cm.btf, control._btf, _INONE)
            _set_if_not_none(&self._l_cm.ordering, control._ordering, _INONE)
            _set_if_not_none(&self._l_cm.scale, control._scale, _INONE)

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _init_symbolic(
        self,
        Py_ssize_t N,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data,
    ):
        """Compute the symbolic factorization.

        Parameters
        ----------
        N : int
            Number of rows and columns of the matrix.
        indptr : 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : 1D array of index_t
            The row indices array of the CSC matrix.
        """
        self._is_real = value_t is double

        if self._use_int32:
            self._symbolic = klu_analyze(
                N,
                <int32_t*>&indptr[0],
                <int32_t*>&indices[0],
                self._cm
            )
            _handle_errors(self._cm.status)
        else:
            self._l_symbolic = klu_l_analyze(
                N,
                <int64_t*>&indptr[0],
                <int64_t*>&indices[0],
                self._l_cm
            )
            _handle_errors(self._l_cm.status)

        self.itype = np.dtype(np.int32 if self._use_int32 else np.int64)
        self.dtype = np.dtype(np.float64 if self._is_real else np.complex128)

    def __dealloc__(self):
        """Deallocate KLU objects."""
        if self._use_int32:
            if self._symbolic is not NULL:
                klu_free_symbolic(&self._symbolic, self._cm)
            if self._numeric is not NULL:
                if self._is_real:
                    klu_free_numeric(&self._numeric, self._cm)
                else:
                    klu_z_free_numeric(&self._numeric, self._cm)
        else:
            if self._l_symbolic is not NULL:
                klu_l_free_symbolic(&self._l_symbolic, self._l_cm)
            if self._l_numeric is not NULL:
                if self._is_real:
                    klu_l_free_numeric(&self._l_numeric, self._l_cm)
                else:
                    klu_zl_free_numeric(&self._l_numeric, self._l_cm)

    def __iter__(self):
        for attr in ['L', 'U', 'perm_r', 'perm_c', 'rscale', 'F', 'rblocks']:
            yield getattr(self, attr)

    def __repr__(self):
        cls_name = self.__class__.__name__
        factor_type = 'numeric' if self.is_numeric else 'symbolic'
        return (
            f"<{cls_name} {factor_type} factor of dtype '{self.dtype}' "
            f"with '{self.itype}' indices:\n"
            f"    L: {self.shape} with {self.lnz} stored elements\n"
            f"    U: {self.shape} with {self.unz} stored elements>"
        )

    def __str__(self):
        return self.__repr__()

    # ---------------------------------------------------------------------------------
    #         Properties
    # ---------------------------------------------------------------------------------
    @property
    def is_numeric(self):
        if self._use_int32:
            return self._symbolic is not NULL and self._numeric is not NULL
        else:
            return self._l_symbolic is not NULL and self._l_numeric is not NULL

    @property
    def lnz(self):
        if self._use_int32:
            if self._numeric is NULL:
                return None
            val = self._numeric.lnz
        else:
            if self._l_numeric is NULL:
                return None
            val = self._l_numeric.lnz
        return int(val) if val >= 0 else None

    @property
    def unz(self):
        if self._use_int32:
            if self._numeric is NULL:
                return None
            val = self._numeric.unz
        else:
            if self._l_numeric is NULL:
                return None
            val = self._l_numeric.unz
        return int(val) if val >= 0 else None

    @property
    def nzoff(self):
        if self._use_int32:
            if self._numeric is NULL:
                return None
            val = self._numeric.nzoff
        else:
            if self._l_numeric is NULL:
                return None
            val = self._l_numeric.nzoff
        return int(val) if val >= 0 else None

    @property
    def nblocks(self):
        if self._use_int32:
            if self._symbolic is NULL:
                return None
            val = self._symbolic.nblocks
        else:
            if self._l_symbolic is NULL:
                return None
            val = self._l_symbolic.nblocks
        return int(val) if val >= 0 else None

    @property
    def nnz(self):
        if self.lnz is None or self.unz is None:
            return None
        return int(self.lnz + self.unz)

    @property
    def shape(self):
        return (self._N, self._N)

    @property
    def L(self):
        if self._L is None:
            self._get_numeric()
        return self._L

    @property
    def U(self):
        if self._U is None:
            self._get_numeric()
        return self._U

    @property
    def F(self):
        if self._F is None:
            self._get_numeric()
        return self._F

    @property
    def perm_r(self):
        if self._P is None:
            self._get_numeric()
        return self._P

    @property
    def perm_c(self):
        if self._Q is None:
            self._get_numeric()
        return self._Q

    @property
    def rscale(self):
        if self._Rs is None:
            self._get_numeric()
        return self._Rs

    @property
    def rblocks(self):
        if self._R is None:
            self._get_numeric()
        return self._R

    @property
    def info(self):
        if self._info is None:
            self._info = KLUInfo()

        # Compute flops to store in info (needs "is_real" info)
        if self._use_int32:
            if self._is_real:
                klu_flops(self._symbolic, self._numeric, self._cm)
            else:
                klu_z_flops(self._symbolic, self._numeric, self._cm)
        else:
            if self._is_real:
                klu_l_flops(self._l_symbolic, self._l_numeric, self._l_cm)
            else:
                klu_zl_flops(self._l_symbolic, self._l_numeric, self._l_cm)

        if self._use_int32:
            self._info.update_from_klu(self._symbolic, self._numeric, self._cm)
        else:
            self._info.update_from_l_klu(self._l_symbolic, self._l_numeric, self._l_cm)

        return self._info

    # ---------------------------------------------------------------------------------
    #         Public API
    # ---------------------------------------------------------------------------------
    def copy(self):
        """Return a deep copy of the current KLUFactor object."""
        cdef KLUFactor klu = KLUFactor.__new__(KLUFactor)

        klu._N = self._N
        klu.itype = self.itype
        klu.dtype = self.dtype
        klu._use_int32 = self._use_int32
        klu._is_real = self._is_real
        klu._info = deepcopy(self._info)

        # settings + output info
        if self._use_int32:
            assert self._cm is not NULL
            assert self._symbolic is not NULL

            klu._cm = &klu._common
            memcpy(klu._cm, self._cm, sizeof(klu_common))

            klu._symbolic = <klu_symbolic*>klu_malloc(1, sizeof(klu_symbolic), klu._cm)
            _handle_errors(klu._cm.status)
            _copy_symbolic(klu._symbolic, self._symbolic, klu._cm)

            if self._numeric is not NULL:
                klu._numeric = <klu_numeric*>klu_malloc(1, sizeof(klu_numeric), klu._cm)
                _handle_errors(klu._cm.status)
                _copy_numeric(klu._numeric, self._numeric, klu._cm, self._is_real)
        else:
            assert self._l_cm is not NULL
            assert self._l_symbolic is not NULL

            klu._l_cm = &klu._l_common
            memcpy(klu._l_cm, self._l_cm, sizeof(klu_l_common))

            klu._l_symbolic = <klu_l_symbolic*>klu_l_malloc(
                1, sizeof(klu_l_symbolic), klu._l_cm
            )
            _handle_errors(klu._l_cm.status)
            _copy_symbolic(klu._l_symbolic, self._l_symbolic, klu._l_cm)

            if self._l_numeric is not NULL:
                klu._l_numeric = <klu_l_numeric*>klu_l_malloc(
                    1, sizeof(klu_l_numeric), klu._l_cm
                )
                _handle_errors(klu._l_cm.status)
                _copy_numeric(klu._l_numeric, self._l_numeric, klu._l_cm, self._is_real)

        # Cached factor objects
        klu._L = self._L.copy() if self._L is not None else None
        klu._U = self._U.copy() if self._U is not None else None
        klu._F = self._F.copy() if self._F is not None else None
        klu._P = self._P.copy() if self._P is not None else None
        klu._Q = self._Q.copy() if self._Q is not None else None
        klu._Rs = self._Rs.copy() if self._Rs is not None else None
        klu._R = self._R.copy() if self._R is not None else None

        return klu

    def factorize(self, object A):
        r"""Compute the numeric factorization of the matrix.

        Computes the numeric factorization of a sparse matrix :math:`A`
        and determines a fill-reducing ordering such that:

        .. math::
            L U + F = R P A Q.

        If given, the matrix :math:`A` must have the same shape and nonzero pattern as
        the one used to create this :class:`KLUFactor` object, but need not have the
        same values.

        .. warning::

            No check is made on the non-zero structure of the input matrix, so if it is
            different from the one used for the symbolic analysis, the results will be
            incorrect without raising an error.

        Parameters
        ----------
        A : (N, N) numpy.ndarray or sparse array
            The input matrix. Must have the same shape and nonzero pattern as
            the matrix used to create this :class:`KLUFactor` object. If not
            provided, the original matrix given to the constructor will be
            used.

        Returns
        -------
        :class:`KLUFactor`
            The current object, for method chaining.
        """
        _msg = "Symbolic analysis not present. Cannot perform numeric factorization."
        if self._use_int32:
            assert self._symbolic is not NULL, _msg
        else:
            assert self._l_symbolic is not NULL, _msg

        A, _, itype = validate_csc_input(A, require_square=True)
        self._check_input_matrix(A, itype)

        # Clear cached factor objects
        self._L = None
        self._U = None
        self._F = None
        self._P = None
        self._Q = None
        self._Rs = None
        self._R = None

        if self._numeric is not NULL or self._l_numeric is not NULL:
            # Refactorize with existing numeric struct
            self._refactorize(A.indptr, A.indices, A.data)
        else:
            # Allocate and compute new numeric struct
            self._factorize(A.indptr, A.indices, A.data)

        # Compute reciprocal pivot growth for KLUInfo O(|A| + |U|)
        self._rgrowth(A.indptr, A.indices, A.data)

        # Compute rough condition number estimate min(abs(diag(U)) / max(abs(diag(U)))
        self._rcond()

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _factorize(
        self,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data,
    ):
        """Compute the numeric factorization given the CSC arrays.

        Parameters
        ----------
        indptr : contiguous 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : contiguous 1D array of index_t
            The row indices array of the CSC matrix.
        data : contiguous 1D array of value_t
            The data array of the CSC matrix.
        """
        # Compute the numeric factorization
        if self._use_int32:
            if self._is_real:
                self._numeric = c_klu_factor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._cm
                )
            else:
                self._numeric = klu_z_factor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._cm
                )
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                self._l_numeric = klu_l_factor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_cm
                )
            else:
                self._l_numeric = klu_zl_factor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_cm
                )
            _handle_errors(self._l_cm.status)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _refactorize(
        self,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data,
    ):
        """Re-compute the numeric factorization given the CSC arrays.

        Parameters
        ----------
        indptr : contiguous 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : contiguous 1D array of index_t
            The row indices array of the CSC matrix.
        data : contiguous 1D array of value_t
            The data array of the CSC matrix.
        """
        # Compute the numeric factorization
        if self._use_int32:
            if self._is_real:
                klu_refactor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._numeric,
                    self._cm
                )
            else:
                klu_z_refactor(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._numeric,
                    self._cm
                )
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_refactor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_numeric,
                    self._l_cm
                )
            else:
                klu_zl_refactor(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_numeric,
                    self._l_cm
                )
            _handle_errors(self._l_cm.status)

    def solve(self, object b, *, bint transpose=False):
        r"""Solve a linear system using the KLU factorization.

        This method solves a linear system for :math:`x` given the right-hand side
        :math:`b` as either a vector or a matrix with multiple right-hand sides.

        If ``transpose=False``, solve

        .. math::
            A x = b

        or, if ``transpose=True``, solve

        .. math::
            x A = b \Longleftrightarrow A^{\top} x^{\top} = b^{\top}.

        The method uses the LU factorization of :math:`A` previously computed by
        :meth:`.factorize`.

        Parameters
        ----------
        b : (N,) or (N, K) numpy.ndarray
            The right-hand side vector or matrix.

        Returns
        -------
        x : (N,) or (N, K) numpy.ndarray or sparse array
            The solution vector or matrix. If ``b`` is a 1D array, then ``x`` is
            returned as a 1D array. If ``b`` is a 2D array with ``K`` columns,
            then ``x`` is returned as a 2D array with ``K`` columns. If ``b``
            is a sparse array, then ``x`` is also returned as a sparse array.
        """
        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")

        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        cdef bint return_1D = b.ndim == 1
        cdef size_t N = b.shape[0] if (b.ndim == 1 or not transpose) else b.shape[1]
        cdef size_t K = 1 if b.ndim == 1 else (b.shape[0] if transpose else b.shape[1])

        if N != self._N:
            raise ValueError(
                "Right-hand side b must have compatible shape with A. "
                f"Got {b.shape=}, but A.shape={self.shape} ({transpose=})."
            )

        if np.can_cast(b.dtype, self.dtype):
            b = b.astype(self.dtype, copy=False)
        else:
            raise TypeError(f"Cannot safely cast {b.dtype=} to {self.dtype=}.")

        # Check the condition number
        self._check_rcond()

        cdef bint return_sparse = issparse(b)

        if return_sparse:
            b = b.toarray()
        else:
            b = np.asarray(b)

        # Ensure columns are contiguous for multiple RHS
        b = np.asfortranarray(b)

        # The klu_solve function overwrites the input with the output
        x = b.copy()

        if transpose:
            x = x.T.conj()

        # klu_solve expects B as a column-oriented 1D array
        x = x.reshape(-1, order='F')

        if transpose:
            self._tsolve(K, x)
        else:
            self._solve(K, x)

        # Reshape X into a  2D array
        x = x.reshape(self._N, K, order='F')

        if return_sparse:
            x = csc_array(x, dtype=b.dtype)
            x.indptr = x.indptr.astype(self.itype, copy=False)
            x.indices = x.indices.astype(self.itype, copy=False)

        if return_1D:
            x = x[:, 0]

        if transpose:
            x = x.T.conj()

        return x

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _solve(self, size_t K, value_t[::1] x):
        """Solve Ax = b.

        Parameters
        ----------
        K : int
            The number of right-hand sides to solve.
        x : (N * K,) array_like
            The right-hand side matrix on input, in column-oriented form, solution on
            output.
        """
        cdef double *x_ptr = <double*>&x[0]

        if self._use_int32:
            if self._is_real:
                c_klu_solve(self._symbolic, self._numeric, self._N, K, x_ptr, self._cm)
            else:
                klu_z_solve(self._symbolic, self._numeric, self._N, K, x_ptr, self._cm)
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_solve(
                    self._l_symbolic, self._l_numeric, self._N, K, x_ptr, self._l_cm
                )
            else:
                klu_zl_solve(
                    self._l_symbolic, self._l_numeric, self._N, K, x_ptr, self._l_cm
                )
            _handle_errors(self._l_cm.status)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _tsolve(self, size_t K, value_t[::1] x):
        """Solve xA = b.

        Parameters
        ----------
        K : int
            The number of right-hand sides to solve.
        x : (N * K,) array_like
            The right-hand side matrix on input, in column-oriented form, solution on
            output.
        """
        cdef double *x_ptr = <double*>&x[0]
        cdef int conj_solve = True

        if self._use_int32:
            if self._is_real:
                klu_tsolve(
                    self._symbolic,
                    self._numeric,
                    self._N,
                    K,
                    x_ptr,
                    self._cm
                )
            else:
                klu_z_tsolve(
                    self._symbolic,
                    self._numeric,
                    self._N,
                    K,
                    x_ptr,
                    conj_solve,
                    self._cm
                )
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_tsolve(
                    self._l_symbolic,
                    self._l_numeric,
                    self._N,
                    K,
                    x_ptr,
                    self._l_cm
                )
            else:
                klu_zl_tsolve(
                    self._l_symbolic,
                    self._l_numeric,
                    self._N,
                    K,
                    x_ptr,
                    conj_solve,
                    self._l_cm
                )
            _handle_errors(self._l_cm.status)

    # ---------------------------------------------------------------------------------
    #         Private API
    # ---------------------------------------------------------------------------------
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

    cdef int _rcond(self) except -1:
        """Compute the condition number estimate."""
        if self._use_int32:
            if self._is_real:
                klu_rcond(self._symbolic, self._numeric, self._cm)
            else:
                klu_z_rcond(self._symbolic, self._numeric, self._cm)
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_rcond(self._l_symbolic, self._l_numeric, self._l_cm)
            else:
                klu_zl_rcond(self._l_symbolic, self._l_numeric, self._l_cm)
            _handle_errors(self._l_cm.status)

    cdef int _check_rcond(self) except -1:
        """Check a rough estimate of the condition number.

        Computes ``min(abs(U.diagonal())) / max(abs(U.diagonal()))``. See
        ``klu_condest`` for more accurate estimate from the full LU decomposition.
        """
        cdef double rcond = self._cm.rcond if self._use_int32 else self._l_cm.rcond
        cdef double eps = np.finfo(np.float64).eps

        cdef int singular_col = (
            self._cm.singular_col if self._use_int32 else self._l_cm.singular_col
        )

        if rcond == 0:
            raise KLUError(
                "Matrix is indefinite or singular to working precision. "
                f"Failed on column {singular_col}."
            )
        elif rcond < eps:
            warnings.warn(
                "Matrix is nearly singular."
                f"  Results may be inaccurate (rcond={rcond:.2e}).",
                KLUSingularMatrixWarning
            )

    def _rgrowth(
        self,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data
    ):
        """Compute the growth factor of the LU factorization."""
        if self._use_int32:
            if self._is_real:
                klu_rgrowth(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._numeric,
                    self._cm
                )
            else:
                klu_z_rgrowth(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    self._numeric,
                    self._cm
                )
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_rgrowth(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_numeric,
                    self._l_cm
                )
            else:
                klu_zl_rgrowth(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._l_symbolic,
                    self._l_numeric,
                    self._l_cm
                )
            _handle_errors(self._l_cm.status)

    cdef void _get_numeric(self) except *:
        """Extract and cache the numeric factors from the klu_numeric struct."""
        if (self._use_int32 and self._numeric is NULL) or (
            not self._use_int32 and self._l_numeric is NULL
        ):
            raise KLUError(
                "Numeric factorization not present. Run `KLUFactor.factorize(A)` first."
            )

        # Create output arrays
        Lp = np.empty(self._N + 1, dtype=self.itype)
        Li = np.empty(self.lnz, dtype=self.itype)
        Lx = np.empty(self.lnz, dtype=np.float64)

        Up = np.empty(self._N + 1, dtype=self.itype)
        Ui = np.empty(self.unz, dtype=self.itype)
        Ux = np.empty(self.unz, dtype=np.float64)

        Fp = np.empty(self._N + 1, dtype=self.itype)
        Fi = np.empty(self.nzoff, dtype=self.itype)
        Fx = np.empty(self.nzoff, dtype=np.float64)

        self._P = np.empty(self._N, dtype=self.itype)
        self._Q = np.empty(self._N, dtype=self.itype)
        self._Rs = np.empty(self._N, dtype=np.float64)  # always real
        self._R = np.empty(self.nblocks + 1, dtype=self.itype)

        # Sort the row indices so the output has canonical format
        if self._use_int32:
            if self._is_real:
                klu_sort(self._symbolic, self._numeric, self._cm)
            else:
                klu_z_sort(self._symbolic, self._numeric, self._cm)
            _handle_errors(self._cm.status)
        else:
            if self._is_real:
                klu_l_sort(self._l_symbolic, self._l_numeric, self._l_cm)
            else:
                klu_zl_sort(self._l_symbolic, self._l_numeric, self._l_cm)
            _handle_errors(self._l_cm.status)

        # Extract the numeric factorization
        if self._is_real:
            self._extract(
                Lp, Li, Lx,
                Up, Ui, Ux,
                Fp, Fi, Fx,
                self._P,
                self._Q,
                self._Rs,
                self._R
            )

            self._L = csc_array((Lx, Li, Lp), shape=self.shape)
            self._U = csc_array((Ux, Ui, Up), shape=self.shape)
            self._F = csc_array((Fx, Fi, Fp), shape=self.shape)
        else:
            # Allocate imaginary parts
            Lz = np.empty(self.lnz, dtype=np.float64)
            Uz = np.empty(self.unz, dtype=np.float64)
            Fz = np.empty(self.nzoff, dtype=np.float64)

            self._z_extract(
                Lp, Li, Lx, Lz,
                Up, Ui, Ux, Uz,
                Fp, Fi, Fx, Fz,
                self._P,
                self._Q,
                self._Rs,
                self._R
            )

            self._L = csc_array((Lx + 1j * Lz, Li, Lp), shape=self.shape)
            self._U = csc_array((Ux + 1j * Uz, Ui, Up), shape=self.shape)
            self._F = csc_array((Fx + 1j * Fz, Fi, Fp), shape=self.shape)

        # Return R as reciprocal so user doesn't have to invert it
        np.reciprocal(self._Rs, out=self._Rs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _extract(
        self,
        index_t[::1] Lp, index_t[::1] Li, value_t[::1] Lx,
        index_t[::1] Up, index_t[::1] Ui, value_t[::1] Ux,
        index_t[::1] Fp, index_t[::1] Fi, value_t[::1] Fx,
        index_t[::1] P,
        index_t[::1] Q,
        double[::1] Rs,
        index_t[::1] R,
    ):
        """Call the appropriate KLU extract function.

        Parameters
        ----------
        Lp, Li, Lx : arrays for the L factor
            The output arrays for the L factor in CSC format.
        Up, Ui, Ux : arrays for the U factor
            The output arrays for the U factor in CSC format.
        Fp, Fi, Fx : arrays for the F factor
            The output arrays for the F factor in CSC format.
        P : array of index_t
            The output row permutation array.
        Q : array of index_t
            The output column permutation array.
        Rs : array of double
            The output row scaling factors.
        R : array of index_t
            The output block boundaries.
        """
        # Extract the numeric factorization
        if self._use_int32:
            klu_extract(
                self._numeric,
                self._symbolic,
                <int32_t*>&Lp[0], <int32_t*>&Li[0], <double*>&Lx[0],
                <int32_t*>&Up[0], <int32_t*>&Ui[0], <double*>&Ux[0],
                <int32_t*>&Fp[0], <int32_t*>&Fi[0], <double*>&Fx[0],
                <int32_t*>&P[0],
                <int32_t*>&Q[0],
                <double*>&Rs[0],
                <int32_t*>&R[0],
                self._cm
            )
            _handle_errors(self._cm.status)
        else:
            klu_l_extract(
                self._l_numeric,
                self._l_symbolic,
                <int64_t*>&Lp[0], <int64_t*>&Li[0], <double*>&Lx[0],
                <int64_t*>&Up[0], <int64_t*>&Ui[0], <double*>&Ux[0],
                <int64_t*>&Fp[0], <int64_t*>&Fi[0], <double*>&Fx[0],
                <int64_t*>&P[0],
                <int64_t*>&Q[0],
                <double*>&Rs[0],
                <int64_t*>&R[0],
                self._l_cm
            )
            _handle_errors(self._l_cm.status)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _z_extract(
        self,
        index_t[::1] Lp, index_t[::1] Li, value_t[::1] Lx, value_t[::1] Lz,
        index_t[::1] Up, index_t[::1] Ui, value_t[::1] Ux, value_t[::1] Uz,
        index_t[::1] Fp, index_t[::1] Fi, value_t[::1] Fx, value_t[::1] Fz,
        index_t[::1] P,
        index_t[::1] Q,
        double[::1] Rs,
        index_t[::1] R,
    ):
        """Call the appropriate KLU extract function.

        Parameters
        ----------
        Lp, Li, Lx, Lz : arrays for the L factor
            The output arrays for the L factor in CSC format.
        Up, Ui, Ux, Uz : arrays for the U factor
            The output arrays for the U factor in CSC format.
        Fp, Fi, Fx, Fz : arrays for the F factor
            The output arrays for the F factor in CSC format.
        P : array of index_t
            The output row permutation array.
        Q : array of index_t
            The output column permutation array.
        Rs : array of double
            The output row scaling factors.
        R : array of index_t
            The output block boundaries.
        """
        # Extract the numeric factorization
        if self._use_int32:
            klu_z_extract(
                self._numeric,
                self._symbolic,
                <int32_t*>&Lp[0], <int32_t*>&Li[0], <double*>&Lx[0], <double*>&Lz[0],
                <int32_t*>&Up[0], <int32_t*>&Ui[0], <double*>&Ux[0], <double*>&Uz[0],
                <int32_t*>&Fp[0], <int32_t*>&Fi[0], <double*>&Fx[0], <double*>&Fz[0],
                <int32_t*>&P[0],
                <int32_t*>&Q[0],
                <double*>&Rs[0],
                <int32_t*>&R[0],
                self._cm
            )
            _handle_errors(self._cm.status)
        else:
            klu_zl_extract(
                self._l_numeric,
                self._l_symbolic,
                <int64_t*>&Lp[0], <int64_t*>&Li[0], <double*>&Lx[0], <double*>&Lz[0],
                <int64_t*>&Up[0], <int64_t*>&Ui[0], <double*>&Ux[0], <double*>&Uz[0],
                <int64_t*>&Fp[0], <int64_t*>&Fi[0], <double*>&Fx[0], <double*>&Fz[0],
                <int64_t*>&P[0],
                <int64_t*>&Q[0],
                <double*>&Rs[0],
                <int64_t*>&R[0],
                self._l_cm
            )
            _handle_errors(self._l_cm.status)


# -----------------------------------------------------------------------------
#         Convenience Functions
# -----------------------------------------------------------------------------
def klu_factor(A, *, KLUControl control=None, **kwargs):
    """Compute the LU factorization of a sparse matrix using KLU.

    This is a convenience function that creates a :class:`KLUFactor` object,
    computes the numeric factorization, and returns the resulting object.

    Parameters
    ----------
    A : (M, N) numpy.ndarray or sparse array
        The input matrix to factorize.
    control : :class:`KLUControl`, optional
        An optional :class:`KLUControl` object to set the factorization parameters.
        If not provided, default parameters are used.
    **kwargs
        Additional keyword arguments passed to the :class:`KLUControl` constructor.

    Returns
    -------
    :class:`KLUFactor`
        The LU factorization of the input matrix.

    Raises
    ------
    :exc:`KLUSingularMatrixWarning`
        If the matrix is exactly singular.

    See Also
    --------
    KLUFactor, klu_solve


    .. versionadded:: 0.5.0

    Examples
    --------
    *See*: Davis, Timothy A. (2006). Direct Methods for Sparse Linear Systems, p 74
    (Figure 5.1)

    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sksparse.klu import klu_factor
    >>> N = 8
    >>> rows = np.array(
    ...    [0, 1, 2, 3, 4, 5, 6, 3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6],
    ...    dtype=np.int32,
    ...)
    >>> cols = np.array(
    ...    [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7],
    ...    dtype=np.int32,
    ...)
    >>> vals = np.ones(len(rows), dtype=np.float64)
    >>> vals[:7] = np.arange(1, 8, dtype=np.float64)  # make diagonal entries non-unit
    >>> A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))
    >>> A
    <Compressed Sparse Column sparse array of dtype 'float64'
            with 23 stored elements and shape (8, 8)>
    >>> # Compute the LU factorization
    >>> f = klu_factor(A)
    >>> f
    <KLUFactor numeric factor of dtype 'float64' with 'int32' indices:
        L: (8, 8) with 16 stored elements
        U: (8, 8) with 17 stored elements>
    >>> L, U, p, q, r, F, _ = f  # unpack the factorization
    >>> LUF = (L @ U + F).toarray()
    >>> RPAQ = (r[:, np.newaxis] * A[p[:, np.newaxis], q]).toarray()
    >>> np.allclose(LUF, RPAQ)
    True
    >>> # Solve a linear system
    >>> expect_x = np.arange(N, dtype=np.float64)
    >>> b = A @ expect_x
    >>> x = f.solve(b)
    >>> np.allclose(x, expect_x)
    True
    """
    if control is None:
        control = KLUControl(**kwargs)
    return KLUFactor(A, control).factorize(A)


def klu_solve(A, b, *, KLUControl control=None, bint transpose=False, **kwargs):
    r"""Solve a linear system using KLU.

    This function solves a linear system for :math:`x` given the right-hand side
    :math:`b` as either a vector or a matrix with multiple right-hand sides.

    If ``transpose=False``, solve

    .. math::
        A x = b

    or, if ``transpose=True``, solve

    .. math::
        x A = b \Longleftrightarrow A^{\top} x^{\top} = b^{\top}.

    This is a convenience function that creates a :class:`KLUFactor` object, computes
    the numeric factorization, and solves the linear system with
    :meth:`KLUFactor.solve`.

    Parameters
    ----------
    A : (N, N) numpy.ndarray or sparse array
        The input matrix to factorize.
    b : (N,) or (N, K) numpy.ndarray
        The right-hand side vector or matrix.
    control : :class:`KLUControl`, optional
        An optional :class:`KLUControl` object to set the factorization parameters.
        If not provided, default parameters are used.
    transpose : bool, optional
        If True, solve :math:`x A = b`, otherwise, solve :math:`A x = b`.
    **kwargs
        Additional keyword arguments passed to the :class:`KLUControl` constructor.

    Returns
    -------
    x : (N,) or (N, K) numpy.ndarray or sparse array
        The solution vector or matrix of the same type and shape as the input
        right-hand side ``b``.

    See Also
    --------
    KLUFactor, klu_solve


    .. versionadded:: 0.5.0

    Examples
    --------
    *See*: Davis, Timothy A. (2006). Direct Methods for Sparse Linear Systems, p 74
    (Figure 5.1)

    >>> import numpy as np
    >>> from scipy import sparse
    >>> from sksparse.klu import klu_solve
    >>> N = 8
    >>> rows = np.array(
    ...    [0, 1, 2, 3, 4, 5, 6, 3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6],
    ...    dtype=np.int32,
    ...)
    >>> cols = np.array(
    ...    [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7],
    ...    dtype=np.int32,
    ...)
    >>> vals = np.ones(len(rows), dtype=np.float64)
    >>> vals[:7] = np.arange(1, 8, dtype=np.float64)  # make diagonal entries non-unit
    >>> A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))
    >>> A
    <Compressed Sparse Column sparse array of dtype 'float64'
            with 23 stored elements and shape (8, 8)>
    >>> # Solve a linear system
    >>> expect_x = np.arange(N, dtype=np.float64)
    >>> b = A @ expect_x
    >>> x = klu_solve(A, b)
    >>> np.allclose(x, expect_x)
    True
    """
    if control is None:
        control = KLUControl(**kwargs)

    # factorize() and solve() will each warn for a singular matrix,
    # so we catch the warnings from factorize() and re-raise only once.
    with warnings.catch_warnings(record=True) as ws:
        x = KLUFactor(A, control).factorize(A).solve(b, transpose=transpose)

    # Raise only the latest singular matrix warning from solve
    if ws:
        w = ws[-1]
        warnings.warn(w.message, w.category)

    return x
