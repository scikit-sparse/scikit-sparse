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


.. klupack-exceptions:

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
cimport numpy as cnp

from copy import deepcopy
import numpy as np
from scipy.sparse import issparse, csc_array
import warnings

from .utils import validate_csc_input


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
    rcond : double
        Estimate of the reciprocal of the condition number.
    singular_col : int
        Index of the first singular column, if any.
    rgrowth : double
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
    tol : double
        The pivot tolerance used.
    memory : int
        Memory usage in bytes.
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
    memory : int | None = None

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
            self.memory = <int>cm.memusage

        if symbolic is not NULL:
            self.nblocks = symbolic.nblocks

        if numeric is not NULL:
            self.lnz = <int>numeric.lnz
            self.unz = <int>numeric.unz
            self.nzoff = <int>numeric.nzoff

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
            self.memory = <int>cm.memusage

        if symbolic is not NULL:
            self.nblocks = symbolic.nblocks

        if numeric is not NULL:
            self.lnz = <int>numeric.lnz
            self.unz = <int>numeric.unz
            self.nzoff = <int>numeric.nzoff

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
        int _btf
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

    # TODO add validation in setters (like 0 <= tol <= 1)
    @property
    def tol(self):
        return None if self._tol == self._FLOAT_NONE else self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value if value is not None else self._FLOAT_NONE

    @property
    def memgrow(self):
        return None if self._memgrow == self._FLOAT_NONE else self._memgrow

    @memgrow.setter
    def memgrow(self, value):
        self._memgrow = value if value is not None else self._FLOAT_NONE

    @property
    def initmem_amd(self):
        return None if self._initmem_amd == self._FLOAT_NONE else self._initmem_amd

    @initmem_amd.setter
    def initmem_amd(self, value):
        self._initmem_amd = value if value is not None else self._FLOAT_NONE

    @property
    def initmem(self):
        return None if self._initmem == self._FLOAT_NONE else self._initmem

    @initmem.setter
    def initmem(self, value):
        self._initmem = value if value is not None else self._FLOAT_NONE

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


# -------------------------------------------------------------------------------------
#         Copy Functions
# -------------------------------------------------------------------------------------
cdef inline void* _malloc_copy(
    const void* src,
    size_t n,
    size_t size,
    const common_t* cm
) except NULL:
    """Allocate memory and copy data from src to the new memory."""
    if cm is NULL:
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


cdef int _copy_symbolic(
    klu_symbolic* dest,
    const klu_symbolic* src,
    const klu_common* cm
) except -1:
    """Deep copy a KLU symbolic struct."""
    if src is NULL or dest is NULL:
        raise ValueError("Source and destination pointers must not be NULL.")

    # Copy the top-level data and pointers
    memcpy(dest, src, sizeof(klu_symbolic))

    cdef size_t n = src.n

    # Deep copy internal arrays
    dest.Lnz = <double*>_malloc_copy(src.Lnz, n, sizeof(double), cm)
    dest.P = <int32_t*>_malloc_copy(src.P, n, sizeof(int32_t), cm)
    dest.Q = <int32_t*>_malloc_copy(src.Q, n, sizeof(int32_t), cm)
    dest.R = <int32_t*>_malloc_copy(src.R, n + 1, sizeof(int32_t), cm)

    return 0


cdef int _copy_l_symbolic(
    klu_l_symbolic* dest,
    const klu_l_symbolic* src,
    const klu_l_common* cm,
) except -1:
    """Deep copy a KLU symbolic struct."""
    if src is NULL or dest is NULL:
        raise ValueError("Source and destination pointers must not be NULL.")

    # Copy the top-level data and pointers
    memcpy(dest, src, sizeof(klu_l_symbolic))

    cdef size_t n = src.n

    # Deep copy internal arrays
    dest.Lnz = <double*>_malloc_copy(src.Lnz, n, sizeof(double), cm)
    dest.P = <int64_t*>_malloc_copy(src.P, n, sizeof(int64_t), cm)
    dest.Q = <int64_t*>_malloc_copy(src.Q, n, sizeof(int64_t), cm)
    dest.R = <int64_t*>_malloc_copy(src.R, n + 1, sizeof(int64_t), cm)

    return 0


cdef int _copy_numeric(
    klu_numeric* dest,
    const klu_numeric* src,
    const klu_common* cm,
    value_t _dummy=0
) except -1:
    """Deep copy a KLU numeric struct."""
    if src is NULL or dest is NULL:
        raise ValueError("Source and destination pointers must not be NULL.")

    # Copy the top-level data and pointers
    memcpy(dest, src, sizeof(klu_numeric))

    cdef size_t n = src.n
    cdef size_t nblocks = src.nblocks

    # Deep copy internal arrays
    dest.Pnum = <int32_t*>_malloc_copy(src.Pnum, n, sizeof(int32_t), cm)
    dest.Pinv = <int32_t*>_malloc_copy(src.Pinv, n, sizeof(int32_t), cm)
    dest.Lip = <int32_t*>_malloc_copy(src.Lip, n, sizeof(int32_t), cm)
    dest.Uip = <int32_t*>_malloc_copy(src.Uip, n, sizeof(int32_t), cm)
    dest.Llen = <int32_t*>_malloc_copy(src.Llen, n, sizeof(int32_t), cm)
    dest.Ulen = <int32_t*>_malloc_copy(src.Ulen, n, sizeof(int32_t), cm)
    dest.Udiag = <value_t*>_malloc_copy(src.Udiag, n, sizeof(value_t), cm)
    dest.LUsize = <size_t*>_malloc_copy(src.LUsize, nblocks, sizeof(size_t), cm)

    cdef size_t k

    if src.LUbx is not NULL and nblocks > 0:
        dest.LUbx = <void**>klu_malloc(nblocks, sizeof(value_t*), cm)
        if dest.LUbx is not NULL:
            for k in range(nblocks):
                dest.LUbx[k] = <value_t*>_malloc_copy(
                    src.LUbx[k], src.LUsize[k], sizeof(value_t), cm
                )

    cdef size_t np1 = n + 1
    cdef size_t nzoffp1 = <size_t>src.nzoff + 1

    dest.Offp = <int32_t*>_malloc_copy(src.Offp, np1, sizeof(int32_t), cm)
    dest.Offi = <int32_t*>_malloc_copy(src.Offi, nzoffp1, sizeof(int32_t), cm)
    dest.Offx = <value_t*>_malloc_copy(src.Offx, nzoffp1, sizeof(value_t), cm)

    dest.Rs = <double*>_malloc_copy(src.Rs, n, sizeof(double), cm)

    # Workspace encompasses Xwork and Iwork, so just copy Work
    dest.Work = _malloc_copy(src.Work, src.worksize, sizeof(value_t), cm)
    dest.Xwork = dest.Work
    dest.Iwork = <int32_t*>(<value_t*>dest.Xwork + n)

    return 0


cdef int _copy_l_numeric(
    klu_l_numeric* dest,
    const klu_l_numeric* src,
    const klu_l_common* cm,
    value_t _dummy=0
) except -1:
    """Deep copy a KLU numeric struct."""
    if src is NULL or dest is NULL:
        raise ValueError("Source and destination pointers must not be NULL.")

    # Copy the top-level data and pointers
    memcpy(dest, src, sizeof(klu_l_numeric))

    cdef size_t n = src.n
    cdef size_t nblocks = src.nblocks

    # Deep copy internal arrays
    dest.Pnum = <int64_t*>_malloc_copy(src.Pnum, n, sizeof(int64_t), cm)
    dest.Pinv = <int64_t*>_malloc_copy(src.Pinv, n, sizeof(int64_t), cm)
    dest.Lip = <int64_t*>_malloc_copy(src.Lip, n, sizeof(int64_t), cm)
    dest.Uip = <int64_t*>_malloc_copy(src.Uip, n, sizeof(int64_t), cm)
    dest.Llen = <int64_t*>_malloc_copy(src.Llen, n, sizeof(int64_t), cm)
    dest.Ulen = <int64_t*>_malloc_copy(src.Ulen, n, sizeof(int64_t), cm)
    dest.Udiag = <value_t*>_malloc_copy(src.Udiag, n, sizeof(value_t), cm)
    dest.LUsize = <size_t*>_malloc_copy(src.LUsize, nblocks, sizeof(size_t), cm)

    cdef size_t k

    if src.LUbx is not NULL and nblocks > 0:
        dest.LUbx = <void**>klu_l_malloc(nblocks, sizeof(value_t*), cm)
        if dest.LUbx is not NULL:
            for k in range(nblocks):
                dest.LUbx[k] = <value_t*>_malloc_copy(
                    src.LUbx[k], src.LUsize[k], sizeof(value_t), cm
                )

    cdef size_t np1 = n + 1
    cdef size_t nzoffp1 = <size_t>src.nzoff + 1

    dest.Offp = <int64_t*>_malloc_copy(src.Offp, np1, sizeof(int64_t), cm)
    dest.Offi = <int64_t*>_malloc_copy(src.Offi, nzoffp1, sizeof(int64_t), cm)
    dest.Offx = <value_t*>_malloc_copy(src.Offx, nzoffp1, sizeof(value_t), cm)

    dest.Rs = <double*>_malloc_copy(src.Rs, n, sizeof(double), cm)

    # Workspace encompasses Xwork and Iwork, so just copy Work
    dest.Work = _malloc_copy(src.Work, src.worksize, sizeof(value_t), cm)
    dest.Xwork = dest.Work
    dest.Iwork = <int64_t*>(<value_t*>dest.Xwork + n)

    return 0


# -------------------------------------------------------------------------------------
#         KLU Class Interface
# -------------------------------------------------------------------------------------
cdef class KLUFactor:
    """Class to compute and store the KLU factorization of a sparse matrix.

    The constructor computes the symbolic analysis of a sparse matrix :math:`A`
    and determines a fill-reducing ordering such that:

    .. math::
        L U + F = R^{-1} P A Q.

    The numeric factorization is not computed until :meth:`.factorize` is called.

    .. note::

        From the ``klu.m`` documentation:

            Note that the use of the scale factor R differs between KLU and UMFPACK
            (and the LU function, which is based on UMFPACK).  In LU, the factorization
            is ``L*U = P*(R1\A)*Q;`` in KLU it is ``L*U+F = R2\(P*A*Q)``.  ``R1`` and
            ``R2`` are related via ``R2 = P*R1*P'``, or equivalently ``R2 = R1(p,p)``.

    Attributes
    ----------
    N : int
        The number of rows/columns in the matrix.
    L : scipy.sparse.csc_array
        The :math:`L` factor as a sparse CSC matrix.
    U : scipy.sparse.csc_array
        The :math:`U` factor as a sparse CSC matrix.
    perm_r, perm_c : numpy.ndarray
        The row and column permutation arrays, :math:`P` and :math:`Q`.

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
            self._cm.tol         = self._cm.tol if control._tol is _FNONE else control._tol
            self._cm.memgrow     = self._cm.memgrow if control._memgrow is _FNONE else control._memgrow
            self._cm.initmem_amd = self._cm.initmem_amd if control._initmem_amd is _FNONE else control._initmem_amd
            self._cm.initmem     = self._cm.initmem if control._initmem is _FNONE else control._initmem
            self._cm.maxwork     = self._cm.maxwork if control._maxwork is _FNONE else control._maxwork
            self._cm.btf         = self._cm.btf if control._btf is _INONE else control._btf
            self._cm.ordering    = self._cm.ordering if control._ordering is _INONE else control._ordering
            self._cm.scale       = self._cm.scale if control._scale is _INONE else control._scale
        else:
            self._l_cm.tol         = self._l_cm.tol if control._tol is _FNONE else control._tol
            self._l_cm.memgrow     = self._l_cm.memgrow if control._memgrow is _FNONE else control._memgrow
            self._l_cm.initmem_amd = self._l_cm.initmem_amd if control._initmem_amd is _FNONE else control._initmem_amd
            self._l_cm.initmem     = self._l_cm.initmem if control._initmem is _FNONE else control._initmem
            self._l_cm.maxwork     = self._l_cm.maxwork if control._maxwork is _FNONE else control._maxwork
            self._l_cm.btf         = self._l_cm.btf if control._btf is _INONE else control._btf
            self._l_cm.ordering    = self._l_cm.ordering if control._ordering is _INONE else control._ordering
            self._l_cm.scale       = self._l_cm.scale if control._scale is _INONE else control._scale

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
        """Get information about the factorization and solve process."""
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
            klu._cm = &klu._common
            memcpy(klu._cm, self._cm, sizeof(klu_common))

            klu._symbolic = <klu_symbolic*>klu_malloc(1, sizeof(klu_symbolic), klu._cm)
            _copy_symbolic(klu._symbolic, self._symbolic, klu._cm)

            klu._numeric = <klu_numeric*>klu_malloc(1, sizeof(klu_numeric), klu._cm)
            if self._is_real:
                _copy_numeric[double](klu._numeric, self._numeric, klu._cm)
            else:
                _copy_numeric[cython.doublecomplex](klu._numeric, self._numeric, klu._cm)
        else:
            klu._l_cm = &klu._l_common
            memcpy(klu._l_cm, self._l_cm, sizeof(klu_l_common))

            klu._l_symbolic = <klu_l_symbolic*>klu_l_malloc(1, sizeof(klu_l_symbolic), klu._l_cm)
            _copy_l_symbolic(klu._l_symbolic, self._l_symbolic, klu._l_cm)

            klu._l_numeric = <klu_l_numeric*>klu_l_malloc(1, sizeof(klu_l_numeric), klu._l_cm)
            if self._is_real:
                _copy_l_numeric[double](klu._l_numeric, self._l_numeric, klu._l_cm)
            else:
                _copy_l_numeric[cython.doublecomplex](klu._l_numeric, self._l_numeric, klu._l_cm)

        # Cached factor objects
        klu._L = None if self._L is None else self._L.copy()
        klu._U = None if self._U is None else self._U.copy()
        klu._F = None if self._F is None else self._F.copy()
        klu._P = None if self._P is None else self._P.copy()
        klu._Q = None if self._Q is None else self._Q.copy()
        klu._Rs = None if self._Rs is None else self._Rs.copy()
        klu._R = None if self._R is None else self._R.copy()

        return klu

    def factorize(self, object A):
        """Compute the numeric factorization of the matrix.

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

        This method solves the linear system for :math:`x` given the right-hand side
        :math:`b` as either a vector or a matrix with multiple right-hand sides,

        .. math::
            A x = b

        if ``transpose=False``, or

        .. math::
            x A = b \Longleftrightarrow A^H x^H = b^H

        if ``transpose=True``, using the LU factorization of :math:`A` previously
        computed by :meth:`.factorize`.

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

        if b.dtype != self.dtype:
            raise ValueError(
                f"LHS and RHS dtypes do not match. {self.dtype=} and {b.dtype=}"
            )

        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        cdef bint return_1D = b.ndim == 1
        cdef size_t N = b.shape[0] if b.ndim == 1 else (b.shape[1] if transpose else b.shape[0])
        cdef size_t K = 1 if b.ndim == 1 else (b.shape[0] if transpose else b.shape[1])

        if N != self._N:
            raise ValueError(
                "Right-hand side b must have compatible shape with A. "
                f"Got {b.shape=}, but A.shape={self.shape} ({transpose=})."
            )

        # Check the condition number
        self._check_rcond()

        cdef bint return_sparse = issparse(b)

        if return_sparse:
            b = b.toarray()
        else:
            b = np.asarray(b)

        # Ensure columns are contiguous for multiple RHS
        b = np.asfortranarray(b)

        # TODO allow overwrite_b=True
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
            x.indptr = x.indptr.astype(self.itype)
            x.indices = x.indices.astype(self.itype)

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
            The right-hand side matrix on input, in column-oriented form, solution on output.
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
            The right-hand side matrix on input, in column-oriented form, solution on output.
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
        cdef int status

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
    """
    if control is None:
        control = KLUControl(**kwargs)
    return KLUFactor(A, control).factorize(A)


def klu_solve(A, b, *, KLUControl control=None, bint transpose=False, **kwargs):
    """Solve a linear system using KLU.

    This is a convenience function that creates a :class:`KLUFactor` object,
    computes the numeric factorization, and solves the linear system.

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
