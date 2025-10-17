# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: umfpack.pyx
#  Created: 2025-10-16 11:35
# =============================================================================

"""
===================================================================
Unsymmetric Multifrontal LU Decomposition (:mod:`sksparse.umfpack`)
===================================================================

.. currentmodule:: sksparse.umfpack

.. versionadded:: 0.5.0


An interface to the SuiteSparse `UMFPACK
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/UMFPACK>`_
package, which computes the LU factorization and solves systems of equations
for sparse, possibly non-symmetric, indefinite matrices.
"""

import numpy as np
cimport numpy as np

import warnings

from .utils import validate_csc_input


# -----------------------------------------------------------------------------
#         Warnings and Errors
# -----------------------------------------------------------------------------
class UMFPACKWarning(Warning):
    """A warning occurred in a UMFPACK routine."""
    pass


class UMFPACKSingularMatrixWarning(UMFPACKWarning):
    """A singular matrix was encountered in a UMFPACK routine."""
    pass


class UMFPACKDeterminantUnderflowWarning(UMFPACKWarning):
    """A determinant underflow was encountered in a UMFPACK routine."""
    pass


class UMFPACKDeterminantOverflowWarning(UMFPACKWarning):
    """A determinant overflow was encountered in a UMFPACK routine."""
    pass


class UMFPACKError(Exception):
    """An error occurred in a UMFPACK routine."""
    pass


class UMFPACKOutOfMemoryError(UMFPACKError):
    """UMFPACK ran out of memory."""
    pass


class UMFPACKInvalidNumericObjectError(UMFPACKError):
    """An invalid Numeric object was passed to a UMFPACK routine."""
    pass


class UMFPACKInvalidSymbolicObjectError(UMFPACKError):
    """An invalid Symbolic object was passed to a UMFPACK routine."""
    pass


class UMFPACKArgumentMissingError(UMFPACKError):
    """A required argument was missing in a UMFPACK routine."""
    pass


class UMFPACKNNonpositiveError(UMFPACKError):
    """A non-positive value for n was passed to a UMFPACK routine."""
    pass


class UMFPACKInvalidMatrixError(UMFPACKError):
    """An invalid matrix was passed to a UMFPACK routine."""
    pass


class UMFPACKDifferentPatternError(UMFPACKError):
    """A matrix with a different nonzero pattern was passed to a UMFPACK routine."""
    pass


class UMFPACKInvalidSystemError(UMFPACKError):
    """An invalid system type was passed to a UMFPACK routine."""
    pass


class UMFPACKInvalidPermutationError(UMFPACKError):
    """An invalid permutation was passed to a UMFPACK routine."""
    pass


class UMFPACKInternalError(UMFPACKError):
    """An internal error occurred in a UMFPACK routine."""
    pass


class UMFPACKFileIOError(UMFPACKError):
    """A file I/O error occurred in a UMFPACK routine."""
    pass


class UMFPACKOrderingFailedError(UMFPACKError):
    """The ordering algorithm failed in a UMFPACK routine."""
    pass


class UMFPACKInvalidBlobError(UMFPACKError):
    """An invalid blob was passed to a UMFPACK routine."""
    pass


cdef _handle_errors(int status) except * with gil:
    """Handle UMFPACK errors by raising Python exceptions or warnings.

    This function should be called with the return ``status`` after any UMFPACK
    C function that may fail.

    Parameters
    ----------
    status : int
        The UMFPACK exit status code.

    Returns
    -------
    None

    Raises
    ------
    :exc:`UMFPACKWarning`
        Raises a warning for non-critical issues.
    :exc:`UMFPACKError` or subclass
        Raises an appropriate Python exception based on the UMFPACK status code.
    """
    if status == UMFPACK_OK:
        return

    status_msg = f"(code {status:d})"

    # Known Errors
    cdef dict error_map = {
        UMFPACK_WARNING_singular_matrix: (
            UMFPACKSingularMatrixWarning,
            "Matrix is singular."
        ),
        UMFPACK_WARNING_determinant_underflow: (
            UMFPACKDeterminantUnderflowWarning,
            "Determinant underflow."
        ),
        UMFPACK_WARNING_determinant_overflow: (
            UMFPACKDeterminantOverflowWarning,
            "Determinant overflow."
        ),
        UMFPACK_ERROR_out_of_memory: (
            UMFPACKOutOfMemoryError,
            "Out of memory."
        ),
        UMFPACK_ERROR_invalid_Numeric_object: (
            UMFPACKInvalidNumericObjectError,
            "Invalid Numeric object."
        ),
        UMFPACK_ERROR_invalid_Symbolic_object: (
            UMFPACKInvalidSymbolicObjectError,
            "Invalid Symbolic object."
        ),
        UMFPACK_ERROR_argument_missing: (
            UMFPACKArgumentMissingError,
            "A required argument is missing."
        ),
        UMFPACK_ERROR_n_nonpositive: (
            UMFPACKNNonpositiveError,
            "Input N is non-positive."
        ),
        UMFPACK_ERROR_invalid_matrix: (
            UMFPACKInvalidMatrixError,
            "Invalid matrix."
        ),
        UMFPACK_ERROR_different_pattern: (
            UMFPACKDifferentPatternError,
            ("Matrix has different nonzero pattern than the matrix that was used"
             "for the symbolic analysis.")
        ),
        UMFPACK_ERROR_invalid_system: (
            UMFPACKInvalidSystemError,
            "Invalid system type argument, or the matrix is not square."
        ),
        UMFPACK_ERROR_invalid_permutation: (
            UMFPACKInvalidPermutationError,
            "Invalid permutation."
        ),
        UMFPACK_ERROR_internal_error: (
            UMFPACKInternalError,
            "An internal error occurred."
        ),
        UMFPACK_ERROR_file_IO: (
            UMFPACKFileIOError,
            "A file I/O error occurred."
        ),
        UMFPACK_ERROR_ordering_failed: (
            UMFPACKOrderingFailedError,
            "The ordering algorithm failed."
        ),
        UMFPACK_ERROR_invalid_blob: (
            UMFPACKInvalidBlobError,
            "Invalid blob."
        ),
    }

    # Fallback to generic error for unknown codes
    exc_class, msg = error_map.get(
        status,
        (UMFPACKError, "An unknown UMFPACK error occurred.")
    )
    full_msg = msg + " " + status_msg

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class)
    else:
        raise exc_class(full_msg)


# -----------------------------------------------------------------------------
#         Helpers
# -----------------------------------------------------------------------------
cdef bint _is_real_dtype(np.dtype dtype):
    if np.issubdtype(dtype, np.float64):
        return True
    elif np.issubdtype(dtype, np.complex128):
        return False
    else:
        raise TypeError(f"dtype must be float64 or complex128. Got {dtype=}")


# -----------------------------------------------------------------------------
#         Control Class
# -----------------------------------------------------------------------------
cdef dict _CONTROL_INDEX = {
    "print_level": UMFPACK_PRL,
    "dense_row": UMFPACK_DENSE_ROW,
    "dense_col": UMFPACK_DENSE_COL,
    "blas3_block_size": UMFPACK_BLOCK_SIZE,
    "strategy": UMFPACK_STRATEGY,
    "ordering_method": UMFPACK_ORDERING,
    "fixQ": UMFPACK_FIXQ,
    "amd_dense": UMFPACK_AMD_DENSE,
    "aggressive": UMFPACK_AGGRESSIVE,
    "singletons": UMFPACK_SINGLETONS,
    "pivot_tol": UMFPACK_PIVOT_TOLERANCE,
    "alloc_init": UMFPACK_ALLOC_INIT,
    "sym_pivot_tol": UMFPACK_SYM_PIVOT_TOLERANCE,
    "row_scale": UMFPACK_SCALE,
    "front_alloc_init": UMFPACK_FRONT_ALLOC_INIT,
    "droptol": UMFPACK_DROPTOL,
    "ir_steps": UMFPACK_IRSTEP,
    "compiles_with_blas": UMFPACK_COMPILED_WITH_BLAS,
    "sym_thresh": UMFPACK_STRATEGY_THRESH_SYM,
    "nnzdiag_thresh": UMFPACK_STRATEGY_THRESH_NNZDIAG,
}


cdef dict _CONTROL_STRATEGY_INDEX = {
    'auto': UMFPACK_STRATEGY_AUTO,
    'unsymmetric': UMFPACK_STRATEGY_UNSYMMETRIC,
    'obsolete': UMFPACK_STRATEGY_OBSOLETE,
    'symmetric': UMFPACK_STRATEGY_SYMMETRIC,
}


cdef dict _CONTROL_SCALE_INDEX = {
    None: UMFPACK_SCALE_NONE,
    'none': UMFPACK_SCALE_NONE,
    'sum': UMFPACK_SCALE_SUM,
    'max': UMFPACK_SCALE_MAX,
}


cdef dict _CONTROL_ORDERING_INDEX = {
    'cholmod': UMFPACK_ORDERING_CHOLMOD,
    'amd': UMFPACK_ORDERING_AMD,
    'given': UMFPACK_ORDERING_GIVEN,
    'none': UMFPACK_ORDERING_NONE,
    None: UMFPACK_ORDERING_NONE,
    'metis': UMFPACK_ORDERING_METIS,
    'best': UMFPACK_ORDERING_BEST,
    'user': UMFPACK_ORDERING_USER,
    'metis_guard': UMFPACK_ORDERING_METIS_GUARD,
}


cdef class UMFControl:
    """The class used to manage UMFPACK control parameters.

    Attributes
    ----------
    print_level : int
        The verbosity level. Values vary depending on the
        function called, but typically "0" means no printing, and higher values
        mean more verbose printing. Default value is 1.
    dense_row, dense_col : int
        A row or column is considered to be dense if it has more than ``max(16,
        dense_[row|col] * 16 * sqrt(n_[row|col])`` entries. Default 0.2.
    blas3_block_size : int
        The block size to use in Level-3 BLAS operations. Default value is 32.
    strategy : str
        The strategy to use in the factorization. Default value is ``'auto'``.
        Possible values are:
        * ``'auto'``: choose the strategy automatically
        * ``'unsymmetric'``: order the columns of :math:`A` with COLAMD
        * ``'symmetric'``: Order the matrix :math:`A + A^{\\top}` with AMD

    ordering_method : str
        The ordering method to use. Default value is ``'amd'``. Possible values
        are:
        * ``'cholmod'``: use AMD/COLAMD, then METIS
        * ``'amd'``: just use AMD or COLAMD
        * ``'given'``: use the user-provided ordering
        * ``'none'``: no ordering
        * ``'metis'``: use METIS on :math:`A + A^{\\top}` or :math:`A^{\\top} A`
        * ``'best'``: try AMD/COLAMD, METIS and NESDIS
        * ``'user'``: use the user-provided function to compute the ordering
        * ``'metis_guard'``: use METIS for symmetric strategy, try METIS for
          unsymmetric and fall back to COLAMD if :math:`A` has many dense rows.

    fixQ : int
        Default 0. Possible values:
        * -1: possibly modify :math:`Q` during numeric factorization.
        * 0: automatic. Modify :math:`Q` only if strategy is unsymmetric.
        * 1: do not modify :math:`Q` during numeric factorization.

    amd_dense : int
        Rows/columns in :math:`A + A^{\top}` with more than ``max(16,
        amd_dense * sqrt(n))`` entries (where ``n = n_row
        = n_col``) are ignored in the AMD pre-ordering. Default 10.
    aggressive : bool
        If True, use aggressive absorption in AMD. Default True.
    singletons : bool
        If True, remove singletons prior to factorization. Default True.
    pivot_tol : float
        The relative pivot tolerance for partial pivoting with row
        interchanges. The absolute value of the entry must be >= ``pivot_tol
        *`` largest absolute value in that column. ``pivot_tol=1.0`` gives true
        partial pivoting. If ``pivot_tol <= 0.0``, then any non-zero entry is
        acceptable as a pivot. Default value is 0.1.
    sym_pivot_tol : float
        The relative pivot tolerance for symmetric strategy. Default 0.001.
    row_scale : str or None
        The row scaling to use. Default value is ``'sum'``. Possible values are:
        * None or ``'none'``: no row scaling
        * ``'sum'``: divide each row by ``sum(abs(A[i,:]))``
        * ``'max'``: divide each row by ``max(abs(A[i,:]))``

    alloc_init : float
        Estimated space for the memory to allocate for numeric factorization.
        Default 0.7.
    front_alloc_init : float
        Estimated space for the memory to allocate for frontal matrices.
        Default 0.5.
    droptol : float
        Drop tolerance for small entries in :math:`L` and :math:`U`. Default
        value is 0.0 (no dropping).
    ir_steps : int
        Number of iterative refinement steps to perform. Default value is 2.
    compiles_with_blas : bool
        True if UMFPACK was compiled with BLAS support. Read-only.
    sym_thresh : float
        Threshold for choosing symmetric strategy. Default 0.3.
    nnzdiag_thresh : float
        Threshold for choosing unsymmetric strategy based on the number of
        diagonal entries. Default 0.9.
    """

    cdef double _arr[UMFPACK_CONTROL]

    def __cinit__(self, **kwargs):
        # NOTE the 4 functions ([dz][il]_defaults) all set the same default
        # values, so just pick one of them.
        umfpack_di_defaults(self._arr)

        # Update with user-provided values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except KeyError:
                raise KeyError(
                    f"Invalid control parameter: {key}. "
                    f"Expected one of {list(_CONTROL_INDEX.keys())}"
                )

    def __getattr__(self, name):
        try:
            return self._arr[_CONTROL_INDEX[name]]
        except KeyError:
            raise AttributeError(f"UMFControl object has no attribute '{name}'")

    def __setattr__(self, name, value):
        try:
            idx = _CONTROL_INDEX[name]
        except KeyError:
            raise AttributeError(f"UMFControl object has no attribute '{name}'")

        # Validate values
        if idx == UMFPACK_STRATEGY:
            try:
                value = _CONTROL_STRATEGY_INDEX[value]
            except KeyError:
                raise ValueError(
                    f"Invalid value for strategy: {value}. "
                    f"Expected one of {list(_CONTROL_STRATEGY_INDEX.keys())}"
                )

            if value == UMFPACK_STRATEGY_OBSOLETE:
                raise ValueError("'obsolete' value is, well, obsolete.")

        elif idx == UMFPACK_ORDERING:
            try:
                value = _CONTROL_ORDERING_INDEX[value]
            except KeyError:
                raise ValueError(
                    f"Invalid value for ordering_method: {value}. "
                    f"Expected one of {list(_CONTROL_ORDERING_INDEX.keys())}"
                )

        elif idx == UMFPACK_SCALE:
            try:
                value = _CONTROL_SCALE_INDEX[value]
            except KeyError:
                raise ValueError(
                    f"Invalid value for row_scale: {value}. "
                    f"Expected one of {list(_CONTROL_SCALE_INDEX.keys())}"
                )

        elif idx == UMFPACK_COMPILED_WITH_BLAS:
            raise AttributeError(f"'{name}' is read-only")

        # Set the value
        self._arr[idx] = value

    def __repr__(self):
        params = ",\n    ".join(
            f"{key}={self._arr[idx]}" for key, idx in _CONTROL_INDEX.items()
        )
        return f"{self.__class__.__name__}(\n    {params}\n)"

    def __str__(self):
        return self.__repr__()

    def report(self):
        """Print a report of the control structure to stdout.

        This method provides more internal details from UMFPACK itself than the
        string representation.

        .. note::

            This method temporarily sets the print level to 2 (print all
            information) and restores the previous value afterwards, so the
            report will *always* show "print level: 2". Use
            ``UMFControl.print_level`` or ``print(UMFControl)`` to see the
            actual print level.
        """
        cdef int old_pl = self.print_level
        self.print_level = 2  # print all info

        # NOTE the 4 functions ([dz][il]_report_control) all print the same.
        umfpack_di_report_control(self._arr)

        # restore old print level
        self.print_level = old_pl


# -----------------------------------------------------------------------------
#         UMFPACK Class Interface
# -----------------------------------------------------------------------------
cdef class UMFFactor:
    """The main object used for creating and using an LU factorization.

    The constructor computes the symbolic analysis of a sparse matrix :math:`A`
    and determines a fill-reducing ordering such that:

    .. math::
        L U = P R A Q.

    The numeric factorization is not computed until :meth:`.numeric` is called.
    """

    cdef void *_symbolic
    cdef UMFControl _control
    cdef double _info[UMFPACK_INFO]
    cdef bint _use_int32
    cdef bint _is_real

    def __cinit__(self, object A, object control=None):
        A, use_int32, _ = validate_csc_input(A)

        self._use_int32 = use_int32

        # Compute the symbolic analysis
        cdef int M = A.shape[0]
        cdef int N = A.shape[1]

        cdef np.ndarray indptr = A.indptr
        cdef np.ndarray indices = A.indices
        cdef np.ndarray real_data = A.data.real
        cdef np.ndarray imag_data = A.data.imag

        self._is_real = _is_real_dtype(A.data.dtype)

        cdef int status

        # Set the control array
        self._control = UMFControl() if control is None else control

        # Compute the symbolic factorization
        if self._is_real:
            if self._use_int32:
                status = umfpack_di_symbolic(
                    M,
                    N,
                    <int32_t*>indptr.data,
                    <int32_t*>indices.data,
                    <double*>real_data.data,
                    &self._symbolic,
                    self._control._arr,
                    self._info
                )
            else:
                status = umfpack_dl_symbolic(
                    M,
                    N,
                    <int64_t*>indptr.data,
                    <int64_t*>indices.data,
                    <double*>real_data.data,
                    &self._symbolic,
                    self._control._arr,
                    self._info
                )
        else:
            if self._use_int32:
                status = umfpack_zi_symbolic(
                    M,
                    N,
                    <int32_t*>indptr.data,
                    <int32_t*>indices.data,
                    <double*>real_data.data,
                    <double*>imag_data.data,
                    &self._symbolic,
                    self._control._arr,
                    self._info
                )
            else:
                status = umfpack_zl_symbolic(
                    M,
                    N,
                    <int64_t*>indptr.data,
                    <int64_t*>indices.data,
                    <double*>real_data.data,
                    <double*>imag_data.data,
                    &self._symbolic,
                    self._control._arr,
                    self._info
                )

        _handle_errors(status)

    def __dealloc__(self):
        if self._symbolic is not NULL:
            if self._is_real:
                if self._use_int32:
                    umfpack_di_free_symbolic(&self._symbolic)
                else:
                    umfpack_dl_free_symbolic(&self._symbolic)
            else:
                if self._use_int32:
                    umfpack_zi_free_symbolic(&self._symbolic)
                else:
                    umfpack_zl_free_symbolic(&self._symbolic)

    # TODO __repr__ and __str__

    # -------------------------------------------------------------------------
    #         Properties
    # -------------------------------------------------------------------------
    @property
    def control(self):
        """The control parameters used for the factorization.

        See :class:`UMFControl` for details.
        """
        return self._control

    @control.setter
    def control(self, UMFControl control):
        self._control = control

    # -------------------------------------------------------------------------
    #         Public Methods
    # -------------------------------------------------------------------------
    def report_control(self):
        self._control.report()

    def report_symbolic(self, object print_level=4):
        """Print a report of the symbolic factorization to stdout.

        Parameters
        ----------
        print_level : int, optional
            The verbosity level. Default value is 4.

            Accepted values are:

            * None: use current print level
            * <= 2: no printing
            * 3: fully check input, and print a short summary of its status
            * 4: as 3, but print first few entries of the input
            * 5: as 3, but print all of the input

        """
        cdef int pl = print_level if print_level is not None else self._control.print_level
        cdef int old_pl = self._control.print_level
        self._control.print_level = pl

        if self._is_real:
            if self._use_int32:
                umfpack_di_report_symbolic(self._symbolic, self._control._arr)
            else:
                umfpack_dl_report_symbolic(self._symbolic, self._control._arr)
        else:
            if self._use_int32:
                umfpack_zi_report_symbolic(self._symbolic, self._control._arr)
            else:
                umfpack_zl_report_symbolic(self._symbolic, self._control._arr)

        # restore old print level
        self._control.print_level = old_pl


# Set docstrings
UMFFactor.report_control.__doc__ = UMFControl.report.__doc__

# =============================================================================
# =============================================================================
