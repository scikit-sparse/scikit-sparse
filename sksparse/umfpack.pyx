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
    cdef double _control[UMFPACK_CONTROL]
    cdef double _info[UMFPACK_INFO]
    cdef bint _use_int32
    cdef bint _is_real

    # TODO set up control array
    def __cinit__(self, object A):
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

        if self._is_real:
            if self._use_int32:
                umfpack_di_defaults(self._control)
                status = umfpack_di_symbolic(
                    M,
                    N,
                    <const int32_t*>indptr.data,
                    <const int32_t*>indices.data,
                    <const double*>real_data.data,
                    &self._symbolic,
                    self._control,
                    self._info
                )
            else:
                umfpack_dl_defaults(self._control)
                status = umfpack_dl_symbolic(
                    M,
                    N,
                    <const int64_t*>indptr.data,
                    <const int64_t*>indices.data,
                    <const double*>real_data.data,
                    &self._symbolic,
                    self._control,
                    self._info
                )
        else:
            if self._use_int32:
                umfpack_zi_defaults(self._control)
                status = umfpack_zi_symbolic(
                    M,
                    N,
                    <const int32_t*>indptr.data,
                    <const int32_t*>indices.data,
                    <const double*>real_data.data,
                    <const double*>imag_data.data,
                    &self._symbolic,
                    self._control,
                    self._info
                )
            else:
                umfpack_zl_defaults(self._control)
                status = umfpack_zl_symbolic(
                    M,
                    N,
                    <const int64_t*>indptr.data,
                    <const int64_t*>indices.data,
                    <const double*>real_data.data,
                    <const double*>imag_data.data,
                    &self._symbolic,
                    self._control,
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
    #         Public API
    # -------------------------------------------------------------------------
    # TODO make a python dataclass for control parameters
    def report_control(self, print_level=2):
        """Print a report of the control structure to stdout.

        Parameters
        ----------
        print_level : int, optional
            The verbosity level. Default value is 2.

            Accepted values are:

            * None: use current print level
            * <= 1: no printing
            * 2: print all of control parameters

        """
        pl = print_level if print_level is not None else self._control[UMFPACK_PRL]
        self._control[UMFPACK_PRL] = pl

        if self._is_real:
            if self._use_int32:
                umfpack_di_report_control(self._control)
            else:
                umfpack_dl_report_control(self._control)
        else:
            if self._use_int32:
                umfpack_zi_report_control(self._control)
            else:
                umfpack_zl_report_control(self._control)

        # restore old print level
        if print_level is None:
            self._control[UMFPACK_PRL] = pl

    def report_symbolic(self, print_level=4):
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
        pl = print_level if print_level is not None else self._control[UMFPACK_PRL]
        self._control[UMFPACK_PRL] = pl

        if self._is_real:
            if self._use_int32:
                umfpack_di_report_symbolic(self._symbolic, self._control)
            else:
                umfpack_dl_report_symbolic(self._symbolic, self._control)
        else:
            if self._use_int32:
                umfpack_zi_report_symbolic(self._symbolic, self._control)
            else:
                umfpack_zl_report_symbolic(self._symbolic, self._control)

        # restore old print level
        if print_level is None:
            self._control[UMFPACK_PRL] = pl

# =============================================================================
# =============================================================================
