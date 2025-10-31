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

import numpy as np
from scipy.sparse import issparse, csr_array, csc_array
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
        warnings.warn(full_msg, exc_class)
    else:
        raise exc_class(full_msg)


# -------------------------------------------------------------------------------------
#         KLU Class Interface
# -------------------------------------------------------------------------------------
# TODO use 2 separate objects for int32 and int64 versions?
cdef class KLUFactor:
    """Class to compute and store the KLU factorization of a sparse matrix.

    The constructor computes the symbolic analysis of a sparse matrix :math:`A`
    and determines a fill-reducing ordering such that:

    .. math::
        L U + F = R P A Q.

    The numeric factorization is not computed until :meth:`.factorize` is called.

    Attributes
    ----------
    N : int
        The number of rows/columns in the matrix.
    L : scipy.sparse.csr_array
        The :math:`L` factor as a sparse CSR matrix.
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
        readonly Py_ssize_t N
        bint _use_int32
        klu_common _common
        klu_common* _cm
        klu_l_common _l_common
        klu_l_common* _l_cm
        klu_symbolic* _symbolic
        klu_l_symbolic* _l_symbolic

    # TODO pass options either via kwargs or struct
    def __init__(self, object A):
        """Compute the KLU factorization of a sparse matrix.

        Parameters
        ----------
        A : (N, N) numpy.ndarray or sparse array
            The input matrix. Any object that can be converted to
            a :class:`~scipy.sparse.csc_array` is accepted.
        """
        A, _, _ = validate_csc_input(A, require_square=True)

        self.N = A.shape[0]

        self._init_symbolic(self.N, A.indptr, A.indices)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _init_symbolic(
        self,
        Py_ssize_t N,
        index_t[::1] indptr,
        index_t[::1] indices
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
        cdef int status

        self._use_int32 = index_t is int32_t

        if self._use_int32:
            self._cm = &self._common
            assert klu_defaults(self._cm)

            self._symbolic = klu_analyze(
                N,
                <int32_t*>&indptr[0],
                <int32_t*>&indices[0],
                self._cm
            )
            _handle_errors(self._cm.status)
        else:
            self._l_cm = &self._l_common
            assert klu_l_defaults(self._l_cm)

            self._l_symbolic = klu_l_analyze(
                N,
                <int64_t*>&indptr[0],
                <int64_t*>&indices[0],
                self._l_cm
            )
            _handle_errors(self._l_cm.status)

    def __dealloc__(self):
        """Deallocate KLU objects."""
        if self._use_int32:
            if self._symbolic is not NULL:
                klu_free_symbolic(&self._symbolic, self._cm)
        else:
            if self._l_symbolic is not NULL:
                klu_l_free_symbolic(&self._l_symbolic, self._l_cm)
