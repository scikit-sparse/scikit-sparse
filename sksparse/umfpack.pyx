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


Function Interface
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    umf_solve - Solve a linear system using the UMFPACK factorization.


Object Interface
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    umf_factor - Compute the LU factorization of a sparse matrix.
    UMFFactor - An object-oriented interface to UMFPACK.


.. umfpack-exceptions:

Warnings and Exceptions
-----------------------

.. autosummary::
    :toctree: generated/

    UMFPACKWarning
    UMFPACKSingularMatrixWarning
    UMFPACKDeterminantUnderflowWarning
    UMFPACKDeterminantOverflowWarning

    UMFPACKError
    UMFPACKOutOfMemoryError
    UMFPACKInvalidNumericObjectError
    UMFPACKInvalidSymbolicObjectError
    UMFPACKArgumentMissingError
    UMFPACKNonpositiveError
    UMFPACKInvalidMatrixError
    UMFPACKDifferentPatternError
    UMFPACKInvalidSystemError
    UMFPACKInvalidPermutationError
    UMFPACKInternalError
    UMFPACKFileIOError
    UMFPACKOrderingFailedError
    UMFPACKInvalidBlobError


References
----------
* `SuiteSparse homepage <https://people.engr.tamu.edu/davis/suitesparse.html>`_
* `SuiteSparse UMFPACK <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/UMFPACK>`_
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


class UMFPACKNonpositiveError(UMFPACKError):
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


# Known Errors
cdef dict _ERROR_INDEX = {
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
        UMFPACKNonpositiveError,
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


cdef int _handle_errors(int status) except -1 with gil:
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
        return 0

    # Fallback to generic error for unknown codes
    exc_class, msg = _ERROR_INDEX.get(
        status,
        (UMFPACKError, "An unknown UMFPACK error occurred.")
    )
    full_msg = f"{msg} (code {status:d})"

    if issubclass(exc_class, Warning):
        warnings.warn(full_msg, exc_class)
    else:
        raise exc_class(full_msg)


# -----------------------------------------------------------------------------
#         Helpers
# -----------------------------------------------------------------------------
cdef bint _is_real_dtype(cnp.dtype dtype):
    if np.issubdtype(dtype, np.float64):
        return True
    elif np.issubdtype(dtype, np.complex128):
        return False
    else:
        raise TypeError(f"dtype must be float64 or complex128. Got {dtype=}")


# -----------------------------------------------------------------------------
#         Parameter Mappings
# -----------------------------------------------------------------------------
cdef dict _INFO_INDEX = {
    "status": UMFPACK_STATUS,
    "n_row": UMFPACK_NROW,
    "n_col": UMFPACK_NCOL,
    "nz": UMFPACK_NZ,
    "size_of_unit": UMFPACK_SIZE_OF_UNIT,
    "size_of_int": UMFPACK_SIZE_OF_INT,
    "size_of_long": UMFPACK_SIZE_OF_LONG,
    "size_of_pointer": UMFPACK_SIZE_OF_POINTER,
    "size_of_entry": UMFPACK_SIZE_OF_ENTRY,
    "ndense_row": UMFPACK_NDENSE_ROW,
    "nempty_row": UMFPACK_NEMPTY_ROW,
    "ndense_col": UMFPACK_NDENSE_COL,
    "nempty_col": UMFPACK_NEMPTY_COL,
    "symbolic_defrag": UMFPACK_SYMBOLIC_DEFRAG,
    "symbolic_peak_memory": UMFPACK_SYMBOLIC_PEAK_MEMORY,
    "symbolic_size": UMFPACK_SYMBOLIC_SIZE,
    "symbolic_time": UMFPACK_SYMBOLIC_TIME,
    "symbolic_walltime": UMFPACK_SYMBOLIC_WALLTIME,
    "strategy_used": UMFPACK_STRATEGY_USED,
    "ordering_used": UMFPACK_ORDERING_USED,
    "qfixed": UMFPACK_QFIXED,
    "diag_preferred": UMFPACK_DIAG_PREFERRED,
    "pattern_symmetry": UMFPACK_PATTERN_SYMMETRY,
    "nz_a_plus_at": UMFPACK_NZ_A_PLUS_AT,
    "nzdiag": UMFPACK_NZDIAG,
    "symmetric_lunz": UMFPACK_SYMMETRIC_LUNZ,
    "symmetric_flops": UMFPACK_SYMMETRIC_FLOPS,
    "symmetric_ndense": UMFPACK_SYMMETRIC_NDENSE,
    "symmetric_dmax": UMFPACK_SYMMETRIC_DMAX,
    "col_singletons": UMFPACK_COL_SINGLETONS,
    "row_singletons": UMFPACK_ROW_SINGLETONS,
    "n2": UMFPACK_N2,
    "s_symmetric": UMFPACK_S_SYMMETRIC,
    "numeric_size_estimate": UMFPACK_NUMERIC_SIZE_ESTIMATE,
    "peak_memory_estimate": UMFPACK_PEAK_MEMORY_ESTIMATE,
    "flops_estimate": UMFPACK_FLOPS_ESTIMATE,
    "lnz_estimate": UMFPACK_LNZ_ESTIMATE,
    "unz_estimate": UMFPACK_UNZ_ESTIMATE,
    "variable_init_estimate": UMFPACK_VARIABLE_INIT_ESTIMATE,
    "variable_peak_estimate": UMFPACK_VARIABLE_PEAK_ESTIMATE,
    "variable_final_estimate": UMFPACK_VARIABLE_FINAL_ESTIMATE,
    "max_front_size_estimate": UMFPACK_MAX_FRONT_SIZE_ESTIMATE,
    "max_front_nrows_estimate": UMFPACK_MAX_FRONT_NROWS_ESTIMATE,
    "max_front_ncols_estimate": UMFPACK_MAX_FRONT_NCOLS_ESTIMATE,
    "numeric_size": UMFPACK_NUMERIC_SIZE,
    "peak_memory": UMFPACK_PEAK_MEMORY,
    "flops": UMFPACK_FLOPS,
    "lnz": UMFPACK_LNZ,
    "unz": UMFPACK_UNZ,
    "variable_init": UMFPACK_VARIABLE_INIT,
    "variable_peak": UMFPACK_VARIABLE_PEAK,
    "variable_final": UMFPACK_VARIABLE_FINAL,
    "max_front_size": UMFPACK_MAX_FRONT_SIZE,
    "max_front_nrows": UMFPACK_MAX_FRONT_NROWS,
    "max_front_ncols": UMFPACK_MAX_FRONT_NCOLS,
    "numeric_defrag": UMFPACK_NUMERIC_DEFRAG,
    "numeric_realloc": UMFPACK_NUMERIC_REALLOC,
    "numeric_costly_realloc": UMFPACK_NUMERIC_COSTLY_REALLOC,
    "compressed_pattern": UMFPACK_COMPRESSED_PATTERN,
    "lu_entries": UMFPACK_LU_ENTRIES,
    "numeric_time": UMFPACK_NUMERIC_TIME,
    "nz_udiag": UMFPACK_UDIAG_NZ,
    "rcond": UMFPACK_RCOND,
    "was_scaled": UMFPACK_WAS_SCALED,
    "rsmin": UMFPACK_RSMIN,
    "rsmax": UMFPACK_RSMAX,
    "umin": UMFPACK_UMIN,
    "umax": UMFPACK_UMAX,
    "alloc_init_used": UMFPACK_ALLOC_INIT_USED,
    "forced_updates": UMFPACK_FORCED_UPDATES,
    "numeric_walltime": UMFPACK_NUMERIC_WALLTIME,
    "noff_diag": UMFPACK_NOFF_DIAG,
    "all_lnz": UMFPACK_ALL_LNZ,
    "all_unz": UMFPACK_ALL_UNZ,
    "nzdropped": UMFPACK_NZDROPPED,
    "ir_taken": UMFPACK_IR_TAKEN,
    "ir_attempted": UMFPACK_IR_ATTEMPTED,
    "omega1": UMFPACK_OMEGA1,
    "omega2": UMFPACK_OMEGA2,
    "solve_flops": UMFPACK_SOLVE_FLOPS,
    "solve_time": UMFPACK_SOLVE_TIME,
    "solve_walltime": UMFPACK_SOLVE_WALLTIME,
}


cdef list _INFO_INT_NAMES = [
    'status',
    'n_row',
    'n_col',
    'nz',
    'size_of_unit',
    'size_of_int',
    'size_of_long',
    'size_of_pointer',
    'size_of_entry',
    'ndense_row',
    'nempty_row',
    'ndense_col',
    'nempty_col',
    'symbolic_defrag',
    'symbolic_peak_memory',
    'symbolic_size',
    'strategy_used',
    'ordering_used',
    'qfixed',
    'diag_preferred',
    'nz_a_plus_at',
    'nzdiag',
    'symmetric_lunz',
    'symmetric_flops',
    'symmetric_ndense',
    'symmetric_dmax',
    'col_singletons',
    'row_singletons',
    'n2',
    's_symmetric',
    'numeric_size_estimate',
    'peak_memory_estimate',
    'flops_estimate',
    'lnz_estimate',
    'unz_estimate',
    'variable_init_estimate',
    'variable_peak_estimate',
    'variable_final_estimate',
    'max_front_size_estimate',
    'max_front_nrows_estimate',
    'max_front_ncols_estimate',
    'numeric_size',
    'peak_memory',
    'flops',
    'lnz',
    'unz',
    'variable_init',
    'variable_peak',
    'variable_final',
    'max_front_size',
    'max_front_nrows',
    'max_front_ncols',
    'numeric_defrag',
    'numeric_realloc',
    'numeric_costly_realloc',
    'compressed_pattern',
    'lu_entries',
    'nz_udiag',
    'forced_updates',
    'noff_diag',
    'all_lnz',
    'all_unz',
    'nzdropped',
    'ir_taken',
    'ir_attempted',
    'solve_flops',
]

cdef list _INFO_BOOL_NAMES = [
    "qfixed",
    "diag_preferred",
    "aggressive",
]


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
    "compiled_with_blas": UMFPACK_COMPILED_WITH_BLAS,
    "sym_thresh": UMFPACK_STRATEGY_THRESH_SYM,
    "nnzdiag_thresh": UMFPACK_STRATEGY_THRESH_NNZDIAG,
}


cdef list _CONTROL_INT_NAMES = [
    "print_level",
    "blas3_block_size",
    "fixQ",
    "amd_dense",
    "ir_steps",
]


cdef list _CONTROL_BOOL_NAMES = [
    "aggressive",
    "singletons",
    "compiled_with_blas",
]


cdef dict _CONTROL_STRATEGY_INDEX = {
    "auto": UMFPACK_STRATEGY_AUTO,
    "unsymmetric": UMFPACK_STRATEGY_UNSYMMETRIC,
    "obsolete": UMFPACK_STRATEGY_OBSOLETE,
    "symmetric": UMFPACK_STRATEGY_SYMMETRIC,
}

cdef dict _CONTROL_STRATEGY_INVERSE_INDEX = {
    v: k for k, v in _CONTROL_STRATEGY_INDEX.items()
}


cdef dict _CONTROL_SCALE_INDEX = {
    "none": UMFPACK_SCALE_NONE,
    "sum": UMFPACK_SCALE_SUM,
    "max": UMFPACK_SCALE_MAX,
}

cdef dict _CONTROL_SCALE_INVERSE_INDEX = {
    v: k for k, v in _CONTROL_SCALE_INDEX.items()
}


cdef dict _CONTROL_ORDERING_INDEX = {
    "cholmod": UMFPACK_ORDERING_CHOLMOD,
    "amd": UMFPACK_ORDERING_AMD,
    "given": UMFPACK_ORDERING_GIVEN,
    "none": UMFPACK_ORDERING_NONE,
    "metis": UMFPACK_ORDERING_METIS,
    "best": UMFPACK_ORDERING_BEST,
    "user": UMFPACK_ORDERING_USER,
    "metis_guard": UMFPACK_ORDERING_METIS_GUARD,
}

cdef dict _CONTROL_ORDERING_INVERSE_INDEX = {
    v: k for k, v in _CONTROL_ORDERING_INDEX.items()
}


cdef dict _CONTROL_DISPATCH = {
    "strategy": _CONTROL_STRATEGY_INVERSE_INDEX,
    "row_scale": _CONTROL_SCALE_INVERSE_INDEX,
    "ordering_method": _CONTROL_ORDERING_INVERSE_INDEX,
}


cdef dict _INFO_DISPATCH = {
    "strategy_used": _CONTROL_STRATEGY_INVERSE_INDEX,
    "was_scaled": _CONTROL_SCALE_INVERSE_INDEX,
    "ordering_used": _CONTROL_ORDERING_INVERSE_INDEX,
}


# -----------------------------------------------------------------------------
#         Info and Control Classes
# -----------------------------------------------------------------------------
cdef class UMFInfo:
    """A data class to store UMFPACK info.

    Attributes
    ----------
    status : int
        Return status of the last UMFPACK call.
    n_row : int
        Number of rows in the input matrix.
    n_col : int
        Number of columns in the input matrix.
    nz : int
        Number of nonzeros in the input matrix.
    size_of_unit : int
        Size of a unit in bytes.
    size_of_int : int
        Size of an `int32_t` in bytes.
    size_of_long : int
        Size of an `int64_t` in bytes.
    size_of_pointer : int
        Size of a `void *` pointer in bytes.
    size_of_entry : int
        Size of an entry in bytes, real or complex.
    ndense_row : int
        Number of dense rows in the input matrix.
    nempty_row : int
        Number of empty rows in the input matrix.
    ndense_col : int
        Number of dense columns in the input matrix.
    nempty_col : int
        Number of empty columns in the input matrix.
    symbolic_defrag : int
        Number of memory compactions performed.
    symbolic_peak_memory : int
        Peak memory usage during symbolic factorization.
    symbolic_size : int
        Size of symbolic factorization, in Units.
    symbolic_time : float
        Time spent in symbolic factorization, in seconds.
    symbolic_walltime : float
        Wall-clock time spent in symbolic factorization, in seconds.
    strategy_used : str in ['auto', 'unsymmetric', 'symmetric']
        Strategy used in the factorization.
    ordering_used : str in ['cholmod', 'amd', 'given', 'none', 'metis',\
                             'best', 'user', 'metis_guard']
        Ordering method used in the factorization.
    qfixed : bool
        Whether the column permutation Q was fixed.
    diag_preferred : bool
        Whether diagonal pivoting was preferred.
    pattern_symmetry : float
        Symmetry of the nonzero pattern of the input matrix, excluding dense
        rows and columns (aka :math:`S`).
    nz_a_plus_at : int
        Number of nonzeros in :math:`S + S^{\\top}`, excluding the diagonal.
    nzdiag : int
        Number of nonzeros on the diagonal of :math:`S`.
    symmetric_lunz : int
        Number of non-zeros in :math:`L + U`, if AMD ordering was used.
    symmetric_flops : int
        Number of floating-point operations for the factorization, if AMD
        ordering was used.
    symmetric_ndense : int
        Number of dense rows and columns in :math:`S + S^{\\top}`.
    symmetric_dmax : int
        Maximum number of entries in any column of :math:`L`, for AMD.
    col_singletons : int
        Number of column singletons.
    row_singletons : int
        Number of row singletons.
    n2 : int
        Size of :math:`S`.
    s_symmetric : int
        1 if :math:`S` is square and symmetrically permuted.
    numeric_size_estimate : int
        Estimated size of numeric factorization, in Units.
    peak_memory_estimate : int
        Estimated peak memory usage during numeric factorization.
    flops_estimate : int
        Estimated number of floating-point operations for the factorization.
    lnz_estimate : int
        Estimated number of nonzeros in :math:`L`.
    unz_estimate : int
        Estimated number of nonzeros in :math:`U`.
    variable_init_estimate : int
        Initial size of memory usage in numeric factorization.
    variable_peak_estimate : int
        Peak size of memory usage in numeric factorization.
    variable_final_estimate : int
        Final size of memory usage in numeric factorization.
    max_front_size_estimate : int
        Maximum frontal matrix size, estimated.
    max_front_nrows_estimate : int
        Maximum number of rows in any frontal matrix, estimated.
    max_front_ncols_estimate : int
        Maximum number of columns in any frontal matrix, estimated.
    numeric_size : int
        Size of numeric factorization, in Units.
    peak_memory : int
        Peak memory usage during symbolic and numeric factorization.
    flops : int
        Number of floating-point operations for the factorization.
    lnz : int
        Number of nonzeros in :math:`L`.
    unz : int
        Number of nonzeros in :math:`U`.
    variable_init : int
        Initial size of memory usage in numeric factorization.
    variable_peak : int
        Peak size of memory usage in numeric factorization.
    variable_final : int
        Final size of memory usage in numeric factorization.
    max_front_size : int
        Maximum frontal matrix size.
    max_front_nrows : int
        Maximum number of rows in any frontal matrix.
    max_front_ncols : int
        Maximum number of columns in any frontal matrix.
    numeric_defrag : int
        Number of memory compactions performed.
    numeric_realloc : int
        Number of memory reallocations performed.
    numeric_costly_realloc : int
        Number of costly memory reallocations performed.
    compressed_pattern : int
        Number of integers in LU pattern.
    lu_entries : int
        Number of real entries in :math:`L` and :math:`U`.
    numeric_time : float
        Time spent in numeric factorization, in seconds.
    nz_udiag : int
        Number of nonzeros on the diagonal of :math:`U`.
    rcond : float
        Estimate of the reciprocal of the condition number of :math:`A`.
    was_scaled : str in ['none', 'sum', 'max']
        Scaling method used.
    rsmin : float
        `min(max(row))` or `min(sum(row))`, depending on the scaling method.
    rsmax : float
        `max(max(row))` or `max(sum(row))`, depending on the scaling method.
    umin : float
        Minimum absolute value of a diagonal entry of :math:`U`.
    umax : float
        Maximum absolute value of a diagonal entry of :math:`U`.
    alloc_init_used : float
        Initial memory allocation used, as a fraction of total numeric memory.
    forced_updates : int
        Number of forced updates during numeric factorization.
    numeric_walltime : float
        Wall-clock time spent in numeric factorization, in seconds.
    noff_diag : int
        Number of off-diagonal pivots.
    all_lnz : int
        Total number of entries in :math:`L`, if no dropped entries.
    all_unz : int
        Total number of entries in :math:`U`, if no dropped entries.
    nzdropped : int
        Number of dropped entries in :math:`L` and :math:`U`.
    ir_taken : int
        Number of iterative refinement steps taken.
    ir_attempted : int
        Number of iterative refinement steps attempted.
    omega1 : int
        Factor for sparse backdward error estimate.
    omega2 : int
        Factor for sparse backdward error estimate.
    solve_flops : int
        Number of floating-point operations for `solve`.
    solve_time : float
        Time spent in `solve`, in seconds.
    solve_walltime : float
        Wall-clock time spent in `solve`, in seconds.


    .. versionadded:: 0.5.0
    """

    cdef double data[UMFPACK_INFO]

    def __getattr__(self, name):
        try:
            value = self.data[_INFO_INDEX[name]]
        except KeyError:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{name}'"
            )

        mapper = _INFO_DISPATCH.get(name, None)
        if mapper is not None:
            value = mapper[value]
        elif name in _INFO_INT_NAMES:
            value = int(value)
        elif name in _INFO_BOOL_NAMES:
            value = bool(value)

        return value

    def __setattr__(self, name, value):
        # Info values are read-only
        raise AttributeError(f"cannot assign to '{name}'.")

    def __iter__(self):
        cdef int idx
        for key, idx in _INFO_INDEX.items():
            yield (key, getattr(self, key))

    def __repr__(self):
        params = ",\n    ".join(f"{k}={repr(v)}" for k, v in self)
        return f"{self.__class__.__name__}(\n    {params}\n)"

    def __str__(self):
        return self.__repr__()


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


    .. versionadded:: 0.5.0
    """

    cdef double data[UMFPACK_CONTROL]

    def __cinit__(self, **kwargs):
        # NOTE the 4 functions ([dz][il]_defaults) all set the same default
        # values, so just pick one of them.
        umfpack_di_defaults(self.data)

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
            value = self.data[_CONTROL_INDEX[name]]
        except KeyError:
            raise AttributeError(f"UMFControl object has no attribute '{name}'")

        # Convert types where appropriate
        mapper = _CONTROL_DISPATCH.get(name, None)
        if mapper is not None:
            value = mapper[value]
        elif name in _CONTROL_INT_NAMES:
            value = int(value)
        elif name in _CONTROL_BOOL_NAMES:
            value = bool(value)

        return value

    def __setattr__(self, name, value):
        try:
            idx = _CONTROL_INDEX[name]
        except KeyError:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{name}'"
            )

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
        self.data[idx] = value

    def __iter__(self):
        cdef int idx
        for key, idx in _CONTROL_INDEX.items():
            yield (key, getattr(self, key))

    def __repr__(self):
        params = ",\n    ".join(f"{k}={repr(v)}" for k, v in self)
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
        umfpack_di_report_control(self.data)

        # restore old print level
        self.print_level = old_pl


# -----------------------------------------------------------------------------
#         UMFPACK Class Interface
# -----------------------------------------------------------------------------
# TODO include all solver options?
cdef dict _TRANS_INDEX = {
    'N': UMFPACK_A,    # Ax = b
    'T': UMFPACK_Aat,  # A^T x = b
    'H': UMFPACK_At,   # A^H x = b
}


cdef class UMFFactor:
    """The main object used for creating and using an LU factorization.

    The constructor computes the symbolic analysis of a sparse matrix :math:`A`
    and determines a fill-reducing ordering such that:

    .. math::
        L U = P R A Q.

    The numeric factorization is not computed until :meth:`.factorize` is called.

    Properties
    ----------
    is_numeric : bool
        Whether the numeric factorization has been computed.
    lnz, unz : int
        Number of nonzeros in :math:`L` and :math:`U`, respectively.
    nnz : int
        Total number of nonzeros in :math:`L` and :math:`U`.
    n_row, n_col : int
        Number of rows and columns in the input matrix.
    nz_udiag : int
        Number of nonzeros on the diagonal of :math:`U`.
    dtype : :obj:`np.dtype`
        The data type of the matrix entries (``float64`` or ``complex128``).
    itype : :obj:`np.dtype`
        The integer type used for indexing (``int32`` or ``int64``).
    L : :obj:`scipy.sparse.csr_array`
        The :math:`L` factor as a sparse CSR matrix.
    U : :obj:`scipy.sparse.csc_array`
        The :math:`U` factor as a sparse CSC matrix.
    perm_r, perm_c : :obj:`np.ndarray`
        The row and column permutation arrays, :math:`P` and :math:`Q`.
    R : :obj:`np.ndarray`
        The row scaling diagonal matrix as a 1D array.
    info : :obj:`UMFInfo`
        An object containing information about the factorization.
    control : :obj:`UMFControl`
        An object containing settings for the factorization.

    See Also
    --------
    UMFControl, umf_factor, umf_solve

    Notes
    -----
    This object is an interface to the SuiteSparse UMFPACK library [#umfpack_url]_.


    .. versionadded:: 0.5.0

    References
    ----------
    .. [#umfpack_url] SuiteSparse UMFPACK
        https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/UMFPACK
    """

    cdef:
        void *_symbolic
        void *_numeric
        UMFControl _control
        UMFInfo _info
        bint _use_int32
        bint _is_real
        # Store A matrix for use in factorize and solve
        cnp.ndarray _Ap
        cnp.ndarray _Ai
        cnp.ndarray _Ax
        # cached "output" arrays, only extracted from _numeric upon request
        cnp.ndarray _Lp
        cnp.ndarray _Lj
        cnp.ndarray _Lx
        cnp.ndarray _Up
        cnp.ndarray _Ui
        cnp.ndarray _Ux
        cnp.ndarray _P
        cnp.ndarray _Q
        cnp.ndarray _Rs
        # TODO Dx for diagonal of U?

    def __init__(self, object A, object control=None):
        """Compute the symbolic analysis.

        Parameters
        ----------
        A : :obj:`np.ndarray` or sparse array
            The input matrix. Any object that can be converted to
            a :obj:`~scipy.sparse.csc_array` is accepted.
        control : :obj:`UMFControl`, optional
            An object containing settings for the factorization. Default values
            will be used if not provided.
        """
        A, _, _ = validate_csc_input(A)

        # Cache the matrix data
        self._Ap = A.indptr
        self._Ai = A.indices
        self._Ax = A.data

        # Initialize the control and info arrays
        self._control = UMFControl() if control is None else control
        self._info = UMFInfo()

        # Compute the symbolic analysis
        self._init_symbolic(A.shape[0], A.shape[1], self._Ap, self._Ai, self._Ax)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _init_symbolic(
        self,
        int M,
        int N,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data
    ):
        """Compute the symbolic factorization.

        Parameters
        ----------
        M, N : int
            Number of rows and columns of the matrix.
        indptr : 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : 1D array of index_t
            The row indices array of the CSC matrix.
        data : 1D array of value_t
            The data array of the CSC matrix.
        """
        cdef int status

        self._use_int32 = index_t is int32_t
        self._is_real = value_t is double

        # Compute the symbolic factorization
        # NOTE numpy complex arrays store real and imag parts interleaved,
        # so we can just pass the pointer to the data as double*
        if self._is_real:
            if self._use_int32:
                status = umfpack_di_symbolic(
                    M,
                    N,
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    &self._symbolic,
                    self._control.data,
                    self._info.data
                )
            else:
                status = umfpack_dl_symbolic(
                    M,
                    N,
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    &self._symbolic,
                    self._control.data,
                    self._info.data
                )
        else:
            if self._use_int32:
                status = umfpack_zi_symbolic(
                    M,
                    N,
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    NULL,
                    &self._symbolic,
                    self._control.data,
                    self._info.data
                )
            else:
                status = umfpack_zl_symbolic(
                    M,
                    N,
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    NULL,
                    &self._symbolic,
                    self._control.data,
                    self._info.data
                )

        _handle_errors(status)

    def __dealloc__(self):
        """Free UMFPACK symbolic and numeric objects."""
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

        if self._numeric is not NULL:
            if self._is_real:
                if self._use_int32:
                    umfpack_di_free_numeric(&self._numeric)
                else:
                    umfpack_dl_free_numeric(&self._numeric)
            else:
                if self._use_int32:
                    umfpack_zi_free_numeric(&self._numeric)
                else:
                    umfpack_zl_free_numeric(&self._numeric)

    def __repr__(self):
        cls_name = self.__class__.__name__
        dtype = 'float64' if self._is_real else 'complex128'
        itype = 'int32' if self._use_int32 else 'int64'
        factor_type = 'numeric' if self.is_numeric else 'symbolic'
        min_MN = min(self.n_row, self.n_col)
        L_shape = (self.n_row, min_MN)
        U_shape = (min_MN, self.n_col)
        return (
            f"<{cls_name} {factor_type} factor of dtype '{dtype}' "
            f"with '{itype}' indices:\n"
            f"    L: {L_shape} with {self.lnz} stored elements\n"
            f"    U: {U_shape} with {self.unz} stored elements>"
        )

    def __str__(self):
        return self.__repr__()

    # -------------------------------------------------------------------------
    #         Properties
    # -------------------------------------------------------------------------
    @property
    def is_numeric(self):
        return self._symbolic is not NULL and self._numeric is not NULL

    @property
    def lnz(self):
        return int(self._info.lnz if self._info.lnz >= 0 else 0)

    @property
    def unz(self):
        return int(self._info.unz if self._info.unz >= 0 else 0)

    @property
    def nnz(self):
        return int(self.lnz + self.unz)

    @property
    def n_row(self):
        return int(self._info.n_row if self._info.n_row >= 0 else 0)

    @property
    def n_col(self):
        return int(self._info.n_col if self._info.n_col >= 0 else 0)

    @property
    def nz_udiag(self):
        return int(self._info.nz_udiag if self._info.nz_udiag >= 0 else 0)

    @property
    def dtype(self):
        return np.float64 if self._is_real else np.complex128

    @property
    def itype(self):
        return np.int32 if self._use_int32 else np.int64

    @property
    def L(self):
        if self._Lp is None or self._Lj is None or self._Lx is None:
            self._get_numeric()

        L_shape = (self.n_row, min(self.n_row, self.n_col))
        return csr_array((self._Lx, self._Lj, self._Lp), shape=L_shape)

    @property
    def U(self):
        if self._Up is None or self._Ui is None or self._Ux is None:
            self._get_numeric()

        U_shape = (min(self.n_row, self.n_col), self.n_col)
        return csc_array((self._Ux, self._Ui, self._Up), shape=U_shape)

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
    def R(self):
        if self._Rs is None:
            self._get_numeric()
        return self._Rs

    @property
    def info(self):
        return self._info

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, UMFControl control):
        self._control = control

    # -------------------------------------------------------------------------
    #         Public Methods
    # -------------------------------------------------------------------------
    def copy(self):
        """Return a deep copy of the current UMFFactor object."""
        cdef UMFFactor umf = UMFFactor.__new__(UMFFactor)

        umf._use_int32 = self._use_int32
        umf._is_real = self._is_real

        cdef int status

        if self._is_real:
            if self._use_int32:
                status = umfpack_di_copy_symbolic(&umf._symbolic, self._symbolic)
            else:
                status = umfpack_dl_copy_symbolic(&umf._symbolic, self._symbolic)
        else:
            if self._use_int32:
                status = umfpack_zi_copy_symbolic(&umf._symbolic, self._symbolic)
            else:
                status = umfpack_zl_copy_symbolic(&umf._symbolic, self._symbolic)

        _handle_errors(status)

        if self._is_real:
            if self._use_int32:
                status = umfpack_di_copy_numeric(&umf._numeric, self._numeric)
            else:
                status = umfpack_dl_copy_numeric(&umf._numeric, self._numeric)
        else:
            if self._use_int32:
                status = umfpack_zi_copy_numeric(&umf._numeric, self._numeric)
            else:
                status = umfpack_zl_copy_numeric(&umf._numeric, self._numeric)

        _handle_errors(status)

        umf._control = self._control
        umf._info = self._info

        umf._Lp = None if self._Lp is None else self._Lp.copy()
        umf._Lj = None if self._Lj is None else self._Lj.copy()
        umf._Lx = None if self._Lx is None else self._Lx.copy()
        umf._Up = None if self._Up is None else self._Up.copy()
        umf._Ui = None if self._Ui is None else self._Ui.copy()
        umf._Ux = None if self._Ux is None else self._Ux.copy()
        umf._P = None if self._P is None else self._P.copy()
        umf._Q = None if self._Q is None else self._Q.copy()
        umf._Rs = None if self._Rs is None else self._Rs.copy()

        return umf

    def factorize(self, object A=None):
        """Compute the numeric factorization of a sparse matrix.

        Given the symbolic analysis performed in the constructor,
        compute the numeric factorization of a sparse matrix :math:`A`
        and determine a fill-reducing ordering such that:

        .. math::
            L U = P R A Q.

        If given, the matrix :math:`A` must have the same shape and nonzero
        pattern as the one used to create this :class:`UMFFactor` object, but
        need not have the same values.

        Parameters
        ----------
        A : *(M, N)* ndarray or sparse array, optional
            The input matrix. Must have the same shape and nonzero pattern as
            the matrix used to create this :class:`UMFFactor` object. If not
            provided, the original matrix given to the constructor will be
            used.

        Returns
        -------
        :class:`UMFFactor`
            The current object, for method chaining.

        Raises
        ------
        :exc:`UMFPACKError` or subclass
            If an error occurs during the numeric factorization.
        """
        assert self._symbolic is not NULL, (
            "Symbolic factorization not present. "
            "Cannot perform numeric factorization."
        )

        if A is not None:
            A, _, itype = validate_csc_input(A)
            self._check_input_matrix(A, itype)
            # Update cached matrix data
            self._Ap = A.indptr
            self._Ai = A.indices
            self._Ax = A.data

        # Clear cached output arrays
        self._Lp = None
        self._Lj = None
        self._Lx = None
        self._Up = None
        self._Ui = None
        self._Ux = None
        self._P = None
        self._Q = None
        self._Rs = None

        self._factorize(self._Ap, self._Ai, self._Ax)

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
        cdef int status

        # Compute the symbolic factorization
        if self._is_real:
            if self._use_int32:
                status = umfpack_di_numeric(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    &self._numeric,
                    self._control.data,
                    self._info.data
                )
            else:
                status = umfpack_dl_numeric(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    self._symbolic,
                    &self._numeric,
                    self._control.data,
                    self._info.data
                )
        else:
            if self._use_int32:
                status = umfpack_zi_numeric(
                    <int32_t*>&indptr[0],
                    <int32_t*>&indices[0],
                    <double*>&data[0],
                    NULL,
                    self._symbolic,
                    &self._numeric,
                    self._control.data,
                    self._info.data
                )
            else:
                status = umfpack_zl_numeric(
                    <int64_t*>&indptr[0],
                    <int64_t*>&indices[0],
                    <double*>&data[0],
                    NULL,
                    self._symbolic,
                    &self._numeric,
                    self._control.data,
                    self._info.data
                )

        _handle_errors(status)

    # TODO allow x as input?
    def solve(self, object b, object A=None, *, object trans='N'):
        """Solve a linear system using the LU factorization.

        This method solves one of the following linear systems:

        * :math:`A x = b` (if ``trans='N'``)
        * :math:`A^{\\top} x = b` (if ``trans='T'`` and :math:`A` is real)
        * :math:`A^{H} x = b` (if ``trans='H'`` and :math:`A` is complex)

        The matrix :math:`A` must have the same shape and nonzero pattern as
        the one used to create this :class:`UMFFactor` object, but need not
        have the same values. No check is performed to ensure that the
        input matrix is compatible with the existing factorization.

        Parameters
        ----------
        b : *(N,)* :obj:`ndarray` or sparse array
            The right-hand side vector.
        A : *(N, N)* :obj:`ndarray` or sparse array, optional
            The input matrix. Must have the same shape and nonzero pattern as
            the matrix used to create this :class:`UMFFactor` object.
        trans : str, optional
            The type of system to solve. Possible values are:

            * ``'N'``: solve :math:`A x = b` (default)
            * ``'T'``: solve :math:`A^{\\top} x = b`
            * ``'H'``: solve :math:`A^{H} x = b`

            .. note::

                If :math:`A` is real, then ``'T'`` and ``'H'`` are equivalent.

        Returns
        -------
        x : *(N,)* or *(N, K)* :obj:`ndarray` or sparse array
            The solution vector or matrix. If ``b`` is a 1D array, then ``x`` is
            returned as a 1D array. If ``b`` is a 2D array with ``K`` columns,
            then ``x`` is returned as a 2D array with ``K`` columns. If ``b``
            is a sparse array, then ``x`` is also returned as a sparse array.

        Warns
        -----
        :exc:`UMFPACKSingularMatrixWarning`
            If the matrix is detected to be singular to working precision.
            In that case, the solution will have infinite or NaN values,
            but other entries may still be valid.

        Raises
        ------
        :exc:`UMFPACKError` or subclass
            If an error occurs during the solve.
        """
        cdef int sys
        try:
            sys = _TRANS_INDEX[trans]
        except KeyError:
            raise ValueError(
                f"Invalid value for trans: {trans}. "
                f"Expected one of {list(_TRANS_INDEX.keys())}"
            )

        if not (isinstance(b, np.ndarray) or issparse(b)):
            raise ValueError("b must be an ndarray or sparse matrix.")

        if A is not None:
            A, _, itype = validate_csc_input(A, require_square=True)
            self._check_input_matrix(A, itype)
            # Update cached matrix data
            self._Ap = A.indptr
            self._Ai = A.indices
            self._Ax = A.data

        if b.dtype != self.dtype:
            raise ValueError(
                f"LHS and RHS dtypes do not match. {self.dtype=} and {b.dtype=}"
            )

        if b.ndim not in (1, 2):
            raise ValueError("b must be a 1D or 2D array.")

        cdef size_t N = <size_t>self._info.data[UMFPACK_NROW]
        cdef bint return_1D = b.ndim == 1

        if b.shape[0] != N:
            raise ValueError(
                "Right-hand side b must have the same number of rows as A."
            )

        cdef bint return_sparse = issparse(b)

        if return_sparse:
            b = b.toarray()
        else:
            b = np.asarray(b)

        if b.ndim == 1:
            b = b.reshape((N, 1))

        # TODO warn here?
        # Prepare to solve the system
        if self._numeric is NULL:
            self.factorize(A)

        # Check the condition number
        self._check_rcond()

        # Ensure columns are contiguous for multiple RHS
        b = np.asfortranarray(b)

        # Allocate the output array
        x = np.empty_like(b, order='F')

        self._solve(sys, b, self._Ap, self._Ai, self._Ax, x)

        if return_sparse:
            x = csc_array(x, dtype=b.dtype)
            x.indptr = x.indptr.astype(self.itype)
            x.indices = x.indices.astype(self.itype)

        if return_1D:
            x = x[:, 0]

        return x

    # TODO see umfpack_wsolve. Provide workspace for multiple solves?
    @cython.boundscheck(False)  # for-loop guaranteed in-bounds
    @cython.wraparound(False)
    def _solve(
        self,
        int sys,
        value_t[::1, :] b,
        index_t[::1] indptr,
        index_t[::1] indices,
        value_t[::1] data,
        value_t[::1, :] x
    ):
        """Solve multiple RHS systems.

        Parameters
        ----------
        sys : int
            The system type (UMFPACK_A, UMFPACK_Aat, UMFPACK_At).
        b : 2D array of value_t, shape (N, K)
            The right-hand side matrix.
        indptr : contiguous 1D array of index_t
            The index pointer array of the CSC matrix.
        indices : contiguous 1D array of index_t
            The row indices array of the CSC matrix.
        data : contiguous 1D array of value_t
            The data array of the CSC matrix.
        x : 2D array of value_t, shape (N, K)
            The output solution matrix.
        """
        cdef:
            Py_ssize_t k
            Py_ssize_t K = b.shape[1]
            double* data_ptr
            double* x_ptr
            double* b_ptr

        for k in range(K):
            # NOTE numpy complex arrays store real and imag parts interleaved,
            # so we can just pass the pointer to the data as double*
            data_ptr = <double*>&data[0]
            x_ptr = <double*>&x[0, k]
            b_ptr = <double*>&b[0, k]

            # Solve the system
            if self._is_real:
                if self._use_int32:
                    status = umfpack_di_solve(
                        sys,
                        <int32_t*>&indptr[0],
                        <int32_t*>&indices[0],
                        data_ptr,
                        x_ptr,
                        b_ptr,
                        self._numeric,
                        self._control.data,
                        self._info.data
                    )
                else:
                    status = umfpack_dl_solve(
                        sys,
                        <int64_t*>&indptr[0],
                        <int64_t*>&indices[0],
                        data_ptr,
                        x_ptr,
                        b_ptr,
                        self._numeric,
                        self._control.data,
                        self._info.data
                    )
            else:
                if self._use_int32:
                    status = umfpack_zi_solve(
                        sys,
                        <int32_t*>&indptr[0],
                        <int32_t*>&indices[0],
                        data_ptr,
                        NULL,
                        x_ptr,
                        NULL,
                        b_ptr,
                        NULL,
                        self._numeric,
                        self._control.data,
                        self._info.data
                    )
                else:
                    status = umfpack_zl_solve(
                        sys,
                        <int64_t*>&indptr[0],
                        <int64_t*>&indices[0],
                        data_ptr,
                        NULL,
                        x_ptr,
                        NULL,
                        b_ptr,
                        NULL,
                        self._numeric,
                        self._control.data,
                        self._info.data
                    )

            _handle_errors(status)

    def slogdet(self):
        """Return the determinant of the matrix as (sign, logabsdet).

        See Also
        --------
        numpy.linalg.slogdet
        """
        if not self.is_numeric:
            raise ValueError(
                "Numeric factorization not present. "
                "Cannot compute determinant."
            )

        Mx = np.empty(1, dtype=np.float64 if self._is_real else np.complex128)
        Ex = np.empty(1, dtype=np.float64)

        self._slogdet(Mx, Ex)

        # Compute the result
        m = Mx[0]
        e = Ex[0]
        sign = np.sign(m)
        # log(|det(A)|) = log(|m| * 10**e) = log(|m|) + e * log(10)
        logabsdet = np.log(abs(m)) + e * np.log(10.0)

        return (sign, logabsdet)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _slogdet(self, value_t[::1] Mx, double[::1] Ex):
        """Compute the determinant of the matrix.

        Parameters
        ----------
        Mx : 1D array of value_t, shape (1,)
            The mantissa of the determinant.
        Ex : 1D array of double, shape (1,)
            The exponent of the determinant.
        """
        cdef int status
        cdef double* mx_ptr = <double*>&Mx[0]
        cdef double* ex_ptr = &Ex[0]

        if self._is_real:
            if self._use_int32:
                status = umfpack_di_get_determinant(
                    mx_ptr, ex_ptr, self._numeric, self._info.data
                )
            else:
                status = umfpack_dl_get_determinant(
                    mx_ptr, ex_ptr, self._numeric, self._info.data
                )
        else:
            if self._use_int32:
                status = umfpack_zi_get_determinant(
                    mx_ptr, NULL, ex_ptr, self._numeric, self._info.data
                )
            else:
                status = umfpack_zl_get_determinant(
                    mx_ptr, NULL, ex_ptr, self._numeric, self._info.data
                )

        _handle_errors(status)

    # -------------------------------------------------------------------------
    #         Reporting
    # -------------------------------------------------------------------------
    def report_info(self, object print_level=2):
        """Print a report of the UMFInfo structure.

        This method provides more internal details from UMFPACK itself than the
        string representation.

        Parameters
        ----------
        print_level : int, optional
            The verbosity level. Default value is 2.

            Accepted values are:

            * None: use current print level
            * <= 0: no output
            * 1: error messages only
            * >= 2: error messages and print all of UMFInfo

        """
        cdef int pl
        if print_level is None:
            pl = self._control.print_level
        else:
            pl = print_level

        cdef int old_pl = self._control.print_level
        self._control.print_level = pl

        umfpack_di_report_info(self._control.data, self._info.data)

        # restore old print level
        self._control.print_level = old_pl

    def report_control(self):
        self._control.report()

    def report_symbolic(self, object print_level=4):
        cdef int pl
        if print_level is None:
            pl = self._control.print_level
        else:
            pl = print_level

        cdef int old_pl = self._control.print_level
        self._control.print_level = pl

        if self._is_real:
            if self._use_int32:
                umfpack_di_report_symbolic(self._symbolic, self._control.data)
            else:
                umfpack_dl_report_symbolic(self._symbolic, self._control.data)
        else:
            if self._use_int32:
                umfpack_zi_report_symbolic(self._symbolic, self._control.data)
            else:
                umfpack_zl_report_symbolic(self._symbolic, self._control.data)

        # restore old print level
        self._control.print_level = old_pl

    def report_numeric(self, object print_level=4):
        cdef int pl
        if print_level is None:
            pl = self._control.print_level
        else:
            pl = print_level

        cdef int old_pl = self._control.print_level
        self._control.print_level = pl

        if self._is_real:
            if self._use_int32:
                umfpack_di_report_numeric(self._numeric, self._control.data)
            else:
                umfpack_dl_report_numeric(self._numeric, self._control.data)
        else:
            if self._use_int32:
                umfpack_zi_report_numeric(self._numeric, self._control.data)
            else:
                umfpack_zl_report_numeric(self._numeric, self._control.data)

        # restore old print level
        self._control.print_level = old_pl

    # -------------------------------------------------------------------------
    #         Private Methods
    # -------------------------------------------------------------------------
    def _check_input_matrix(self, object A, object itype):
        """Check that the input matrix matches the existing factorization."""
        if A.shape != (self.n_row, self.n_col):
            raise ValueError(
                "The shape of the input matrix does not match "
                "the one used for symbolic factorization. "
                f"Expected {(self.n_row, self.n_col)}, got {A.shape}."
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

    cdef int _check_rcond(self) except -1:
        """Check the condition number."""
        cdef double rcond = self._info.rcond
        cdef double eps = np.finfo(np.float64).eps

        if rcond == 0:
            warnings.warn(
                "Matrix is indefinite or singular to working precision."
                "  Results may contain infinite or NaN values.",
                UMFPACKSingularMatrixWarning
            )
        elif rcond < eps:
            warnings.warn(
                "Matrix is nearly singular."
                f"  Results may be inaccurate (rcond={rcond:.2e}).",
                UMFPACKSingularMatrixWarning
            )

    cdef void _get_numeric(self) except *:
        """Extract the numeric factorization data from UMFPACK."""
        if self._numeric is NULL:
            raise UMFPACKError(
                "Numeric factorization not present. "
                "Run `UMFFactor.factorize(A)` first."
            )

        cdef:
            size_t lnz = self._info.lnz
            size_t unz = self._info.unz
            size_t n_row = self._info.n_row
            size_t n_col = self._info.n_col

        dtype = np.dtype(np.double if self._is_real else np.cdouble)
        itype = np.dtype(np.int32 if self._use_int32 else np.int64)

        # Create output arrays
        self._Lp = np.empty(n_row + 1, dtype=itype)
        self._Lj = np.empty(lnz, dtype=itype)
        self._Lx = np.empty(lnz, dtype=dtype)

        self._Up = np.empty(n_col + 1, dtype=itype)
        self._Ui = np.empty(unz, dtype=itype)
        self._Ux = np.empty(unz, dtype=dtype)

        self._P = np.empty(n_row, dtype=itype)
        self._Q = np.empty(n_col, dtype=itype)
        self._Rs = np.empty(n_row, dtype=np.float64)  # always real

        self._dispatch_get_numeric(
            self._Lp, self._Lj, self._Lx,
            self._Up, self._Ui, self._Ux,
            self._P,
            self._Q,
            self._Rs
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _dispatch_get_numeric(
        self,
        index_t[::1] Lp, index_t[::1] Lj, value_t[::1] Lx,
        index_t[::1] Up, index_t[::1] Ui, value_t[::1] Ux,
        index_t[::1] P,
        index_t[::1] Q,
        double[::1] Rs,
    ):
        """Call the appropriate UMFPACK get_numeric function.

        Parameters
        ----------
        Lp, Lj, Lx : arrays for the L factor
            The output arrays for the L factor in CSC format.
        Up, Ui, Ux : arrays for the U factor
            The output arrays for the U factor in CSC format.
        P : array of index_t
            The output row permutation array.
        Q : array of index_t
            The output column permutation array.
        Rs : array of double
            The output row scaling factors.
        """
        cdef int status
        cdef bint do_recip

        # Extract the numeric factorization
        if self._is_real:
            if self._use_int32:
                status = umfpack_di_get_numeric(
                    <int32_t*>&Lp[0],
                    <int32_t*>&Lj[0],
                    <double*>&Lx[0],
                    <int32_t*>&Up[0],
                    <int32_t*>&Ui[0],
                    <double*>&Ux[0],
                    <int32_t*>&P[0],
                    <int32_t*>&Q[0],
                    NULL,  # Dx
                    <int32_t*>&do_recip,
                    <double*>&Rs[0],
                    self._numeric
                )
            else:
                status = umfpack_dl_get_numeric(
                    <int64_t*>&Lp[0],
                    <int64_t*>&Lj[0],
                    <double*>&Lx[0],
                    <int64_t*>&Up[0],
                    <int64_t*>&Ui[0],
                    <double*>&Ux[0],
                    <int64_t*>&P[0],
                    <int64_t*>&Q[0],
                    NULL,  # Dx
                    <int64_t*>&do_recip,
                    <double*>&Rs[0],
                    self._numeric
                )
        else:
            if self._use_int32:
                status = umfpack_zi_get_numeric(
                    <int32_t*>&Lp[0],
                    <int32_t*>&Lj[0],
                    <double*>&Lx[0],
                    NULL,  # Lz
                    <int32_t*>&Up[0],
                    <int32_t*>&Ui[0],
                    <double*>&Ux[0],
                    NULL,  # Uz
                    <int32_t*>&P[0],
                    <int32_t*>&Q[0],
                    NULL,  # Dx
                    NULL,  # Dz
                    <int32_t*>&do_recip,
                    <double*>&Rs[0],
                    self._numeric
                )
            else:
                status = umfpack_zl_get_numeric(
                    <int64_t*>&Lp[0],
                    <int64_t*>&Lj[0],
                    <double*>&Lx[0],
                    NULL,  # Lz
                    <int64_t*>&Up[0],
                    <int64_t*>&Ui[0],
                    <double*>&Ux[0],
                    NULL,  # Uz
                    <int64_t*>&P[0],
                    <int64_t*>&Q[0],
                    NULL,  # Dx
                    NULL,  # Dz
                    <int64_t*>&do_recip,
                    <double*>&Rs[0],
                    self._numeric
                )

        _handle_errors(status)

        # From umfpack.h:
        #   If do_recip is TRUE (one), then the scale factors Rs [i] are to be used
        #   by multiplying row i by Rs [i].  Otherwise, the entries in row i are to
        #   be divided by Rs [i].
        #
        # Always return R s.t. (R[:, np.newaxis] * A) scales the rows.
        if not do_recip:
            np.reciprocal(self._Rs, out=self._Rs)


# Set docstrings
_REPORT_DOC = """Print a report of the {kind} factorization.

This method provides more internal details from UMFPACK itself than the
string representation.

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

UMFFactor.report_symbolic.__doc__ = _REPORT_DOC.format(kind="symbolic")
UMFFactor.report_numeric.__doc__ = _REPORT_DOC.format(kind="numeric")


# -----------------------------------------------------------------------------
#         Convenience Functions
# -----------------------------------------------------------------------------
def umf_factor(object A, *, object control=None, **kwargs):
    """Compute the LU factorization of a sparse matrix using UMFPACK.

    This is a convenience function that creates a :class:`UMFFactor` object,
    computes the numeric factorization, and returns the resulting object.

    Parameters
    ----------
    A : *(M, N)* :obj:`ndarray` or sparse array
        The input matrix to factorize.
    control : :class:`UMFControl`, optional
        The control parameters to use for the factorization. If not provided,
        default parameters are used.
    kwargs : keyword arguments, optional
        Additional keyword arguments to pass to :class:`UMFControl` if
        `control` is not provided.

    Returns
    -------
    :class:`UMFFactor`
        The LU factorization of the input matrix.

    Warns
    -----
    :exc:`UMFPACKSingularMatrixWarning`
        If the matrix is exactly singular.

    Raises
    ------
    :exc:`UMFPACKError` or subclass
        If an error occurs during the factorization or solve.

    See Also
    --------
    UMFFactor, UMFControl, umf_solve

    .. versionadded:: 0.5.0
    """
    if control is None:
        control = UMFControl(**kwargs)
    return UMFFactor(A, control).factorize()


def umf_solve(object A, object b, *, object trans='N', object control=None, **kwargs):
    """Solve a linear system using UMFPACK.

    This is a convenience function that creates a :class:`UMFFactor` object,
    computes the numeric factorization, and solves the linear system.

    Parameters
    ----------
    A : *(N, N)* :obj:`ndarray` or sparse array
        The input matrix.
    b : *(N,)* :obj:`ndarray` or sparse array
        The right-hand side vector.
    trans : str, optional
        The type of system to solve. Possible values are:

        * ``'N'``: solve :math:`A x = b` (default)
        * ``'T'``: solve :math:`A^{\\top} x = b`
        * ``'H'``: solve :math:`A^{H} x = b`

        .. note::

            If :math:`A` is real, then ``'T'`` and ``'H'`` are equivalent.

    control : :class:`UMFControl`, optional
        The control parameters to use for the factorization. If not provided,
        default parameters are used.
    kwargs : keyword arguments, optional
        Additional keyword arguments to pass to :class:`UMFControl` if
        `control` is not provided.

    Returns
    -------
    x : *(N,)* :obj:`ndarray` or sparse array
        The solution vector of the same type as the input right-hand side `b`.

    Warns
    -----
    :exc:`UMFPACKSingularMatrixWarning`
        If the matrix is detected to be singular to working precision.
        In that case, the solution will have infinite or NaN values,
        but other entries may still be valid.

    Raises
    ------
    :exc:`UMFPACKError` or subclass
        If an error occurs during the factorization or solve.

    See Also
    --------
    UMFFactor, UMFControl, umf_factor


    .. versionadded:: 0.5.0
    """
    if control is None:
        control = UMFControl(**kwargs)

    # factorize() and solve() will each warn for a singular matrix,
    # so we catch the warnings from factorize() and re-raise only once.
    with warnings.catch_warnings(record=True) as ws:
        x = UMFFactor(A, control).factorize().solve(b, trans=trans)

    # Raise only the latest singular matrix warning from solve
    if ws:
        w = ws[-1]
        warnings.warn(w.message, w.category)

    return x


# =============================================================================
# =============================================================================
