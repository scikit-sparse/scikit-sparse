# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: utils.py
#  Created: 2025-08-12 21:15
# =============================================================================

"""Utility functions for scikit-sparse."""

import warnings

import numpy as np
from packaging import version
from scipy import __version__ as scipy_version
from scipy.sparse import SparseEfficiencyWarning, csc_array, issparse


def validate_csc_input(A, require_square=False, ensure_double=True):
    """Validate and convert input matrix to CSC format.

    Integer and boolean data types are converted to float32 by default.

    Parameters
    ----------
    A : (M, N) array_like
        Input matrix to be validated and converted to CSC format, if possible.
        If `A` is already a `csc_array`, and not in canonical format, it will
        be converted to canonical format in-place (a copy is not made).
    require_square : bool, optional
        If True, the input matrix must be square (M == N). Default is False.
    ensure_double : bool, optional
        If True, the input matrix will be converted to double precision (either
        float64 or complex128) if it is not already in double precision.
        Otherwise, it will retain its original floating-point precision
        (float32 or complex64), or the minimum precision required to convert
        integer or boolean types.

    Returns
    -------
    A : (M, N) csc_array
        The input matrix converted to canonical CSC format.
    use_int32 : bool
        Indicates whether the index arrays use int32 (True) or int64 (False).
    out_itype : dtype
        The data type of the output matrix indices, which is determined based
        on the input matrix's index data type.

    Raises
    ------
    SparseEfficiencyWarning
        If the input matrix is not in CSC format and is converted to CSC.
    ValueError
        If the input matrix is not 2D, or cannot be converted to CSC format, or
        if it is not square when `require_square` is True.

    .. versionadded:: 0.5.0
    """
    # Convert dense to sparse CSC
    if not issparse(A):
        A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("Input must be 2D.")

    M, N = A.shape

    if require_square and M != N:
        raise ValueError("Input must be square.")

    try:
        if not isinstance(A, csc_array):
            warnings.warn(
                f"Input matrix ({type(A)}) not in CSC array format. Converting to CSC.",
                SparseEfficiencyWarning,
                stacklevel=3,
            )
            A = csc_array(A)
    except ValueError:
        raise ValueError("Input must be convertible to CSC format.")

    # Coerce bool or int data to float
    if np.issubdtype(A.dtype, np.bool_) or np.issubdtype(A.dtype, np.integer):
        dtype = np.result_type(A.dtype, np.float32)
        A = A.astype(dtype)

    # Ensure double precision if requested
    if ensure_double:
        if np.issubdtype(A.dtype, np.floating):
            A = A.astype(np.float64, copy=False)
        elif np.issubdtype(A.dtype, np.complexfloating):
            A = A.astype(np.complex128, copy=False)

    # NOTE as of scipy 1.16.2, A.has_sorted_indices and A.has_canonical_format
    # are not always set correctly! In particular, A.setdiag(...) can lead to
    # incorrect flags.
    #
    # A copy will reset the flags, and avoid modifying the input. Generally,
    # users would not expect the input matrix to be modified.
    # A = A.copy()
    #
    # To save on memory, we manually set the flags to False to force
    # sum_duplicates() to re-sort and re-sum any duplicates.
    if version.parse(scipy_version) < version.parse("1.17.0"):
        A.has_sorted_indices = False
        A.has_canonical_format = False

    A.sum_duplicates()  # sort indices and sum duplicates

    assert A.has_sorted_indices
    assert A.has_canonical_format

    # Choose index width: int32 or int64
    use_int32 = A.indptr.dtype == np.int32 and A.indices.dtype == np.int32
    out_itype = np.dtype(np.int32 if use_int32 else np.int64)

    return A, use_int32, out_itype
