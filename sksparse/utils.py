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
from scipy.sparse import SparseEfficiencyWarning, csc_array, issparse


def validate_csc_input(A, require_square=False):
    """Validate and convert input matrix to CSC format.

    Parameters
    ----------
    A : (M, N) array_like
        Input matrix to be validated and converted to CSC format, if possible.
        If `A` is already a `csc_array`, and not in canonical format, it will
        be converted to canonical format in-place (a copy is not made).
    require_square : bool, optional
        If True, the input matrix must be square (M == N). Default is False.

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

    # A copy will reset the flags, and avoid modifying the input. Generally,
    # users would not expect the input matrix to be modified.
    # A = A.copy()

    # NOTE as of scipy 1.16.2, A.has_sorted_indices and A.has_canonical_format
    #   are not always set correctly!
    # Manually set the flags to False to force fixing the format.
    A.has_sorted_indices = False
    A.has_canonical_format = False
    A.sum_duplicates()  # sort indices and sum duplicates

    assert A.has_sorted_indices
    assert A.has_canonical_format

    # Choose index width: int32 or int64
    use_int32 = A.indptr.dtype == np.int32 and A.indices.dtype == np.int32
    out_itype = np.int32 if use_int32 else np.int64

    return A, use_int32, out_itype
