# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_utils.py
#  Created: 2025-08-12 21:32
# =============================================================================

"""Test cases for utility functions in scikit-sparse."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.utils import validate_csc_input


def test_1D_input():
    with pytest.raises(ValueError, match="Input must be 2D"):
        validate_csc_input(np.arange(10))


def test_nonsquare_require_square():
    with pytest.raises(ValueError, match="Input must be square"):
        validate_csc_input(sparse.csc_array((3, 4)), require_square=True)


def test_ND_input():
    with pytest.raises(ValueError, match="Input must be 2D"):
        validate_csc_input(np.empty((2, 3, 4)))


@pytest.mark.parametrize("matrix_type", ["dense", "csc", "coo", "csc_matrix"])
def test_input_conversion(matrix_type):
    A = sparse.csc_array(np.arange(12).reshape(3, 4))

    match matrix_type:
        case "dense":
            A = A.toarray()
        case "csc":
            A = A.tocsc()
        case "coo":
            A = A.tocoo()
        case "csc_matrix":
            A = sparse.csc_matrix(A)
        case _:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

    if matrix_type == "csc":
        result, use_int32, out_itype = validate_csc_input(A)
    else:
        with pytest.warns(sparse.SparseEfficiencyWarning, match="not in CSC array format"):
            result, use_int32, out_itype = validate_csc_input(A)

    assert isinstance(result, sparse.csc_array)
    assert_array_equal(result.toarray(), A.toarray() if sparse.issparse(A) else A, strict=True)
    assert use_int32
    assert out_itype == np.int32
