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
from scipy import sparse

from sksparse.utils import validate_csc_input


def test_1D_input():
    with pytest.raises(ValueError, match="Input must be 2D"):
        validate_csc_input(np.arange(10))


class TestNonSquareInput:
    def test_nonsquare_require_square(self):
        with pytest.raises(ValueError, match="Input must be square"):
            validate_csc_input(sparse.csc_array((3, 4)), require_square=True)

    def test_nonsquare_no_require_square(self):
        shape = (3, 4)
        A = sparse.csc_array(shape)
        itype = np.int32
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        result, use_int32, out_itype = validate_csc_input(A, require_square=False)
        assert isinstance(result, sparse.csc_array)
        assert result.shape == shape
        assert use_int32 is True
        assert out_itype == itype


def test_ND_input():
    with pytest.raises(ValueError, match="Input must be 2D"):
        validate_csc_input(np.empty((2, 3, 4)))
