#!/usr/bin/env python3
# =============================================================================
#     File: test_amd.py
#  Created: 2025-07-28 13:34
#   Author: Bernie Roesler
#
"""Test code for the sksparse.amd module."""
# =============================================================================

import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sksparse.amd import amd


def is_valid_permutation(p):
    """Check if a vector is a valid permutation."""
    return np.array_equal(np.sort(p), np.arange(len(p)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    """Test that an empty input raises an error."""
    empty_A = sparse.csc_matrix((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    assert_array_equal(amd(empty_A), np.array([], dtype=itype), strict=True)


def test_1D_input():
    """Test that an invalid input raises a ValueError."""
    with pytest.raises(ValueError, match="Input must be square"):
        amd(np.arange(10))


def test_ND_input():
    """Test that an invalid input raises a ValueError."""
    rng = np.random.default_rng(565656)
    with pytest.raises(ValueError, match="Input must be convertible to CSC format"):
        amd(rng.random((2, 3, 4)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    """Test that a matrix of all zeros returns the identity permutation."""
    N = 10  # arbitrary
    zero_A = sparse.csc_matrix((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    assert_array_equal(amd(zero_A), np.arange(N, dtype=itype), strict=True)

# =============================================================================
# =============================================================================
