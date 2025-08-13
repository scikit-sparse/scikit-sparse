# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_btf_btf.py
#  Created: 2025-08-06 18:59
# =============================================================================

"""Test cases for the sksparse.btf.btf function."""

# from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.btf import btf, btf_q_permutation

from ..helpers import generate_random_matrices, is_valid_permutation


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    p, q, r = btf(empty_A)
    assert_array_equal(p, np.array([], dtype=itype), strict=True)
    assert_array_equal(q, np.array([], dtype=itype), strict=True)
    assert_array_equal(r, np.zeros(1, dtype=itype), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    p, q, r = btf(zero_A)
    assert_array_equal(p, np.arange(N, dtype=itype), strict=True)
    assert_array_equal(q, -np.arange(N, dtype=itype) - 2, strict=True)
    assert_array_equal(r, np.arange(N + 1, dtype=itype), strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_array([[1]])
    p, q, r = btf(singleton_A)
    assert_array_equal(p, np.array([0], dtype=np.int32), strict=True)
    assert_array_equal(q, np.array([0], dtype=np.int32), strict=True)
    assert_array_equal(r, np.array([0, 1], dtype=np.int32), strict=True)


def test_q_permutation():
    """Test that the q permutation is correct for a simple case."""
    N = 10
    A = sparse.csc_array((N, N))  # empty array
    _, q, _ = btf(A)
    expect_q = -np.arange(N, dtype=np.int32) - 2
    assert_array_equal(q, expect_q, strict=True)
    assert_array_equal(btf_q_permutation(q), np.abs(q + 1) - 1, strict=True)


@pytest.mark.parametrize(
    "A",
    list(
        generate_random_matrices(
            N_trials=100, N_max=200, d_scale=0.05, square_only=True
        )
    ),
)
class TestRandomSquareMatrices:
    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        p, q, r = btf(A)
        assert p.dtype == itype
        assert p.shape == (A.shape[0],)
        assert is_valid_permutation(p)
        assert is_valid_permutation(btf_q_permutation(q))


# =============================================================================
# =============================================================================
