# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_btf_strongcomp.py
#  Created: 2025-08-06 15:11
# =============================================================================

"""Test cases for the sksparse.btf.strongcomp function."""

# from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sksparse.btf import strongcomp

from ..helpers import generate_random_matrices, is_valid_permutation


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    p, r = strongcomp(empty_A)
    assert_array_equal(p, np.array([], dtype=itype), strict=True)
    assert_array_equal(r, np.zeros(1, dtype=itype), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    p, r = strongcomp(zero_A)
    expect_r = np.zeros(N + 1, dtype=itype)
    expect_r[-1] = N
    assert_array_equal(p, np.arange(N, dtype=itype), strict=True)
    assert_array_equal(r, expect_r, strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_array([[1]])
    p, r = strongcomp(singleton_A)
    assert_array_equal(p, np.array([0], dtype=np.int32), strict=True)
    assert_array_equal(r, np.array([0, 1], dtype=np.int32), strict=True)


@pytest.mark.parametrize(
    "A",
    list(
        generate_random_matrices(
            N_trials=100, N_max=200, d_scale=0.05, shape_kind="square"
        )
    ),
)
class TestRandomSquareMatrices:
    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        p, r = strongcomp(A)
        assert p.dtype == itype
        assert p.shape == (A.shape[0],)
        assert is_valid_permutation(p)


def test_column_permutation():
    rng = np.random.default_rng(565656)
    N = 100
    A = sparse.random_array((N, N), density=0.2, format="csc", rng=rng)
    qin = rng.permutation(N)
    AQ = A[:, qin].tocsc()
    p_, r_ = strongcomp(AQ)
    p, q, r = strongcomp(A, qin)
    assert_array_equal(q, qin[p_])
    assert_allclose(A[p][:, q].toarray(), AQ[p_][:, p_].toarray(), atol=1e-15)


# =============================================================================
# =============================================================================
