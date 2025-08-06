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
from scipy.sparse import SparseEfficiencyWarning

from sksparse.btf import btf

from .helpers import generate_random_matrices, is_valid_permutation

# TODO check q outputs

@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    p, q, r = btf(empty_A)
    assert_array_equal(p, np.array([], dtype=itype), strict=True)
    assert_array_equal(r, np.zeros(1, dtype=itype), strict=True)


def test_1D_input():
    with pytest.raises(ValueError, match="must be 2D"):
        btf(np.arange(10))


def test_nonsquare_input():
    with pytest.raises(ValueError, match="Input must be square"):
        btf(sparse.csc_array((3, 4)))


def test_ND_input():
    with pytest.raises(ValueError, match="must be 2D"):
        btf(np.empty((2, 3, 4)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    p, q, r = btf(zero_A)
    expect_r = np.zeros(N + 1, dtype=itype)
    expect_r[-1] = N
    assert_array_equal(p, np.arange(N, dtype=itype), strict=True)
    assert_array_equal(r, expect_r, strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_array([[1]])
    p, q, r = btf(singleton_A)
    assert_array_equal(p, np.array([0], dtype=np.int32), strict=True)
    assert_array_equal(r, np.array([0, 1], dtype=np.int32), strict=True)


@pytest.mark.parametrize(
    "A",
    list(
        generate_random_matrices(
            N_trials=100, N_max=200, d_scale=0.05, square_only=True
        )
    ),
)
class TestRandomSquareMatrices:
    @pytest.mark.parametrize("matrix_type", ["dense", "csc", "coo"])
    def test_input_type(self, A, matrix_type):
        match matrix_type:
            case "dense":
                A = A.toarray()
            case "csc":
                A = A.tocsc()
            case "coo":
                A = A.tocoo()
            case _:
                raise ValueError(f"Unknown matrix type: {matrix_type}")

        if matrix_type != "csc":
            with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
                p, q, r = btf(A)
        else:
            p, q, r = btf(A)

        assert is_valid_permutation(p)
        # TODO check q
        # assert is_valid_permutation(q)

    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        p, q, r = btf(A)
        assert p.dtype == itype
        assert p.shape == (A.shape[0],)
        assert is_valid_permutation(p)
        # TODO check q
        # assert is_valid_permutation(q)


# =============================================================================
# =============================================================================
