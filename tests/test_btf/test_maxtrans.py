# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_btf.py
#  Created: 2025-08-04 21:04
# =============================================================================

"""Test cases for the sksparse.btf.maxtrans function."""

# from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning

from sksparse.btf import maxtrans

from ..helpers import generate_random_matrices, is_valid_permutation


def is_valid_match(p):
    """Check if a maximum matching is valid."""
    if -1 not in p:
        return is_valid_permutation(p)
    else:
        # Check uniqueness of non-negative entries
        x = np.array(p)
        x = np.sort(x[x >= 0])
        all_x_unique = np.all(x[:-1] < x[1:])
        # Check range of all entries [-1, len(p))
        return all((p >= -1) & (p < len(p))) and all_x_unique


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    assert_array_equal(maxtrans(empty_A), np.array([], dtype=itype), strict=True)


def test_1D_input():
    with pytest.raises(ValueError, match="must be 2D"):
        maxtrans(np.arange(10))


def test_ND_input():
    rng = np.random.default_rng(565656)
    with pytest.raises(ValueError, match="must be 2D"):
        maxtrans(rng.random((2, 3, 4)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    M, N = 10, 8  # arbitrary
    zero_A = sparse.csc_array((M, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    assert_array_equal(maxtrans(zero_A), np.full(M, -1, dtype=itype), strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_array([[1]])
    assert_array_equal(
        maxtrans(singleton_A), np.array([0], dtype=np.int32), strict=True
    )


@pytest.mark.parametrize(
    "A",
    list(generate_random_matrices(N_trials=100, N_max=200, d_scale=0.05)),
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
                p = maxtrans(A)
        else:
            p = maxtrans(A)

        assert is_valid_match(p)

    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        p = maxtrans(A)
        assert p.dtype == itype
        assert p.shape == (A.shape[0],)
        assert is_valid_match(p)


# =============================================================================
# =============================================================================
