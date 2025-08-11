# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_cholmod.py
#  Created: 2025-08-12 14:52
# =============================================================================

"""Unit tests for the cholmod module."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

# from scipy.sparse import SparseEfficiencyWarning
from sksparse.cholmod import cholesky

# from .helpers import generate_random_matrices, is_valid_permutation


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    L = cholesky(empty_A)
    assert_array_equal(L.toarray(), empty_A.toarray(), strict=True)


def test_1D_input():
    with pytest.raises(ValueError, match="must be 2D"):
        cholesky(np.arange(10))


def test_nonsquare_input():
    with pytest.raises(ValueError, match="Input must be square"):
        cholesky(sparse.csc_array((3, 4)))


def test_ND_input():
    with pytest.raises(ValueError, match="must be 2D"):
        cholesky(np.empty((2, 3, 4)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    with pytest.raises(ValueError, match="not positive definite"):
        cholesky(zero_A)


@pytest.mark.parametrize(
    # TODO integer dtypes currently lead to a ValueError in CHOLMOD
    # "dtype", [np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_singleton_matrix(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    L = cholesky(singleton_A)
    expect_L = singleton_A.copy()
    assert_array_equal(L.toarray(), expect_L.toarray(), strict=True)


# @pytest.mark.parametrize(
#     "A",
#     list(
#         generate_random_matrices(
#             N_trials=100, N_max=200, d_scale=0.05, square_only=True
#         )
#     ),
# )
# class TestRandomSquareMatrices:
#     @pytest.mark.parametrize("matrix_type", ["dense", "csc", "coo"])
#     def test_input_type(self, A, matrix_type):
#         match matrix_type:
#             case "dense":
#                 A = A.toarray()
#             case "csc":
#                 A = A.tocsc()
#             case "coo":
#                 A = A.tocoo()
#             case _:
#                 raise ValueError(f"Unknown matrix type: {matrix_type}")

#         if matrix_type != "csc":
#             with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
#                 L = cholesky(A)
#         else:
#             L = cholesky(A)

#         assert is_valid_permutation(p)
#         assert is_valid_permutation(btf_q_permutation(q))

#     @pytest.mark.parametrize("itype", [np.int32, np.int64])
#     def test_itype(self, A, itype):
#         A.indptr = A.indptr.astype(itype)
#         A.indices = A.indices.astype(itype)
#         L = cholesky(A)
#         assert p.dtype == itype
#         assert p.shape == (A.shape[0],)
#         assert is_valid_permutation(p)
#         assert is_valid_permutation(btf_q_permutation(q))
