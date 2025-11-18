# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_etree.py
#  Created: 2025-08-20 09:07
# =============================================================================

"""Unit tests for the cholmod.etree function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.cholmod import etree

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.fixture
def A_default():
    return sparse.csc_array([[1, 2], [3, 4]])


def test_bad_kind(A_default):
    with pytest.raises(ValueError, match="Unknown factorization kind"):
        etree(A_default, kind="invalid")


@pytest.mark.parametrize("kind", [None, "sym"])
def test_nonsquare_input(kind):
    A = sparse.csc_array((10, 8))
    with pytest.raises(ValueError, match="must be square"):
        etree(A, kind=kind)


def test_empty_defaults():
    empty_A = sparse.csc_array((0, 0))
    parent, post = etree(empty_A, return_post=True)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(parent, empty_p, strict=True)
    assert_array_equal(post, empty_p, strict=True)


def test_empty_row():
    empty_A = sparse.csc_array((0, 3))
    parent, post = etree(empty_A, kind="row", return_post=True)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(parent, empty_p, strict=True)
    assert_array_equal(post, empty_p, strict=True)


def test_empty_col():
    empty_A = sparse.csc_array((3, 0))
    parent, post = etree(empty_A, kind="col", return_post=True)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(parent, empty_p, strict=True)
    assert_array_equal(post, empty_p, strict=True)


def test_square_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    parent, post = etree(zero_A, return_post=True)
    assert_array_equal(parent, np.full(N, -1, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(post, np.arange(N, dtype=zero_A.indptr.dtype), strict=True)


def test_nonsquare_zero_input_row():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    parent, post = etree(zero_A, return_post=True, kind="row")
    assert_array_equal(parent, np.full(M, -1, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(post, np.arange(M, dtype=zero_A.indptr.dtype), strict=True)


def test_nonsquare_zero_input_col():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    parent, post = etree(zero_A, return_post=True, kind="col")
    assert_array_equal(parent, np.full(N, -1, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(post, np.arange(N, dtype=zero_A.indptr.dtype), strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    itype = singleton_A.indptr.dtype
    parent, post = etree(singleton_A, return_post=True)
    assert_array_equal(parent, np.array([-1], dtype=itype), strict=True)
    assert_array_equal(post, np.array([0], dtype=itype), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_etree_known(davis_example_chol, itype):
    A = davis_example_chol
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    print("\nA_example:")
    print(A.toarray())
    print()
    parent, post = etree(A, return_post=True)
    expect_parent = np.array([5, 2, 7, 5, 7, 6, 8, 9, 9, 10, -1], dtype=itype)
    expect_post = np.array([1, 2, 4, 7, 0, 3, 5, 6, 8, 9, 10], dtype=itype)
    assert_array_equal(parent, expect_parent, strict=True)
    assert_array_equal(post, expect_post, strict=True)


def test_symlo(davis_example_chol):
    # kind="lo" is the same as kind="sym" with A.T (lower triangular)
    A = davis_example_chol
    S_sym = etree(A, kind="sym", return_post=True)
    S_lo = etree(A.T.tocsc(), kind="lo", return_post=True)
    for x, y in zip(S_lo, S_sym):
        if sparse.issparse(x):
            assert_array_equal(x.toarray(), y.toarray(), strict=True)
        else:
            assert_array_equal(x, y, strict=True)


def test_rowcol(davis_example_chol):
    A = davis_example_chol
    S_col = etree(A, kind="col", return_post=True)
    # Check that the factorization is correct
    N = A.shape[0]
    itype = A.indptr.dtype
    expect_parent = np.array([3, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1], dtype=itype)
    expect_post = np.arange(N, dtype=itype)
    parent, post = S_col
    assert_array_equal(parent, expect_parent, strict=True)
    assert_array_equal(post, expect_post, strict=True)
    # Compare with the factorization of the transpose
    # A is symmetric (A = A.T), so A @ A.T == A.T @ A
    S_row = etree(A.T.tocsc(), kind="row", return_post=True)
    for x, y in zip(S_row, S_col):
        if sparse.issparse(x):
            assert_array_equal(x.toarray(), x.toarray(), strict=True)
        else:
            assert_array_equal(x, y, strict=True)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
pos_def_As = list(
    generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, spd_only=True)
)
general_As = list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))


def _test_kind(A, kind):
    N = A.shape[0]
    parent, post = etree(A, kind=kind, return_post=True)
    assert len(parent) == N
    assert np.all(parent >= -1)
    assert np.all(parent < N)
    assert len(post) == N
    assert is_valid_permutation(post, N)


@pytest.mark.parametrize("A", pos_def_As)
@pytest.mark.parametrize("kind", [None, "sym"])
def test_kind(A, kind):
    _test_kind(A, kind)


@pytest.mark.parametrize("A", general_As)
@pytest.mark.parametrize("kind", ["row", "col"])
def test_rowcol_kind(A, kind):
    _test_kind(A, kind)
