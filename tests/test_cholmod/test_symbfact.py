# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_symbfact.py
#  Created: 2025-08-19 12:36
# =============================================================================

"""Unit tests for the cholmod.symbfact function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.cholmod import symbfact

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.fixture
def A_default():
    return sparse.csc_array([[1, 2], [3, 4]])


def test_bad_kind(A_default):
    with pytest.raises(ValueError, match="Unknown factorization kind"):
        symbfact(A_default, kind="invalid")


@pytest.mark.parametrize("kind", [None, "sym"])
def test_nonsquare_input(kind):
    A = sparse.csc_array((10, 8))
    with pytest.raises(ValueError, match="must be square"):
        symbfact(A, kind=kind)


def test_empty_defaults():
    empty_A = sparse.csc_array((0, 0))
    count, h, parent, post, L = symbfact(empty_A, return_factor=True)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(count, empty_p, strict=True)
    assert h == 0
    assert_array_equal(parent, empty_p, strict=True)
    assert_array_equal(post, empty_p, strict=True)
    assert_array_equal(L.toarray(), empty_A.toarray(), strict=True)


def test_empty_row():
    empty_A = sparse.csc_array((0, 3))
    count, h, parent, post, L = symbfact(empty_A, kind="row", return_factor=True)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(count, empty_p, strict=True)
    assert h == 0
    assert_array_equal(parent, empty_p, strict=True)
    assert_array_equal(post, empty_p, strict=True)
    assert_array_equal(L.toarray(), empty_A.toarray(), strict=True)


def test_empty_col():
    empty_A = sparse.csc_array((3, 0))
    count, h, parent, post, L = symbfact(empty_A, kind="col", return_factor=True)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(count, empty_p, strict=True)
    assert h == 0
    assert_array_equal(parent, empty_p, strict=True)
    assert_array_equal(post, empty_p, strict=True)
    assert_array_equal(L.toarray(), empty_A.toarray(), strict=True)


def test_square_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    count, h, parent, post, L = symbfact(zero_A, return_factor=True)
    assert_array_equal(count, np.zeros(N, dtype=zero_A.indptr.dtype), strict=True)
    assert h == 1
    assert_array_equal(parent, np.full(N, -1, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(post, np.arange(N, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(L.toarray(), np.eye(N, dtype=zero_A.dtype), strict=True)


def test_nonsquare_zero_input_row():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    count, h, parent, post, L = symbfact(zero_A, return_factor=True, kind="row")
    assert_array_equal(count, np.zeros(M, dtype=zero_A.indptr.dtype), strict=True)
    assert h == 1
    assert_array_equal(parent, np.full(M, -1, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(post, np.arange(M, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(L.toarray(), np.eye(M, dtype=zero_A.dtype), strict=True)


def test_nonsquare_zero_input_col():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    count, h, parent, post, L = symbfact(zero_A, return_factor=True, kind="col")
    assert_array_equal(count, np.zeros(N, dtype=zero_A.indptr.dtype), strict=True)
    assert h == 1
    assert_array_equal(parent, np.full(N, -1, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(post, np.arange(N, dtype=zero_A.indptr.dtype), strict=True)
    assert_array_equal(L.toarray(), np.eye(N, dtype=zero_A.dtype), strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    itype = singleton_A.indptr.dtype
    count, h, parent, post, L = symbfact(singleton_A, return_factor=True)
    assert_array_equal(count, np.array([1], dtype=itype), strict=True)
    assert h == 1
    assert_array_equal(parent, np.array([-1], dtype=itype), strict=True)
    assert_array_equal(post, np.array([0], dtype=itype), strict=True)
    assert_array_equal(L.toarray(), np.array([[1]], dtype=bool), strict=True)


# Declare a single matrix fixture for some tests
# See: Davis, Timothy A. (2006). Direct Methods for Sparse Linear Systems,
# pp 708 (Equation 2.1).
@pytest.fixture
def A_example():
    N = 11
    rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    vals = np.ones(len(rows), dtype=np.float64)
    L = sparse.coo_array((vals, (rows, cols)), shape=(N, N))
    A = (L + L.T).tocsc()  # make it symmetric
    A.setdiag(1)
    return A


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_symbfact_known(A_example, itype):
    A = A_example
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    print("\nA_example:")
    print(A.toarray())
    print()
    count, h, parent, post, L = symbfact(A, lower=True, return_factor=True)
    expect_count = np.array([3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1], dtype=itype)
    expect_h = 6
    expect_parent = np.array([5, 2, 7, 5, 7, 6, 8, 9, 9, 10, -1], dtype=itype)
    expect_post = np.array([1, 2, 4, 7, 0, 3, 5, 6, 8, 9, 10], dtype=itype)
    # Check results
    assert_array_equal(count, expect_count, strict=True)
    assert h == expect_h
    assert_array_equal(parent, expect_parent, strict=True)
    assert_array_equal(post, expect_post, strict=True)
    # Check self-consistency of L
    colcount = L.sum(axis=0)
    indptr = np.cumulative_sum(colcount, include_initial=True, dtype=itype)
    assert_array_equal(colcount, expect_count)
    assert_array_equal(L.indptr, indptr, strict=True)


def test_symlo(A_example):
    # kind="lo" is the same as kind="sym" with A.T (lower triangular)
    A = A_example
    S_sym = symbfact(A, kind="sym", return_factor=True)
    S_lo = symbfact(A.T.tocsc(), kind="lo", return_factor=True)
    for x, y in zip(S_lo, S_sym):
        if sparse.issparse(x):
            assert_array_equal(x.toarray(), y.toarray(), strict=True)
        else:
            assert_array_equal(x, y, strict=True)


def test_rowcol(A_example):
    A = A_example
    S_col = symbfact(A, kind="col", return_factor=True)
    # Check that the factorization is correct
    N = A.shape[0]
    itype = A.indptr.dtype
    expect_count = np.array([7, 6, 8, 8, 7, 6, 5, 4, 3, 2, 1], dtype=itype)
    expect_h = 10
    expect_parent = np.array([3, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1], dtype=itype)
    expect_post = np.arange(N, dtype=itype)
    count, h, parent, post, L = S_col
    assert_array_equal(count, expect_count, strict=True)
    assert h == expect_h
    assert_array_equal(parent, expect_parent, strict=True)
    assert_array_equal(post, expect_post, strict=True)
    # Compare with the factorization of the transpose
    # A is symmetric (A = A.T), so A @ A.T == A.T @ A
    S_row = symbfact(A.T.tocsc(), kind="row", return_factor=True)
    for x, y in zip(S_row, S_col):
        if sparse.issparse(x):
            assert_array_equal(x.toarray(), x.toarray(), strict=True)
        else:
            assert_array_equal(x, y, strict=True)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
pos_def_As = list(
    generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True)
)
general_As = list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))


def _test_kind(A, kind):
    N = A.shape[0]
    count, h, parent, post, L = symbfact(A, kind=kind, return_factor=True)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)
    assert h >= 1
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
