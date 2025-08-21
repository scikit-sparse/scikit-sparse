# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_nesdis.py
#  Created: 2025-08-21 12:31
# =============================================================================

"""Unit tests for the cholmod.nesdis function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.cholmod import nesdis

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.fixture
def A_default():
    return sparse.csc_array([[1, 2], [3, 4]])


def test_bad_kind(A_default):
    with pytest.raises(ValueError, match="Unknown factorization kind"):
        nesdis(A_default, kind="invalid")


@pytest.mark.parametrize("kind", [None, "sym"])
def test_nonsquare_input(kind):
    A = sparse.csc_array((10, 8))
    with pytest.raises(ValueError, match="must be square"):
        nesdis(A, kind=kind)


def test_empty_defaults():
    empty_A = sparse.csc_array((0, 0))
    itype = empty_A.indptr.dtype
    p, cp, cmember = nesdis(empty_A, return_separator=True)
    expect_p = np.array([], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_p, strict=True)


def test_empty_row():
    empty_A = sparse.csc_array((0, 3))
    itype = empty_A.indptr.dtype
    p, cp, cmember = nesdis(empty_A, kind="row", return_separator=True)
    expect_p = np.array([], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_p, strict=True)


def test_empty_col():
    empty_A = sparse.csc_array((3, 0))
    itype = empty_A.indptr.dtype
    p, cp, cmember = nesdis(empty_A, kind="col", return_separator=True)
    expect_p = np.array([], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_p, strict=True)


def test_square_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    itype = zero_A.indptr.dtype
    p, cp, cmember = nesdis(zero_A, return_separator=True)
    expect_p = np.arange(N, dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(N, dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_cmember, strict=True)


def test_nonsquare_zero_input_row():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    itype = zero_A.indptr.dtype
    p, cp, cmember = nesdis(zero_A, kind="row", return_separator=True)
    expect_p = np.arange(M, dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(M, dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_cmember, strict=True)


def test_nonsquare_zero_input_col():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    itype = zero_A.indptr.dtype
    p, cp, cmember = nesdis(zero_A, kind="col", return_separator=True)
    expect_p = np.arange(N, dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(N, dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_cmember, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    itype = singleton_A.indptr.dtype
    p, cp, cmember = nesdis(singleton_A, return_separator=True)
    expect_p = np.array([0], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_p, strict=True)


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
def test_bisect_known(A_example, itype):
    A = A_example
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    print("\nA_example:")
    print(A.toarray())
    print()
    p, cp, cmember = nesdis(A, return_separator=True)
    # Computed with MATLAB CHOLMOD nesdis function p = nesdis(A, 'sym')
    expect_p = np.array([1, 4, 6, 8, 0, 3, 5, 2, 9, 10, 7], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(A.shape[0], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(cp, expect_cp, strict=True)
    assert_array_equal(cmember, expect_cmember, strict=True)


def test_rowcol(A_example):
    A = A_example
    s_col = nesdis(A, kind="col", return_separator=True)
    p_col, cp_col, cmember_col = s_col
    # Check that the factorization is correct
    itype = A.indptr.dtype
    # Computed with MATLAB CHOLMOD nesdis function p = nesdis(A, 'col')
    expect_p = np.array([8, 3, 5, 0, 6, 10, 9, 7, 4, 2, 1], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(A.shape[0], dtype=itype)
    assert_array_equal(p_col, expect_p, strict=True)
    assert_array_equal(cp_col, expect_cp, strict=True)
    assert_array_equal(cmember_col, expect_cmember, strict=True)
    # Compare with the factorization of the transpose
    # A is symmetric (A = A.T), so A @ A.T == A.T @ A
    s_row = nesdis(A.T.tocsc(), kind="row", return_separator=True)
    for x, y in zip(s_col, s_row):
        assert_array_equal(x, y, strict=True)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "A",
    list(
        generate_random_matrices(
            N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True
        )
    ),
)
@pytest.mark.parametrize("kind", [None, "sym"])
def test_kind(A, kind):
    N = A.shape[0]
    p, cp, cmember = nesdis(A, kind=kind, return_separator=True)
    assert is_valid_permutation(p, N)
    assert len(cmember) == N
    assert np.all(cmember >= 0)
    assert np.all(cmember < N)
    assert len(cp) == cmember.max() + 1


@pytest.mark.parametrize(
    "A",
    list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05)),
)
@pytest.mark.parametrize("kind", ["row", "col"])
def test_rowcol_kind(A, kind):
    N = A.shape[0]
    p, cp, cmember = nesdis(A, kind=kind, return_separator=True)
    assert is_valid_permutation(p, N)
    assert len(cmember) == N
    assert np.all(cmember >= 0)
    assert np.all(cmember < N)
    assert len(cp) == cmember.max() + 1
