# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_bisect.py
#  Created: 2025-08-21 11:15
# =============================================================================

"""Unit tests for the cholmod.bisect function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.cholmod import bisect

from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.fixture
def A_default():
    return sparse.csc_array([[1, 2], [3, 4]])


def test_bad_kind(A_default):
    with pytest.raises(ValueError, match="Unknown factorization kind"):
        bisect(A_default, kind="invalid")


@pytest.mark.parametrize("kind", [None, "sym"])
def test_nonsquare_input(kind):
    A = sparse.csc_array((10, 8))
    with pytest.raises(ValueError, match="must be square"):
        bisect(A, kind=kind)


def test_empty_defaults():
    empty_A = sparse.csc_array((0, 0))
    s = bisect(empty_A)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(s, empty_p, strict=True)


def test_empty_row():
    empty_A = sparse.csc_array((0, 3))
    s = bisect(empty_A, kind="row")
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(s, empty_p, strict=True)


def test_empty_col():
    empty_A = sparse.csc_array((3, 0))
    s = bisect(empty_A, kind="col")
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(s, empty_p, strict=True)


def _expect_s_zero_bisect(N, itype):
    expect_s = np.empty(N, dtype=itype)
    k = N // 2
    expect_s[:k] = 0
    expect_s[k:] = 1
    expect_s[-1] = 2  # last node is the separator
    return expect_s


def test_square_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    s = bisect(zero_A)
    expect_s = _expect_s_zero_bisect(N, zero_A.indptr.dtype)
    assert_array_equal(s, expect_s, strict=True)


def test_nonsquare_zero_input_row():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    s = bisect(zero_A, kind="row")
    expect_s = _expect_s_zero_bisect(M, zero_A.indptr.dtype)
    assert_array_equal(s, expect_s, strict=True)


def test_nonsquare_zero_input_col():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    s = bisect(zero_A, kind="col")
    expect_s = _expect_s_zero_bisect(N, zero_A.indptr.dtype)
    assert_array_equal(s, expect_s, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    itype = singleton_A.indptr.dtype
    s = bisect(singleton_A)
    # The only node *is* the separator
    assert_array_equal(s, np.array([2], dtype=itype), strict=True)


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
    s = bisect(A)
    # Computed with MATLAB CHOLMOD bisect function s = bisect(A, 'sym')
    expect_s = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 2, 2], dtype=itype)
    assert_array_equal(s, expect_s, strict=True)


def test_rowcol(A_example):
    A = A_example
    s_col = bisect(A, kind="col")
    # Check that the factorization is correct
    itype = A.indptr.dtype
    # Computed with MATLAB CHOLMOD bisect function s = bisect(A, 'col')
    expect_s = np.array([2, 1, 1, 2, 1, 2, 1, 1, 0, 2, 1], dtype=itype)
    assert_array_equal(s_col, expect_s, strict=True)
    # Compare with the factorization of the transpose
    # A is symmetric (A = A.T), so A @ A.T == A.T @ A
    s_row = bisect(A.T.tocsc(), kind="row")
    assert_array_equal(s_row, s_col, strict=True)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
pos_def_As = list(
    generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True)
)
general_As = list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))


def _test_kind(A, kind):
    N = A.shape[0]
    s = bisect(A, kind=kind)
    assert len(s) == N
    assert np.isin(s, [0, 1, 2]).all()


@pytest.mark.parametrize("A", pos_def_As)
@pytest.mark.parametrize("kind", [None, "sym"])
def test_kind(A, kind):
    _test_kind(A, kind)


@pytest.mark.parametrize("A", general_As)
@pytest.mark.parametrize("kind", ["row", "col"])
def test_rowcol_kind(A, kind):
    _test_kind(A, kind)
