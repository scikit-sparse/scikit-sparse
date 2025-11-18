# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_metis.py
#  Created: 2025-08-21 13:27
# =============================================================================

"""Unit tests for the cholmod.metis function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.cholmod import metis

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.fixture
def A_default():
    return sparse.csc_array([[1, 2], [3, 4]])


def test_bad_kind(A_default):
    with pytest.raises(ValueError, match="Unknown factorization kind"):
        metis(A_default, kind="invalid")


@pytest.mark.parametrize("kind", [None, "sym"])
def test_nonsquare_input(kind):
    A = sparse.csc_array((10, 8))
    with pytest.raises(ValueError, match="must be square"):
        metis(A, kind=kind)


def test_empty_defaults():
    empty_A = sparse.csc_array((0, 0))
    p = metis(empty_A)
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(p, empty_p, strict=True)


def test_empty_row():
    empty_A = sparse.csc_array((0, 3))
    p = metis(empty_A, kind="row")
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(p, empty_p, strict=True)


def test_empty_col():
    empty_A = sparse.csc_array((3, 0))
    p = metis(empty_A, kind="col")
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(p, empty_p, strict=True)


def test_square_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    p = metis(zero_A)
    expect_p = np.arange(N, dtype=zero_A.indptr.dtype)
    assert_array_equal(p, expect_p, strict=True)


def test_nonsquare_zero_input_row():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    p = metis(zero_A, kind="row")
    expect_p = np.arange(M, dtype=zero_A.indptr.dtype)
    assert_array_equal(p, expect_p, strict=True)


def test_nonsquare_zero_input_col():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    p = metis(zero_A, kind="col")
    expect_p = np.arange(N, dtype=zero_A.indptr.dtype)
    assert_array_equal(p, expect_p, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    itype = singleton_A.indptr.dtype
    p = metis(singleton_A)
    assert_array_equal(p, np.array([0], dtype=itype), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_metis_known(davis_example_chol, itype):
    A = davis_example_chol
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    print("\nA_example:")
    print(A.toarray())
    print()
    p = metis(A)
    # Computed with MATLAB CHOLMOD metis function p = metis(A, 'sym')
    expect_p = np.array([8, 3, 6, 0, 5, 2, 4, 7, 1, 9, 10], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)


def test_rowcol(davis_example_chol):
    A = davis_example_chol
    s_col = metis(A, kind="col")
    # Check that the factorization is correct
    itype = A.indptr.dtype
    # Computed with MATLAB CHOLMOD metis function p = metis(A, 'col')
    expect_p = np.array([6, 1, 4, 2, 7, 10, 8, 3, 0, 5, 9], dtype=itype)
    assert_array_equal(s_col, expect_p, strict=True)
    # Compare with the factorization of the transpose
    # A is symmetric (A = A.T), so A @ A.T == A.T @ A
    s_row = metis(A.T.tocsc(), kind="row")
    assert_array_equal(s_row, s_col, strict=True)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
pos_def_As = list(
    generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, spd_only=True)
)
general_As = list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))


def _test_kind(A, kind):
    N = A.shape[0]
    p = metis(A, kind=kind)
    assert is_valid_permutation(p, N)


@pytest.mark.parametrize("A", pos_def_As)
@pytest.mark.parametrize("kind", [None, "sym"])
def test_kind(A, kind):
    _test_kind(A, kind)


@pytest.mark.parametrize("A", general_As)
@pytest.mark.parametrize("kind", ["row", "col"])
def test_rowcol_kind(A, kind):
    _test_kind(A, kind)
