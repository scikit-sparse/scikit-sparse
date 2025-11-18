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
    p, st = nesdis(empty_A, return_separator=True)
    expect_p = np.array([], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_p, strict=True)


def test_empty_row():
    empty_A = sparse.csc_array((0, 3))
    itype = empty_A.indptr.dtype
    p, st = nesdis(empty_A, kind="row", return_separator=True)
    expect_p = np.array([], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_p, strict=True)


def test_empty_col():
    empty_A = sparse.csc_array((3, 0))
    itype = empty_A.indptr.dtype
    p, st = nesdis(empty_A, kind="col", return_separator=True)
    expect_p = np.array([], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_p, strict=True)


def test_square_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    itype = zero_A.indptr.dtype
    p, st = nesdis(zero_A, return_separator=True)
    expect_p = np.arange(N, dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(N, dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_cmember, strict=True)


def test_nonsquare_zero_input_row():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    itype = zero_A.indptr.dtype
    p, st = nesdis(zero_A, kind="row", return_separator=True)
    expect_p = np.arange(M, dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(M, dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_cmember, strict=True)


def test_nonsquare_zero_input_col():
    M, N = 10, 7
    zero_A = sparse.csc_array((M, N))
    itype = zero_A.indptr.dtype
    p, st = nesdis(zero_A, kind="col", return_separator=True)
    expect_p = np.arange(N, dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(N, dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_cmember, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    itype = singleton_A.indptr.dtype
    p, st = nesdis(singleton_A, return_separator=True)
    expect_p = np.array([0], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_p, strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_bisect_known(davis_example_chol, itype):
    A = davis_example_chol
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    print("\nA_example:")
    print(A.toarray())
    print()
    p, st = nesdis(A, return_separator=True)
    # Computed with MATLAB CHOLMOD nesdis function p = nesdis(A, 'sym')
    expect_p = np.array([1, 4, 6, 8, 0, 3, 5, 2, 9, 10, 7], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(A.shape[0], dtype=itype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(st.cp, expect_cp, strict=True)
    assert_array_equal(st.cmember, expect_cmember, strict=True)


def test_rowcol(davis_example_chol):
    A = davis_example_chol
    s_col = nesdis(A, kind="col", return_separator=True)
    p_col, st_col = s_col
    # Check that the factorization is correct
    itype = A.indptr.dtype
    # Computed with MATLAB CHOLMOD nesdis function p = nesdis(A, 'col')
    expect_p = np.array([8, 3, 5, 0, 6, 10, 9, 7, 4, 2, 1], dtype=itype)
    expect_cp = np.array([-1], dtype=itype)
    expect_cmember = np.zeros(A.shape[0], dtype=itype)
    assert_array_equal(p_col, expect_p, strict=True)
    assert_array_equal(st_col.cp, expect_cp, strict=True)
    assert_array_equal(st_col.cmember, expect_cmember, strict=True)
    # Compare with the factorization of the transpose
    # A is symmetric (A = A.T), so A @ A.T == A.T @ A
    p_row, st_row = nesdis(A.T.tocsc(), kind="row", return_separator=True)
    assert_array_equal(p_row, p_col, strict=True)
    assert_array_equal(st_row.cp, st_col.cp, strict=True)
    assert_array_equal(st_row.cmember, st_col.cmember, strict=True)


# TODO test options nd_small, etc.

# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
pos_def_As = list(
    generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, spd_only=True)
)
general_As = list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))


def _test_kind(A, kind):
    N = A.shape[0]
    p, st = nesdis(A, kind=kind, return_separator=True)
    assert is_valid_permutation(p, N)
    assert len(st.cmember) == N
    assert np.all(st.cmember >= 0)
    assert np.all(st.cmember < N)
    assert len(st.cp) == st.cmember.max() + 1


@pytest.mark.parametrize("A", pos_def_As)
@pytest.mark.parametrize("kind", [None, "sym"])
def test_kind(A, kind):
    _test_kind(A, kind)


@pytest.mark.parametrize("A", general_As)
@pytest.mark.parametrize("kind", ["row", "col"])
def test_rowcol_kind(A, kind):
    _test_kind(A, kind)


# -----------------------------------------------------------------------------
#         Test pruning of the separator tree
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("A", general_As)
def test_prune_septree(A):
    N = A.shape[0]
    p, st = nesdis(A, return_separator=True)
    assert is_valid_permutation(p, N)
    assert len(st.cmember) == N
    assert np.all(st.cmember >= 0)
    assert np.all(st.cmember < N)
    assert len(st.cp) == st.cmember.max() + 1

    st_pruned = st.prune()
    assert st_pruned is not st
    assert len(st_pruned.cmember) == N
    assert np.all(st_pruned.cmember >= 0)
    assert np.all(st_pruned.cmember < N)
    assert len(st_pruned.cp) == st_pruned.cmember.max() + 1


# =============================================================================
# =============================================================================
