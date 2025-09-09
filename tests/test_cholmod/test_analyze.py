# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_analyze.py
#  Created: 2025-08-18 21:08
# =============================================================================

"""Unit tests for the cholmod.analyze function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.cholmod import CholeskyFactor, CholmodNotPositiveDefiniteError

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.fixture
def A_default():
    return sparse.csc_array([[1, 2], [3, 4]])


def test_bad_kind(A_default):
    with pytest.raises(ValueError, match="Unknown symmetry kind"):
        CholeskyFactor(A_default, sym_kind="invalid")


def test_bad_supernodal(A_default):
    with pytest.raises(ValueError, match="Unknown factorization mode"):
        CholeskyFactor(A_default, supernodal_mode="invalid")


def test_bad_order(A_default):
    with pytest.raises(ValueError, match="Unknown ordering method"):
        CholeskyFactor(A_default, order="invalid")


def test_empty_input():
    empty_A = sparse.csc_array((0, 0))
    f = CholeskyFactor(empty_A)
    p = f.perm
    count = f.colcount
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(p, empty_p, strict=True)
    assert_array_equal(count, empty_p, strict=True)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    with pytest.raises(CholmodNotPositiveDefiniteError, match="not positive definite"):
        CholeskyFactor(zero_A)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    f = CholeskyFactor(singleton_A)
    p = f.perm
    count = f.colcount
    expect_p = np.array([0], dtype=singleton_A.indptr.dtype)
    expect_count = np.array([1], dtype=singleton_A.indptr.dtype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(count, expect_count, strict=True)


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
def test_itype(A_example, itype):
    A = A_example
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    f = CholeskyFactor(A)
    p = f.perm
    count = f.colcount
    expect_count = np.array([3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1], dtype=itype)
    # expect_count = sum(lchol(A) != 0, 1) in MATLAB (natural ordering)
    assert p.dtype == itype
    assert count.dtype == itype
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)
    assert_array_equal(count, expect_count, strict=True)
    assert f.nnz == 33  # == nnz(lchol(A)) in MATLAB (natural ordering)


ORDERS = [
    None,
    "default",
    "best",
    "natural",
    "amd",
    "metis",
    "nesdis",
    "colamd",
    "postordered",
]


@pytest.mark.parametrize("order", ORDERS)
def test_ordering(A_example, order):
    f = CholeskyFactor(A_example, order=order)
    assert f.order in ORDERS
    if order is None:
        assert f.order == "natural"
    elif order not in ("default", "best"):
        # default and best may return any ordering
        assert f.order == order
    else:
        print(order, f.order)  # still passes, but just for info


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
posdef_As = list(
    generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True)
)


@pytest.mark.parametrize("A", posdef_As)
@pytest.mark.parametrize("supernodal_mode", [None, "auto", "simplicial", "supernodal"])
def test_supernodal_mode(A, supernodal_mode):
    N = A.shape[0]
    f = CholeskyFactor(A, supernodal_mode=supernodal_mode)
    p = f.perm
    count = f.colcount
    if supernodal_mode not in (None, "auto"):
        assert f.is_super == (supernodal_mode == "supernodal")
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)


@pytest.mark.parametrize("A", posdef_As)
@pytest.mark.parametrize("order", ORDERS)
def test_order(A, order):
    N = A.shape[0]
    f = CholeskyFactor(A, order=order)
    p = f.perm
    count = f.colcount
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)


@pytest.mark.parametrize("A", posdef_As)
@pytest.mark.parametrize("sym_kind", [None, "sym"])
def test_kind_sym(A, sym_kind):
    N = A.shape[0]
    f = CholeskyFactor(A, sym_kind=sym_kind)
    p = f.perm
    count = f.colcount
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)


@pytest.mark.parametrize(
    "A", list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))
)
@pytest.mark.parametrize("sym_kind", ["row", "col"])
def test_kind_rowcol(A, sym_kind):
    N = A.shape[0] if sym_kind == "row" else A.shape[1]
    f = CholeskyFactor(A, sym_kind=sym_kind)
    p = f.perm
    count = f.colcount
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)
