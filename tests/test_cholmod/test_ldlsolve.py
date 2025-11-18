# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_ldlsolve.py
#  Created: 2025-08-15 09:03
# =============================================================================

"""Unit tests for the cholmod.ldlsolve function."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as la
from scipy import sparse

from sksparse.cholmod import CholmodError, CholmodWarning, ldl_factor

from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


class TestBadBShapeDense:
    @pytest.fixture(scope="class")
    def N(self):
        return 5

    @pytest.fixture(scope="class")
    def f(self, N):
        """Create a pair of empty L and D matrices. Only shapes matter."""
        A = sparse.eye_array(N).tocsc()
        return ldl_factor(A)

    def test_b_0D_dense(self, f):
        b = np.empty([])
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(b)

    def test_b_3D_dense(self, f):
        b = np.empty((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(b)

    def test_b_3D_sparse(self, f):
        b = sparse.coo_array((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(b)

    def test_b_KD_dense(self, f, N):
        b = np.empty((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as L"):
            f.solve(b)

    def test_b_KD_sparse(self, f, N):
        b = sparse.csc_array((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as L"):
            f.solve(b)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
def test_empty_dense_input(K):
    empty_A = sparse.csc_array((0, 0))
    empty_b = np.empty((0, K))
    x = ldl_factor(empty_A).solve(empty_b)
    assert_array_equal(x, empty_b, strict=True)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
def test_empty_sparse_input(K):
    empty_A = sparse.csc_array((0, 0))
    empty_b = sparse.csc_array((0, K))
    x = ldl_factor(empty_A).solve(empty_b)
    assert_array_equal(x.toarray(), empty_b.toarray(), strict=True)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    with pytest.raises(CholmodError, match="not positive definite"):
        ldl_factor(zero_A).solve(np.zeros((N,)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = np.array([1], dtype=dtype)
    x = ldl_factor(singleton_A).solve(b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = sparse.coo_array([1], dtype=dtype)
    x = ldl_factor(singleton_A).solve(b)
    assert_allclose(x.toarray(), b.toarray())


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype_1D(davis_example_chol, itype):
    A = davis_example_chol
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = ldl_factor(A).solve(b)
    assert isinstance(x, sparse.coo_array)
    assert x.coords[0].dtype == itype


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype_2D(davis_example_chol, itype):
    A = davis_example_chol
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    K = 3  # arbitrary number of rhs
    s = np.arange(1, N + 1, dtype=A.dtype)
    data = np.array([i * s for i in range(1, K + 1)]).T
    expect_x = sparse.csc_array(data, dtype=A.dtype)
    b = A @ expect_x
    x = ldl_factor(A).solve(b)
    assert isinstance(x, sparse.csc_array)
    assert x.indptr.dtype == itype
    assert x.indices.dtype == itype


# NOTE *exactly* singular matrices are not positive definite, so they fail in
# the ldl() function.
def test_nearly_singular(davis_example_chol):
    A = davis_example_chol.todok()
    N = A.shape[0]
    lam0 = la.eigvalsh(A.toarray()).min()

    # Make A nearly singular
    A[:, -1] = 0.0
    A[-1, :] = 0.0
    A[-1, -1] = 0.5 * np.finfo(A.dtype).eps
    A = A.tocsc()

    lam1 = la.eigvalsh(A.toarray()).min()
    print(f"\nMin eigenvalue: {lam0:.2e} -> {lam1:.2e}\n")

    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    with pytest.warns(CholmodWarning, match="nearly singular"):
        ldl_factor(A).solve(b)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True, dtype=dtype
    )
]


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("order", [None, "amd"])
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_ldlsolve(A, order, K, is_sparse):
    rtol = 1e-08 if A.dtype in (np.float64, np.complex128) else 1e-3
    atol = 1e-15 if A.dtype in (np.float64, np.complex128) else 1e-8

    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # improve conditioning

    # Build RHS
    N = A.shape[0]
    s = np.arange(1, N + 1, dtype=A.dtype)

    if K == 0:
        data = s  # (N,)
    else:
        data = np.array([i * s for i in range(1, K + 1)], dtype=A.dtype).T  # (N, K)

    if is_sparse:
        expect_x = sparse.coo_array(data, dtype=A.dtype)
    else:
        expect_x = np.asarray(data, dtype=A.dtype)

    # Solve the system
    b = A @ expect_x
    x = ldl_factor(A, order=order).solve(b)

    # Compare
    if is_sparse:
        assert_allclose(x.toarray(), expect_x.toarray(), rtol=rtol, atol=atol)
    else:
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)
