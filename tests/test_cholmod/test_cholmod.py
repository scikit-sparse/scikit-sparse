# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_cholmod.py
#  Created: 2025-08-15 23:03
# =============================================================================

"""Unit tests for the cholmod.cholmod function."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as la
from scipy import sparse

from sksparse.cholmod import (
    CholmodError,
    CholmodNotPositiveDefiniteError,
    CholmodWarning,
    cho_factor,
    cho_solve,
)

from ..helpers import generate_random_matrices, load_problem

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


class TestBadBShape:
    @pytest.fixture(scope="class")
    def N(self):
        return 5

    @pytest.fixture(scope="class")
    def f(self, N):
        A = sparse.eye_array(N).tocsc()
        return cho_factor(A)

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
    x = cho_factor(empty_A).solve(empty_b)
    assert_array_equal(x, empty_b, strict=True)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
def test_empty_sparse_input(K):
    empty_A = sparse.csc_array((0, 0))
    empty_b = sparse.csc_array((0, K))
    x = cho_factor(empty_A).solve(empty_b)
    assert_array_equal(x.toarray(), empty_b.toarray(), strict=True)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    with pytest.raises(CholmodError, match="not positive definite"):
        cho_factor(zero_A).solve(np.zeros((N,)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = np.array([1], dtype=dtype)
    x = cho_factor(singleton_A).solve(b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = sparse.coo_array([1], dtype=dtype)
    x = cho_factor(singleton_A).solve(b)
    assert_allclose(x.toarray(), b.toarray())


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype_1D(davis_example_chol, itype):
    A = davis_example_chol
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = cho_factor(A).solve(b)
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
    x = cho_factor(A).solve(b)
    assert isinstance(x, sparse.csc_array)
    assert x.indptr.dtype == itype
    assert x.indices.dtype == itype


def test_exactly_singular(davis_example_chol):
    A = davis_example_chol.todok()
    N = A.shape[0]
    lam0 = la.eigvalsh(A.toarray()).min()

    # Make A exactly singular
    A[:, -1] = 0.0
    A[-1, :] = 0.0
    A = A.tocsc()

    lam1 = la.eigvalsh(A.toarray()).min()
    print(f"\nMin eigenvalue: {lam0:.2e} -> {lam1:.2e}\n")

    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    with pytest.raises(CholmodNotPositiveDefiniteError, match="not positive definite"):
        cho_factor(A).solve(b)


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
    f = cho_factor(A)
    assert f.rcond < np.finfo(A.dtype).eps
    with pytest.warns(CholmodWarning, match="nearly singular"):
        f.solve(b)


class TestRHSCasting:
    @pytest.fixture
    def example_system(self, davis_example_chol):
        A = davis_example_chol.astype(np.float32)
        N = A.shape[0]
        f = cho_factor(A)
        expect_x = np.arange(1, N + 1, dtype=A.dtype)
        b = A @ expect_x
        return f, expect_x, b

    def test_upcast_rhs(self, example_system):
        f, expect_x, b = example_system
        x = f.solve(b.astype(np.float16))
        assert_allclose(x, expect_x, strict=True, rtol=1e-3)

    def test_downcast_rhs(self, example_system):
        f, expect_x, b = example_system
        with pytest.raises(TypeError, match="Cannot safely cast"):
            f.solve(b.astype(np.float64))


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=5, N_max=200, d_scale=0.05, spd_only=True, dtype=dtype
    )
]


orders = [
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


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_solve(A, order, K, is_sparse):
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
    x = cho_factor(A, order=order).solve(b)
    xs = cho_solve(A, b, order=order)

    # Compare
    if is_sparse:
        assert_allclose(x.toarray(), expect_x.toarray(), rtol=rtol, atol=atol)
        assert_allclose(xs.toarray(), expect_x.toarray(), rtol=rtol, atol=atol)
    else:
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)
        assert_allclose(xs, expect_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("order", [None, "default"])
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
class TestSolveSystems:
    @staticmethod
    @pytest.fixture
    def factor_system(davis_example_chol, order, is_sparse):
        A = davis_example_chol.copy()
        A.setdiag(A.diagonal() + 1.0)  # improve conditioning

        rtol = 1e-08 if A.dtype in (np.float64, np.complex128) else 1e-3
        atol = 1e-15 if A.dtype in (np.float64, np.complex128) else 1e-8

        # Build RHS
        N = A.shape[0]
        expect_x = np.arange(1, N + 1, dtype=A.dtype)

        if is_sparse:
            expect_x = sparse.coo_array(expect_x, dtype=A.dtype)

        f = cho_factor(A, order=order)

        return f, expect_x, rtol, atol

    def test_system_LDLt(self, factor_system):
        f, expect_x, rtol, atol = factor_system
        L, D = f.L, f.D
        x = f.solve(L @ D @ L.T.conj() @ expect_x, system="LDLt")
        if sparse.issparse(expect_x):
            x = x.toarray()
            expect_x = expect_x.toarray()
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)

    def test_system_LD(self, factor_system):
        f, expect_x, rtol, atol = factor_system
        L, D = f.L, f.D
        x = f.solve(L @ D @ expect_x, system="LD")
        if sparse.issparse(expect_x):
            x = x.toarray()
            expect_x = expect_x.toarray()
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)

    def test_system_DLt(self, factor_system):
        f, expect_x, rtol, atol = factor_system
        L, D = f.L, f.D
        x = f.solve(D @ L.T.conj() @ expect_x, system="DLt")
        if sparse.issparse(expect_x):
            x = x.toarray()
            expect_x = expect_x.toarray()
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)

    def test_system_L(self, factor_system):
        f, expect_x, rtol, atol = factor_system
        L = f.L
        x = f.solve(L @ expect_x, system="L")
        if sparse.issparse(expect_x):
            x = x.toarray()
            expect_x = expect_x.toarray()
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)

    def test_system_Lt(self, factor_system):
        f, expect_x, rtol, atol = factor_system
        Lt = f.L.T
        x = f.solve(Lt @ expect_x, system="Lt")
        if sparse.issparse(expect_x):
            x = x.toarray()
            expect_x = expect_x.toarray()
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)

    def test_system_D(self, factor_system):
        f, expect_x, rtol, atol = factor_system
        D = f.D
        x = f.solve(D @ expect_x, system="D")
        if sparse.issparse(expect_x):
            x = x.toarray()
            expect_x = expect_x.toarray()
        assert_allclose(x, expect_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("problem", ["well1033", "illc1033", "well1850", "illc1850"])
def test_solve_real(problem):
    A, b = load_problem(problem)
    # Solve the normal equations A^T A x = A^T b
    ATA = (A.T @ A).tocsc()
    ATb = A.T @ b
    atol = 1e-7 if A.dtype in (np.float64, np.complex128) else 1e-3
    expect_x = np.linalg.lstsq(A.toarray(), b)[0]
    f = cho_factor(ATA)
    assert_allclose(f.solve(ATb), expect_x, atol=atol)
