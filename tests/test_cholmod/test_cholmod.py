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

from sksparse.cholmod import CholmodError, CholmodNotPositiveDefiniteError, cholmod

from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


class TestBadBShape:
    @pytest.fixture(scope="class")
    def A(self):
        N = 5
        A = sparse.csc_array((N, N))
        return A

    def test_b_0D_dense(self, A):
        b = np.empty([])
        with pytest.raises(ValueError, match="must be a vector or matrix"):
            cholmod(A, b)

    def test_b_3D_dense(self, A):
        b = np.empty((2, 3, 4))
        with pytest.raises(ValueError, match="must be a vector or matrix"):
            cholmod(A, b)

    def test_b_3D_sparse(self, A):
        b = sparse.coo_array((2, 3, 4))
        with pytest.raises(ValueError, match="must be a vector or matrix"):
            cholmod(A, b)

    def test_b_KD_dense(self, A):
        N = A.shape[0]
        b = np.empty((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as A"):
            cholmod(A, b)

    def test_b_KD_sparse(self, A):
        N = A.shape[0]
        b = sparse.csc_array((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as A"):
            cholmod(A, b)


def test_order_and_p():
    N = 10  # arbitrary
    A = sparse.csc_array((N, N))
    b = np.arange(1, N + 1, dtype=A.dtype)
    p = np.arange(N, dtype=A.indptr.dtype)
    with pytest.raises(ValueError, match="one of 'order' or 'p'"):
        cholmod(A, b, order="amd", p=p)


def test_invalid_p():
    N = 10  # arbitrary
    A = sparse.csc_array((N, N))
    b = np.arange(1, N + 1, dtype=A.dtype)
    p = np.arange(N, dtype=A.indptr.dtype)
    p[7] = 3  # duplicate entry
    with pytest.raises(ValueError, match="p is not valid"):
        cholmod(A, b, p=p)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
def test_empty_dense_input(K):
    empty_A = sparse.csc_array((0, 0))
    empty_b = np.empty((0, K))
    x = cholmod(empty_A, empty_b)
    assert_array_equal(x, empty_b, strict=True)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
def test_empty_sparse_input(K):
    empty_A = sparse.csc_array((0, 0))
    empty_b = sparse.csc_array((0, K))
    x = cholmod(empty_A, empty_b)
    assert_array_equal(x.toarray(), empty_b.toarray(), strict=True)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    with pytest.raises(CholmodError, match="is empty"):
        cholmod(zero_A, np.zeros((N,)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = np.array([1], dtype=dtype)
    x = cholmod(singleton_A, b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = sparse.coo_array([1], dtype=dtype)
    x = cholmod(singleton_A, b)
    assert_allclose(x.toarray(), b.toarray())


# Declare a single random matrix fixture for some tests
@pytest.fixture(
    params=list(
        generate_random_matrices(
            N_trials=1, N_max=200, d_scale=0.05, pos_def_only=True
        ),
    )
)
def Arandom(request):
    return request.param


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype_1D(Arandom, itype):
    A = Arandom
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = cholmod(A, b)
    assert isinstance(x, sparse.coo_array)
    assert x.coords[0].dtype == itype


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype_2D(Arandom, itype):
    A = Arandom
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    K = 3  # arbitrary number of rhs
    s = np.arange(1, N + 1, dtype=A.dtype)
    data = np.array([i * s for i in range(1, K + 1)]).T
    expect_x = sparse.csc_array(data, dtype=A.dtype)
    b = A @ expect_x
    x = cholmod(A, b)
    assert isinstance(x, sparse.csc_array)
    assert x.indptr.dtype == itype
    assert x.indices.dtype == itype


def test_exactly_singular(Arandom):
    A = Arandom.todok()
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
    with pytest.raises(CholmodNotPositiveDefiniteError, match="indefinite or singular"):
        cholmod(A, b)


def test_nearly_singular(Arandom):
    A = Arandom.todok()
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
    with pytest.raises(CholmodNotPositiveDefiniteError, match="nearly singular"):
        cholmod(A, b)


def test_manual_permutation(Arandom):
    A = Arandom
    N = A.shape[0]
    p = np.arange(N, dtype=A.indptr.dtype)
    rng = np.random.default_rng(56)  # For reproducibility
    rng.shuffle(p)  # Shuffle the permutation
    print(f"\nManual Permutation: {p}\n")
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = cholmod(A, b, p=p)
    assert_allclose(x.toarray(), expect_x.toarray(), atol=1e-12)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
test_As = [
    A
    # for dtype in DTYPES  # FIXME? single precision dtypes are not close
    for dtype in [np.float64, np.complex128]
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True, dtype=dtype
    )
]


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize(
    "order",
    [
        None,
        "default",
        "best",
        "natural",
        "amd",
        "metis",
        "nesdis",
        "colamd",
        "postordered",
    ],
)
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_ldlsolve(A, order, K, is_sparse):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5

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
    x = cholmod(A, b, order=order)

    # Compare
    if is_sparse:
        assert_allclose(x.toarray(), expect_x.toarray(), atol=atol)
    else:
        assert_allclose(x, expect_x, atol=atol)
