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

from sksparse.cholmod import CholmodError, ldl, ldlsolve

from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]

# TODO: Add tests for the `ldlsolve` function, including:
# * Permuted systems (need to manually permute b, x for now)
# * *nearly* singular matrices (check for rcond errors)


def test_bad_D_shape():
    N = 5
    L = sparse.csc_array((N, N))
    D = sparse.csc_array((N - 1, N))
    with pytest.raises(ValueError, match="must match"):
        ldlsolve(L, D, np.zeros((N,)))


class TestBadBShapeDense:
    @pytest.fixture(scope="class")
    def LD_matrices(self):
        """Create a pair of empty L and D matrices. Only shapes matter."""
        N = 5
        L = sparse.csc_array((N, N))
        D = L.copy()
        return L, D

    def test_b_0D_dense(self, LD_matrices):
        L, D = LD_matrices
        b = np.empty([])
        with pytest.raises(ValueError, match="must be a vector or matrix"):
            ldlsolve(L, D, b)

    def test_b_3D_dense(self, LD_matrices):
        L, D = LD_matrices
        b = np.empty((2, 3, 4))
        with pytest.raises(ValueError, match="must be a vector or matrix"):
            ldlsolve(L, D, b)

    def test_b_3D_sparse(self, LD_matrices):
        L, D = LD_matrices
        b = sparse.coo_array((2, 3, 4))
        with pytest.raises(ValueError, match="must be a vector or matrix"):
            ldlsolve(L, D, b)

    def test_b_KD_dense(self, LD_matrices):
        L, D = LD_matrices
        N = L.shape[0]
        b = np.empty((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as L"):
            ldlsolve(L, D, b)

    def test_b_KD_sparse(self, LD_matrices):
        L, D = LD_matrices
        N = L.shape[0]
        b = sparse.csc_array((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as L"):
            ldlsolve(L, D, b)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
def test_empty_dense_input(K):
    empty_A = sparse.csc_array((0, 0))
    empty_b = np.empty((0, K))
    L, D = ldl(empty_A)
    x = ldlsolve(L, D, empty_b)
    assert_array_equal(x, empty_b, strict=True)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
def test_empty_sparse_input(K):
    empty_A = sparse.csc_array((0, 0))
    empty_b = sparse.csc_array((0, K))
    L, D = ldl(empty_A)
    x = ldlsolve(L, D, empty_b)
    assert_array_equal(x.toarray(), empty_b.toarray(), strict=True)


def test_zero_input():
    N = 10  # arbitrary
    zero_LD = sparse.csc_array((N, N))
    with pytest.raises(CholmodError, match="is empty"):
        ldlsolve(zero_LD, zero_LD, np.zeros((N,)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    L, D = ldl(singleton_A)
    b = np.array([1], dtype=dtype)
    x = ldlsolve(L, D, b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    L, D = ldl(singleton_A)
    b = sparse.coo_array([1], dtype=dtype)
    x = ldlsolve(L, D, b)
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
    L, D = ldl(A)
    assert L.indptr.dtype == itype
    assert L.indices.dtype == itype
    N = A.shape[0]
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = ldlsolve(L, D, b)
    assert isinstance(x, sparse.coo_array)
    assert x.coords[0].dtype == itype


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype_2D(Arandom, itype):
    A = Arandom
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    L, D = ldl(A)
    assert L.indptr.dtype == itype
    assert L.indices.dtype == itype
    N = A.shape[0]
    K = 3  # arbitrary number of rhs
    s = np.arange(1, N + 1, dtype=A.dtype)
    data = np.array([i * s for i in range(1, K + 1)]).T
    expect_x = sparse.csc_array(data, dtype=A.dtype)
    b = A @ expect_x
    x = ldlsolve(L, D, b)
    assert isinstance(x, sparse.csc_array)
    assert x.indptr.dtype == itype
    assert x.indices.dtype == itype


# NOTE *exactly* singular matrices are not positive definite, so they fail in
# the ldl() function.
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

    L, D = ldl(A)
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    with pytest.raises(CholmodError, match="nearly singular"):
        ldlsolve(L, D, b)


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


@pytest.fixture(params=test_As)
def Am(request):
    return request.param


@pytest.fixture(params=[None, "amd"], ids=lambda x: f"order={x}")
def ldl_decomp(Am, request):
    order = request.param
    if order is None:
        L, D = ldl(Am)
        p = None
    else:
        L, D, p = ldl(Am, order=order)
    return Am, L, D, p, order


@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_ldlsolve(ldl_decomp, K, is_sparse):
    A, L, D, p, order = ldl_decomp
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
    x = ldlsolve(L, D, b, p)

    # Compare
    if is_sparse:
        assert_allclose(x.toarray(), expect_x.toarray(), atol=atol)
    else:
        assert_allclose(x, expect_x, atol=atol)
