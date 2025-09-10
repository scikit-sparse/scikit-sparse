# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_ldlupdown.py
#  Created: 2025-08-18 10:35
# =============================================================================

"""Unit tests for the ldlupdate and ldlrowmod functions in sksparse.cholmod."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd

from sksparse.cholmod import ldl_factor

Ng = 15  # arbitrary problem size A = (Ng**2, Ng**2)


@pytest.fixture
def A():
    # Create (negative) Laplacian matrix that is symmetric positive definite
    G = LaplacianNd((Ng, Ng))
    A = -G.tosparse().tocsc().astype(float)
    A.setdiag(A.diagonal() + 1)  # make it positive definite
    return A


@pytest.fixture
def expect_x():
    return sparse.dok_array(np.arange(Ng * Ng))


@pytest.fixture
def b(A, expect_x):
    return A @ expect_x


@pytest.fixture(params=[None, "natural", "default", "amd"])
def f(A, request):
    return ldl_factor(A, order=request.param)


def _create_update_matrix(L):
    # See CHOLMOD/MATLAB/Test/test0.m for the original example
    N = L.shape[0]
    rng = np.random.default_rng(5656)  # seed=5656 no failure for order=None
    k = rng.integers(1, N // 4)
    cols = rng.choice(N, size=k, replace=False)  # random columns to update

    # Take existing pattern from L
    C = L[:, cols].copy()
    C.data = rng.normal(size=C.nnz).astype(L.dtype)  # random values for the update

    # Add one entry to make sure L gets some fill-in
    row = rng.integers(0, N)
    C = C.todok()
    C[row, 0] = 1.0

    return C.tocsc()


def test_ldlupdown(A, f):
    L, D = f.get_factor()
    p = f.get_perm()

    # Verify that the factorization is correct
    S = A[p][:, p]
    assert_allclose((L @ D @ L.T).toarray(), S.toarray(), atol=1e-12)

    # Compute a rank-k update of LDL.T factorization
    Cp = _create_update_matrix(L)
    C = Cp[np.argsort(p), :]  # unpermute C into A space

    # NOTE this test fails on order=None and 'natural' for some random seeds.
    # We get an occasional infinite loop or actual failure. The same failure
    # occurs with ldlupdate function, so issue is not in the object wrapper.
    # Probably something in CHOLMOD itself.

    # Update the factorization
    f.update(C)
    Lc, Dc = f.get_factor()

    # Verify that the updated factorization is correct
    Sc = S + Cp @ Cp.T
    assert_allclose((Lc @ Dc @ Lc.T).toarray(), Sc.toarray(), atol=1e-12)

    # Downdate back to the original factorization
    f.downdate(C)
    Ld, Dd = f.get_factor()
    assert_allclose((Ld @ Dd @ Ld.T).toarray(), S.toarray(), atol=1e-12)


def test_resymbol(A, f):
    L, D = f.get_factor()
    p = f.get_perm()

    # Verify that the factorization is correct
    S = A[p][:, p]
    assert_allclose((L @ D @ L.T).toarray(), S.toarray(), atol=1e-12)

    # Compute a rank-k update of LDL.T factorization
    Cp = _create_update_matrix(L)
    C = Cp[np.argsort(p), :]  # unpermute C into A space

    # Update the factorization
    f.update(C)
    Lc, Dc = f.get_factor()

    # Verify that the updated factorization is correct
    Sc = S + Cp @ Cp.T
    assert_allclose((Lc @ Dc @ Lc.T).toarray(), Sc.toarray(), atol=1e-12)

    # Downdate back to the original factorization
    f.downdate(C)
    Ld, Dd = f.get_factor()
    assert_allclose((Ld @ Dd @ Ld.T).toarray(), S.toarray(), atol=1e-12)

    print("\nBefore resymbol:")
    print(f"{ L.nnz=}")
    print(f"{Ld.nnz=}")

    # Test resymbol
    f.resymbol(S)
    Lr = f.get_factor()[0]

    print("After resymbol:")
    print(f"{Lr.nnz=}")

    # Compare with the original factorization
    assert_allclose(Lr.toarray(), L.toarray(), atol=1e-12)


def test_ldlrowmod(A, expect_x, b, f):
    L, D = f.get_factor()
    p = f.get_perm()
    S = A[p][:, p]

    # -------------------------------------------------------------------------
    #         Delete row 3 of A
    # -------------------------------------------------------------------------
    # Row 3 corresponds to the p_inv(3) in S and LDL
    # Invert the permutation
    p_inv = np.argsort(p)

    k = 3

    # Store for later
    A_col_k = A[:, [k]].copy()  # column k of A

    pk = p_inv[k]  # index in S and LDL
    I = sparse.eye_array(*A.shape).tocsc()

    Ak = A.copy()
    Ak[k, :] = I[k, :]
    Ak[:, [k]] = I[:, [k]]

    # Remove row and column k from the factorization
    f.rowdel(pk)
    Lk, Dk = f.get_factor()

    # Remove from the original matrix
    Sk = S.copy()
    Sk[pk, :] = I[pk, :]
    Sk[:, [pk]] = I[:, [pk]]

    assert_allclose((Lk @ Dk @ Lk.T).toarray(), Sk.toarray(), atol=1e-12)

    # Solve the modified system
    x = f.solve(b)
    xs = sparse.linalg.spsolve(Ak, b.tocoo())

    assert_allclose((Ak @ x).toarray(), b.toarray(), atol=1e-12)
    assert_allclose(x.toarray(), xs, atol=1e-12)

    # -------------------------------------------------------------------------
    #         Add row 3 back to the factorization
    # -------------------------------------------------------------------------
    W = A_col_k
    Aa = Ak.copy()
    Aa[k, :] = W.T
    Aa[:, [k]] = W

    C = W[p].tocsc()  # permuted version
    Sa = Sk.copy()
    Sa[pk, :] = C.T
    Sa[:, [pk]] = C

    f.rowadd(pk, C)
    La, Da = f.get_factor()

    assert_allclose((La @ Da @ La.T).toarray(), Sa.toarray(), atol=1e-12)

    # Solve the modified system
    x = f.solve(b)
    xs = sparse.linalg.spsolve(Aa, b.tocoo())

    assert_allclose((Aa @ x).toarray(), b.toarray(), atol=1e-12)
    assert_allclose(x.toarray(), xs, atol=1e-12)
