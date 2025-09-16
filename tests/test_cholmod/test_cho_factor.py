# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_cho_factor.py
#  Created: 2025-09-04 19:32
# =============================================================================

"""Unit tests for the CholeskyFactor object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from sksparse.cholmod import cho_factor

from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


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
    A.setdiag(N)
    return A


def test_convert_factor(A_example):
    A = A_example
    f = cho_factor(A, lower=True)
    L, D = f.get_factor(kind="LDL")
    assert_allclose((L @ D @ L.T.conj()).toarray(), A.toarray(), atol=1e-15)


@pytest.mark.parametrize("order", [None, "amd"])
def test_view_vs_get(A_example, order):
    A = A_example
    f = cho_factor(A, lower=True, order=order)
    Lv = f.factor
    pv = f.perm
    L = f.get_factor()
    p = f.get_perm()
    assert Lv is not L  # different objects
    assert pv is not p
    assert_allclose(Lv.toarray(), L.toarray(), atol=1e-15)
    assert_allclose(pv, p, atol=1e-15)


@pytest.fixture
def A_small():
    return sparse.csc_array(
        np.array(
            [[10,  0, 3,  0],
              [0,  5, 0, -2],
              [3,  0, 5,  0],
              [0, -2, 0,  2]]
        ),
        dtype=np.float64,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_determinant(A_small, dtype):
    A = A_small.astype(dtype)
    rtol = 1e-7 if A.dtype in (np.float64, np.complex128) else 1e-6

    f = cho_factor(A, lower=True)

    if A.dtype in (np.complex64, np.complex128):
        with pytest.warns(RuntimeWarning, match="(divide by zero|invalid value)"):
            expect_det = np.linalg.det(A.toarray())
            expect_sign, expect_logdet = np.linalg.slogdet(A.toarray())
    else:
        expect_det = np.linalg.det(A.toarray())
        expect_sign, expect_logdet = np.linalg.slogdet(A.toarray())

    assert_allclose(f.det(), expect_det, rtol=rtol, strict=True)
    assert_allclose(f.slogdet(), (expect_sign, expect_logdet), rtol=rtol, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inv(A_small, dtype):
    atol = 1e-12 if dtype in (np.float64, np.complex128) else 1e-3
    A = A_small.astype(dtype)
    f = cho_factor(A)
    Ainv = f.inv()
    I = np.eye(A.shape[0], dtype=A.dtype)
    assert_allclose((A @ Ainv).toarray(), I, atol=atol, strict=True)


test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True, dtype=dtype
    )
]


def _create_randomized_matrix(A):
    """Create a new matrix with the same sparsity pattern as A but different values."""
    Bl = sparse.tril(A, -1).copy()
    rng = np.random.default_rng(56)
    if np.issubdtype(A.dtype, np.complexfloating):
        Bl.data = rng.random(Bl.nnz, dtype=A.real.dtype) + 1j * rng.random(
            Bl.nnz, dtype=A.real.dtype
        )
    else:
        Bl.data = rng.random(Bl.nnz, dtype=A.dtype)
    B = Bl + Bl.T.conj()
    # Ensure positive definiteness by adding to the diagonal
    B.setdiag(A.diagonal())
    B += sparse.diags_array(np.full(B.shape[0], B.shape[0], dtype=B.dtype))
    return B.tocsc()


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("A", test_As)
def test_refactor(A, copy):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-3
    f = cho_factor(A, lower=True)
    L = f.get_factor()
    assert_allclose((L @ L.T.conj()).toarray(), A.toarray(), atol=atol)
    # Create a new matrix with the same sparsity pattern but different values
    B = _create_randomized_matrix(A)
    # Factor the new matrix with the same sparsity pattern
    if copy:
        # Use a copy of the factorization object to ensure that we are taking
        # the relevant parameters from the underlying cholmod_common object.
        g = f.copy()
        g.factorize(B)
        Lb = g.get_factor()
    else:
        f.factorize(B)
        Lb = f.get_factor()
    assert_allclose((Lb @ Lb.T.conj()).toarray(), B.toarray(), atol=atol)
