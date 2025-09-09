# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_ldl.py
#  Created: 2025-08-14 14:12
# =============================================================================

"""Unit tests for the cholmod.ldl function."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sksparse.cholmod import CholmodNotPositiveDefiniteError, ldl

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    L, D = ldl(empty_A)
    assert_array_equal(L.toarray(), empty_A.toarray(), strict=True)
    assert_array_equal(D.toarray(), empty_A.toarray(), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    with pytest.raises(CholmodNotPositiveDefiniteError, match="not positive definite"):
        ldl(zero_A)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_matrix(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    L, D = ldl(singleton_A)
    expect_L = expect_D = singleton_A.copy()
    assert_array_equal(L.toarray(), expect_L.toarray(), strict=True)
    assert_array_equal(D.toarray(), expect_D.toarray(), strict=True)


@pytest.mark.parametrize(
    "A",
    generate_random_matrices(N_trials=1, N_max=200, d_scale=0.05, pos_def_only=True),
)
@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype(A, itype):
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    L, _ = ldl(A)
    assert L.indptr.dtype == itype
    assert L.indices.dtype == itype


@pytest.mark.parametrize("dtype", DTYPES)
def test_not_positive_definite(dtype):
    # Create a simple non-positive definite matrix
    A = sparse.csc_array([[0, 2], [2, 1]], dtype=dtype)
    with pytest.raises(CholmodNotPositiveDefiniteError):
        ldl(A)


test_As = [
    A
    for dtype in DTYPES
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
def test_ordering(A, order):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    if order is None:
        L, D = ldl(A, order=order)
        assert_allclose((L @ D @ L.T.conj()).toarray(), A.toarray(), atol=atol)
    else:
        L, D, p = ldl(A, order=order)
        assert is_valid_permutation(p)
        PAPT = A[p][:, p]
        assert_allclose((L @ D @ L.T.conj()).toarray(), PAPT.toarray(), atol=atol)


@pytest.mark.parametrize("A", test_As)
def test_lower(A):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    R, Dr = ldl(A, lower=False)
    L, Dl = ldl(A)
    assert_allclose(R.T.conj().toarray(), L.toarray(), atol=atol)
    assert_allclose(Dr.toarray(), Dl.toarray(), atol=atol)


@pytest.mark.parametrize("A", test_As)
def test_beta(A):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-4
    N = A.shape[0]
    beta = 17.0  # arbitrary positive value
    L, D = ldl(A, beta)
    expect_LDL = (A @ A.T.conj() + beta * sparse.eye_array(N)).toarray()
    assert_allclose((L @ D @ L.T.conj()).toarray(), expect_LDL, atol=atol)
