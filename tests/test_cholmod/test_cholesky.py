# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_cholmod.py
#  Created: 2025-08-12 14:52
# =============================================================================

"""Unit tests for the cholmod module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sksparse.cholmod import CholmodNotPositiveDefiniteError, cholesky

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    R = cholesky(empty_A)
    assert_array_equal(R.toarray(), empty_A.toarray(), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    with pytest.raises(CholmodNotPositiveDefiniteError, match="not positive definite"):
        cholesky(zero_A)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_matrix(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    L = cholesky(singleton_A, lower=True)
    expect_L = singleton_A.copy()
    assert_array_equal(L.toarray(), expect_L.toarray(), strict=True)


# See: Davis, Timothy A. (2006). Direct Methods for Sparse Linear Systems,
# pp 708 (Equation 2.1).
@pytest.fixture
def noncanonical_A():
    """Return a small non-canonical example matrix from Davis (2006)."""
    N = 11
    rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
    cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])
    rng = np.random.default_rng(565656)
    vals = rng.random(len(rows), dtype=np.float64)
    L = sparse.coo_array((vals, (rows, cols)), shape=(N, N))
    A = (L + L.T).tocsc()  # make it symmetric
    # NOTE As of scipy v1.16.2, sparse.csc_array.setdiag() does not guarantee
    # sorted indices. This line breaks A.has_canonical_format and
    # A.has_sorted_indices! A.has_sorted_indices returns True, but A.indices is
    # NOT sorted!
    A.setdiag(N)  # make it strongly positive definite
    return A


def test_noncanonical_input(noncanonical_A):
    A = noncanonical_A
    expect_unsorted_cols = [0, 1, 2, 3, 4, 5, 6, 7, 9]

    # Show that A is not in canonical format (unsorted indices)
    for p in range(A.shape[1]):
        col_idx = A.indices[A.indptr[p] : A.indptr[p + 1]]
        if not np.all(np.diff(col_idx) > 0):
            assert p in expect_unsorted_cols
            print(f"Column {p} is not sorted: {col_idx}")

    R = cholesky(A)
    assert_allclose((R.T.conj() @ R).toarray(), A.toarray(), atol=1e-12)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype(davis_example_chol, itype):
    A = davis_example_chol
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    R = cholesky(A)
    assert R.indptr.dtype == itype
    assert R.indices.dtype == itype


@pytest.mark.parametrize("dtype", DTYPES)
def test_not_positive_definite(dtype):
    # Create a simple non-positive definite matrix
    A = sparse.eye_array(10, dtype=dtype).todok()
    A[5:, 5:] = 0  # make it not positive definite
    A = A.tocsc()
    with pytest.raises(
        CholmodNotPositiveDefiniteError, match="not positive definite.*column 5"
    ):
        cholesky(A)


test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True, dtype=dtype
    )
]


@pytest.mark.parametrize("A", test_As[:1])
def test_natural_ordering(A):
    _L, p = cholesky(A, order="natural", lower=True)
    assert_array_equal(p, np.arange(A.shape[0]))


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
        L = cholesky(A, order=order, lower=True)
        assert_allclose((L @ L.T.conj()).toarray(), A.toarray(), atol=atol)
    else:
        L, p = cholesky(A, order=order, lower=True)
        assert is_valid_permutation(p)
        PAPT = A[p][:, p]
        assert_allclose((L @ L.T.conj()).toarray(), PAPT.toarray(), atol=atol)


@pytest.mark.parametrize("A", test_As)
def test_lower(A):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    R = cholesky(A)
    L = cholesky(A, lower=True)
    assert_allclose(R.T.conj().toarray(), L.toarray(), atol=atol)


@pytest.mark.parametrize("beta", [0.0, 1.0, 3.4])
@pytest.mark.parametrize("order", [None, "amd"])
@pytest.mark.parametrize("A", test_As)
def test_beta(A, beta, order):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    N = A.shape[0]

    if order is None:
        L = cholesky(A, beta, lower=True)
        expect_LL = (A + beta * sparse.eye_array(N)).toarray()
    else:
        L, p = cholesky(A, beta, lower=True, order=order)
        expect_LL = (A[p][:, p] + beta * sparse.eye_array(N)).toarray()

    assert_allclose((L @ L.T.conj()).toarray(), expect_LL, atol=atol)
