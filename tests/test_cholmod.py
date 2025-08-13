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

from .helpers import generate_random_matrices, is_valid_permutation

# TODO integer dtypes currently lead to a ValueError
# DTYPES + [np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
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


@pytest.mark.parametrize(
    "A",
    generate_random_matrices(N_trials=1, N_max=200, d_scale=0.05, pos_def_only=True),
)
@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype(A, itype):
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    R = cholesky(A)
    assert R.indptr.dtype == itype
    assert R.indices.dtype == itype


@pytest.mark.parametrize("dtype", DTYPES)
def test_not_positive_definite(dtype):
    # Create a simple non-positive definite matrix
    A = sparse.csc_array([[1, 2], [2, 1]], dtype=dtype)
    with pytest.raises(CholmodNotPositiveDefiniteError):
        cholesky(A)


test_As = []

# for dtype in DTYPES:
for dtype in [np.float32, np.float64]:
    test_As.extend(
        generate_random_matrices(
            N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True, dtype=dtype
        )
    )


# FIXME
# This test fails for some matrices with complex64, and complex128 dtypes.
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
        R = cholesky(A, order=order, lower=True).T
        assert_allclose((R.T.conj() @ R).toarray(), A.toarray(), atol=atol)
    else:
        R, p = cholesky(A, order=order, lower=True)
        R = R.T
        assert is_valid_permutation(p)
        PAPT = A[p][:, p]
        assert_allclose((R.T.conj() @ R).toarray(), PAPT.toarray(), atol=atol)


# FIXME fails see cholmod.pyx for details
@pytest.mark.parametrize("A", test_As)
def test_lower(A):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    R = cholesky(A)
    L = cholesky(A, lower=True)
    assert_allclose(R.T.conj().toarray(), L.toarray(), atol=atol)
