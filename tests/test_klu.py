# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_klu.py
#  Created: 2025-10-31 12:25
# =============================================================================

"""Unit tests for the klu module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sksparse.klu import (
    KLUError,
    KLUFactor,
    KLUInvalidError,
    klu_factor,
)

from .helpers import generate_random_matrices

ITYPES = [np.int32, np.int64]
DTYPES = [np.float64, np.complex128]


def assert_LU_equals_A(f, A, atol=1e-15):
    """Check that L U = P R A Q."""
    L, U, F, p, q, r = f.L, f.U, f.F, f.perm_r, f.perm_c, f.rscale
    LUF = (L @ U + F).toarray()
    PRinvAQ = ((1 / r)[:, np.newaxis] * A[p][:, q]).toarray()
    assert_allclose(LUF, PRinvAQ, atol=atol, strict=True)


# -----------------------------------------------------------------------------
#         Simple Tests
# -----------------------------------------------------------------------------
def test_empty_input():
    empty_A = sparse.csc_array((0, 0))
    with pytest.raises(KLUInvalidError, match="invalid input"):
        _f = KLUFactor(empty_A)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    f = KLUFactor(zero_A)
    assert not f.is_numeric
    assert f.lnz is None
    assert f.unz is None
    assert f.nnz is None
    assert f.shape == (N, N)
    assert f.itype == zero_A.indptr.dtype
    assert f.dtype == zero_A.dtype
    with pytest.raises(KLUError, match="Numeric factorization not present"):
        _L = f.L


def test_singleton():
    dtype = np.float64
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    f = KLUFactor(singleton_A).factorize(singleton_A)
    assert f.is_numeric
    assert f.lnz == 1
    assert f.unz == 1
    assert f.nnz == 2  # nnz(L) + nnz(U)
    assert f.shape == (1, 1)
    assert f.itype == singleton_A.indptr.dtype
    assert f.dtype == dtype


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_types(davis_example_qr, itype, dtype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    A.data = A.data.astype(dtype)
    f = KLUFactor(A)
    assert f.itype == itype
    assert f.dtype == dtype


# -----------------------------------------------------------------------------
#         Numeric Factorization
# -----------------------------------------------------------------------------
def test_bad_factorize_itype(davis_example_qr):
    A = davis_example_qr
    A.indptr = A.indptr.astype(np.int32)
    A.indices = A.indices.astype(np.int32)
    f = KLUFactor(A)
    B = A.copy()
    B.indptr = B.indptr.astype(np.int64)
    B.indices = B.indices.astype(np.int64)
    with pytest.raises(ValueError, match="integer.*does not match"):
        f.factorize(B)


def test_bad_factorize_dtype(davis_example_qr):
    A = davis_example_qr.astype(np.float64)
    f = KLUFactor(A)
    with pytest.raises(ValueError, match="type.*does not match"):
        f.factorize(A.astype(np.complex128))


def test_bad_factorize_shape(davis_example_qr):
    A = davis_example_qr
    f = KLUFactor(A)
    with pytest.raises(ValueError, match="shape.*does not match"):
        f.factorize(A[:-1, :-1])  # remove last row and col


# TODO test bad factorize *structure*


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_davis_example_qr(davis_example_qr, itype, dtype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    A.data = A.data.astype(dtype)

    f = klu_factor(A)
    assert f.is_numeric
    assert f.shape == A.shape

    # Get the factors
    p, q = f.perm_r, f.perm_c

    # Values from MATLAB klu
    # >> [LU, info, c] = klu(A);
    # >> LU.p - 1
    # >> LU.q - 1
    expect_p = np.array([4, 7, 5, 1, 2, 0, 6, 3], dtype=itype)
    expect_q = np.array([4, 5, 7, 1, 2, 0, 6, 3], dtype=itype)

    assert_array_equal(p, expect_p)
    assert_array_equal(q, expect_q)
    assert f.lnz == 16  # == nnz(L) in MATLAB
    assert f.unz == 17  # == nnz(U) in MATLAB
    assert f.nnz == 33  # == nnz(L) + nnz(U) in MATLAB
    assert_LU_equals_A(f, A)


def test_iter(davis_example_qr):
    A = davis_example_qr
    L, U, p, q, r, F, _rblocks = klu_factor(A)
    LUF = (L @ U + F).toarray()
    PRinvAQ = ((1 / r)[:, np.newaxis] * A[p][:, q]).toarray()
    assert_allclose(LUF, PRinvAQ, atol=1e-15, strict=True)


test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, dtype=dtype)
]


@pytest.mark.parametrize("copy", [False])
@pytest.mark.parametrize("A", test_As)
def test_refactor(A, copy):
    atol = 1e-10
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = klu_factor(A)
    assert_LU_equals_A(f, A, atol=atol)
    # Create a new matrix with the same sparsity pattern but different values
    B = A.copy()
    rng = np.random.default_rng(56)
    B.data = rng.random(len(B.data)).astype(dtype=B.dtype)
    # Factor the new matrix with the same sparsity pattern
    # $ TODO
    # if copy:
    #     g = f.copy()
    #     assert g is not f
    #     g.factorize(B)
    #     assert_LU_equals_A(g, B, atol=atol)
    # else:
    f.factorize(B)
    assert_LU_equals_A(f, B, atol=atol)


# =============================================================================
# =============================================================================
