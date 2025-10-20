# Part of the scikit-sparse project.
# Copyright (C) 2025 the scikit-sparse developers. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_umfpack.py
#  Created: 2025-10-16 11:54
# =============================================================================

"""Unit tests for the umfpack module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sksparse.umfpack import UMFFactor, UMFPACKError, UMFPACKNonpositiveError

from .helpers import generate_random_matrices

ITYPES = [np.int32, np.int64]
DTYPES = [np.float64, np.complex128]


def assert_LU_equals_A(f, A, atol=1e-15):
    """Check that L U = P R A Q."""
    L, U, p, q, r = f.L, f.U, f.perm_r, f.perm_c, f.R
    LU = (L @ U).toarray()
    PRAQ = (r[:, np.newaxis] * A).tocsc()[p][:, q].toarray()
    assert_allclose(LU, PRAQ, atol=atol, strict=True)


def test_empty_input():
    empty_A = sparse.csc_array((0, 0))
    with pytest.raises(UMFPACKNonpositiveError, match="non-positive"):
        _f = UMFFactor(empty_A)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    f = UMFFactor(zero_A)
    assert f.nnz == 0
    assert f.n_row == N
    assert f.n_col == N
    assert f.itype == zero_A.indptr.dtype
    assert f.dtype == zero_A.dtype
    with pytest.raises(UMFPACKError, match="Numeric factorization not present"):
        _L = f.L


def test_singleton():
    dtype = np.float64
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    f = UMFFactor(singleton_A)
    assert f.nnz == 0
    assert f.n_row == 1
    assert f.n_col == 1
    assert f.itype == singleton_A.indptr.dtype
    assert f.dtype == dtype


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_types(davis_example_qr, itype, dtype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    A.data = A.data.astype(dtype)
    f = UMFFactor(A)
    assert f.itype == itype
    assert f.dtype == dtype


# -----------------------------------------------------------------------------
#         Numeric Factorization
# -----------------------------------------------------------------------------
def test_bad_factorize_type(davis_example_qr):
    A = davis_example_qr
    A.data = A.data.astype(np.float64)
    f = UMFFactor(A)
    with pytest.raises(ValueError, match="type.*does not match"):
        f.factorize(A.astype(np.complex128))


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_davis_example_qr(davis_example_qr, itype, dtype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    A.data = A.data.astype(dtype)

    f = UMFFactor(A)
    f.factorize(A)
    assert f.is_numeric

    # Get the factors
    p, q = f.perm_r, f.perm_c

    # Values from MATLAB umfpack
    # >> [L, U, P, Q, R] = umfpack(A);
    # >> [p j x] = find(P');
    # >> [q j x] = find(Q);
    expect_p = np.array([0, 3, 1, 2, 6, 7, 4, 5], dtype=itype)
    expect_q = np.array([0, 3, 1, 2, 6, 7, 5, 4], dtype=itype)

    assert_array_equal(p, expect_p)
    assert_array_equal(q, expect_q)
    assert f.lnz == 15  # == nnz(L) in MATLAB
    assert f.unz == 16  # == nnz(U) in MATLAB
    assert f.nnz == 31  # == nnz(L) + nnz(U) in MATLAB
    assert f.nz_udiag == 8  # == nnz(diag(U)) in MATLAB
    assert_LU_equals_A(f, A)


test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, dtype=dtype)
]


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("A", test_As)
def test_refactor(A, copy):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-6
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = UMFFactor(A)
    f.factorize(A)
    assert_LU_equals_A(f, A, atol=atol)
    # Create a new matrix with the same sparsity pattern but different values
    B = A.copy()
    rng = np.random.default_rng(56)
    B.data = rng.random(len(B.data)).astype(dtype=B.dtype)
    # Factor the new matrix with the same sparsity pattern
    if copy:
        g = f.copy()
        assert g is not f
        g.factorize(B)
        assert_LU_equals_A(g, B, atol=atol)
    else:
        f.factorize(B)
        assert_LU_equals_A(f, B, atol=atol)



# @pytest.mark.parametrize("itype", ITYPES)
# @pytest.mark.parametrize("dtype", DTYPES)
# def test_demo(itype, dtype):
#     rng = np.random.default_rng(56)
#     # Random matrix
#     # N = 10
#     # A = sparse.random_array((N, N), density=0.5, format="csc", rng=56, dtype=dtype)
#     # A.setdiag(1.0)

#     # Laplaceian grid
#     A = -LaplacianNd((3, 3), dtype=dtype).tosparse().tocsc()
#     A[-1, -1] += 1.0  # make non-singular
#     N = A.shape[0]

#     A.indptr = A.indptr.astype(itype)
#     A.indices = A.indices.astype(itype)
#     f = UMFFactor(A)
#     print()
#     print(f)
#     print('---------- report_control():')
#     f.report_control()
#     print('---------- report_symbolic():')
#     f.report_symbolic()
#     print('---------- print(f.control):')
#     print(f.control)
#     f.factorize(A)
#     print('---------- report_numeric():')
#     f.report_numeric()
#     print(f)
#     # Solve a system
#     expect_x = np.arange(1, N + 1, dtype=dtype)
#     # Ensure non-zero complex parts
#     if np.issubdtype(dtype, np.complexfloating):
#         expect_x += 1j * 0.1 * rng.random(N)
#     expect_x = np.r_[expect_x, 2 * expect_x].reshape((-1, 2))  # multiple RHS
#     print(f"{expect_x=}")
#     b = A @ expect_x
#     x = f.solve(A, b)
#     assert_allclose(x, expect_x, atol=1e-12, strict=True)
#     print('---------- print(f.info):')
#     print(f.info)
#     print('---------- report_info():')
#     f.report_info()
#     # Print the factors
#     print('---------- factors:')
#     print(repr(f.L))
#     print(repr(f.U))
#     print(repr(f.perm_r))
#     print(repr(f.perm_c))
#     print(repr(f.R))
#     L, U, p, q, r = f.L, f.U, f.perm_r, f.perm_c, f.R
#     # Check that L U = P R A Q
#     LU = (L @ U).toarray()
#     PRAQ = (r[:, np.newaxis] * A).tocsc()[p][:, q].toarray()
#     assert_allclose(LU, PRAQ, atol=1e-12, strict=True)

# =============================================================================
# =============================================================================
