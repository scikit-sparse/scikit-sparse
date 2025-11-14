# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_ldl_factor.py
#  Created: 2025-09-05 11:05
# =============================================================================

"""Unit tests for the CholeskyFactor object."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from sksparse.cholmod import CholeskyFactor, ldl_factor

from ..helpers import generate_random_matrices

ITYPES = [np.int32, np.int64]
DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


def assert_LDLT_equals_A(f, A, atol=1e-15):
    """Assert that L @ @ D L.T.conj() equals A."""
    L, D = f.get_factor(kind="LDL", lower=True)
    assert_allclose((L @ D @ L.T.conj()).toarray(), A.toarray(), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_convert_factor(davis_example_chol, dtype):
    atol = 1e-15 if dtype in (np.float64, np.complex128) else 1e-6
    A = davis_example_chol.astype(dtype)
    f = ldl_factor(A)
    L = f.get_factor(kind="LL")
    assert_allclose((L @ L.T.conj()).toarray(), A.toarray(), atol=atol)


@pytest.mark.parametrize("order", [None, "amd"])
def test_view_vs_get(davis_example_chol, order):
    A = davis_example_chol
    f = ldl_factor(A)
    LDv = f.factor
    pv = f.perm
    L, D = f.get_factor()
    p = f.get_perm()
    assert LDv is not L  # different objects
    assert LDv is not D
    assert pv is not p
    # Split the view into L and D
    Dv = sparse.diags_array(LDv.diagonal())
    with pytest.raises(ValueError, match="read-only"):
        LDv.setdiag(1.0)
    LDv = LDv.copy()
    LDv.setdiag(1.0)
    assert_allclose(LDv.toarray(), L.toarray(), atol=1e-15)
    assert_allclose(Dv.toarray(), D.toarray(), atol=1e-15)
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

    f = ldl_factor(A, lower=True)

    # In some versions of numpy, a warning is raised by slogdet for
    # these complex types: (np.complex64, np.complex128). Make sure that is the
    # warning that is raised, and not some other warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        expect_det = np.linalg.det(A.toarray())
        expect_sign, expect_logdet = np.linalg.slogdet(A.toarray())

    if record:
        assert record[0].category is RuntimeWarning
        assert "divide by zero" in str(record[0].message) or "invalid value" in str(
            record[0].message
        )
    else:
        pass

    assert_allclose(f.det(), expect_det, rtol=rtol, strict=True)
    assert_allclose(f.slogdet(), (expect_sign, expect_logdet), rtol=rtol, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inv(A_small, dtype):
    atol = 1e-12 if dtype in (np.float64, np.complex128) else 1e-3
    A = A_small.astype(dtype)
    f = ldl_factor(A)
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


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("A", test_As)
def test_copy_symbolic(A, itype):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = CholeskyFactor(A, lower=True)
    g = f.copy()
    assert g is not f
    # Test that numeric factorization can be done on the copy
    f.factorize(A)
    assert_LDLT_equals_A(f, A, atol=atol)
    del f  # ensure no shared state
    g.factorize(A)
    assert_LDLT_equals_A(g, A, atol=atol)


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("A", test_As)
def test_copy_numeric(A, itype):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = ldl_factor(A)
    g = f.copy()
    assert g is not f
    assert_LDLT_equals_A(f, A, atol=atol)
    del f  # ensure no shared state
    assert_LDLT_equals_A(g, A, atol=atol)


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


@pytest.mark.parametrize("copy", [False])
@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("A", test_As)
def test_refactor(A, itype, copy):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-3
    A = A.copy()
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = ldl_factor(A)
    assert_LDLT_equals_A(f, A, atol=atol)
    # Create a new matrix with the same sparsity pattern but different values
    B = _create_randomized_matrix(A)
    # Factor the new matrix with the same sparsity pattern
    if copy:
        # Use a copy of the factorization object to ensure that we are taking
        # the relevant parameters from the underlying cholmod_common object.
        g = f.copy()
        assert g is not f
        assert_LDLT_equals_A(f, A, atol=atol)  # original still works
        del f  # ensure no shared state
        g.factorize(B)
        assert_LDLT_equals_A(g, B, atol=atol)
    else:
        f.factorize(B)
        assert_LDLT_equals_A(f, B, atol=atol)
