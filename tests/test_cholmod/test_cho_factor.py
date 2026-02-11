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

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as la
from scipy import sparse

from sksparse.cholmod import (
    CholeskyFactor,
    CholmodInvalidInputError,
    CholmodNotPositiveDefiniteError,
    cho_factor,
)

from ..helpers import generate_random_matrices

ITYPES = [np.int32, np.int64]
DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


def assert_LLT_equals_A(f, A, rtol=1e-7, atol=1e-15):
    """Assert that L @ L.T.conj() equals A."""
    L = f.get_factor(lower=True)
    p = f.get_perm()
    LLT = (L @ L.T.conj()).toarray()
    PAPT = A[p[:, np.newaxis], p].toarray()
    assert_allclose(LLT, PAPT, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_convert_factor(davis_example_chol, dtype):
    atol = 1e-15 if dtype in (np.float64, np.complex128) else 1e-6
    A = davis_example_chol.astype(dtype)
    f = cho_factor(A, lower=True, order=None)
    L, D = f.get_factor(kind="LDL")
    assert_allclose((L @ D @ L.T.conj()).toarray(), A.toarray(), atol=atol)


@pytest.mark.parametrize("lower", [False, True])
def test_properties(davis_example_chol, lower):
    atol = 1e-15
    A = davis_example_chol
    f = cho_factor(A, order=None, lower=lower)
    N = A.shape[0]
    assert f.N == N
    assert f.nnz == f.factor.nnz
    assert f.is_lower == lower
    # Check that L, R are independent of lower
    L, R, D = f.L, f.R, f.D
    assert_array_equal(D.toarray(), np.eye(N))
    assert_allclose((L @ L.T).toarray(), A.toarray(), atol=atol)
    assert_allclose((R.T @ R).toarray(), A.toarray(), atol=atol)


@pytest.mark.parametrize("order", [None, "amd"])
def test_view_vs_get(davis_example_chol, order):
    A = davis_example_chol
    f = cho_factor(A, lower=True, order=order)
    Lv = f.factor
    pv = f.perm
    cc = f.colcount
    assert Lv is f.factor
    assert pv is f.perm
    assert cc is f.colcount
    L = f.get_factor()
    p = f.get_perm()
    assert Lv is not L  # different objects
    assert pv is not p
    Lp = f.L
    assert Lp is not Lv
    assert Lp is not L
    R = f.get_factor(lower=False)
    Rp = f.R
    assert Rp is f.R
    assert Rp is not R
    assert_allclose(Lv.toarray(), L.toarray(), atol=1e-15)
    assert_allclose(Lp.toarray(), L.toarray(), atol=1e-15)
    assert_allclose(Rp.toarray(), R.toarray(), atol=1e-15)
    assert_allclose(R.T.conj().toarray(), L.toarray(), atol=1e-15)
    assert_allclose(pv, p, atol=1e-15)


def test_change_factor(davis_example_chol):
    A = davis_example_chol
    f = cho_factor(A, lower=True)
    assert f.is_ll
    L = f.L
    D = f.D
    p = f.perm
    LLT = (L @ L.T.conj()).toarray()
    PAPT = A[p[:, np.newaxis], p].toarray()
    assert_allclose(LLT, PAPT, atol=1e-15)
    assert_array_equal(D.toarray(), np.eye(A.shape[0]))
    # Change to "nothing"
    f.change_factor(kind="LL")
    assert f.is_ll
    # Convert to LDL^T
    f.change_factor()
    assert not f.is_ll
    Ld = f.L
    Dd = f.D
    LDLT = (Ld @ Dd @ Ld.T.conj()).toarray()
    assert_allclose(LDLT, PAPT, atol=1e-15)
    # Convert back to LL^T
    f.change_factor(kind="LL")
    assert f.is_ll


@pytest.fixture
def A_small():
    return sparse.csc_array(
        np.array([[10, 0, 3, 0], [0, 5, 0, -2], [3, 0, 5, 0], [0, -2, 0, 2]]),
        dtype=np.float64,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_determinant(A_small, dtype):
    A = A_small.astype(dtype)
    rtol = 1e-7 if A.dtype in (np.float64, np.complex128) else 1e-6

    f = cho_factor(A, lower=True)

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
    f = cho_factor(A)
    Ainv = f.inv()
    I = np.eye(A.shape[0], dtype=A.dtype)
    assert_allclose((A @ Ainv).toarray(), I, atol=atol, strict=True)


def test_bad_refactor_type(A_small):
    A = A_small.astype(np.float64)
    f = cho_factor(A)
    with pytest.raises(CholmodInvalidInputError):
        f.factorize(A.astype(np.complex128))


test_As = [
    A
    for itype in ITYPES
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=10,
        N_max=200,
        d_scale=0.05,
        spd_only=True,
        itype=itype,
        dtype=dtype,
    )
]


@pytest.mark.parametrize("A", test_As)
def test_copy_symbolic(A):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = CholeskyFactor(A)
    g = f.copy()
    assert g is not f
    # Test that numeric factorization can be done on the copy
    f.factorize(A)
    assert_LLT_equals_A(f, A, atol=atol)
    del f  # ensure no shared state
    g.factorize(A)
    assert_LLT_equals_A(g, A, atol=atol)


@pytest.mark.parametrize("A", test_As)
def test_copy_numeric(A):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5
    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = cho_factor(A)
    g = f.copy()
    assert g is not f
    assert_LLT_equals_A(f, A, atol=atol)
    del f  # ensure no shared state
    assert_LLT_equals_A(g, A, atol=atol)


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
    A = A.copy()
    f = cho_factor(A, lower=True)
    assert_LLT_equals_A(f, A, atol=atol)
    # Create a new matrix with the same sparsity pattern but different values
    B = _create_randomized_matrix(A)
    # Factor the new matrix with the same sparsity pattern
    if copy:
        # Use a copy of the factorization object to ensure that we are taking
        # the relevant parameters from the underlying cholmod_common object.
        g = f.copy()
        assert g is not f
        assert_LLT_equals_A(f, A, atol=atol)  # original still works
        del f  # ensure no shared state
        g.factorize(B)
        assert_LLT_equals_A(g, B, atol=atol)
    else:
        f.factorize(B)
        assert_LLT_equals_A(f, B, atol=atol)


# -----------------------------------------------------------------------------
#         Non-Square or Unsymmetric Matrices
# -----------------------------------------------------------------------------
test_As = [
    A
    for itype in ITYPES
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=5,
        N_max=200,
        d_scale=0.05,
        shape_kind="M > N",
        itype=itype,
        dtype=dtype,
    )
]


def test_bad_sym(A_small):
    A = A_small[:-1, :]  # make non-square
    with pytest.raises(ValueError, match="Expected square matrix"):
        CholeskyFactor(A, sym_kind="sym")


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonspd_sym(dtype):
    # NOTE If A is (M, N) with M > N, then A @ A.T is (M, M) but A can have at
    # most rank(N) < M, so A @ A.T is symmetric, but not positive definite.
    # Similarly, for the "col" case with A.T, A.T @ A is (N, N) but rank
    # at most rank(M) < N.
    A = sparse.random_array((10, 7), density=0.6, format="csc", dtype=dtype, rng=56)
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

    AAT = A @ A.T.conj()
    AAT = (AAT + AAT.T.conj()) / 2  # make *exactly* Hermitian
    lam = la.eigvals(AAT.toarray()).min()
    print(f"\nmin(eig(AAT)): {lam:.2e}")

    f = CholeskyFactor(A, sym_kind="row")
    with pytest.raises(
        CholmodNotPositiveDefiniteError, match="matrix is not positive definite"
    ):
        f.factorize(A)

    # Test the "col" case with A.T
    A = A.T.conj().tocsc()

    ATA = A.T.conj() @ A
    ATA = (ATA + ATA.T.conj()) / 2  # make *exactly* Hermitian
    lam = la.eigvals(ATA.toarray()).min()
    print(f"\nmin(eig(ATA)): {lam:.2e}")

    f = CholeskyFactor(A, sym_kind="col")
    with pytest.raises(
        CholmodNotPositiveDefiniteError, match="matrix is not positive definite"
    ):
        f.factorize(A)


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("sym_kind", ["row", "col"])
def test_sym(A, sym_kind):
    rtol = 1e-7 if A.dtype in (np.float64, np.complex128) else 1e-3
    atol = 1e-14 if A.dtype in (np.float64, np.complex128) else 1e-5
    A = A.copy()

    if sym_kind == "row":
        A = A.T.conj().tocsc()  # M > N -> M < N, so A @ A.T is (M, M)
        assert A.shape[0] < A.shape[1]

    A.setdiag(A.diagonal() + 10.0)  # make non-singular

    if sym_kind == "row":
        AXX = A @ A.T.conj()
    else:
        AXX = (A.T.conj() @ A).tocsc()

    if np.iscomplexobj(AXX):
        AXX = (AXX + AXX.T.conj()) / 2  # make *exactly* Hermitian

    lamAXX = la.eigvals(AXX.toarray()).min()
    A_str = "AAT" if sym_kind == "row" else "ATA"
    print(f"\nmin(eig({A_str})): {lamAXX:.2e}")

    # Test symbolic analysis
    f = CholeskyFactor(A, sym_kind=sym_kind, order=None)
    g = CholeskyFactor(AXX, sym_kind="sym", order=None)
    assert_array_equal(f.colcount, g.colcount)

    # Test numeric factorization
    f.factorize(A)
    g.factorize(AXX)

    # Check self-consistency
    assert_LLT_equals_A(f, AXX, rtol=rtol, atol=atol)
    assert_LLT_equals_A(g, AXX, rtol=rtol, atol=atol)

    # Check that the factors are equal
    Lf = f.get_factor().toarray()
    Lg = g.get_factor().toarray()
    assert_allclose(Lf, Lg, rtol=rtol, atol=atol)
