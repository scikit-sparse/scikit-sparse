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

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as la
from scipy import sparse
from scipy.io import mmread

from sksparse.klu import (
    KLUError,
    KLUFactor,
    KLUInvalidError,
    KLUSingularMatrixWarning,
    klu_factor,
    klu_solve,
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


def test_bad_factorize_structure(davis_example_qr):
    A = davis_example_qr
    f = KLUFactor(A)
    B = A.copy().todok()
    # Change the structure of the matrix by adding a new non-zero
    B[0, 1] = 2.3
    B = B.tocsc()
    f.factorize(B)  # passes
    assert_LU_equals_A(f, B)


@pytest.mark.xfail(reason="Does not error, but gives wrong answer.")
def test_bad_refactorize_structure(davis_example_qr):
    A = davis_example_qr
    f = klu_factor(A)
    B = A.copy().todok()
    # Change the structure of the matrix by adding a new non-zero
    B[0, 1] = 2.3
    B = B.tocsc()
    f.factorize(B)  # just gives wrong answer without error
    assert_LU_equals_A(f, B)


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
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, square_only=True, dtype=dtype
    )
]


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("A", test_As)
def test_refactor(A, itype, copy):
    atol = 1e-8
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = klu_factor(A)
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


@pytest.mark.parametrize("A", test_As[:1])
def test_sorted(A):
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = klu_factor(A)
    L, U = f.L, f.U
    assert L.has_sorted_indices
    assert L.has_canonical_format
    assert U.has_sorted_indices
    assert U.has_canonical_format


# -----------------------------------------------------------------------------
#         Solve
# -----------------------------------------------------------------------------
class TestBadBShape:
    @pytest.fixture(scope="class")
    def N(self):
        return 5

    @pytest.fixture(scope="class")
    def A(self, N):
        return sparse.eye_array(N).tocsc()

    @pytest.fixture(scope="class")
    def f(self, A):
        return klu_factor(A)

    def test_b_0D_dense(self, f, A):
        b = np.empty([])
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(b)

    def test_b_3D_dense(self, f, A):
        b = np.empty((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(b)

    def test_b_3D_sparse(self, f, A):
        b = sparse.coo_array((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(b)

    def test_b_KD_dense(self, f, A, N):
        b = np.empty((N - 1, N))
        with pytest.raises(ValueError, match="compatible shape with A"):
            f.solve(b)

    def test_b_KD_sparse(self, f, A, N):
        b = sparse.csc_array((N - 1, N))
        with pytest.raises(ValueError, match="compatible shape with A"):
            f.solve(b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = np.array([1], dtype=dtype)
    x = KLUFactor(singleton_A).factorize(singleton_A).solve(b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = sparse.coo_array([1], dtype=dtype)
    x = KLUFactor(singleton_A).factorize(singleton_A).solve(b)
    assert_allclose(x.toarray(), b.toarray())


@pytest.mark.parametrize("itype", ITYPES)
def test_itype_1D(davis_example_qr, itype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = klu_solve(A, b)
    assert isinstance(x, sparse.coo_array)
    assert x.coords[0].dtype == itype


@pytest.mark.parametrize("itype", ITYPES)
def test_itype_2D(davis_example_qr, itype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    K = 3  # arbitrary number of rhs
    s = np.arange(1, N + 1, dtype=A.dtype)
    data = np.array([i * s for i in range(1, K + 1)]).T
    expect_x = sparse.csc_array(data, dtype=A.dtype)
    b = A @ expect_x
    x = klu_solve(A, b)
    assert isinstance(x, sparse.csc_array)
    assert x.indptr.dtype == itype
    assert x.indices.dtype == itype


def test_exactly_singular(davis_example_qr):
    A = davis_example_qr.todok()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

    N = A.shape[0]
    lam0 = la.eigvalsh(A.toarray()).min()

    # Make A exactly singular
    s = -3
    A[:, s] = 0.0
    A[s, :] = 0.0
    A = A.tocsc()

    lam1 = la.eigvalsh(A.toarray()).min()
    print(f"\nMin eigenvalue: {lam0:.2e} -> {lam1:.2e}\n")

    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x

    with pytest.raises(KLUError, match="indefinite or singular to working precision"):
        klu_solve(A, b)


def test_nearly_singular(davis_example_qr):
    A = davis_example_qr.todok()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

    N = A.shape[0]
    lam0 = la.eigvalsh(A.toarray()).min()

    # Make A nearly singular
    A[:, -1] = 0.0
    A[-1, :] = 0.0
    A[-1, -1] = 0.5 * np.finfo(A.dtype).eps
    A = A.tocsc()

    lam1 = la.eigvalsh(A.toarray()).min()
    print(f"\nMin eigenvalue: {lam0:.2e} -> {lam1:.2e}\n")

    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    f = klu_factor(A, scale="none")  # disable row-scaling to trigger warning
    print(f"{f.info.scale=}")
    with pytest.warns(KLUSingularMatrixWarning, match="nearly singular"):
        f.solve(b)


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_solve(A, K, is_sparse):
    atol = 1e-12
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

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
    x = klu_solve(A, b)
    xt = klu_solve(A.T.tocsc(), b.T, transpose=True)

    # Compare
    if is_sparse:
        assert_allclose(x.toarray(), expect_x.toarray(), atol=atol)
        assert_allclose(xt.toarray(), expect_x.T.toarray(), atol=atol)
    else:
        assert_allclose(x, expect_x, atol=atol)
        assert_allclose(xt, expect_x.T, atol=atol)


# Test solve on "real-world" matrices
def _load_problem(name):
    """Load a matrix and RHS from a Matrix Market file."""
    data_path = Path(__file__).parent / "data"
    matrix_file = data_path / f"{name}.mtx.gz"

    if not matrix_file.exists():
        raise FileNotFoundError(f"Matrix Market file {matrix_file} not found.")

    A = mmread(matrix_file, spmatrix=False).tocsc()

    # Possibly load RHS
    rhs_file = data_path / f"{name}_rhs1.mtx.gz"

    if not rhs_file.exists():
        raise FileNotFoundError(f"Matrix Market file {rhs_file} not found.")

    b = mmread(rhs_file)

    return A, b


# TODO @pytest.mark.slow
@pytest.mark.parametrize("problem", ["well1033", "illc1033", "well1850", "illc1850"])
def test_solve_real(problem):
    A, b = _load_problem(problem)
    # Solve the normal equations A^T A x = A^T b
    ATA = (A.T @ A).tocsc()
    ATb = A.T @ b
    expect_x = np.linalg.lstsq(A.toarray(), b)[0]
    x = klu_solve(ATA, ATb)
    assert_allclose(x, expect_x, atol=1e-7)


# -----------------------------------------------------------------------------
#         Test Control and Info
# -----------------------------------------------------------------------------
def test_info(davis_example_qr):
    A = davis_example_qr
    f = klu_factor(A)
    info = f.info
    # Values from MATLAB klu
    # >> [LU, info, c] = klu(A);
    assert info.noffdiag == 0
    assert info.nrealloc == 0
    assert_allclose(info.rcond, 0.084848, rtol=1e-4)
    assert info.singular_col == 8  # dimension of A
    assert_allclose(info.rgrowth, 0.509, rtol=1e-3)
    assert info.flops == 28
    assert info.nblocks == 1
    assert info.ordering == "AMD"
    assert info.scale == "max"
    assert info.lnz == 16
    assert info.unz == 17
    assert info.nzoff == 0
    assert info.tol == 0.001
    assert info.memory != 0  # number varies with system


@pytest.mark.parametrize("scale", [None, "none_no_check", "none", "sum", "max"])
def test_row_scale(davis_example_qr, scale):
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = klu_factor(A, scale=scale)
    if scale is None:
        assert f.info.scale == "max"  # default
    else:
        assert f.info.scale == scale
    assert_LU_equals_A(f, A)
    if scale == "none":
        assert_allclose(f.L.diagonal(), 1.0)


ORDERINGS = [None, "AMD", "COLAMD"]


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_ordering(davis_example_qr, ordering):
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    # Ordering must be done *before* symbolic factorization
    f = klu_factor(A, ordering=ordering)
    expect_x = np.arange(1, A.shape[0] + 1, dtype=A.dtype)
    b = A @ expect_x
    x = f.solve(b)
    assert_LU_equals_A(f, A)
    assert_allclose(x, expect_x, atol=1e-15, strict=True)
    if ordering is None:
        assert f.info.ordering == "AMD"  # default
    else:
        assert f.info.ordering == ordering


@pytest.mark.parametrize("ordering", ["user_perm", "user_func"])
def test_bad_ordering(davis_example_qr, ordering):
    A = davis_example_qr
    with pytest.raises(NotImplementedError, match="not yet supported"):
        klu_factor(A, ordering=ordering)


# =============================================================================
# =============================================================================
