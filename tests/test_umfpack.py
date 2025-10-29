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

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as la
from scipy import sparse
from scipy.io import mmread
from sksparse.umfpack import (
    UMFControl,
    UMFFactor,
    UMFPACKError,
    UMFPACKNonpositiveError,
    UMFPACKSingularMatrixWarning,
    umf_factor,
    umf_solve,
)

from .helpers import generate_random_matrices

ITYPES = [np.int32, np.int64]
DTYPES = [np.float64, np.complex128]


def assert_LU_equals_A(f, A, atol=1e-15):
    """Check that L U = P R A Q."""
    L, U, p, q, r = f.L, f.U, f.perm_r, f.perm_c, f.rscale
    LU = (L @ U).toarray()
    PRAQ = (r[:, np.newaxis] * A).tocsc()[p][:, q].toarray()
    assert_allclose(LU, PRAQ, atol=atol, strict=True)


# -----------------------------------------------------------------------------
#         Simple Tests
# -----------------------------------------------------------------------------
def test_empty_input():
    empty_A = sparse.csc_array((0, 0))
    with pytest.raises(UMFPACKNonpositiveError, match="non-positive"):
        _f = UMFFactor(empty_A)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    f = UMFFactor(zero_A)
    assert f.nnz == 0
    assert f.shape == (N, N)
    assert f.itype == zero_A.indptr.dtype
    assert f.dtype == zero_A.dtype
    with pytest.raises(UMFPACKError, match="Numeric factorization not present"):
        _L = f.L


def test_singleton():
    dtype = np.float64
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    f = UMFFactor(singleton_A)
    assert f.nnz == 0
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


def test_bad_factorize_shape(davis_example_qr):
    A = davis_example_qr
    f = UMFFactor(A)
    with pytest.raises(ValueError, match="shape.*does not match"):
        f.factorize(A[:-1, :])  # remove last row


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_davis_example_qr(davis_example_qr, itype, dtype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    A.data = A.data.astype(dtype)

    f = umf_factor(A)
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


def _numpy_slogdet(A):
    """Compute sign and logdet of A using numpy. Suppress warnings."""
    # In some versions of numpy, a warning is raised by slogdet for
    # these complex types: (np.complex64, np.complex128). Make sure that is the
    # warning that is raised, and not some other warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sign, logdet = np.linalg.slogdet(A.toarray())

    if record:
        assert record[0].category is RuntimeWarning
        assert "divide by zero" in str(record[0].message) or "invalid value" in str(
            record[0].message
        )
    else:
        pass

    return sign, logdet


@pytest.mark.parametrize("dtype", DTYPES)
def test_eye_determinant(dtype):
    N = 3
    A = 10 * sparse.eye_array(N, dtype=dtype).tocsc()
    f = umf_factor(A)
    rtol = 1e-7
    expect_sign, expect_logdet = _numpy_slogdet(A)
    assert expect_sign == 1
    assert expect_logdet == N * np.log(10)
    assert_allclose(f.slogdet(), (expect_sign, expect_logdet), rtol=rtol, strict=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_determinant(davis_example_qr, dtype):
    A = davis_example_qr
    # Set the data to random values
    rng = np.random.default_rng(56)
    A.data = rng.random(len(A.data)).astype(dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        A.data += 1j * rng.random(len(A.data)).astype(dtype=dtype)
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = umf_factor(A)
    rtol = 1e-7
    expect_sign, expect_logdet = _numpy_slogdet(A)
    assert_allclose(f.slogdet(), (expect_sign, expect_logdet), rtol=rtol, strict=True)


test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, dtype=dtype)
]


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("A", test_As)
def test_refactor(A, copy):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-6
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = umf_factor(A)
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
        return umf_factor(A)

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
        with pytest.raises(ValueError, match="same number of rows as A"):
            f.solve(b)

    def test_b_KD_sparse(self, f, A, N):
        b = sparse.csc_array((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as A"):
            f.solve(b)


def test_bad_A_shape_solve():
    A = sparse.csc_array([[1, 2, 3], [3, 4, 4]]).astype(float)
    assert A.shape == (2, 3)
    b = np.array([1, 2]).astype(float)
    f = umf_factor(A)
    with pytest.raises(ValueError, match="must be square"):
        f.solve(b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = np.array([1], dtype=dtype)
    x = UMFFactor(singleton_A).factorize().solve(b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = sparse.coo_array([1], dtype=dtype)
    x = UMFFactor(singleton_A).factorize().solve(b)
    assert_allclose(x.toarray(), b.toarray())


@pytest.mark.parametrize("itype", ITYPES)
def test_itype_1D(davis_example_qr, itype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = umf_solve(A, b)
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
    x = umf_solve(A, b)
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

    # NOTE umf_solve does some trickery to only warn once, so we expect only
    # one warning here. pytest.warns(), however, overrides the
    # "warnings.catch_warnings" context and captures all warnings, so we
    # manually check the warnings instead.
    with warnings.catch_warnings(record=True) as ws:
        x = umf_solve(A, b)

    assert len(ws) == 1
    w = ws[0]
    assert w.category == UMFPACKSingularMatrixWarning
    assert "indefinite or singular to working precision" in str(w.message)

    assert np.isnan(x.toarray()[s])
    assert_allclose((A @ x).toarray(), b.toarray(), atol=1e-12)
    idx = ~np.isnan(x.toarray())
    assert_allclose(x.toarray()[idx], expect_x.toarray()[idx], atol=1e-12)


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
    f = umf_factor(A, row_scale="none")  # turn off scaling to trigger warning
    with pytest.warns(UMFPACKSingularMatrixWarning, match="nearly singular"):
        f.solve(b)


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_solve(A, K, is_sparse):
    atol = 1e-12

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
    x = umf_solve(A, b)

    # Compare
    if is_sparse:
        assert_allclose(x.toarray(), expect_x.toarray(), atol=atol)
    else:
        assert_allclose(x, expect_x, atol=atol)


# Test solve on "real-world" matrices
def _load_problem(name):
    """Load a matrix and RHS from a Matrix Market file."""
    data_path = Path(__file__).parent / "test_data"
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
    x = umf_solve(ATA, ATb)
    assert_allclose(x, expect_x, atol=1e-7)


# -----------------------------------------------------------------------------
#         Test Info and Control
# -----------------------------------------------------------------------------
# Copied from umfpack.h, subject to change
CONTROL_DEFAULTS = {
    "print_level": 1,
    "dense_row": 0.2,
    "dense_col": 0.2,
    "pivot_tol": 0.1,
    "sym_pivot_tol": 0.001,
    "blas3_block_size": 32,
    "alloc_init": 0.7,
    "front_alloc_init": 0.5,
    "ir_steps": 2,
    "row_scale": "sum",  # UMFPACK_SCALE_SUM
    "strategy": "auto",  # UMFPACK_STRATEGY_AUTO
    "amd_dense": 10.0,  # AMD_DEFAULT_DENSE
    "fixQ": 0,
    "aggressive": True,
    "droptol": 0.0,
    "ordering_method": "amd",  # UMFPACK_ORDERING_AMD
    "singletons": True,
    "sym_thresh": 0.3,
    "nnzdiag_thresh": 0.9,
}


def test_default_controls():
    c = UMFControl()
    for key, expect_value in CONTROL_DEFAULTS.items():
        actual_value = getattr(c, key)
        assert actual_value == expect_value, (
            f"Control '{key}': expected {expect_value}, got {actual_value}"
        )


def test_ir_steps(davis_example_qr):
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    N_steps = 0
    c = UMFControl(ir_steps=N_steps)  # arbitrary > default
    f = umf_factor(A, control=c)
    assert f.control.ir_steps == N_steps
    expect_x = np.arange(1, A.shape[0] + 1, dtype=A.dtype)
    b = A @ expect_x
    x = f.solve(b)
    assert_allclose(x, expect_x, atol=1e-15, strict=True)
    print(f"{f.info.ir_attempted=}, {f.info.ir_attempted=}")
    assert f.info.ir_attempted == N_steps


@pytest.mark.parametrize("scale", ["none", "sum", "max"])
def test_row_scale(davis_example_qr, scale):
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = UMFFactor(A)
    # Row scaling can be done *after* symbolic, but *before* numeric
    f.control.row_scale = scale
    assert f.control.row_scale == scale
    f.factorize()
    assert f.info.was_scaled == scale
    assert_LU_equals_A(f, A)
    if scale in [None, "none"]:
        assert_allclose(f.rscale, 1.0)
        assert_allclose(f.L.diagonal(), 1.0)


ORDERINGS = ["none", "cholmod", "amd", "metis", "best", "metis_guard"]


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_ordering(davis_example_qr, ordering):
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    # Ordering must be done *before* symbolic factorization
    f = umf_factor(A, ordering_method=ordering)
    expect_x = np.arange(1, A.shape[0] + 1, dtype=A.dtype)
    b = A @ expect_x
    x = f.solve(b)
    assert_LU_equals_A(f, A)
    assert_allclose(x, expect_x, atol=1e-15, strict=True)
    if ordering in [None, "none", "amd", "metis"]:
        assert f.info.ordering_used == ordering
    else:  # ["cholmod", "best", "metis_guard"]
        # May choose AMD or METIS
        assert f.info.ordering_used in ["amd", "metis"]


# =============================================================================
# =============================================================================
