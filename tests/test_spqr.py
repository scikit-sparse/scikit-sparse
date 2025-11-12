# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_spqr.py
#  Created: 2025-11-07 13:31
# =============================================================================

"""Unit tests for the scikit-sparse.spqr module."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg as la
from scipy import sparse
from scipy.io import mmread

from sksparse.spqr import (
    SPQRFactor,
    SPQRRankDeficiencyWarning,
    spqr,
    spqr_factor,
    spqr_solve,
)

from .helpers import generate_random_matrices

ITYPES = [np.int32, np.int64]
DTYPES = [np.float64, np.complex128]


def assert_solve_dense(A, f, atol=1e-15):
    N = f.shape[1]
    expect_x = np.arange(1, N + 1, dtype=f.dtype)
    b = A @ expect_x
    x = f.solve(b)
    assert_allclose(x, expect_x, atol=atol, strict=True)


# -----------------------------------------------------------------------------
#         Simple Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("itype", ITYPES)
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    f = SPQRFactor(empty_A)
    assert f.rank == 0
    # assert_allclose(f.Q.toarray(), empty_A.toarray(), strict=True)
    # assert_allclose(f.R.toarray(), empty_A.toarray(), strict=True)
    assert_array_equal(f.perm, np.array([], dtype=itype), strict=True)


@pytest.mark.parametrize("itype", ITYPES)
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    f = spqr_factor(zero_A)
    assert f.rank == 0
    # assert_allclose(f.Q.toarray(), np.eye(N, dtype=A.dtype), strict=True)
    # assert_allclose(f.R.toarray(), np.array([], dtype=A.dtype), strict=True)
    assert_array_equal(f.perm, np.arange(N, dtype=itype), strict=True)


def test_singleton():
    dtype = np.float64
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    f = spqr_factor(singleton_A)
    assert f.is_numeric
    assert f.rank == 1
    assert f.shape == (1, 1)
    assert_array_equal(f.perm, np.array([0], dtype=np.int32), strict=True)


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_types(davis_example_qr, itype, dtype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    A.data = A.data.astype(dtype)
    f = SPQRFactor(A)
    assert f.itype == itype
    assert f.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_type_promotion(davis_example_qr, dtype):
    A = davis_example_qr.astype(dtype)
    f = SPQRFactor(A)
    expect_dtype = np.float64 if np.issubdtype(dtype, np.floating) else np.complex128
    assert f.dtype == expect_dtype


# -----------------------------------------------------------------------------
#         Numeric Factorization
# -----------------------------------------------------------------------------
def test_bad_factorize_itype(davis_example_qr):
    A = davis_example_qr
    A.indptr = A.indptr.astype(np.int32)
    A.indices = A.indices.astype(np.int32)
    f = SPQRFactor(A)
    B = A.copy()
    B.indptr = B.indptr.astype(np.int64)
    B.indices = B.indices.astype(np.int64)
    with pytest.raises(ValueError, match="integer.*does not match"):
        f.factorize(B)


def test_bad_factorize_dtype(davis_example_qr):
    A = davis_example_qr.astype(np.float64)
    f = SPQRFactor(A)
    with pytest.raises(ValueError, match="type.*does not match"):
        f.factorize(A.astype(np.complex128))


def test_bad_factorize_shape(davis_example_qr):
    A = davis_example_qr
    f = SPQRFactor(A)
    with pytest.raises(ValueError, match="shape.*does not match"):
        f.factorize(A[:-1, :])  # remove last row


@pytest.mark.xfail(reason="No pattern check is done in SPQRFactor")
def test_bad_factorize_structure(davis_example_qr):
    A = davis_example_qr
    f = SPQRFactor(A)
    assert_solve_dense(A, f)
    B = A.copy().todok()
    # Change the structure of the matrix by adding a new non-zero
    B[0, 1] = 2.3
    B = B.tocsc()
    B.indptr = B.indptr.astype(A.indptr.dtype)
    B.indices = B.indices.astype(A.indices.dtype)
    # No error is raised since there is no pattern check
    f.factorize(B)
    # Try solving a system to ensure factorization was successful
    assert_solve_dense(B, f)


@pytest.mark.xfail(reason="No pattern check is done in SPQRFactor")
def test_bad_refactorize_structure(davis_example_qr):
    A = davis_example_qr
    f = spqr_factor(A)
    assert_solve_dense(A, f)
    B = A.copy().todok()
    # Change the structure of the matrix by adding a new non-zero
    B[0, 1] = 2.3
    B = B.tocsc()
    B.indptr = B.indptr.astype(A.indptr.dtype)
    B.indices = B.indices.astype(A.indices.dtype)
    # No error is raised since there is no pattern check
    f.factorize(B)
    # Try solving a system to ensure factorization was successful
    assert_solve_dense(B, f)


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_davis_example_qr(davis_example_qr, itype, dtype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    A.data = A.data.astype(dtype)

    f = SPQRFactor(A).factorize(A)
    assert f.is_numeric

    # Get the factors
    p = f.perm

    # Values from MATLAB spqr
    # >> [Q, R, E] = spqr(A);
    # >> [p j x] = find(E);
    expect_p = np.array([0, 3, 2, 1, 7, 4, 5, 6], dtype=itype)

    assert_array_equal(p, expect_p)
    assert_solve_dense(A, f)


test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, shape_kind="M >= N", dtype=dtype
    )
]


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("A", test_As)
def test_copy_symbolic(A, itype):
    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = SPQRFactor(A)
    g = f.copy()
    assert g is not f
    assert g.shape == f.shape
    assert g.itype == f.itype
    assert g.dtype == f.dtype
    # Test that numeric factorization + solve can be done on the copy
    f.factorize(A)
    assert_solve_dense(A, f)
    del f  # ensure no shared state
    g.factorize(A)
    assert_solve_dense(A, g)


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("A", test_As)
def test_copy_numeric(A, itype):
    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = spqr_factor(A)
    g = f.copy()
    assert g is not f
    assert_solve_dense(A, f)
    del f  # ensure no shared state
    assert_solve_dense(A, g)


@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("A", test_As)
def test_refactor(A, itype, copy):
    A = A.copy()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = spqr_factor(A)
    # Create a new matrix with the same sparsity pattern but different values
    B = A.copy()
    rng = np.random.default_rng(56)
    B.data = rng.random(len(B.data)).astype(dtype=B.dtype)
    # Factor the new matrix with the same sparsity pattern
    if copy:
        g = f.copy()
        assert g is not f
        assert_solve_dense(A, f)  # make sure copy didn't affect f
        del f  # ensure no shared state
        # NOTE this test *does not* check that the numeric copy is correct,
        # because it entirely recomputes it with the factorize call.
        g.factorize(B)
        assert_solve_dense(B, g)
    else:
        f.factorize(B)
        assert_solve_dense(B, f)


# -----------------------------------------------------------------------------
#         Qmult
# -----------------------------------------------------------------------------
class TestBadQmultShape:
    @pytest.fixture(scope="class")
    def N(self):
        return 5

    @pytest.fixture(scope="class")
    def A(self, N):
        return sparse.eye_array(N).tocsc()

    @pytest.fixture(scope="class")
    def f(self, A):
        return spqr_factor(A)

    def test_X_0D_dense(self, f, A):
        X = np.empty([])
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.qmult(X)

    def test_X_3D_dense(self, f, A):
        X = np.empty((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.qmult(X)

    def test_X_3D_sparse(self, f, A):
        X = sparse.coo_array((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.qmult(X)

    def test_X_KD_dense(self, f, A, N):
        X = np.empty((N - 1, N))
        with pytest.raises(ValueError, match="compatible shape with Q"):
            f.qmult(X)

    def test_X_KD_sparse(self, f, A, N):
        X = sparse.csc_array((N - 1, N))
        with pytest.raises(ValueError, match="compatible shape with Q"):
            f.qmult(X)


@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_qmult(davis_example_qr, is_sparse, itype, dtype):
    A = davis_example_qr.astype(dtype)
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = spqr_factor(A)

    if is_sparse:
        I = sparse.eye_array(A.shape[0], dtype=dtype).tocsc()
        I.indptr = I.indptr.astype(itype)
        I.indices = I.indices.astype(itype)
    else:
        I = np.eye(A.shape[0], dtype=dtype)

    Q = f.qmult(I, "QX")
    QTQ = f.qmult(Q, "QTX")
    QQT = f.qmult(Q, "XQT")

    if is_sparse:
        assert_allclose(QTQ.toarray(), I.toarray(), atol=1e-15, strict=True)
        assert_allclose(QQT.toarray(), I.toarray(), atol=1e-15, strict=True)
    else:
        assert_allclose(QTQ, I, atol=1e-15, strict=True)
        assert_allclose(QQT, I, atol=1e-15, strict=True)

    QT = f.qmult(I, "QTX")
    QTQ = f.qmult(QT, "XQ")

    if is_sparse:
        assert_allclose(QTQ.toarray(), I.toarray(), atol=1e-15, strict=True)
    else:
        assert_allclose(QTQ, I, atol=1e-15, strict=True)


# -----------------------------------------------------------------------------
#         Solve
# -----------------------------------------------------------------------------
class TestBadSolveShape:
    @pytest.fixture(scope="class")
    def N(self):
        return 5

    @pytest.fixture(scope="class")
    def A(self, N):
        return sparse.eye_array(N).tocsc()

    @pytest.fixture(scope="class")
    def f(self, A):
        return spqr_factor(A)

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
    x = SPQRFactor(singleton_A).factorize(singleton_A).solve(b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = sparse.coo_array([1], dtype=dtype)
    x = SPQRFactor(singleton_A).factorize(singleton_A).solve(b)
    assert_allclose(x.toarray(), b.toarray())


@pytest.mark.parametrize("itype", ITYPES)
def test_itype_1D(davis_example_qr, itype):
    A = davis_example_qr
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    x = spqr_solve(A, b)
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
    x = spqr_solve(A, b)
    assert isinstance(x, sparse.csc_array)
    assert x.indptr.dtype == itype
    assert x.indices.dtype == itype


def test_exactly_singular(davis_example_qr):
    A = davis_example_qr.todok()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

    N = A.shape[0]
    lam0 = la.eigvalsh(A.toarray()).min()

    # Make A exactly singular
    s = 7
    A[:, s] = 0.0
    A[s, :] = 0.0
    A = A.tocsc()

    lam1 = la.eigvalsh(A.toarray()).min()
    print(f"\nMin eigenvalue: {lam0:.2e} -> {lam1:.2e}\n")

    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    f = spqr_factor(A)
    assert f.rank == N - 1

    # We just get a "0" in the singular position of the solution
    with pytest.warns(SPQRRankDeficiencyWarning, match="rank deficient"):
        x = f.solve(b)

    x = x.toarray()
    assert x[s] == 0.0
    idx = np.arange(N) != s
    assert_allclose(x[idx], expect_x.toarray()[idx], atol=1e-15, strict=True)
    assert_allclose(A @ x, b.toarray(), atol=1e-15, strict=True)


def test_nearly_singular(davis_example_qr):
    A = davis_example_qr.todok()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

    N = A.shape[0]
    lam0 = la.eigvalsh(A.toarray()).min()

    # Make A nearly singular
    s = 5  # arbitrary singular row/column
    A[:, s] = 0.0
    A[s, :] = 0.0
    A[s, s] = 0.5 * np.finfo(A.dtype).eps
    A = A.tocsc()

    lam1 = la.eigvalsh(A.toarray()).min()
    print(f"\nMin eigenvalue: {lam0:.2e} -> {lam1:.2e}\n")

    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    f = spqr_factor(A)
    assert f.rank == N - 1

    # We just get a "0" in the singular position of the solution
    with pytest.warns(SPQRRankDeficiencyWarning, match="rank deficient"):
        x = f.solve(b)

    x = x.toarray()
    assert x[s] == 0.0
    idx = np.arange(N) != s
    assert_allclose(x[idx], expect_x.toarray()[idx], atol=1e-15, strict=True)
    assert_allclose(A @ x, b.toarray(), atol=1e-15, strict=True)


# Base test function for solving systems
def _test_solve(A, K, is_sparse, transpose, underdetermined):
    atol = 1e-12
    A = A.copy()
    if underdetermined:
        A = A.T.conj().tocsc()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

    # Build RHS
    M, N = A.shape
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
    if not transpose:
        b = A @ expect_x
    else:
        b = A.T.conj() @ expect_x

    f = spqr_factor(A)
    assert f.rank == (N if not underdetermined else M)

    x = f.solve(b, transpose=transpose)

    # Check residuals in all cases
    if not transpose:
        resid = A @ x - b
    else:
        resid = A.T.conj() @ x - b

    if is_sparse:
        resid = resid.toarray()

    assert_allclose(resid, np.zeros_like(resid), atol=1e-10, strict=True)

    # In underdetermined case, the solution is not unique
    if not underdetermined:
        if is_sparse:
            assert_allclose(x.toarray(), expect_x.toarray(), atol=atol, strict=True)
        else:
            assert_allclose(x, expect_x, atol=atol, strict=True)


square_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, shape_kind="square", dtype=dtype
    )
]


@pytest.mark.parametrize("A", square_As)
@pytest.mark.parametrize("transpose", [False, True], ids=["A", "A^T"])
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_solve_square(A, K, is_sparse, transpose):
    _test_solve(A, K, is_sparse, transpose, underdetermined=False)


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize(
    "underdetermined", [False, True], ids=["overdetermined", "underdetermined"]
)
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_solve_overunder(A, K, is_sparse, underdetermined):
    _test_solve(A, K, is_sparse, transpose=False, underdetermined=underdetermined)


def test_min2norm(davis_example_qr):
    A = davis_example_qr.todok()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    A = A[:-2, :].tocsc()  # make underdetermined
    M, N = A.shape

    expect_x = np.arange(1, N + 1, dtype=A.dtype)
    b = A @ expect_x

    # Solve with min 2-norm solver
    x = spqr_solve(A, b, min2norm=True)

    assert_allclose(A @ x, b, atol=1e-15, strict=True)

    # Solve with normal solver and check norm
    xf = spqr_solve(A, b, min2norm=False)

    print()
    print(f"||x||_2  = {la.norm(x):.6e}")
    print(f"||xf||_2 = {la.norm(xf):.6e}")
    assert la.norm(x) <= la.norm(xf)


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


@pytest.mark.parametrize("problem", ["well1033", "illc1033", "well1850", "illc1850"])
def test_solve_real(problem):
    A, b = _load_problem(problem)
    # Solve the normal equations A^T A x = A^T b
    ATA = (A.T @ A).tocsc()
    ATb = A.T @ b
    expect_x = np.linalg.lstsq(A.toarray(), b)[0]
    x = spqr_solve(ATA, ATb)
    assert_allclose(x, expect_x, atol=1e-7)


# -----------------------------------------------------------------------------
#         Test Control and Info
# -----------------------------------------------------------------------------
def test_info(davis_example_qr):
    A = davis_example_qr
    f = spqr_factor(A)
    info = f.info
    print()
    print(info)
    # Values from MATLAB spqr
    # >> [Q, R, E, info] = spqr(A);
    assert info.nnzR_upper_bound == 36  # == 100
    assert info.nnzH_upper_bound == 9
    assert info.nf == 1
    assert info.rank_A_estimate == 8
    assert info.n1cols == 0
    assert info.n1rows == 0
    assert info.ordering == "colamd"
    assert info.memory > 0  # == 6184
    assert info.flops_upper_bound == 303  # == 847
    assert_allclose(info.tol, 5.1728e-13, rtol=1e-4)
    assert info.norm_E_fro == 0
    assert info.analyze_time > 0  # == 6.4135e-05
    assert info.factorize_time > 0  # == 3.0994e-05
    assert info.solve_time == 0  # == 1.1683e-05
    assert_allclose(
        info.total_time,  # == 1.7700e-04
        info.analyze_time + info.factorize_time + info.solve_time,
    )
    assert info.flops == 303  # == 847


ORDERS = [
    None,
    "default",
    "fixed",
    "natural",
    "colamd",
    "cholmod",
    "amd",
    "metis",
    "best",
    "bestamd",
]


def test_bad_ordering(davis_example_qr):
    A = davis_example_qr
    with pytest.raises(ValueError, match="Unknown ordering"):
        SPQRFactor(A, order="invalid")


@pytest.mark.parametrize("order", ORDERS)
def test_ordering(davis_example_qr, order):
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = spqr_factor(A, order=order)
    assert_solve_dense(A, f)


# -----------------------------------------------------------------------------
#         Test spqr
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("itype", ITYPES)
def test_spqr_r(A, itype):
    A = A.copy()
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    R, p = spqr(A, mode="r")
    RTR = (R.T.conj() @ R).toarray()
    ATA = (A.T.conj() @ A)[p[:, np.newaxis], p].toarray()
    assert_allclose(RTR, ATA, atol=1e-14, strict=True)


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("itype", ITYPES)
def test_spqr_full(A, itype):
    A = A.copy()
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    Q, R, p = spqr(A, mode="full")
    assert_allclose((Q @ R).toarray(), A[:, p].toarray(), atol=1e-14, strict=True)


# =============================================================================
# =============================================================================
