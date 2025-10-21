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
from scipy import linalg as la
from scipy import sparse
from sksparse.umfpack import (
    UMFControl,
    UMFFactor,
    UMFPACKError,
    UMFPACKNonpositiveError,
    UMFPACKSingularMatrixWarning,
    UMFPACKWarning,
)

from .helpers import generate_random_matrices

ITYPES = [np.int32, np.int64]
DTYPES = [np.float64, np.complex128]


def assert_LU_equals_A(f, A, atol=1e-15):
    """Check that L U = P R A Q."""
    L, U, p, q, r = f.L, f.U, f.perm_r, f.perm_c, f.R
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


@pytest.mark.parametrize("copy", [False, True])
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
        return UMFFactor(A).factorize(A)

    def test_b_0D_dense(self, f, A):
        b = np.empty([])
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(A, b)

    def test_b_3D_dense(self, f, A):
        b = np.empty((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(A, b)

    def test_b_3D_sparse(self, f, A):
        b = sparse.coo_array((2, 3, 4))
        with pytest.raises(ValueError, match="must be a 1D or 2D array"):
            f.solve(A, b)

    def test_b_KD_dense(self, f, A, N):
        b = np.empty((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as A"):
            f.solve(A, b)

    def test_b_KD_sparse(self, f, A, N):
        b = sparse.csc_array((N - 1, N))
        with pytest.raises(ValueError, match="same number of rows as A"):
            f.solve(A, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = np.array([1], dtype=dtype)
    x = UMFFactor(singleton_A).factorize(singleton_A).solve(singleton_A, b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    b = sparse.coo_array([1], dtype=dtype)
    x = UMFFactor(singleton_A).factorize(singleton_A).solve(singleton_A, b)
    assert_allclose(x.toarray(), b.toarray())


def test_exactly_singular(davis_example_qr):
    A = davis_example_qr.todok()
    A.setdiag(A.diagonal() + 1.0)  # make non-singular

    N = A.shape[0]
    lam0 = la.eigvalsh(A.toarray()).min()

    # Make A exactly singular
    A[:, -1] = 0.0
    A[-1, :] = 0.0
    A = A.tocsc()

    lam1 = la.eigvalsh(A.toarray()).min()
    print(f"\nMin eigenvalue: {lam0:.2e} -> {lam1:.2e}\n")

    expect_x = sparse.coo_array(np.arange(1, N + 1, dtype=A.dtype))
    b = A @ expect_x
    with pytest.raises(UMFPACKError, match="indefinite or singular"):
        # FIXME shouldn't warn *and* raise
        with pytest.warns(UMFPACKSingularMatrixWarning, match="is singular"):
            UMFFactor(A).factorize(A).solve(A, b)


# FIXME doesn't warn?
@pytest.mark.xfail(reason="FIXME")
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
    with pytest.warns(UMFPACKWarning, match="nearly singular"):
        UMFFactor(A).factorize(A).solve(A, b)


@pytest.mark.parametrize("A", test_As)
@pytest.mark.parametrize("K", [0, 1, 3], ids=lambda k: f"K={k}")
@pytest.mark.parametrize("is_sparse", [False, True], ids=["dense", "sparse"])
def test_solve(A, K, is_sparse):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-5

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
    x = UMFFactor(A).factorize(A).solve(A, b)

    # Compare
    if is_sparse:
        assert_allclose(x.toarray(), expect_x.toarray(), atol=atol)
    else:
        assert_allclose(x, expect_x, atol=atol)


# -----------------------------------------------------------------------------
#         Test Info and Control
# -----------------------------------------------------------------------------
# Copied from umfpack.h, subject to change
CONTROL_DEFAULTS = {
    'print_level': 1,
    'dense_row': 0.2,
    'dense_col': 0.2,
    'pivot_tol': 0.1,
    'sym_pivot_tol': 0.001,
    'blas3_block_size': 32,
    'alloc_init': 0.7,
    'front_alloc_init': 0.5,
    'ir_steps': 2,
    'row_scale': 1,  # UMFPACK_SCALE_SUM
    'strategy': 0,  # UMFPACK_STRATEGY_AUTO
    'amd_dense': 10.0,  # AMD_DEFAULT_DENSE
    'fixQ': 0,
    'aggressive': 1,
    'droptol': 0,
    'ordering_method': 1,  # UMFPACK_ORDERING_AMD
    'singletons': True,
    'sym_thresh': 0.3,
    'nnzdiag_thresh': 0.9,
}


def test_default_controls():
    c = UMFControl()
    for key, expect_value in CONTROL_DEFAULTS.items():
        actual_value = getattr(c, key)
        assert actual_value == expect_value


# TODO test IRSTEP == 0 and don't pass in A to solve()
def test_ir_steps(davis_example_qr):
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    N_steps = 0
    c = UMFControl(ir_steps=N_steps)  # arbitrary > default
    c.report()
    f = UMFFactor(A, control=c)
    assert f.control.ir_steps == N_steps
    f.factorize(A)
    expect_x = np.arange(1, A.shape[0] + 1, dtype=A.dtype)
    b = A @ expect_x
    f.solve(A, b)
    print(f"{f.info.ir_attempted=}, {f.info.ir_attempted=}")  # FIXME?
    assert f.info.ir_attempted == N_steps


# NOTE the integer values are from umfpack.h and subject to change
SCALES = [None, "none", "sum", "max"]
SCALE_MAP = {
    None: 0,
    "none": 0,  # UMFPACK_SCALE_NONE
    "sum": 1,   # UMFPACK_SCALE_SUM
    "max": 2    # UMFPACK_SCALE_MAX
}


@pytest.mark.parametrize("scale", SCALE_MAP)
def test_row_scale(davis_example_qr, scale):
    expect_scale = SCALE_MAP[scale]
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    f = UMFFactor(A)
    # Row scaling can be done *after* symbolic, but *before* numeric
    f.control.row_scale = scale
    f.factorize(A)
    assert f.control.row_scale == expect_scale  # TODO
    expect_x = np.arange(1, A.shape[0] + 1, dtype=A.dtype)
    b = A @ expect_x
    f.solve(A, b)
    assert_LU_equals_A(f, A)
    assert f.info.was_scaled == expect_scale


# NOTE the integer values are from umfpack.h and subject to change
ORDERINGS = [None, "none", "cholmod", "amd", "metis", "best"]
ORDERING_MAP = {
    "cholmod": 0,  # UMFPACK_ORDERING_CHOLMOD
    "amd": 1,      # UMFPACK_ORDERING_AMD
    "metis": 3,    # UMFPACK_ORDERING_METIS
    "best": 4,     # UMFPACK_ORDERING_BEST
    "none": 5,     # UMFPACK_ORDERING_NONE
    None: 5,
}


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_ordering(davis_example_qr, ordering):
    expect_ordering = ORDERING_MAP[ordering]
    A = davis_example_qr
    A.setdiag(A.diagonal() + 1.0)  # make non-singular
    # Ordering must be done *before* symbolic factorization
    c = UMFControl(ordering_method=ordering)
    f = UMFFactor(A, control=c)
    assert f.control.ordering_method == expect_ordering  # TODO
    f.factorize(A)
    expect_x = np.arange(1, A.shape[0] + 1, dtype=A.dtype)
    b = A @ expect_x
    f.solve(A, b)
    assert_LU_equals_A(f, A)
    if ordering in [None, "none", "amd", "metis"]:
        assert f.info.ordering_used == expect_ordering
    else:  # ["cholmod", "best"]
        # May choose AMD or METIS
        assert f.info.ordering_used in [1, 3]



# =============================================================================
# =============================================================================
