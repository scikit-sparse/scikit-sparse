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

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.spqr import (
    SPQRFactor,
    spqr_factor,
)

ITYPES = [np.int32, np.int64]
DTYPES = [np.float64, np.complex128]


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


# # TODO unclear what happens here, but no error is raised
# def test_bad_factorize_structure(davis_example_qr):
#     A = davis_example_qr
#     f = SPQRFactor(A)
#     B = A.copy().todok()
#     # Change the structure of the matrix by adding a new non-zero
#     B[0, 1] = 2.3
#     B = B.tocsc()
#     B.indptr = B.indptr.astype(A.indptr.dtype)
#     B.indices = B.indices.astype(A.indices.dtype)
#     f.factorize(B)
#     # with pytest.raises(UMFPACKDifferentPatternError, match="different nonzero pattern"):
#     #     f.factorize(B)


# # TODO unclear what happens here, but no error is raised
# def test_bad_refactorize_structure(davis_example_qr):
#     A = davis_example_qr
#     f = spqr_factor(A)
#     B = A.copy().todok()
#     # Change the structure of the matrix by adding a new non-zero
#     B[0, 1] = 2.3
#     B = B.tocsc()
#     B.indptr = B.indptr.astype(A.indptr.dtype)
#     B.indices = B.indices.astype(A.indices.dtype)
#     f.factorize(B)
#     # with pytest.raises(UMFPACKDifferentPatternError, match="different nonzero pattern"):
#     #     f.factorize(B)


# TODO
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
    # assert_QR_equals_A(f, A) # TODO

