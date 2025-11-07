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
from scipy import sparse

from sksparse.spqr import (
    SPQRFactor,
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
    # assert_allclose(f.perm, np.array([], dtype=itype), strict=True)


@pytest.mark.parametrize("itype", ITYPES)
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    f = SPQRFactor(zero_A)
    assert f.rank == 0
    # assert_allclose(f.Q.toarray(), np.eye(N, dtype=A.dtype), strict=True)
    # assert_allclose(f.R.toarray(), np.array([], dtype=A.dtype), strict=True)
    # assert_allclose(f.perm, np.arange(N, dtype=itype), strict=True)


# def test_singleton():
#     dtype = np.float64
#     singleton_A = sparse.csc_array([[1]], dtype=dtype)
#     f = spqr_factor(singleton_A)
#     assert f.is_numeric


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


# =============================================================================
# =============================================================================
