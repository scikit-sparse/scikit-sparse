# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_ldlsolve.py
#  Created: 2025-08-15 09:03
# =============================================================================

"""Unit tests for the cholmod.ldlsolve function."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sksparse.cholmod import CholmodError, ldl, ldlsolve

# from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]

# TODO: Add tests for the `ldlsolve` function, including:
# * Handling of different data types
# * Correctness of the solution for various input matrices
# * Edge cases such as empty matrices or zero matrices
# * Bad shapes for L, D, and b
# * Dense and sparse right-hand sides
# * Matrix and vector right-hand sides
# * Permuted systems (need to manually permute b, x for now)
# * *nearly* singular matrices (check for rcond errors)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_dense_input(itype, K):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    empty_b = np.empty((0, K))
    L, D = ldl(empty_A)
    x = ldlsolve(L, D, empty_b)
    assert_array_equal(x, empty_b, strict=True)


@pytest.mark.parametrize("K", [0, 1, 3])  # arbitrary number of rhs
@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_sparse_input(itype, K):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    empty_b = sparse.csc_array((0, K))
    L, D = ldl(empty_A)
    x = ldlsolve(L, D, empty_b)
    assert_array_equal(x.toarray(), empty_b.toarray(), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_LD = sparse.csc_array((N, N))
    zero_LD.indptr = zero_LD.indptr.astype(itype)
    zero_LD.indices = zero_LD.indices.astype(itype)
    with pytest.raises(CholmodError, match="is empty"):
        ldlsolve(zero_LD, zero_LD, np.zeros((N,)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_dense(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    L, D = ldl(singleton_A)
    b = np.array([1], dtype=dtype)
    x = ldlsolve(L, D, b)
    assert_allclose(x, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton_sparse(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    L, D = ldl(singleton_A)
    b = sparse.coo_array([1], dtype=dtype)
    x = ldlsolve(L, D, b)
    assert_allclose(x.toarray(), b.toarray())

