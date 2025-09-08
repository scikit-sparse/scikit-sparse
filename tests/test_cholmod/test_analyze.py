# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_analyze.py
#  Created: 2025-08-18 21:08
# =============================================================================

"""Unit tests for the cholmod.analyze function."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.cholmod import CholeskyFactor, CholmodNotPositiveDefiniteError

from ..helpers import generate_random_matrices, is_valid_permutation

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.fixture
def A_default():
    return sparse.csc_array([[1, 2], [3, 4]])


def test_bad_kind(A_default):
    with pytest.raises(ValueError, match="Unknown factorization kind"):
        CholeskyFactor(A_default, kind="invalid")


def test_bad_order(A_default):
    with pytest.raises(ValueError, match="Unknown ordering method"):
        CholeskyFactor(A_default, order="invalid")


def test_empty_input():
    empty_A = sparse.csc_array((0, 0))
    f = CholeskyFactor(empty_A)
    p = f.get_perm()
    count = f.get_colcount()
    empty_p = np.array([], dtype=empty_A.indptr.dtype)
    assert_array_equal(p, empty_p, strict=True)
    assert_array_equal(count, empty_p, strict=True)


def test_zero_input():
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    with pytest.raises(CholmodNotPositiveDefiniteError, match="not positive definite"):
        CholeskyFactor(zero_A)


@pytest.mark.parametrize("dtype", DTYPES)
def test_singleton(dtype):
    singleton_A = sparse.csc_array([[1]], dtype=dtype)
    f = CholeskyFactor(singleton_A)
    p = f.get_perm()
    count = f.get_colcount()
    expect_p = np.array([0], dtype=singleton_A.indptr.dtype)
    expect_count = np.array([1], dtype=singleton_A.indptr.dtype)
    assert_array_equal(p, expect_p, strict=True)
    assert_array_equal(count, expect_count, strict=True)


# TODO change this test to use Davis Cholesky example matrix
# Declare a single random matrix fixture for some tests
@pytest.fixture(
    params=list(
        generate_random_matrices(
            N_trials=1, N_max=200, d_scale=0.05, pos_def_only=True
        ),
    )
)
def A_random(request):
    return request.param


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_itype(A_random, itype):
    A = A_random
    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    N = A.shape[0]
    f = CholeskyFactor(A)
    p = f.get_perm()
    count = f.get_colcount()
    assert p.dtype == itype
    assert count.dtype == itype
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)


# -----------------------------------------------------------------------------
#         Test many random matrices of various dtypes
# -----------------------------------------------------------------------------
posdef_As = list(
    generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True)
)


@pytest.mark.parametrize("A", posdef_As)
@pytest.mark.parametrize(
    "order",
    [
        None,
        "default",
        "best",
        "natural",
        "amd",
        "metis",
        "nesdis",
        "colamd",
        "postordered",
    ],
)
def test_order(A, order):
    N = A.shape[0]
    f = CholeskyFactor(A, order=order)
    p = f.get_perm()
    count = f.get_colcount()
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)


@pytest.mark.parametrize("A", posdef_As)
@pytest.mark.parametrize("kind", [None, "sym"])
def test_kind_sym(A, kind):
    N = A.shape[0]
    f = CholeskyFactor(A, kind=kind)
    p = f.get_perm()
    count = f.get_colcount()
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)


@pytest.mark.parametrize(
    "A", list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))
)
@pytest.mark.parametrize("kind", ["row", "col"])
def test_kind_rowcol(A, kind):
    N = A.shape[0] if kind == "row" else A.shape[1]
    f = CholeskyFactor(A, kind=kind)
    p = f.get_perm()
    count = f.get_colcount()
    assert is_valid_permutation(p)
    assert len(count) == N
    assert np.all(count >= 0)
    assert np.all(count <= N)
