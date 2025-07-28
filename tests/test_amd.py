#!/usr/bin/env python3
# =============================================================================
#     File: test_amd.py
#  Created: 2025-07-28 13:34
#   Author: Bernie Roesler
#
"""Test code for the sksparse.amd module."""
# =============================================================================

import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sksparse.amd import amd


def is_valid_permutation(p):
    """Check if a vector is a valid permutation."""
    return np.array_equal(np.sort(p), np.arange(len(p)))


def generate_random_matrices(
    seed=565656,
    N_trials=100,
    N_max=10,
    square_only=True,
    d_scale=1
):
    """Generate a list of random sparse matrices of maximum size N x N.

    Parameters
    ----------
    seed : int
        The random seed for reproducibility.
    N_trials : int
        Number of random matrices to generate.
    N_max : int
        Maximum size of the matrix (M, N) will be at most ``N_max`` x ``N_max``.
    square_only : bool
        If True, generate only square matrices (M == N).
    d_scale : float
        Scale factor for the density of the sparse matrix. The density will
        be a random value between 0 and ``d_scale``.

    Returns
    -------
    generator
        A generator yielding pytest parameters for random sparse matrices.
    """
    rng = np.random.default_rng(seed)
    for trial in range(N_trials):
        # Generate a random sparse matrix
        if square_only:
            M = N = rng.integers(1, N_max, endpoint=True)
        else:
            M, N = rng.integers(1, N_max, size=2, endpoint=True)

        d = d_scale * rng.random()  # density

        A = sparse.random_array(
            (M, N),
            density=d,
            format='csc',
            random_state=rng
        )

        yield pytest.param(A, id=f"random_{trial:02d}::{A.shape}::{A.nnz}")


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_matrix((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    assert_array_equal(amd(empty_A), np.array([], dtype=itype), strict=True)


def test_1D_input():
    with pytest.raises(ValueError, match="Input must be square"):
        amd(np.arange(10))


def test_nonsquare_input():
    with pytest.raises(ValueError, match="Input must be square"):
        amd(sparse.csc_matrix((3, 4)))


def test_ND_input():
    rng = np.random.default_rng(565656)
    with pytest.raises(ValueError, match="Input must be convertible to CSC format"):
        amd(rng.random((2, 3, 4)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_matrix((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    assert_array_equal(amd(zero_A), np.arange(N, dtype=itype), strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_matrix([[1]])
    assert_array_equal(amd(singleton_A), np.array([0], dtype=np.int32), strict=True)


@pytest.mark.parametrize(
    "A",
    list(generate_random_matrices(
        N_trials=100,
        N_max=200,
        d_scale=0.05,
        square_only=True
    )),
)
class TestRandomSquareMatrices:
    @pytest.mark.parametrize("matrix_type", ['dense', 'csc', 'coo'])
    def test_input_type(self, A, matrix_type):
        match matrix_type:
            case 'dense':
                A = A.toarray()
            case 'csc':
                A = A.tocsc()
            case 'coo':
                A = A.tocoo()
            case _:
                raise ValueError(f"Unknown matrix type: {matrix_type}")

        p = amd(A)
        assert is_valid_permutation(p)

    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        p = amd(A)
        assert p.dtype == itype
        assert p.shape == (A.shape[0],)
        assert is_valid_permutation(p)

# =============================================================================
# =============================================================================
