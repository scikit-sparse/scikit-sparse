# Test cases for the sksparse.camd module.
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_camd.py
#  Created: 2025-07-28 13:34
# =============================================================================

"""Test cases for the sksparse.camd module."""

import numpy as np
import pytest

from numpy.testing import assert_array_equal
from pathlib import Path
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning
from sksparse.camd import CAMDInfo, camd, camd_default_control

from .helpers import is_valid_permutation, generate_random_matrices


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    assert_array_equal(camd(empty_A), np.array([], dtype=itype), strict=True)


def test_1D_input():
    with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
        with pytest.raises(ValueError, match="Input must be square"):
            camd(np.arange(10))


def test_nonsquare_input():
    with pytest.raises(ValueError, match="Input must be square"):
        camd(sparse.csc_array((3, 4)))


def test_ND_input():
    rng = np.random.default_rng(565656)
    with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
        with pytest.raises(ValueError, match="Input must be convertible to CSC format"):
            camd(rng.random((2, 3, 4)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    assert_array_equal(camd(zero_A), np.arange(N, dtype=itype), strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_array([[1]])
    assert_array_equal(camd(singleton_A), np.array([0], dtype=np.int32), strict=True)


@pytest.mark.parametrize(
    "A",
    list(
        generate_random_matrices(
            N_trials=100, N_max=200, d_scale=0.05, square_only=True
        )
    ),
)
class TestRandomSquareMatrices:
    @pytest.mark.parametrize("matrix_type", ["dense", "csc", "coo"])
    def test_input_type(self, A, matrix_type):
        match matrix_type:
            case "dense":
                A = A.toarray()
            case "csc":
                A = A.tocsc()
            case "coo":
                A = A.tocoo()
            case _:
                raise ValueError(f"Unknown matrix type: {matrix_type}")

        if matrix_type != "csc":
            with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
                p = camd(A)
        else:
            p = camd(A)

        assert is_valid_permutation(p)

    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        p = camd(A)
        assert p.dtype == itype
        assert p.shape == (A.shape[0],)
        assert is_valid_permutation(p)

    @pytest.mark.parametrize("aggressive", [True, False])
    def test_aggressive(self, A, aggressive):
        p = camd(A, aggressive=aggressive)
        assert is_valid_permutation(p)


@pytest.mark.parametrize("dense_thresh", [None, 8, 5])
def test_camd_with_dense_rows(dense_thresh):
    N = 1000
    rng = np.random.default_rng(56)
    A = sparse.random_array((N, N), density=0.001, format="lil", rng=rng)

    # Create a known number of dense rows above the threshold
    # thresh is actually dense_thresh * sqrt(N) == dense_thresh * 10
    # for N = 1000, density = 0.001:
    #   A.astype(bool).sum(axis=1).max() ~ 5
    #   (A + A.T).astype(bool).sum(axis=1).max() ~ 10
    CAMD_DEFAULT_DENSE = 10  # default value from camd.h
    thresh = int(
        (dense_thresh if dense_thresh is not None else CAMD_DEFAULT_DENSE)
        * np.sqrt(N)
    )

    N_dense_rows = 10  # arbitrary choice for number of dense rows
    N_elems = min(2 * thresh, N)  # arbitrary choice to ensure enough elements

    dense_row_idx = rng.choice(N, size=N_dense_rows, replace=False)
    col_idx = np.array(
        [rng.choice(N, size=N_elems, replace=False) for _ in range(N_dense_rows)]
    )

    for i, js in zip(dense_row_idx, col_idx):
        A[i, js] = rng.random(size=N_elems)

    # Make sure the matrix is symmetric
    A = A + A.T
    A = A.tocsc()
    p = camd(A, dense_thresh=dense_thresh)

    assert is_valid_permutation(p)

    # Expect dense rows at the end of the permutation, but maybe not in order
    assert_array_equal(np.sort(p[-N_dense_rows:]), np.sort(dense_row_idx))


def test_info_can_24():
    # The can_24 matrix is used in the SuiteSparse CAMD MATLAB/camd_demo.m file.
    expect_info = CAMDInfo.from_array(
        np.array([
            0,     # status
            24,    # N
            160,   # nz
            1,     # symmetry
            24,    # nzdiag
            136,   # nz_A_plus_AT
            0,     # Ndense
            3288,  # memory
            0,     # Ncmpa
            97,    # Lnz
            97,    # Ndiv
            275,   # Nmultsubs_LDL
            453,   # Nmultsubs_LU
            8,     # dmax
        ]
        )
    )

    # Load the can_24 matrix from a file
    can_24_path = Path("tests") / "test_data" / "can_24"
    with can_24_path.open() as fp:
        can_24 = np.genfromtxt(fp, dtype=int)

    A = sparse.csc_array((can_24[:, 2], (can_24[:, 0] - 1, can_24[:, 1] - 1)))
    p, info = camd(A, return_info=True)

    assert is_valid_permutation(p)
    assert info == expect_info


def test_camd_default_control():
    # The default control settings are (from camd.h):
    # - CAMD_DEFAULT_DENSE      -> dense_thresh: 10.0
    # - CAMD_DEFAULT_AGGRESSIVE ->   aggressive: True
    expect_control = {
        "dense_thresh": 10.0,
        "aggressive": True,
    }
    control = camd_default_control()
    assert control == expect_control

    A = sparse.csc_array([[1, 2], [3, 4]])
    p = camd(A, **control)
    assert is_valid_permutation(p)


@pytest.fixture(scope="class")
def rand_matrix_A():
    N = 10
    rng = np.random.default_rng(56)
    A = sparse.random_array((N, N), density=0.4, format="lil", rng=rng)
    A.setdiag(N)
    A = A.tocsc()
    return A, N, rng


class TestConstraints:
    def test_complete_constraints(self, rand_matrix_A):
        A, N, _ = rand_matrix_A
        C = np.arange(N)
        q = camd(A, constraints=C)
        assert_array_equal(q, C)

    def test_general_constraints(self, rand_matrix_A):
        A, N, rng = rand_matrix_A
        # Set some constraints
        k = 3
        C = np.full(N, 2, dtype=int)
        all_idx = rng.permutation(N)
        C[all_idx[:k]] = 0
        C[all_idx[k:2*k]] = 1

        p = camd(A, constraints=C)

        assert is_valid_permutation(p)
        # Check that the constraints are respected
        assert all(C[p][:k] == 0)
        assert all(C[p][k:2*k] == 1)
        assert all(C[p][2*k:] == 2)


# =============================================================================
# =============================================================================
