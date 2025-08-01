# Test cases for the sksparse.colamd module.
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_colamd.py
#  Created: 2025-07-31 11:42
# =============================================================================

"""Test cases for the sksparse.colamd module."""

# import matplotlib.pyplot as plt  # DEBUG only
import numpy as np
import pytest

from numpy.testing import assert_array_equal
from pathlib import Path
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning
from sksparse.colamd import COLAMDStats, colamd, colamd_get_defaults

from .helpers import is_valid_permutation, generate_random_matrices


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    assert_array_equal(colamd(empty_A), np.array([], dtype=itype), strict=True)


def test_1D_row_input():
    with pytest.raises(ValueError, match="must be 2D"):
        colamd(np.arange(10))


def test_2D_row_input():
    with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
        N = 10
        q = colamd(np.arange(N)[np.newaxis, :])  # (1, N)
        assert_array_equal(q, np.arange(N, dtype=np.int32), strict=True)


def test_2D_col_input():
    with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
        N = 10
        q = colamd(np.arange(N)[:, np.newaxis])  # (N, 1)
        assert_array_equal(q, np.zeros(1, dtype=np.int32), strict=True)


def test_ND_input():
    rng = np.random.default_rng(565656)
    with pytest.raises(ValueError, match="must be 2D"):
        colamd(rng.random((2, 3, 4)))


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    assert_array_equal(colamd(zero_A), np.arange(N, dtype=itype), strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_array([[1]])
    assert_array_equal(colamd(singleton_A), np.array([0], dtype=np.int32), strict=True)


@pytest.mark.parametrize(
    "A", list(generate_random_matrices(N_trials=100, N_max=200, d_scale=0.05)),
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
                q = colamd(A)
        else:
            q = colamd(A)

        assert is_valid_permutation(q)

    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        q = colamd(A)
        assert q.dtype == itype
        assert q.shape == (A.shape[0],)
        assert is_valid_permutation(q)

    @pytest.mark.parametrize("aggressive", [True, False])
    def test_aggressive(self, A, aggressive):
        q = colamd(A, aggressive=aggressive)
        assert is_valid_permutation(q)


COLAMD_DEFAULT_DENSE = 10  # NOTE depends on the default in colamd.c
DENSE_THRESHOLDS = [None, 5, 2]

@pytest.mark.parametrize("row_or_col", ["row", "col"])
@pytest.mark.parametrize("dense_thresh", DENSE_THRESHOLDS)
def test_colamd_with_dense(dense_thresh, row_or_col):
    N = 1000
    rng = np.random.default_rng(56)
    A = sparse.random_array((N, N), density=0.001, format="lil", rng=rng)

    # Create a known number of dense rows above the threshold
    # thresh is actually dense_thresh * sqrt(N) == dense_thresh * 10
    # max(A[i] for i in range(N)) is ~ 5 for N = 1000, density = 0.001
    thresh = int(
        (dense_thresh if dense_thresh is not None else COLAMD_DEFAULT_DENSE) * np.sqrt(N)
    )

    N_dense = 10  # arbitrary choice for number of dense rows
    N_elems = min(2 * thresh, N)  # arbitrary choice to ensure enough elements

    dense_idx = rng.choice(N, size=N_dense, replace=False)
    other_idx = rng.choice(N, size=N_elems, replace=False)
    for i in dense_idx:
        if row_or_col == "row":
            A[i, other_idx] = rng.random(size=len(other_idx))
        else:
            A[other_idx, i] = rng.random(size=len(other_idx))

    A = A.tocsc()
    if row_or_col == "row":
        q, stats = colamd(A, dense_row_thresh=dense_thresh, return_info=True)
    else:
        q, stats = colamd(A, dense_col_thresh=dense_thresh, return_info=True)

    assert is_valid_permutation(q)

    # DEBUG: plot the matrix before and after permutation
    # fig, axs = plt.subplots(num=1, ncols=2, clear=True)
    # axs[0].spy(A, markersize=1)
    # axs[1].spy(A[:, q], markersize=1)
    # plt.show()

    # Expect dense cols at the end of the permutation, but maybe not in order
    # NOTE *empty* columns are also moved to the end of the matrix,
    # so we need to check the stats.N_cols_ignored value
    if row_or_col == "col":
        assert_array_equal(
            np.sort(q[-stats.N_cols_ignored:-(stats.N_cols_ignored - N_dense)]),
            np.sort(dense_idx)
        )


def test_info_can_24():
    # The can_24 matrix is used in the SuiteSparse AMD MATLAB/amd_demo.m file.
    expect_info = COLAMDStats.from_array(
        np.array([
            0,   # N_rows_ignored
            0,   # N_cols_ignored
            1,   # Ncmpa
            0,   # status
            -1,  # info1
            -1,  # info2
            0,   # info3
        ])
    )

    # Load the can_24 matrix from a file
    can_24_path = Path("tests") / "test_data" / "can_24"
    with can_24_path.open() as fp:
        can_24 = np.genfromtxt(fp, dtype=int)

    A = sparse.csc_array((can_24[:, 2], (can_24[:, 0] - 1, can_24[:, 1] - 1)))
    q, info = colamd(A, return_info=True)

    assert is_valid_permutation(q)
    assert info == expect_info


def test_colamd_defaults():
    """Test that COLAMD uses the default control settings."""
    # The default control settings are (from colamd.c:1095-1097):
    # knobs[COLAMD_DENSE_ROW] = 10 ;
    # knobs[COLAMD_DENSE_COL] = 10 ;
    # knobs[COLAMD_AGGRESSIVE] = TRUE ;
    expect_knobs = {
        "dense_row_thresh": 10,
        "dense_col_thresh": 10,
        "aggressive": True,
    }
    knobs = colamd_get_defaults()
    assert knobs == expect_knobs

    # A = sparse.csc_array([[1, 2], [3, 4]])
    # p = amd(A, **knobs)
    # assert is_valid_permutation(p)

# # =============================================================================
# # =============================================================================
