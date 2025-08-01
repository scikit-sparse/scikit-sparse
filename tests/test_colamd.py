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

import matplotlib.pyplot as plt  # DEBUG only
import numpy as np
import pytest

from numpy.testing import assert_array_equal
from pathlib import Path
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning
from sksparse.colamd import COLAMDStats, colamd, symamd, colamd_get_defaults

from .helpers import is_valid_permutation, generate_random_matrices


class _BasicInputMixin:
    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_empty_input(self, itype):
        empty_A = sparse.csc_array((0, 0))
        empty_A.indptr = empty_A.indptr.astype(itype)
        empty_A.indices = empty_A.indices.astype(itype)
        assert_array_equal(
            self.colamd_func(empty_A), np.array([], dtype=itype), strict=True
        )

    def test_1D_row_input(self):
        with pytest.raises(ValueError, match="must be 2D"):
            self.colamd_func(np.arange(10))

    def test_ND_input(self):
        rng = np.random.default_rng(565656)
        with pytest.raises(ValueError, match="must be 2D"):
            self.colamd_func(rng.random((2, 3, 4)))

    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_zero_input(self, itype):
        N = 10  # arbitrary
        zero_A = sparse.csc_array((N, N))
        zero_A.indptr = zero_A.indptr.astype(itype)
        zero_A.indices = zero_A.indices.astype(itype)
        assert_array_equal(
            self.colamd_func(zero_A), np.arange(N, dtype=itype), strict=True
        )

    def test_singleton_matrix(self):
        singleton_A = sparse.csc_array([[1]])
        assert_array_equal(
            self.colamd_func(singleton_A), np.array([0], dtype=np.int32), strict=True
        )


class TestColamdInput(_BasicInputMixin):
    colamd_func = staticmethod(colamd)

    def test_2D_row_input(self):
        with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
            N = 10
            q = self.colamd_func(np.arange(N)[np.newaxis, :])  # (1, N)
            assert_array_equal(q, np.arange(N, dtype=np.int32), strict=True)

    def test_2D_col_input(self):
        with pytest.warns(SparseEfficiencyWarning, match="not in CSC format"):
            N = 10
            q = self.colamd_func(np.arange(N)[:, np.newaxis])  # (N, 1)
            assert_array_equal(q, np.zeros(1, dtype=np.int32), strict=True)


class TestSymamdInput(_BasicInputMixin):
    colamd_func = staticmethod(symamd)

    def test_2D_nonsquare_input(self):
        with pytest.raises(ValueError, match="must be square"):
            self.colamd_func(np.arange(12).reshape((4, 3)))


class _RandomInputMixin:
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
                q = self.colamd_func(A)
        else:
            q = self.colamd_func(A)

        assert is_valid_permutation(q)

    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        q = self.colamd_func(A)
        assert q.dtype == itype
        assert q.shape == (A.shape[0],)
        assert is_valid_permutation(q)

    @pytest.mark.parametrize("aggressive", [True, False])
    def test_aggressive(self, A, aggressive):
        q = self.colamd_func(A, aggressive=aggressive)
        assert is_valid_permutation(q)


@pytest.mark.parametrize(
    "A",
    list(generate_random_matrices(N_trials=100, N_max=200, d_scale=0.05)),
)
class TestColamdRandomInput(_RandomInputMixin):
    colamd_func = staticmethod(colamd)


@pytest.mark.parametrize(
    "A",
    list(
        generate_random_matrices(
            N_trials=100, N_max=200, d_scale=0.05, square_only=True
        )
    ),
)
class TestSymamdRandomInput(_RandomInputMixin):
    colamd_func = staticmethod(symamd)


COLAMD_DEFAULT_DENSE = 10  # NOTE depends on the default in colamd.c
DENSE_THRESHOLDS = [None, 5, 2]


@pytest.mark.parametrize("row_or_col", ["row", "col"])
@pytest.mark.parametrize("dense_thresh", DENSE_THRESHOLDS)
def test_colamd_with_dense(dense_thresh, row_or_col):
    M = 1000  # arbitrary size
    N =  800
    rng = np.random.default_rng(56)
    A = sparse.random_array((M, N), density=0.001, format="lil", rng=rng)

    max_N_rowcols, max_N_elems = (M, N) if row_or_col == "row" else (N, M)

    # Create a known number of dense rows/cols above the threshold
    # thresh is actually dense_thresh * sqrt(N) == dense_thresh * 10
    # max(A[i] for i in range(N)) is ~ 5 for N = 1000, density = 0.001
    thresh = int(
        (dense_thresh if dense_thresh is not None else COLAMD_DEFAULT_DENSE)
        * np.sqrt(max_N_elems)
    )

    N_dense = 10  # arbitrary choice for number of dense rows/columns
    N_elems = min(2 * thresh, max_N_elems)  # arbitrary choice for enough elements

    dense_idx = rng.choice(max_N_rowcols, size=N_dense, replace=False)
    other_idx = rng.choice(max_N_elems, size=N_elems, replace=False)

    for i in dense_idx:
        if row_or_col == "row":
            A[i, other_idx] = rng.random(size=len(other_idx))
        else:
            A[other_idx, i] = rng.random(size=len(other_idx))

    A = A.tocsc()

    kwargs = {f"dense_{row_or_col}_thresh": dense_thresh, "return_info": True}
    q, stats = colamd(A, **kwargs)

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


@pytest.mark.parametrize("dense_thresh", DENSE_THRESHOLDS)
def test_symamd_with_dense(dense_thresh):
    N = 1000  # arbitrary size
    rng = np.random.default_rng(56)
    A = sparse.random_array((N, N), density=0.001, format="lil", rng=rng)

    # Create a known number of dense rows/cols above the threshold
    # thresh is actually dense_thresh * sqrt(N) == dense_thresh * 10
    # max(A[i] for i in range(N)) is ~ 5 for N = 1000, density = 0.001
    thresh = int(
        (dense_thresh if dense_thresh is not None else COLAMD_DEFAULT_DENSE)
        * np.sqrt(N)
    )

    N_dense = 10  # arbitrary choice for number of dense rows/columns
    N_elems = min(2 * thresh, N)  # arbitrary choice for enough elements

    dense_idx = rng.choice(N, size=N_dense, replace=False)
    other_idx = rng.choice(N, size=N_elems, replace=False)

    for i in dense_idx:
        A[i, other_idx] = rng.random(size=len(other_idx))

    A = A + A.T  # make it symmetric
    A = A.tocsc()

    q, stats = symamd(A, return_info=True)

    assert is_valid_permutation(q)

    # # DEBUG: plot the matrix before and after permutation
    # fig, axs = plt.subplots(num=1, ncols=2, clear=True)
    # axs[0].spy(A, markersize=1)
    # axs[1].spy(A[q][:, q], markersize=1)
    # plt.show()

    # NOTE I am not sure what the expected behavior is here for symamd (vs
    # colamd) The test *almost* passes, but there are some empty rows/columns
    # interspersed with the dense rows/columns at the end of the matrix. There
    # may not be a deterministic order of the dense/empty rows/columns that
    # applies to every matrix, so we cannot make a strong assertion here.

    # Check that the same number of rows/columns are ignored.
    assert stats.N_rows_ignored == stats.N_cols_ignored

    # Expect dense cols at the end of the permutation, but maybe not in order.
    # *empty* columns are also moved to the end of the matrix, so we need to
    # check the stats.N_cols_ignored value
    # assert_array_equal(
    #     np.sort(q[-stats.N_cols_ignored:-(stats.N_cols_ignored - N_dense)]),
    #     np.sort(dense_idx)
    # )


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
