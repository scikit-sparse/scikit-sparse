# Test cases for the sksparse.amd module.
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_amd.py
#  Created: 2025-07-28 13:34
# =============================================================================

"""Test cases for the sksparse.amd module."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from sksparse.amd import AMDInfo, amd, amd_default_control

from .helpers import generate_random_matrices, is_valid_permutation


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_empty_input(itype):
    empty_A = sparse.csc_array((0, 0))
    empty_A.indptr = empty_A.indptr.astype(itype)
    empty_A.indices = empty_A.indices.astype(itype)
    assert_array_equal(amd(empty_A), np.array([], dtype=itype), strict=True)


@pytest.mark.parametrize("itype", [np.int32, np.int64])
def test_zero_input(itype):
    N = 10  # arbitrary
    zero_A = sparse.csc_array((N, N))
    zero_A.indptr = zero_A.indptr.astype(itype)
    zero_A.indices = zero_A.indices.astype(itype)
    assert_array_equal(amd(zero_A), np.arange(N, dtype=itype), strict=True)


def test_singleton_matrix():
    singleton_A = sparse.csc_array([[1]])
    assert_array_equal(amd(singleton_A), np.array([0], dtype=np.int32), strict=True)


@pytest.mark.parametrize(
    "A",
    list(
        generate_random_matrices(
            N_trials=100, N_max=200, d_scale=0.05, shape_kind="square"
        )
    ),
)
class TestRandomSquareMatrices:
    @pytest.mark.parametrize("itype", [np.int32, np.int64])
    def test_itype(self, A, itype):
        A.indptr = A.indptr.astype(itype)
        A.indices = A.indices.astype(itype)
        p = amd(A)
        assert p.dtype == itype
        assert p.shape == (A.shape[0],)
        assert is_valid_permutation(p)

    @pytest.mark.parametrize("aggressive", [True, False])
    def test_aggressive(self, A, aggressive):
        p = amd(A, aggressive=aggressive)
        assert is_valid_permutation(p)


DENSE_THRESHOLDS = [None, 5, 2]


@pytest.mark.parametrize("dense_thresh", DENSE_THRESHOLDS)
def test_amd_with_dense_rows(dense_thresh):
    N = 1000
    rng = np.random.default_rng(56)
    A = sparse.random_array((N, N), density=0.001, format="lil", rng=rng)

    # Create a known number of dense rows above the threshold
    # thresh is actually dense_thresh * sqrt(N) == dense_thresh * 10
    # max(A[i] for i in range(N)) is ~ 5 for N = 1000, density = 0.001
    AMD_DEFAULT_DENSE = 10
    thresh = int(
        (dense_thresh if dense_thresh is not None else AMD_DEFAULT_DENSE) * np.sqrt(N)
    )

    N_dense_rows = 10  # arbitrary choice for number of dense rows
    N_elems = min(2 * thresh, N)  # arbitrary choice to ensure enough elements

    dense_row_idx = rng.choice(N, size=N_dense_rows, replace=False)
    col_idx = rng.choice(N, size=N_elems, replace=False)
    for i in dense_row_idx:
        # Ensure the row is dense enough
        A[i, col_idx] = rng.random(size=len(col_idx))

    A = A + A.T
    A = A.tocsc()
    p = amd(A, dense_thresh=dense_thresh)

    assert is_valid_permutation(p)

    # Expect dense row at the end of the permutation, but maybe not in order
    assert_array_equal(np.sort(p[-N_dense_rows:]), np.sort(dense_row_idx))


def test_info_can_24():
    # The can_24 matrix is used in the SuiteSparse AMD MATLAB/amd_demo.m file.
    expect_info = AMDInfo.from_array(
        np.array([
            0,     # status
            24,    # N
            160,   # nz
            1,     # symmetry
            24,    # nzdiag
            136,   # nz_A_plus_AT
            0,     # Ndense
            3032,  # memory
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
    can_24_path = Path("tests") / "data" / "can_24"
    with can_24_path.open() as fp:
        can_24 = np.genfromtxt(fp, dtype=int)

    A = sparse.csc_array((can_24[:, 2], (can_24[:, 0] - 1, can_24[:, 1] - 1)))
    p, info = amd(A, return_info=True)

    assert is_valid_permutation(p)
    assert info == expect_info


def test_amd_default_control():
    """Test that AMD uses the default control settings."""
    # The default control settings are (from amd.h):
    # - AMD_DEFAULT_DENSE      -> dense_thresh: 10.0
    # - AMD_DEFAULT_AGGRESSIVE ->   aggressive: True
    expect_control = {
        "dense_thresh": 10.0,
        "aggressive": True,
    }
    control = amd_default_control()
    assert control == expect_control

    A = sparse.csc_array([[1, 2], [3, 4]])
    p = amd(A, **control)
    assert is_valid_permutation(p)


# =============================================================================
# =============================================================================
