# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: helpers.py
#  Created: 2025-07-31 11:42
# =============================================================================

"""Helper functions for the scikit-sparse project unit tests."""

import numpy as np
import pytest

from scipy import sparse


def is_valid_permutation(p):
    """Check if a vector is a valid permutation."""
    return np.array_equal(np.sort(p), np.arange(len(p)))


def generate_random_matrices(
    seed=565656, N_trials=100, N_max=10, square_only=True, d_scale=1
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

        A = sparse.random_array((M, N), density=d, format="csc", rng=rng)

        yield pytest.param(A, id=f"random_{trial:02d}::{A.shape}::{A.nnz}")
