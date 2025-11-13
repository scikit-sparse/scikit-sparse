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

import operator

import numpy as np
import pytest
from scipy import sparse


def is_valid_permutation(p, N=None):
    """Check if a vector is a valid permutation."""
    if N is None:
        N = len(p)
    return np.array_equal(np.sort(p), np.arange(N))


def _get_dims(op_str, N_max, rng):
    """Compute random dimensions M, N with given relationship.

    Parameters
    ----------
    op_str : str
        A string representing the relationship between M and N.
    N_max : int
        Maximum size for M and N.
    rng : np.random.Generator
        Random number generator.
    """
    if op_str == "==":
        M = N = rng.integers(1, N_max, endpoint=True)
    elif op_str in [">", "<"]:
        if N_max < 2:
            raise ValueError(
                "'N_max' must be at least 2 for over/underdetermined shapes."
            )
        dim_small = rng.integers(1, N_max - 1, endpoint=True)
        dim_large = rng.integers(dim_small + 1, N_max, endpoint=True)

        if op_str == ">":
            M, N = dim_large, dim_small
        else:  # op == "<"
            M, N = dim_small, dim_large

    else:  # op in ["any", ">=", "<=", "!="]
        M, N = rng.integers(1, N_max, size=2, endpoint=True)

    return M, N


def generate_random_matrices(
    seed=565656,
    N_trials=100,
    N_max=10,
    shape_kind="square",
    d_scale=1,
    pos_def_only=False,
    dtype=float,
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
    shape_kind : str, optional
        The shape of the generated matrices. Default is 'square'. Options are:

        * ``any``: M and N are chosen independently.
        * ``square`` or ``M == N``
        * ``overdetermined`` or ``M > N``
        * ``underdetermined`` or ``M < N``
        * Combinations ``M >= N``, ``M <= N`` and ``M != N`` are also accepted.

    d_scale : float
        Scale factor for the density of the sparse matrix. The density will
        be a random value between 0 and ``d_scale``.
    pos_def_only : bool
        If True, generate only positive definite matrices. This requires
        that the matrix is square and symmetric.

    Returns
    -------
    generator
        A generator yielding pytest parameters for random sparse matrices.
    """
    rng = np.random.default_rng(seed)

    _valid_op_strs = {
        "any": "any",
        "square": "==",
        "M == N": "==",
        "overdetermined": ">",
        "M > N": ">",
        "underdetermined": "<",
        "M < N": "<",
        "M >= N": ">=",
        "M <= N": "<=",
        "M != N": "!=",
    }

    _ops = {
        "==": operator.eq,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "!=": operator.ne,
    }

    for trial in range(N_trials):
        try:
            op_str = _valid_op_strs[shape_kind]
        except KeyError as e:
            raise ValueError(f"Invalid shape_kind: {shape_kind}") from e

        if pos_def_only and op_str != "==":
            raise ValueError(
                "Positive definite matrices must be square. "
                "Set shape_kind to 'square' or 'M == N'."
            )

        if op_str == "any":
            M, N = _get_dims(op_str, N_max, rng)
        else:
            op = _ops[op_str]
            # Keep generating until the condition is met
            MAX_TRIES = 10
            for _ in range(MAX_TRIES):
                M, N = _get_dims(op_str, N_max, rng)
                if op(M, N):
                    break

        d = d_scale * rng.random()  # density

        A = sparse.random_array((M, N), density=d, format="csc", dtype=dtype, rng=rng)

        if pos_def_only:
            # Ensure the matrix is positive definite
            A = A.T.conj() @ A
            A = 0.5 * (A + A.T.conj())  # make it strictly Hermitian
            # Add a small value to the diagonal to ensure positive definiteness
            A += sparse.diags_array(np.full(N, 1e-6, dtype=dtype))
            A = A.tocsc()

        yield pytest.param(
            A, id=f"random_{trial:02d}::{A.shape}::{A.nnz}::{dtype.__name__}"
        )
