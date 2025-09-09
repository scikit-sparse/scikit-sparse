# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_cho_factor.py
#  Created: 2025-09-04 19:32
# =============================================================================

"""Unit tests for the CholeskyFactor object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from sksparse.cholmod import cho_factor

from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("order", [None, "amd"])
@pytest.mark.parametrize(
    "A", generate_random_matrices(N_trials=1, N_max=10, d_scale=0.2, pos_def_only=True)
)
def test_view_vs_get(A, order):
    f = cho_factor(A, lower=True, order=order)
    Lv = f.view_factor()
    pv = f.perm
    L = f.get_factor()
    p = f.get_perm()
    assert Lv is not L  # different objects
    assert pv is not p
    assert_allclose(Lv.toarray(), L.toarray(), atol=1e-15)
    assert_allclose(pv, p, atol=1e-15)


test_As = [
    A
    for dtype in DTYPES
    for A in generate_random_matrices(
        N_trials=10, N_max=200, d_scale=0.05, pos_def_only=True, dtype=dtype
    )
]


@pytest.mark.parametrize("A", test_As)
def test_refactor(A):
    atol = 1e-12 if A.dtype in (np.float64, np.complex128) else 1e-3
    f = cho_factor(A, lower=True)
    L = f.get_factor()
    assert_allclose((L @ L.T.conj()).toarray(), A.toarray(), atol=atol)
    # Create a new matrix with the same sparsity pattern but different values
    Bl = sparse.tril(A, -1).copy()
    rng = np.random.default_rng(56)
    if np.issubdtype(A.dtype, np.complexfloating):
        Bl.data = rng.random(Bl.nnz, dtype=A.real.dtype) + 1j * rng.random(
            Bl.nnz, dtype=A.real.dtype
        )
    else:
        Bl.data = rng.random(Bl.nnz, dtype=A.dtype)
    B = Bl + Bl.T.conj()
    # Ensure positive definiteness by adding to the diagonal
    B.setdiag(A.diagonal())
    B += sparse.diags_array(np.full(B.shape[0], B.shape[0], dtype=B.dtype))
    B = B.tocsc()
    # Factor the new matrix with the same sparsity pattern
    f.factorize(B, lower=True)
    Lb = f.get_factor()
    assert_allclose((Lb @ Lb.T.conj()).toarray(), B.toarray(), atol=atol)
