# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_ldl_factor.py
#  Created: 2025-09-05 11:05
# =============================================================================

"""Unit tests for the CholeskyFactor object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from sksparse.cholmod import ldl_factor

from ..helpers import generate_random_matrices

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("order", [None, "amd"])
@pytest.mark.parametrize(
    "A", generate_random_matrices(N_trials=1, N_max=10, d_scale=0.2, pos_def_only=True)
)
def test_view_vs_get(A, order):
    f = ldl_factor(A)
    LDv = f.view_factor()
    pv = f.perm
    L, D = f.get_factor()
    p = f.get_perm()
    assert LDv is not L  # different objects
    assert LDv is not D
    assert pv is not p
    # Split the view into L and D
    Dv = sparse.diags_array(LDv.diagonal())
    with pytest.raises(ValueError, match="read-only"):
        LDv.setdiag(1.0)
    LDv = LDv.copy()
    LDv.setdiag(1.0)
    assert_allclose(LDv.toarray(), L.toarray(), atol=1e-15)
    assert_allclose(Dv.toarray(), D.toarray(), atol=1e-15)
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
    f = ldl_factor(A, lower=True)
    L, D = f.get_factor()
    assert_allclose((L @ D @ L.T.conj()).toarray(), A.toarray(), atol=atol)
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
    Lb, Db = f.get_factor()
    assert_allclose((Lb @ Db @ Lb.T.conj()).toarray(), B.toarray(), atol=atol)
