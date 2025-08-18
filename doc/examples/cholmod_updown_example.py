# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: cholmod_updown_example.py
#  Created: 2025-08-18 10:35
# =============================================================================

"""An example of using CHOLMOD to compute the Cholesky factorization of a
sparse matrix and update it with a new row and column.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd

from sksparse.cholmod import ldl, ldlsolve, ldlupdate

# Create (negative) Laplacian matrix that is symmetric positive definite
N = 15
G = LaplacianNd((N, N))
A = -G.tosparse().tocsc().astype(float)
A.setdiag(A.diagonal() + 1)  # make it positive definite

# Make the RHS
expect_x = sparse.dok_array(np.arange(N * N))
b = A @ expect_x

# Compute the Cholesky factorization
L, D, p = ldl(A, order="default", remove_zeros=False)

# Verify that the factorization is correct
S = A[p][:, p]
assert_allclose((L @ D @ L.T).toarray(), S.toarray(), atol=1e-12)

# TODO print timings

# Solve Ax = b using LDL.T of A[p][:, p]
x = ldlsolve(L, D, b, p=p)

assert_allclose(x.toarray(), expect_x.toarray(), atol=1e-12)

# Solve using scipy sparse
Pxs = sparse.linalg.spsolve(A[p][:, p], b.todok()[p])
xs = Pxs[np.argsort(p)]

assert_allclose(x.toarray(), xs, atol=1e-12)

# -----------------------------------------------------------------------------
#         Compute a rank-1 update of LDL.T factorization
# -----------------------------------------------------------------------------
# Arbitrary values to update (see CHOLMOD/MATLAB/cholmod_updown_demo.m)
# These indices are in the original A, so the non-zero pattern does not change
W = sparse.coo_array(
    ([5, -1, -1, -1], ([0, 1, 2, 151], [0, 0, 0, 0])),
    shape=(N * N, 1),
    dtype=A.dtype,
).todok()

C = W[p].tocsc()  # permuted version

Lp, Dp = ldlupdate(L, D, C, update=True)

Ap = A + W @ W.T  # (N, N) + (N, 1) @ (1, N) = (N, N)
Sp = S + C @ C.T

# Verify that the updated factorization is correct
assert_allclose((Lp @ Dp @ Lp.T).toarray(), Sp.toarray(), atol=1e-12)
