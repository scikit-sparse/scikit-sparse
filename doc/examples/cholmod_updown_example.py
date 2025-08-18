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

from sksparse.cholmod import ldl, ldlrowmod, ldlsolve, ldlupdate

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
xs = sparse.linalg.spsolve(A, b.tocoo())

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

Lc, Dc = ldlupdate(L, D, C, update=True)

Aw = A + W @ W.T  # (N, N) + (N, 1) @ (1, N) = (N, N)
Sc = S + C @ C.T

# Verify that the updated factorization is correct
assert_allclose((Lc @ Dc @ Lc.T).toarray(), Sc.toarray(), atol=1e-12)

# -----------------------------------------------------------------------------
#         Solve the updated system
# -----------------------------------------------------------------------------
x = ldlsolve(Lc, Dc, b, p=p)

Pxs = sparse.linalg.spsolve(Aw[p][:, p], b.todok()[p])
xs = Pxs[np.argsort(p)]

assert_allclose((Aw @ x).toarray(), b.toarray(), atol=1e-12)
assert_allclose(x.toarray(), xs, atol=1e-12)

# -----------------------------------------------------------------------------
#         Delete row 3 of A
# -----------------------------------------------------------------------------
# Row 3 corresponds to the p_inv(3) in S and LDL
# Invert the permutation
p_inv = np.argsort(p)

k = 3

# Store for later
A_col_k = A[:, [k]].copy()  # column k of A

pk = p_inv[k]  # index in S and LDL
I = sparse.eye_array(*A.shape).tocsc()

Ak = A.copy()
Ak[k, :] = I[k, :]
Ak[:, [k]] = I[:, [k]]

# Remove row and column k from the factorization
Lk, Dk = ldlrowmod(L, D, pk)

# Remove from the original matrix
Sk = S.copy()
Sk[pk, :] = I[pk, :]
Sk[:, [pk]] = I[:, [pk]]

assert_allclose((Lk @ Dk @ Lk.T).toarray(), Sk.toarray(), atol=1e-12)

# Solve the modified system
x = ldlsolve(Lk, Dk, b, p=p)
xs = sparse.linalg.spsolve(Ak, b.tocoo())

assert_allclose((Ak @ x).toarray(), b.toarray(), atol=1e-12)
assert_allclose(x.toarray(), xs, atol=1e-12)

# -----------------------------------------------------------------------------
#         Add row 3 back to the factorization
# -----------------------------------------------------------------------------
W = A_col_k
Aa = Ak.copy()
Aa[k, :] = W.T
Aa[:, [k]] = W

C = W[p].tocsc()  # permuted version
Sa = Sk.copy()
Sa[pk, :] = C.T
Sa[:, [pk]] = C

La, Da = ldlrowmod(Lk, Dk, pk, C=C)

assert_allclose((La @ Da @ La.T).toarray(), Sa.toarray(), atol=1e-12)

# Solve the modified system
x = ldlsolve(La, Da, b, p=p)
xs = sparse.linalg.spsolve(Aa, b.tocoo())

assert_allclose((Aa @ x).toarray(), b.toarray(), atol=1e-12)
assert_allclose(x.toarray(), xs, atol=1e-12)
