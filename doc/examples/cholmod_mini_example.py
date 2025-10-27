# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: cholmod_mini_example.py
#  Created: 2025-08-13 10:13
# =============================================================================

"""An example of using CHOLMOD to compute the Cholesky factorization of a
sparse matrix.
"""

import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd

from sksparse.cholmod import cholesky

# Create (negative) Laplacian matrix that is symmetric positive definite
N = 5
G = LaplacianNd((N, N))
A = -G.tosparse().tocsc().astype(float)
A.setdiag(A.diagonal() + 1)  # make it positive definite

# Make it complex
A = A + 0.5j * sparse.tril(A)
A = 0.5 * (A + A.conj().T)  # ensure symmetry
A = A.tocsc()

# compute the Cholesky factorization
R = cholesky(A)
Rp, p = cholesky(A, order="amd")

PAPT = A[p][:, p]  # apply the ordering to the matrix

# Make sure the factorization is correct
assert_allclose((R.T.conj() @ R).toarray(), A.toarray(), atol=1e-15)
assert_allclose((Rp.T.conj() @ Rp).toarray(), PAPT.toarray(), atol=1e-15)

# Plot the original and permuted matrices
plt.rcParams.update({"font.size": 10})
MSIZE = 5  # marker size for the spy plots

fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
fig.set_size_inches((6, 6), forward=True)
fig.set_constrained_layout(True)
fig.suptitle("CHOLMOD Example: Cholesky Factor of Laplacian Matrix")

ax = axs[0, 0]
ax.spy(A, markersize=MSIZE)
ax.set_title(r"Original Matrix $A$")

ax = axs[0, 1]
ax.spy(PAPT, markersize=MSIZE)
ax.set_title(r"Permuted Matrix $PAP^{\top}$")

ax = axs[1, 0]
ax.spy(R, markersize=MSIZE)
ax.set_title(r"Original Cholesky Factor $R$")
ax.set_xlabel(f"{R.nnz:,} non-zeros")

ax = axs[1, 1]
ax.spy(Rp, markersize=MSIZE)
ax.set_title(r"Permuted R Factor $R_p$")
ax.set_xlabel(f"{Rp.nnz:,} non-zeros")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
