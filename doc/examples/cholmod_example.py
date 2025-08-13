# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: cholmod_example.py
#  Created: 2025-08-12 20:12
# =============================================================================

"""An example of using CHOLMOD to compute the Cholesky factorization of a
sparse matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import mmread

from sksparse.cholmod import cholesky

# Load the west0479 matrix (downloaded from the SuiteSparse Matrix Collection:
# <https://sparse.tamu.edu/HB/west0479>)
filepath = Path("data") / "west0479.mtx"
A = mmread(filepath, spmatrix=False)  # read the matrix
A = (A.T @ A).tocsc()  # make it symmetric positive definite

# compute the Cholesky factorization
R = cholesky(A, order="natural")
Rp, p = cholesky(A, order="amd")

PAPT = A[p][:, p]  # apply the ordering to the matrix

# Plot the original and permuted matrices
plt.rcParams.update({"font.size": 10})

fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
fig.set_size_inches((6, 6), forward=True)
fig.set_constrained_layout(True)
fig.suptitle("CHOLMOD Example: Cholesky Factor of west0479")

ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title(r"Original Matrix $A$")

ax = axs[0, 1]
ax.spy(PAPT, markersize=1)
ax.set_title(r"Permuted Matrix $PAP^{\top}$")

ax = axs[1, 0]
ax.spy(R, markersize=1)
ax.set_title(r"Original Cholesky Factor $R$")
ax.set_xlabel(f"{R.nnz:,} non-zeros")

ax = axs[1, 1]
ax.spy(Rp, markersize=1)
ax.set_title(r"Permuted R Factor $R_p$")
ax.set_xlabel(f"{Rp.nnz:,} non-zeros")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
fig.savefig("cholesky_example.svg")
