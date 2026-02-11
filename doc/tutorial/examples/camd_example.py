#!/usr/bin/env python3
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: camd_example.py
#  Created: 2025-09-30 10:16
# =============================================================================

"""An example of using the AMD (Approximate Minimum Degree) algorithm to find
a fill-reducing ordering of a sparse matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread

from sksparse.camd import camd
from sksparse.cholmod import cholesky

# Load the bcsstk06 matrix (downloaded from the SuiteSparse Matrix Collection:
# <https://sparse.tamu.edu/HB/bcsstk06>)
filepath = Path("data") / "bcsstk06.mtx"
A = mmread(filepath, spmatrix=False).tocsc()  # read the matrix

N = A.shape[0]
K = N // 2                    # number of constrained variables
C = np.full(N, K)             # fully constrained
C[:K] = np.arange(K)          # first K variables are constrained
Nc = np.count_nonzero(C < K)  # number of constrained variables

p = camd(A, constraints=C)  # compute the AMD ordering
PAPT = A[p][:, p]           # apply the ordering to the matrix

# Compute the Cholesky factorization of each matrix
L = cholesky(A, lower=True)
Lp = cholesky(PAPT, lower=True)

# Plot the original and permuted matrices
plt.rcParams.update({"font.size": 10})

fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
fig.set_size_inches((6, 6), forward=True)
fig.set_constrained_layout(True)
fig.suptitle(
    f"CAMD Example: Fill-Reducing Ordering of bcsstk06\n{Nc} constrained variables"
)

ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title(r"Original Matrix $A$")

ax = axs[0, 1]
ax.spy(PAPT, markersize=1)
ax.set_title(r"Permuted Matrix $PAP^T$")

ax = axs[1, 0]
ax.spy(L, markersize=1)
ax.set_title(r"Original Cholesky Factor $L$")
ax.set_xlabel(f"{L.nnz:,} non-zeros")

ax = axs[1, 1]
ax.spy(Lp, markersize=1)
ax.set_title(r"Permuted Cholesky Factor $L_p$")
ax.set_xlabel(f"{Lp.nnz:,} non-zeros")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
fig.savefig("camd_example.svg")
