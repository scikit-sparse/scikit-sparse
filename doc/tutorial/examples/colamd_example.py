# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: colamd_example.py
#  Created: 2025-07-31 14:18
# =============================================================================

"""An example of using the COLAMD (Column Approximate Minimum Degree) algorithm
to find a fill-reducing ordering of a sparse matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse.linalg import splu

from sksparse.colamd import colamd

# from numpy.testing import assert_allclose, assert_array_equal


# Load the west0479 matrix (downloaded from the SuiteSparse Matrix Collection:
# <https://sparse.tamu.edu/HB/west0479>)
filepath = Path("data") / "west0479.mtx"
A = mmread(filepath, spmatrix=False).tocsc()  # read the matrix

q = colamd(A)  # compute the COLAMD ordering
AQ = A[:, q]   # apply the column ordering to the matrix

# Compute the LU factorization of the original and permuted matrices
lu = splu(A, permc_spec='NATURAL')
L_, U_ = lu.L, lu.U

luq = splu(AQ, permc_spec='NATURAL')
Lq, Uq = luq.L, luq.U

# Plot the original and permuted matrices
plt.rcParams.update({'font.size': 10})

fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
fig.set_size_inches((6, 6), forward=True)
fig.set_constrained_layout(True)
fig.suptitle("COLAMD Example: Fill-Reducing Ordering of west0479")

ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title(r"Original Matrix $A$")

ax = axs[0, 1]
ax.spy(AQ, markersize=1)
ax.set_title(r"Permuted Matrix $AQ$")

ax = axs[1, 0]
ax.spy(L_ + U_, markersize=1)
ax.set_title(r"Original LU Factors $L + U$")
ax.set_xlabel(f"{(L_ + U_).nnz:,} total non-zeros")

ax = axs[1, 1]
ax.spy(Lq + Uq, markersize=1)
ax.set_title(r"Permuted LU Factors $L_q + U_q$")
ax.set_xlabel(f"{(Lq + Uq).nnz:,} total non-zeros")

for ax in axs.flat:
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
fig.savefig("colamd_example.svg")
