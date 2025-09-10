# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: nesdis_example.py
#  Created: 2025-08-22 11:04
# =============================================================================

"""An example of using CHOLMOD to compute the Cholesky factorization of a
sparse matrix with various orderings.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse.linalg import splu

from sksparse.amd import amd
from sksparse.cholmod import nesdis

# Load the west0479 matrix (downloaded from the SuiteSparse Matrix Collection:
# <https://sparse.tamu.edu/HB/west0479>)
filepath = Path("data") / "west0479.mtx"
A = mmread(filepath, spmatrix=False).tocsc()  # read the matrix

# Compute the AMD permutation
p = amd(A)
PAPT = A[p][:, p]

# Compute the nested dissection A.T @ A permutation
q = nesdis(A)
QAQT = A[q][:, q]

# Compute the LU decompositions
lu = splu(A)
lup = splu(PAPT)
luq = splu(QAQT)

LU = lu.L + lu.U
LUp = lup.L + lup.U
LUq = luq.L + luq.U

# Plot the original and permuted matrices
plt.rcParams.update({"font.size": 10})

fig, axs = plt.subplots(num=1, nrows=3, ncols=2, clear=True)
fig.set_size_inches((6, 10), forward=True)
fig.set_constrained_layout(True)
fig.suptitle("CHOLMOD Example: Cholesky Factor of west0479")

ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title(r"Original Matrix $A$")

ax = axs[1, 0]
ax.spy(PAPT, markersize=1)
ax.set_title(r"AMD-Ordered Matrix $PAP^{\top}$")

ax = axs[0, 1]
ax.spy(LU, markersize=1)
ax.set_title(r"LU Factors $LU = A$")
ax.set_xlabel(f"{LU.nnz:,} non-zeros")

ax = axs[1, 1]
ax.spy(LUp, markersize=1)
ax.set_title(r"AMD Factors $LU = PAP^{\top}$")
ax.set_xlabel(f"{LUp.nnz:,} non-zeros")

ax = axs[2, 0]
ax.spy(QAQT, markersize=1)
ax.set_title(r"Nested Dissection Matrix $QAQ^{\top}$")

ax = axs[2, 1]
ax.spy(LUq, markersize=1)
ax.set_title(r"Nesdis Factors $LU = QAQ^{\top}$")
ax.set_xlabel(f"{LUq.nnz:,} non-zeros")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
fig.savefig("nesdis_example.svg")
