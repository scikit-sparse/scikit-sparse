# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: spqr_example.py
#  Created: 2025-11-17 11:58
# =============================================================================

"""An example of using SPQR to compute the QR factorization of a
sparse matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from scipy.io import mmread

from sksparse.spqr import spqr

# Load the west0479 matrix (downloaded from the SuiteSparse Matrix Collection:
# <https://sparse.tamu.edu/HB/west0479>)
filepath = Path("data") / "west0479.mtx"
A = mmread(filepath, spmatrix=False)  # read the matrix
A = A.tocsc()

# compute the LU factorization
Q, R, _ = spqr(A, order="fixed")
Qp, Rp, p = spqr(A, order="colamd")

# Permute A for visualization
AE = A[:, p]

# Make sure the factorization is correct
assert_allclose((Q @ R).toarray(), A.toarray(), atol=1e-9)
assert_allclose((Qp @ Rp).toarray(), AE.toarray(), atol=1e-9)

# Plot the original and permuted matrices
plt.rcParams.update({"font.size": 10})

fig, axs = plt.subplots(num=1, nrows=3, ncols=2, clear=True)
fig.set_size_inches((6, 10), forward=True)
fig.set_constrained_layout(True)
fig.suptitle("SPQR Example: QR Factors of west0479")

ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title(r"Original Matrix $A$")

ax = axs[0, 1]
ax.spy(AE, markersize=1)
ax.set_title(r"Permuted Matrix $AE$")

ax = axs[1, 0]
ax.spy(Q, markersize=1)
ax.set_title(r"Original Q Factor $Q$")
ax.set_xlabel(f"{Q.nnz:,} non-zeros")

ax = axs[1, 1]
ax.spy(Qp, markersize=1)
ax.set_title(r"Permuted Q Factor $Q_p$")
ax.set_xlabel(f"{Qp.nnz:,} non-zeros")

ax = axs[2, 0]
ax.spy(R, markersize=1)
ax.set_title(r"Original R Factor $R$")
ax.set_xlabel(f"{R.nnz:,} non-zeros")

ax = axs[2, 1]
ax.spy(Rp, markersize=1)
ax.set_title(r"Permuted R Factor $R_p$")
ax.set_xlabel(f"{Rp.nnz:,} non-zeros")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
fig.savefig("spqr_example.svg")
