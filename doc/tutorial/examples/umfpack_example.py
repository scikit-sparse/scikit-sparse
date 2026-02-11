# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: umfpack_example.py
#  Created: 2025-11-17 09:35
# =============================================================================

"""An example of using UMFPACK to compute the umf_factor factorization of a
sparse matrix.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from scipy.io import mmread

from sksparse.umfpack import umf_factor

# Load the west0479 matrix (downloaded from the SuiteSparse Matrix Collection:
# <https://sparse.tamu.edu/HB/west0479>)
filepath = Path("data") / "west0479.mtx"
A = mmread(filepath, spmatrix=False)  # read the matrix

# compute the LU factorization
L, U, p, q, rscale = umf_factor(A, ordering_method="none")
Lp, Up, pp, qp, rscalep = umf_factor(A, ordering_method="amd")

# Make sure the factorization is correct
PRAQ = (rscale[:, np.newaxis] * A).tocsc()[p[:, np.newaxis], q]
PRAQp = (rscalep[:, np.newaxis] * A).tocsc()[pp[:, np.newaxis], qp]
assert_allclose((L @ U).toarray(), PRAQ.toarray(), atol=1e-9)
assert_allclose((Lp @ Up).toarray(), PRAQp.toarray(), atol=1e-9)

# Plot the original and permuted matrices
plt.rcParams.update({"font.size": 10})

fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
fig.set_size_inches((6, 6), forward=True)
fig.set_constrained_layout(True)
fig.suptitle("UMFPACK Example: LU Factors of west0479")

ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title(r"Original Matrix $A$")

ax = axs[0, 1]
ax.spy(PRAQ, markersize=1)
ax.set_title(r"Permuted Matrix $PAQ$")

ax = axs[1, 0]
ax.spy(L + U, markersize=1)
ax.set_title(r"Original LU Factors $L + U$")
ax.set_xlabel(f"{L.nnz + U.nnz:,} non-zeros")

ax = axs[1, 1]
ax.spy(Lp + Up, markersize=1)
ax.set_title(r"Permuted LU Factors $L_p + U_p$")
ax.set_xlabel(f"{Lp.nnz + Up.nnz:,} non-zeros")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
fig.savefig("umfpack_example.svg")
