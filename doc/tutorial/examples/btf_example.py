#!/usr/bin/env python3
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: btf_example.py
#  Created: 2025-08-07 14:03
# =============================================================================

"""An example of using the BTF (Block Triangular Form) algorithm."""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.io import mmread

from sksparse.btf import btf, btf_q_permutation, maxtrans, strongcomp


def plot_btf(A, p, q, r, ax=None):
    """Plot the blocks of the BTF matrix.

    Adapted from the SuiteSparse ``drawbtf.m`` [#drawbtf]_.

    Parameters
    ----------
    A : (N, N) sparse matrix
        A square, sparse matrix in CSC format.
    p, q : (N,) ndarray of int
        The row and column permutations that put ``A`` into upper block
        triangular form.
    r : (Nb + 1,) ndarray of int
        The indices of the block boundaries in the permuted matrix. The number
        of blocks is ``len(r) - 1``.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, the current axes will be used.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the BTF blocks plotted.

    References
    ----------
    .. [#drawbtf] MATLAB BTF plotting function:
        https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/BTF/MATLAB/drawbtf.m
    """
    if ax is None:
        ax = plt.gca()

    Nb = len(r) - 1
    q = btf_q_permutation(q)  # ensure q is a valid permutation

    for k in range(Nb):
        r0 = r[k]
        r1 = r[k + 1]
        Nk = r1 - r0
        if Nk > 1:
            # Plot the block
            ax.add_patch(
                plt.Rectangle(
                    (r0 - 0.5, r0 - 0.5),
                    Nk,
                    Nk,
                    fill=False,
                    edgecolor="C3",
                    linewidth=1.5,
                )
            )

    return ax


# Load the west0479 matrix (downloaded from the SuiteSparse Matrix Collection:
# <https://sparse.tamu.edu/HB/west0479>)
filepath = Path("data") / "west0479.mtx"
A = mmread(filepath, spmatrix=False).tocsc()  # read the matrix

# Compute the BTF ordering
p, q, r = btf(A)
q = btf_q_permutation(q)
PAQ = A[p][:, q]

# Compute the maximum transversal
qm = maxtrans(A)
A_mt = A[:, qm]

# Compute the strongly connected components
pcc, rcc = strongcomp(A)
A_scc = A[pcc][:, pcc]

# Plot the original and permuted matrices
plt.rcParams.update({"font.size": 10})

fig, axs = plt.subplots(num=1, nrows=2, ncols=2, clear=True)
fig.set_size_inches((6, 6), forward=True)
fig.set_constrained_layout(True)
fig.suptitle("BTF Example: Block Triangular Ordering of west0479")

ax = axs[0, 0]
ax.spy(A, markersize=1)
ax.set_title(r"Original Matrix $A$")

ax = axs[0, 1]
ax.spy(PAQ, markersize=1)
plot_btf(PAQ, p, q, r, ax=ax)
ax.set_title(r"BTF Matrix $PAQ$")

ax = axs[1, 0]
ax.spy(A_mt, markersize=1)
ax.set_title(r"Maximum Transveral")

ax = axs[1, 1]
ax.spy(A_scc, markersize=1)
plot_btf(A_scc, pcc, pcc, rcc, ax=ax)
ax.set_title(r"Strongly Connected Components")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
fig.savefig("btf_example.svg")
