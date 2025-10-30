# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: compare_umfpack.py
#  Created: 2025-10-30 14:18
# =============================================================================

"""
Compare the scikit-sparse UMFPACK interface with the existing scikit-umfpack
interface.
"""

import timeit
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scikits.umfpack import splu, spsolve
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd
from tqdm import tqdm

from sksparse.umfpack import umf_factor, umf_solve

SEED = 565656

Ns = np.unique(np.logspace(1, 3, num=20, dtype=int))
sqrtNs = [int(np.sqrt(N)) for N in Ns]

SAVE_FIGS = True
FORCE_UPDATE = False

filestem = "umf_perf"
data_path = Path(__file__).absolute().parent / "data"
df_file = data_path / (filestem + "_results.pkl")
fig_file = data_path / (filestem + "_results.pdf")

if not FORCE_UPDATE and df_file.exists():
    df = pd.read_pickle(df_file)
    print(f"Loaded existing results from: {df_file}")
else:
    print("Running performance tests...")

    N_repeats = 5  # length of output vector from %timeit (5 is default)

    pkg_names = ["sksparse", "scikits"]
    func_types = ["factorize", "solve"]
    index = pd.MultiIndex.from_product(
        [pkg_names, func_types, Ns], names=["package", "function", "N"]
    )
    df = pd.DataFrame(index=index, columns=["time"], dtype=float)

    # Test performance of multiple solves
    for sqrtN in tqdm(sqrtNs):
        A = -LaplacianNd((sqrtN, sqrtN), dtype=float).tosparse().tocsc()
        A[-1, -1] += 1.0  # make sure A is non-singular
        N = A.shape[0]

        x_col = np.arange(1, N + 1, dtype=float)
        expect_x = np.outer(x_col, x_col)  # multiple RHS columns
        B = A @ expect_x

        Am = sparse.csc_matrix(A)  # scikits does not accept csc_array

        funcs = {
            ("sksparse", "factorize"): partial(umf_factor, A),
            ("scikits", "factorize"): partial(splu, Am),
            ("sksparse", "solve"): partial(umf_solve, A, B),
            ("scikits", "solve"): partial(spsolve, Am, B),
        }

        for key, func in tqdm(funcs.items(), leave=False):
            timer = timeit.Timer(func)
            N_samples, _ = timer.autorange()
            # N_samples = 1  # fast for testing

            ts = timer.repeat(repeat=N_repeats, number=N_samples)
            ts = np.array(ts) / N_samples

            df.loc[key[0], key[1], N] = np.min(ts)  # time per loop

    df.to_pickle(df_file)

# -----------------------------------------------------------------------------
#         Plot Results
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=1, clear=True)
fig.suptitle("scikit-sparse vs scikits-umfpack Performance")
sns.lineplot(
    ax=ax,
    data=df,
    x="N",
    y="time",
    hue="package",
    style="function",
    markers=True,
)

ax.set(
    xlabel="Number of Rows/Columns (N)",
    ylabel="time [s]",
    xscale="log",
    yscale="log",
)

ax.grid(True, which="both")

if SAVE_FIGS:
    fig.savefig(fig_file)
    print(f"Saved figure to: {fig_file}")

# =============================================================================
# =============================================================================
