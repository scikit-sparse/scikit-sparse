# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: klu_batches.py
#  Created: 2025-12-15 10:32
# =============================================================================

"""Compare RHS batch sizes for sparse solves in KLU."""

from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd
from tqdm import tqdm

from scripts.utils import measure_perf
from sksparse.klu import klu_factor
from sksparse.spqr import spqr_factor

SEED = 565656

SAVE_FIGS = False

DATA_PATH = Path(__file__).absolute().parent.parent.parent / "_dev_data"
DATA_PATH.mkdir(parents=True, exist_ok=True)


def run_batch_comparison(df_file, force_update=False):
    """Compare performance on batch solve with multiple RHS."""
    if not force_update and df_file.exists():
        print(f"Loaded existing results from: {df_file}")
        return pd.read_pickle(df_file)

    assert df_file.parent.exists(), f"Data path does not exist: {df_file.parent}"
    print(f"Running batch solve tests for {df_file}...")

    # Build the results DataFrame
    densities = [0.01, 0.1, 0.5, 1.0]
    batch_sizes = [1, 3, 10, 30, 100, 300, 1_000, 3_000]

    Nsq = 100
    Ng = np.sqrt(Nsq).astype(int)
    A = -LaplacianNd((Ng, Ng), dtype=float).tosparse().tocsc()
    A[-1, -1] += 1.0  # make sure A is non-singular
    N = A.shape[0]
    K = 1_056  # arbitrary number of RHS

    # Pre-factor the matrix
    lu = klu_factor(A)
    qr = spqr_factor(A)

    results = []

    # Sparse solve with batches
    for d in tqdm(densities):
        b = sparse.random_array((N, K), density=d, format="csc", random_state=SEED)

        for rhs_batch_size in tqdm(batch_sizes, leave=False):
            for name, factor in zip(["klu", "spqr"], [lu, qr]):
                solve_func = partial(factor.solve, b, rhs_batch_size=rhs_batch_size)
                time, mem = measure_perf(solve_func)
                results.append(
                    {
                        "solver": name,
                        "rhs_batch_size": rhs_batch_size,
                        "density": d,
                        "time": time,
                        "memory": mem,
                    }
                )

    # Build the results DataFrame
    df = (
        pd.DataFrame(results)
        .set_index(["solver", "rhs_batch_size", "density"])
        .sort_index()
    )
    df.columns.name = "metric"

    df.to_pickle(df_file)
    return df


# -----------------------------------------------------------------------------
#         Run the Tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df_batch = run_batch_comparison(DATA_PATH / "batch_results.pkl", force_update=False)

    for fignum, name in enumerate(["klu", "spqr"]):
        df = df_batch.xs(name, level="solver")
        # Plot results
        fig, axs = plt.subplots(num=fignum, nrows=2, sharex=True, clear=True)
        fig.suptitle(
            f"Batch RHS {name.upper()} Solve Performance\n"
            "A (100, 100) 2D Laplacian, B (100, 1,056)"
        )
        fig.set_size_inches((6.4, 8), forward=True)

        for i, col in enumerate(["time", "memory"]):
            sns.lineplot(
                ax=axs[i],
                data=df,
                x="rhs_batch_size",
                y=col,
                hue="density",
                hue_norm=mpl.colors.LogNorm(),
                palette="flare",
                marker="o",
                legend=(i == 0),
            )
            axs[i].grid(True, which="both")

        axs[0].legend(title="density")
        axs[0].set(
            ylabel="time [s]",
            yscale="log",
        )

        axs[1].set(
            xscale="log",
            xlabel="RHS Batch Size",
            ylabel="peak memory [MB]",
        )

        if SAVE_FIGS:
            fig_file = DATA_PATH / f"{name}_batch.pdf"
            fig.savefig(fig_file)
            print(f"Saved figure to: {fig_file}")

        plt.show()


# =============================================================================
# =============================================================================
