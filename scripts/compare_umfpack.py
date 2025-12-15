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

import gc
import timeit
import tracemalloc
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scikits.umfpack import splu
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd
from tqdm import tqdm

from sksparse.umfpack import umf_factor

SEED = 565656

SAVE_FIGS = False

DATA_PATH = Path(__file__).absolute().parent.parent.parent / "_dev_data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

PKG_NAMES = ["sksparse", "scikit-umfpack"]


def measure_perf(func, N_repeats=5, N_samples=None):
    """Measure time and memory usage of a function.

    Parameters
    ----------
    func : callable
        The function to measure.

    Returns
    -------
    time : float
        The minimum execution time in seconds.
    peak_mb : float
        The peak memory usage in megabytes.
    """
    # Measure timing (multiple runs)
    timer = timeit.Timer(func)
    if N_samples is None:
        N_samples, _ = timer.autorange()
    ts = timer.repeat(repeat=N_repeats, number=N_samples)
    ts = np.array(ts) / N_samples
    time = np.min(ts)

    # Measure memory usage (single pass)
    gc.collect()  # force garbage collection before measuring
    tracemalloc.start()

    try:
        func()
    except Exception:
        tracemalloc.stop()
        raise

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024**2)  # convert to MB

    return time, peak_mb


def run_package_comparison(df_file, force_update=False):
    """Run performance comparison between scikit-sparse and scikits-umfpack."""
    if not force_update and df_file.exists():
        print(f"Loaded existing results from: {df_file}")
        return pd.read_pickle(df_file)

    assert df_file.parent.exists(), f"Data path does not exist: {df_file.parent}"
    print(f"Running performance tests for {df_file}...")
    Ns = np.unique(np.logspace(1, 4, num=10, dtype=int))
    sqrtNs = np.unique([int(np.sqrt(N)) for N in Ns])

    results = []

    # Test performance of multiple solves
    for sqrtN in tqdm(sqrtNs):
        A = -LaplacianNd((sqrtN, sqrtN), dtype=float).tosparse().tocsc()
        A[-1, -1] += 1.0  # make sure A is non-singular
        N = A.shape[0]

        x_col = np.arange(1, N + 1, dtype=float)
        expect_x = np.outer(x_col, np.arange(1, 1000))  # many RHS columns
        B = A @ expect_x  # C order
        Bf = np.asfortranarray(B)  # Fortran order
        Bsp = sparse.csc_array(B)

        Am = sparse.csc_matrix(A)  # scikits does not accept csc_array

        # Pre-factor matrices to test solve performance
        lu = umf_factor(A)
        umf = splu(Am)

        funcs = {
            ("sksparse", "factorize"): partial(umf_factor, A),
            ("scikit-umfpack", "factorize"): partial(splu, Am),
            ("sksparse", "solve dense C"): partial(lu.solve, B),
            ("scikit-umfpack", "solve dense C"): partial(umf.solve, B),
            ("sksparse", "solve dense F"): partial(lu.solve, Bf),
            ("scikit-umfpack", "solve dense F"): partial(umf.solve, Bf),
            ("sksparse", "solve sparse"): partial(lu.solve, Bsp, rhs_batch_size=1),
            ("scikit-umfpack", "solve sparse"): partial(umf.solve_sparse, Bsp),
        }

        for key, func in tqdm(funcs.items(), leave=False):
            time, mem = measure_perf(func)
            results.append(
                {
                    "package": key[0],
                    "function": key[1],
                    "N": N,
                    "time": time,
                    "memory": mem,
                }
            )

    # Build the results DataFrame
    df = pd.DataFrame(results).set_index(["package", "function", "N"]).sort_index()
    df.columns.name = "metric"

    df.to_pickle(df_file)
    return df


def run_batch_comparison(df_file, force_update=False):
    """Compare performance on batch solve with multiple RHS."""
    if not force_update and df_file.exists():
        print(f"Loaded existing results from: {df_file}")
        return pd.read_pickle(df_file)

    assert df_file.parent.exists(), f"Data path does not exist: {df_file.parent}"
    print(f"Running batch solve tests for {df_file}...")

    # Build the results DataFrame
    densities = [0.01, 0.1, 0.5, 1.0]
    batch_sizes = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000]

    Nsq = 100
    Ng = np.sqrt(Nsq).astype(int)
    A = -LaplacianNd((Ng, Ng), dtype=float).tosparse().tocsc()
    A[-1, -1] += 1.0  # make sure A is non-singular
    N = A.shape[0]
    K = 9_056  # arbitrary number of RHS

    # NOTE scikit-umfpack just converts sparse -> dense in spsolve, so we need
    # to compare against Umfpack.solve_sparse that loops over columns (rhs_batch_size=1)
    # The comparison is between a python loop and sksparse.umfpack Cython loop.

    # Pre-factor the matrices
    umf = splu(sparse.csc_matrix(A))  # scikits does not accept csc_array
    lu = umf_factor(A)

    results = []

    # Sparse solve with batches
    for d in tqdm(densities):
        b = sparse.random_array((N, K), density=d, format="csc", random_state=SEED)

        # Scikits-umfpack solve (no batching)
        umf_func = partial(umf.solve_sparse, b)
        time, mem = measure_perf(umf_func)
        results.append(
            {
                "package": "scikit-umfpack",
                "rhs_batch_size": 1,
                "density": d,
                "time": time,
                "memory": mem,
            }
        )

        # Scikit-sparse umfpack solve (with batching)
        for rhs_batch_size in tqdm(batch_sizes, leave=False):
            solve_func = partial(lu.solve, b, rhs_batch_size=rhs_batch_size)
            time, mem = measure_perf(solve_func)
            results.append(
                {
                    "package": "sksparse",
                    "rhs_batch_size": rhs_batch_size,
                    "density": d,
                    "time": time,
                    "memory": mem,
                }
            )

    # Build the results DataFrame
    df = (
        pd.DataFrame(results)
        .set_index(["package", "rhs_batch_size", "density"])
        .sort_index()
    )
    df.columns.name = "metric"

    df.to_pickle(df_file)
    return df


# -----------------------------------------------------------------------------
#         Run the Tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- Package Tests
    df_pkg = run_package_comparison(
        DATA_PATH / "umf_perf_pkg_results.pkl", force_update=False
    )

    fig, axs = plt.subplots(num=1, nrows=2, sharex=True, clear=True)
    fig.suptitle(
        "sksparse.umfpack vs scikit-umfpack\nA (N, N) 2D Laplacian, B (N, 1000)"
    )
    fig.set_size_inches((6.4, 8), forward=True)

    for i, col in enumerate(["time", "memory"]):
        sns.lineplot(
            ax=axs[i],
            data=df_pkg,
            x="N",
            y=col,
            hue="package",
            style="function",
            markers=True,
            legend=(i == 1),
        )
        axs[i].grid(True, which="both")
        axs[i].set(yscale="log")

    axs[0].set(
        ylabel="time [s]",
    )

    axs[1].legend(loc="lower right")
    axs[1].set(
        xscale="log",
        xlabel="Number of Rows/Columns (N)",
        ylabel="peak memory [MB]",
    )

    plt.show()

    if SAVE_FIGS:
        fig_file = DATA_PATH / "umf_perf_pkg.pdf"
        fig.savefig(fig_file)
        print(f"Saved figure to: {fig_file}")

    # ---------- Plot ratios of sksparse / scikit-umfpack
    # NOTE we get linter warnings for "modern" pandas usage on this code:
    # df_ratio = (
    #     df_pkg.stack("metric")
    #     .unstack("package")
    #     .assign(ratio=lambda x: x["sksparse"] / x["scikit-umfpack"])
    #     .unstack("metric")["ratio"]
    # )

    df_ratio = (
        df_pkg.reset_index()
        .melt(
            id_vars=["package", "function", "N"],
            value_vars=["time", "memory"],
            var_name="metric",
            value_name="value",
        )
        .pivot_table(
            index=["function", "N", "metric"],
            columns="package",
            values="value",
        )
        .assign(ratio=lambda x: x["sksparse"] / x["scikit-umfpack"])["ratio"]
        .reset_index()
        .pivot_table(index=["function", "N"], columns="metric", values="ratio")
    )

    fig, axs = plt.subplots(num=3, nrows=2, sharex=True, sharey=True, clear=True)
    fig.suptitle(
        "sksparse.umfpack / scikit-umfpack\nA (N, N) 2D Laplacian, B (N, 1000)"
    )
    fig.set_size_inches((6.4, 8), forward=True)

    for i, col in enumerate(["time", "memory"]):
        sns.lineplot(
            ax=axs[i],
            data=df_ratio,
            x="N",
            y=col,
            style="function",
            markers=True,
            legend=(i == 0),
        )
        axs[i].grid(True, which="both")

    axs[0].legend(loc="lower right")
    axs[0].set(
        ylabel="ratio of runtime",
        ylim=(-0.05, None),
    )

    axs[1].set(
        xscale="log",
        xlabel="Number of Rows/Columns (N)",
        ylabel="ratio of peak memory",
    )

    plt.show()

    if SAVE_FIGS:
        fig_file = DATA_PATH / "umf_perf_pkg_ratio.pdf"
        fig.savefig(fig_file)
        print(f"Saved figure to: {fig_file}")

    # ---------- Batch Solve Tests
    df_batch = run_batch_comparison(
        DATA_PATH / "umf_perf_batch_results.pkl", force_update=False
    )

    tf = df_batch.xs("sksparse")

    # Plot results
    fig, axs = plt.subplots(num=2, nrows=2, sharex=True, clear=True)
    fig.suptitle(
        "Batch RHS Sparse Solve Performance\nA (100, 100) 2D Laplacian, B (100, 9,056)"
    )
    fig.set_size_inches((6.4, 8), forward=True)

    for i, col in enumerate(["time", "memory"]):
        # Plot marker to compare scikit-umfpack (no batching)
        sns.scatterplot(
            ax=axs[i],
            data=df_batch.xs("scikit-umfpack"),
            x="rhs_batch_size",
            y=col,
            hue="density",
            hue_norm=mpl.colors.LogNorm(),
            palette="mako_r",
            marker="X",
            s=100,
            legend=(i == 0),
        )

        # Plot sksparse results
        sns.lineplot(
            ax=axs[i],
            data=tf,
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
        fig_file = DATA_PATH / "umf_perf_batch.pdf"
        fig.savefig(fig_file)
        print(f"Saved figure to: {fig_file}")

    plt.show()


# =============================================================================
# =============================================================================
