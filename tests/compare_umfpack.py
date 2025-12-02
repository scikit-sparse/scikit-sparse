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
from scikits.umfpack import splu, spsolve
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd
from tqdm import tqdm

from sksparse.umfpack import umf_factor, umf_solve

SEED = 565656

SAVE_FIGS = True

DATA_PATH = Path(__file__).absolute().parent.parent.parent / "_dev_data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

PKG_NAMES = ["sksparse", "scikits"]


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
    Ns = np.unique(np.logspace(1, 3, num=20, dtype=int))
    sqrtNs = np.unique([int(np.sqrt(N)) for N in Ns])

    results = []

    # Test performance of multiple solves
    for sqrtN in tqdm(sqrtNs):
        A = -LaplacianNd((sqrtN, sqrtN), dtype=float).tosparse().tocsc()
        A[-1, -1] += 1.0  # make sure A is non-singular
        N = A.shape[0]

        x_col = np.arange(1, N + 1, dtype=float)
        expect_x = np.outer(x_col, np.arange(1, 1000))  # many RHS columns
        B = A @ expect_x

        Am = sparse.csc_matrix(A)  # scikits does not accept csc_array

        # NOTE the solve tested here is a dense solve.
        funcs = {
            ("sksparse", "factorize"): partial(umf_factor, A),
            ("scikits", "factorize"): partial(splu, Am),
            ("sksparse", "solve"): partial(umf_solve, A, B, rhs_batch_size=1),
            ("scikits", "solve"): partial(spsolve, Am, B),
        }

        for key, func in tqdm(funcs.items(), leave=False):
            time, mem = measure_perf(func)
            results.append({
                "package": key[0],
                "function": key[1],
                "N": N,
                "time": time,
                "memory": mem,
            })

    # Build the results DataFrame
    df = (
        pd.DataFrame(results)
        .set_index(["package", "function", "N"])
        .sort_index()
    )

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
        results.append({
            "package": "scikits",
            "rhs_batch_size": 1,
            "density": d,
            "time": time,
            "memory": mem,
        })

        # Scikit-sparse umfpack solve (with batching)
        for rhs_batch_size in tqdm(batch_sizes, leave=False):
            solve_func = partial(lu.solve, b, rhs_batch_size=rhs_batch_size)
            time, mem = measure_perf(solve_func)
            results.append({
                "package": "sksparse",
                "rhs_batch_size": rhs_batch_size,
                "density": d,
                "time": time,
                "memory": mem,
            })

    # Build the results DataFrame
    df = (
        pd.DataFrame(results)
        .set_index(["package", "rhs_batch_size", "density"])
        .sort_index()
    )

    df.to_pickle(df_file)
    return df


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #         Package Tests
    # -------------------------------------------------------------------------
    df_pkg = run_package_comparison(
        DATA_PATH / "umf_perf_pkg_results.pkl", force_update=False
    )

    fig, axs = plt.subplots(num=1, nrows=2, sharex=True, clear=True)
    fig.suptitle("scikit-sparse vs scikits-umfpack Performance")
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
            legend=(i == 0),
        )
        axs[i].grid(True, which="both")
        axs[i].set(yscale="log")

    axs[0].set(
        xlabel="Number of Rows/Columns (N)",
        ylabel="time [s]",
        xscale="log",
    )

    axs[1].set(
        ylabel="peak memory [MB]",
    )

    plt.show()

    if SAVE_FIGS:
        fig_file = DATA_PATH / "umf_perf_pkg.pdf"
        fig.savefig(fig_file)
        print(f"Saved figure to: {fig_file}")

    # -------------------------------------------------------------------------
    #         Batch Solve Tests
    # -------------------------------------------------------------------------
    df = run_batch_comparison(
        DATA_PATH / "umf_perf_batch_results.pkl", force_update=False
    )

    tf = df.xs("sksparse")

    # Plot results
    fig, axs = plt.subplots(num=2, nrows=2, sharex=True, clear=True)
    fig.suptitle("sksparse.umfpack Batch RHS Solve Performance")
    fig.set_size_inches((6.4, 8), forward=True)

    for i, col in enumerate(["time", "memory"]):
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
        xlabel="RHS Batch Size",
        ylabel="time [s]",
        xscale="log",
        yscale="log",
    )

    axs[1].set(ylabel="peak memory [MB]")

    if SAVE_FIGS:
        fig_file = DATA_PATH / "umf_perf_batch.pdf"
        fig.savefig(fig_file)
        print(f"Saved figure to: {fig_file}")

    # -------------------------------------------------------------------------
    #         Plot sksparse vs scikits vs density
    # -------------------------------------------------------------------------
    # Only compare batch_size=1 case
    tf = df.xs(1, level="rhs_batch_size")

    fig, axs = plt.subplots(num=3, nrows=2, sharex=True, clear=True)
    fig.suptitle("Sparse RHS Solve Comparison")
    fig.set_size_inches((6.4, 8), forward=True)

    for i, col in enumerate(["time", "memory"]):
        sns.lineplot(
            ax=axs[i],
            data=tf,
            x="density",
            y=col,
            hue="package",
            ls="-",
            marker="o",
            legend=(i == 0),
        )
        axs[i].grid(True, which="both")

    axs[0].set(
        xlabel="RHS Density",
        ylabel="time [s]",
        xscale="log",
    )

    axs[1].set(
        xlabel="RHS Density",
        ylabel="peak memory [MB]",
    )

    if SAVE_FIGS:
        fig_file = DATA_PATH / "umf_compare_sparse_solve.pdf"
        fig.savefig(fig_file)
        print(f"Saved figure to: {fig_file}")

    plt.show()


# =============================================================================
# =============================================================================
