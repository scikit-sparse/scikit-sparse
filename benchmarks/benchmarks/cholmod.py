"""Benchmark for factoring and solving sparse systems with CHOLMOD."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LaplacianNd

from sksparse import cholmod

Nsqs = [int(x) for x in np.unique(np.logspace(1, 4, num=10, dtype=int))]


def laplacian_2d(Nsq):
    """Create a 2D Laplacian matrix."""
    Ng = np.sqrt(Nsq).astype(int)
    A = -LaplacianNd((Ng, Ng), dtype=float).tosparse().tocsc()
    A[-1, -1] += 1  # make A non-singular
    # Guarantee canonical format
    A.has_sorted_indices = False
    A.has_canonical_format = False
    A.sum_duplicates()
    return A


class AnalyzeSuite:
    param_names = ["Nsq"]
    params = [Nsqs]

    def setup(self, Nsq):
        self.A = laplacian_2d(Nsq)

    def _run_analysis(self, Nsq):
        try:
            # dev branch uses CholeskyFactor.__init__
            cholmod.CholeskyFactor(self.A, lower=False, order="default")
        except AttributeError:
            # master branch uses analyze
            cholmod.analyze(self.A, ordering_method="default")

    def time_analyze(self, Nsq):
        self._run_analysis(Nsq)

    def peakmem_analyze(self, Nsq):
        self._run_analysis(Nsq)


class FactorizeSuite:
    param_names = ["Nsq"]
    params = [Nsqs]

    def setup(self, Nsq):
        self.A = laplacian_2d(Nsq)

    def _run_factorization(self, Nsq):
        try:
            # dev branch uses cho_factor
            cholmod.cho_factor(self.A, order="default")
        except AttributeError:
            # master branch uses cholesky
            cholmod.cholesky(self.A, ordering_method="default")

    def time_factorize(self, Nsq):
        self._run_factorization(Nsq)

    def peakmem_factorize(self, Nsq):
        self._run_factorization(Nsq)


class SolveSuite:
    param_names = ["Nsq", "K", "is_sparse"]
    params = [Nsqs, [1, 100, 1000], [True, False]]

    def setup(self, Nsq, K, is_sparse):
        self.A = laplacian_2d(Nsq)
        N = self.A.shape[0]

        # Define the RHS
        x_col = np.arange(1, N + 1, dtype=float)

        if K == 1:
            expect_x = x_col
        else:
            expect_x = np.outer(x_col, np.arange(1, K))  # many RHS columns

        self.b = self.A @ expect_x

        if is_sparse:
            if self.b.ndim == 1:
                self.b = self.b[:, np.newaxis]
            self.b = sparse.csc_array(self.b)

        # Pre-factorize the matrix to only time solver
        try:
            # dev branch uses cho_factor
            self.f = cholmod.cho_factor(self.A, order="default")
        except AttributeError:
            # master branch uses cholesky
            self.f = cholmod.cholesky(self.A, ordering_method="default")

    def _run_solve(self, Nsq, K, is_sparse):
        try:
            self.f.solve(self.b)
        except AttributeError:
            self.f.solve_A(self.b)

    def time_solve(self, Nsq, K, is_sparse):
        self._run_solve(Nsq, K, is_sparse)

    def peakmem_solve(self, Nsq, K, is_sparse):
        self._run_solve(Nsq, K, is_sparse)
