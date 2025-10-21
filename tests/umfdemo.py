# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: umfdemo.py
#  Created: 2025-10-20 19:51
# =============================================================================

"""Demo of UMFPACK usage via UMFFactor."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse.linalg import LaplacianNd

from sksparse.umfpack import UMFFactor

ITYPES = [np.int32, np.int64]
DTYPES = [np.float64, np.complex128]


def run_demo(itype, dtype):
    rng = np.random.default_rng(56)
    # Random matrix
    # N = 10
    # A = sparse.random_array((N, N), density=0.5, format="csc", rng=56, dtype=dtype)
    # A.setdiag(1.0)

    # Laplaceian grid
    A = -LaplacianNd((3, 3), dtype=dtype).tosparse().tocsc()
    A[-1, -1] += 1.0  # make non-singular
    N = A.shape[0]

    A.indptr = A.indptr.astype(itype)
    A.indices = A.indices.astype(itype)
    f = UMFFactor(A)
    print()
    print(f)
    print("---------- report_control():")
    f.report_control()
    print("---------- report_symbolic():")
    f.report_symbolic()
    print("---------- print(f.control):")
    print(f.control)
    f.factorize(A)
    print("---------- report_numeric():")
    f.report_numeric()
    print(f)
    # Solve a system
    expect_x = np.arange(1, N + 1, dtype=dtype)
    # Ensure non-zero complex parts
    if np.issubdtype(dtype, np.complexfloating):
        expect_x += 1j * 0.1 * rng.random(N)
    expect_x = np.r_[expect_x, 2 * expect_x].reshape((-1, 2))  # multiple RHS
    print(f"{expect_x=}")
    b = A @ expect_x
    x = f.solve(A, b)
    assert_allclose(x, expect_x, atol=1e-12, strict=True)
    print("---------- print(f.info):")
    print(f.info)
    print("---------- report_info():")
    f.report_info()
    # Print the factors
    print("---------- factors:")
    print(repr(f.L))
    print(repr(f.U))
    print(repr(f.perm_r))
    print(repr(f.perm_c))
    print(repr(f.R))
    L, U, p, q, r = f.L, f.U, f.perm_r, f.perm_c, f.R
    # Check that L U = P R A Q
    LU = (L @ U).toarray()
    PRAQ = (r[:, np.newaxis] * A).tocsc()[p][:, q].toarray()
    assert_allclose(LU, PRAQ, atol=1e-12, strict=True)


if __name__ == "__main__":
    for itype in ITYPES:
        for dtype in DTYPES:
            print(f"===== Testing itype={itype}, dtype={dtype} =====")
            run_demo(itype, dtype)

# =============================================================================
# =============================================================================
