# Part of the scikit-sparse project.
# Copyright (C) 2025 the scikit-sparse developers. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_umfpack.py
#  Created: 2025-10-16 11:54
# =============================================================================

"""Unit tests for the umfpack module."""

import pytest

import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose

from scipy.sparse.linalg import LaplacianNd

from sksparse.umfpack import UMFFactor


@pytest.mark.parametrize("itype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_symbolic(itype, dtype):
    # Random matrix
    # N = 10
    # A = sparse.random_array((N, N), density=0.5, format="csc", dtype=dtype)
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
    print('---------- report_control():')
    f.report_control()
    print('---------- report_symbolic():')
    f.report_symbolic()
    print('---------- print(f.control):')
    print(f.control)
    f.factorize(A)
    print('---------- report_numeric():')
    f.report_numeric()
    print(f)
    # Solve a system
    expect_x = np.arange(1, N + 1, dtype=dtype)
    b = A @ expect_x
    x = f.solve(A, b)
    assert_allclose(x, expect_x, atol=1e-12, strict=True)
    print('---------- print(f.info):')
    print(f.info)
    print('---------- report_info():')
    f.report_info()
    # Print the factors
    print('---------- factors:')
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

