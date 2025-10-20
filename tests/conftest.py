# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: conftest.py
#  Created: 2025-09-18 16:27
# =============================================================================

"""Common test fixtures for the scikit-sparse project unit tests."""

import numpy as np
import pytest
from scipy import sparse


# See: Davis, Timothy A. (2006). Direct Methods for Sparse Linear Systems,
# p 74 (Figure 5.1)
@pytest.fixture
def davis_example_qr():
    """Return a small example matrix from Davis (2006)."""
    N = 8
    rows = np.array([0, 1, 2, 3, 4, 5, 6,
                     3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6])
    cols = np.array([0, 1, 2, 3, 4, 5, 6,
                     0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7])
    # rng = np.random.default_rng(565656)
    # vals = rng.random(len(rows), dtype=np.float64)
    vals = np.ones(len(rows), dtype=np.float64)
    vals[:7] = np.arange(1, 8, dtype=np.float64)  # make diagonal entries non-unit
    A = sparse.csc_array((vals, (rows, cols)), shape=(N, N))
    return A
