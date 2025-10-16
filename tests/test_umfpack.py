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

from sksparse.umfpack import UMFFactor


def test_symbolic():
    N = 10
    A = sparse.random_array((N, N), density=0.5, format="csc", dtype=float)
    A.setdiag(1.0)
    f = UMFFactor(A)
    f.report_control()
    f.report_symbolic()
