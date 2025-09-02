# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: test_septree.py
#  Created: 2025-09-02 16:41
# =============================================================================

"""Unit tests for the cholmod.prune_septree function."""

import numpy as np
import pytest

from sksparse.cholmod import nesdis, prune_septree

from ..helpers import generate_random_matrices, is_valid_permutation

general_As = list(generate_random_matrices(N_trials=10, N_max=200, d_scale=0.05))


@pytest.mark.parametrize("A", general_As)
def test_prune_septree(A):
    N = A.shape[0]
    p, cp, cmember = nesdis(A, return_separator=True)
    assert is_valid_permutation(p, N)
    assert len(cmember) == N
    assert np.all(cmember >= 0)
    assert np.all(cmember < N)
    assert len(cp) == cmember.max() + 1

    cp_pruned, cmember_pruned = prune_septree(cp, cmember)
    assert len(cmember) == N
    assert np.all(cmember >= 0)
    assert np.all(cmember < N)
    assert len(cp) == cmember.max() + 1
