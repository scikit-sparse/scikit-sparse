# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: umfpack.pyx
#  Created: 2025-10-16 11:35
# =============================================================================

"""
===================================================================
Unsymmetric Multifrontal LU Decomposition (:mod:`sksparse.umfpack`)
===================================================================

.. currentmodule:: sksparse.umfpack

.. versionadded:: 0.5.0


An interface to the SuiteSparse `UMFPACK
<https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/UMFPACK>`_
package, which computes the LU factorization and solves systems of equations
for sparse, possibly non-symmetric, indefinite matrices.
"""

import numpy as np
cimport numpy as np

def dummy_func():
    """A simple function to test that the Cython module compiles correctly."""
    cdef void* symbolic = NULL
    cdef double control[UMFPACK_CONTROL]
    cdef double info[UMFPACK_INFO]

    # Initialize control parameters to 0
    for i in range(UMFPACK_CONTROL):
        control[i] = 0.0

    umfpack_di_symbolic(0, 0, NULL, NULL, NULL, &symbolic, control, info)

    return True
