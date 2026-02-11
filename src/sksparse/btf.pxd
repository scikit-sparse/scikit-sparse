# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: btf.pxd
#  Created: 2025-08-04 20:12
# =============================================================================
# distutils: language = c
# cython: language_level=3

from libc.stdint cimport int32_t, int64_t


cdef extern from "btf.h":
    int32_t btf_maxtrans(
        int32_t nrow,
        int32_t ncol,
        int32_t Ap[],
        int32_t Ai[],
        double maxwork,
        double *work,
        int32_t Match[],
        int32_t Work[]
    )

    int64_t btf_l_maxtrans(
        int64_t nrow,
        int64_t ncol,
        int64_t Ap[],
        int64_t Ai[],
        double maxwork,
        double *work,
        int64_t Match[],
        int64_t Work[]
    )

    int32_t btf_strongcomp(
        int32_t n,
        int32_t Ap[],
        int32_t Ai[],
        int32_t Q[],
        int32_t P[],
        int32_t R[],
        int32_t Work[]
    )

    int64_t btf_l_strongcomp(
        int64_t n,
        int64_t Ap[],
        int64_t Ai[],
        int64_t Q[],
        int64_t P[],
        int64_t R[],
        int64_t Work[]
    )

    int32_t btf_order(
        int32_t n,
        int32_t Ap[],
        int32_t Ai[],
        double maxwork,
        double *work,
        int32_t P[],
        int32_t Q[],
        int32_t R[],
        int32_t *nmatch,
        int32_t Work[]
    )

    int64_t btf_l_order(
        int64_t n,
        int64_t Ap[],
        int64_t Ai[],
        double maxwork,
        double *work,
        int64_t P[],
        int64_t Q[],
        int64_t R[],
        int64_t *nmatch,
        int64_t Work[]
    )
