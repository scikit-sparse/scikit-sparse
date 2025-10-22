# Cython CCOLAMD header interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: ccolamd.pxd
#  Created: 2025-07-31 09:45
# =============================================================================
# distutils: language = c
# cython: language_level=3

from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t
from libc.stdlib cimport calloc, free


cdef extern from "ccolamd.h":
    ctypedef void* (*alloc_func)(size_t, size_t)
    ctypedef void (*free_func)(void *)

    # Get all #defined constants
    enum:
        # sizes of input and output arrays
        CCOLAMD_KNOBS
        CCOLAMD_STATS

        # indices of knobs
        CCOLAMD_DENSE_ROW
        CCOLAMD_DENSE_COL
        CCOLAMD_AGGRESSIVE
        CCOLAMD_LU

        # indices of stats
        CCOLAMD_DEFRAG_COUNT
        CCOLAMD_STATUS
        CCOLAMD_INFO1
        CCOLAMD_INFO2
        CCOLAMD_INFO3

        # return values of ccolamd
        CCOLAMD_OK
        CCOLAMD_OK_BUT_JUMBLED
        CCOLAMD_ERROR_A_not_present
        CCOLAMD_ERROR_p_not_present
        CCOLAMD_ERROR_nrow_negative
        CCOLAMD_ERROR_ncol_negative
        CCOLAMD_ERROR_nnz_negative
        CCOLAMD_ERROR_p0_nonzero
        CCOLAMD_ERROR_A_too_small
        CCOLAMD_ERROR_col_length_negative
        CCOLAMD_ERROR_row_index_out_of_bounds
        CCOLAMD_ERROR_out_of_memory
        CCOLAMD_ERROR_internal_error

    size_t ccolamd_recommended(int32_t nnz, int32_t n_row, int32_t n_col)
    size_t ccolamd_l_recommended(int64_t nnz, int64_t n_row, int64_t n_col)

    void ccolamd_set_defaults(double knobs[CCOLAMD_KNOBS])
    void ccolamd_l_set_defaults(double knobs[CCOLAMD_KNOBS])

    int c_ccolamd "ccolamd"(
        int32_t n_row,
        int32_t n_col,
        int32_t Alen,
        int32_t A[],
        int32_t p[],
        double knobs[CCOLAMD_KNOBS],
        int32_t stats[CCOLAMD_STATS],
        int32_t cmember[]
    )

    int c_ccolamd_l "ccolamd_l"(
        int64_t n_row,
        int64_t n_col,
        int64_t Alen,
        int64_t A[],
        int64_t p[],
        double knobs[CCOLAMD_KNOBS],
        int64_t stats[CCOLAMD_STATS],
        int64_t cmember[]
    )

    int c_csymamd "csymamd"(
        int32_t n,
        int32_t A[],
        int32_t p[],
        int32_t perm[],
        double knobs[CCOLAMD_KNOBS],
        int32_t stats[CCOLAMD_STATS],
        alloc_func allocate,
        free_func release,
        int32_t cmember[],
        int32_t stype
    )

    int c_csymamd_l "csymamd_l"(
        int64_t n,
        int64_t A[],
        int64_t p[],
        int64_t perm[],
        double knobs[CCOLAMD_KNOBS],
        int64_t stats[CCOLAMD_STATS],
        alloc_func allocate,
        free_func release,
        int64_t cmember[],
        int64_t stype
    )
