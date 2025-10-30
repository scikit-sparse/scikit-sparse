# Cython COLAMD header interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: colamd.pxd
#  Created: 2025-07-31 09:45
# =============================================================================
# distutils: language = c
# cython: language_level=3

from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t
from libc.stdlib cimport calloc, free


cdef extern from "colamd.h":
    ctypedef void* (*alloc_func)(size_t, size_t)
    ctypedef void (*free_func)(void *)

    # Get all #defined constants
    enum:
        # sizes of input and output arrays
        COLAMD_KNOBS
        COLAMD_STATS

        # indices of knobs
        COLAMD_DENSE_ROW
        COLAMD_DENSE_COL
        COLAMD_AGGRESSIVE

        # indices of stats
        COLAMD_DEFRAG_COUNT
        COLAMD_STATUS
        COLAMD_INFO1
        COLAMD_INFO2
        COLAMD_INFO3

        # return values of colamd
        COLAMD_OK
        COLAMD_OK_BUT_JUMBLED
        COLAMD_ERROR_A_not_present
        COLAMD_ERROR_p_not_present
        COLAMD_ERROR_nrow_negative
        COLAMD_ERROR_ncol_negative
        COLAMD_ERROR_nnz_negative
        COLAMD_ERROR_p0_nonzero
        COLAMD_ERROR_A_too_small
        COLAMD_ERROR_col_length_negative
        COLAMD_ERROR_row_index_out_of_bounds
        COLAMD_ERROR_out_of_memory
        COLAMD_ERROR_internal_error

    size_t colamd_recommended(int32_t nnz, int32_t n_row, int32_t n_col)
    size_t colamd_l_recommended(int64_t nnz, int64_t n_row, int64_t n_col)

    void colamd_set_defaults(double knobs[COLAMD_KNOBS])
    void colamd_l_set_defaults(double knobs[COLAMD_KNOBS])

    int c_colamd "colamd"(
        int32_t n_row,
        int32_t n_col,
        int32_t Alen,
        int32_t A[],
        int32_t p[],
        double knobs[COLAMD_KNOBS],
        int32_t stats[COLAMD_STATS]
    )

    int c_colamd_l "colamd_l"(
        int64_t n_row,
        int64_t n_col,
        int64_t Alen,
        int64_t A[],
        int64_t p[],
        double knobs[COLAMD_KNOBS],
        int64_t stats[COLAMD_STATS]
    )

    int c_symamd "symamd"(
        int32_t n,
        int32_t A[],
        int32_t p[],
        int32_t perm[],
        double knobs[COLAMD_KNOBS],
        int32_t stats[COLAMD_STATS],
        alloc_func allocate,
        free_func release
    )

    int c_symamd_l "symamd_l"(
        int64_t n,
        int64_t A[],
        int64_t p[],
        int64_t perm[],
        double knobs[COLAMD_KNOBS],
        int64_t stats[COLAMD_STATS],
        alloc_func allocate,
        free_func release
    )
