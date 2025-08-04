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

    # sizes of input and output arrays
    int CCOLAMD_KNOBS
    int CCOLAMD_STATS

    # indices of knobs
    int CCOLAMD_DENSE_ROW
    int CCOLAMD_DENSE_COL
    int CCOLAMD_AGGRESSIVE
    int CCOLAMD_LU

    # indices of stats
    int CCOLAMD_DEFRAG_COUNT
    int CCOLAMD_STATUS
    int CCOLAMD_INFO1
    int CCOLAMD_INFO2
    int CCOLAMD_INFO3

    # return values of ccolamd
    int CCOLAMD_OK
    int CCOLAMD_OK_BUT_JUMBLED
    int CCOLAMD_ERROR_A_not_present
    int CCOLAMD_ERROR_p_not_present
    int CCOLAMD_ERROR_nrow_negative
    int CCOLAMD_ERROR_ncol_negative
    int CCOLAMD_ERROR_nnz_negative
    int CCOLAMD_ERROR_p0_nonzero
    int CCOLAMD_ERROR_A_too_small
    int CCOLAMD_ERROR_col_length_negative
    int CCOLAMD_ERROR_row_index_out_of_bounds
    int CCOLAMD_ERROR_out_of_memory
    int CCOLAMD_ERROR_internal_error

    size_t ccolamd_recommended(int32_t nnz, int32_t n_row, int32_t n_col)
    size_t ccolamd_l_recommended(int64_t nnz, int64_t n_row, int64_t n_col)

    void ccolamd_set_defaults(double knobs[])
    void ccolamd_l_set_defaults(double knobs[])

    int c_ccolamd "ccolamd"(
        int32_t n_row,
        int32_t n_col,
        int32_t Alen,
        int32_t A[],
        int32_t p[],
        double knobs[],
        int32_t stats[],
        int32_t cmember[]
    )

    int c_ccolamd_l "ccolamd_l"(
        int64_t n_row,
        int64_t n_col,
        int64_t Alen,
        int64_t A[],
        int64_t p[],
        double knobs[],
        int64_t stats[],
        int64_t cmember[]
    )

    int c_csymamd "csymamd"(
        int32_t n,
        int32_t A[],
        int32_t p[],
        int32_t perm[],
        double knobs[],
        int32_t stats[],
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
        double knobs[],
        int64_t stats[],
        alloc_func allocate,
        free_func release,
        int64_t cmember[],
        int64_t stype
    )
