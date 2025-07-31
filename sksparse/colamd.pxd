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


cdef extern from "colamd.h":
    # sizes of input and output arrays
    int COLAMD_KNOBS
    int COLAMD_STATS

    # indices of knobs
    int COLAMD_DENSE_ROW
    int COLAMD_DENSE_COL
    int COLAMD_AGGRESSIVE

    # indices of stats
    int COLAMD_DEFRAG_COUNT
    int COLAMD_STATUS
    int COLAMD_INFO1
    int COLAMD_INFO2
    int COLAMD_INFO3

    # return values of colamd
    int COLAMD_OK
    int COLAMD_OK_BUT_JUMBLED
    int COLAMD_ERROR_A_not_present
    int COLAMD_ERROR_p_not_present
    int COLAMD_ERROR_nrow_negative
    int COLAMD_ERROR_ncol_negative
    int COLAMD_ERROR_nnz_negative
    int COLAMD_ERROR_p0_nonzero
    int COLAMD_ERROR_A_too_small
    int COLAMD_ERROR_col_length_negative
    int COLAMD_ERROR_row_index_out_of_bounds
    int COLAMD_ERROR_out_of_memory
    int COLAMD_ERROR_internal_error

    size_t colamd_recommended(int32_t nnz, int32_t n_row, int32_t n_col)
    size_t colamd_l_recommended(int64_t nnz, int64_t n_row, int64_t n_col)

    void colamd_set_defaults(double knobs[])
    void colamd_l_set_defaults(double knobs[])

    int c_colamd "colamd"(
        int32_t n_row,
        int32_t n_col,
        int32_t Alen,
        int32_t A[],
        int32_t p[],
        double knobs[],
        int32_t stats[]
    )

    int c_colamd_l "colamd_l"(
        int64_t n_row,
        int64_t n_col,
        int64_t Alen,
        int64_t A[],
        int64_t p[],
        double knobs[],
        int64_t stats[]
    )

    # int symamd(
    #     int32_t n,
    #     int32_t A[],
    #     int32_t p[],
    #     int32_t perm[],
    #     double knobs[COLAMD_KNOBS],
    #     int32_t stats[COLAMD_STATS],
    #     void * (*allocate) (size_t, size_t),
    #     void (*release) (void *)
    # )

    # int symamd_l
    # (
    #     int64_t n,
    #     int64_t A[],
    #     int64_t p[],
    #     int64_t perm[],
    #     double knobs[COLAMD_KNOBS],
    #     int64_t stats[COLAMD_STATS],
    #     void * (*allocate) (size_t, size_t),
    #     void (*release) (void *)
    # )
