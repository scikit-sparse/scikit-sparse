# Cython AMD header interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: amd.pxd
#  Created: 2025-07-28 10:27
# =============================================================================
# distutils: language = c
# cython: language_level=3

from libc.stdint cimport int32_t, int64_t


cdef extern from "amd.h":
    # sizes of Control and Info
    int AMD_CONTROL
    int AMD_INFO

    # indices of Control
    int AMD_DENSE
    int AMD_AGGRESSIVE

    # indices of Info
    int AMD_STATUS             # return value of amd_order and amd_l_order
    int AMD_N                  # A is n-by-n
    int AMD_NZ                 # number of nonzeros in A
    int AMD_SYMMETRY           # symmetry of pattern ( is sym.,  is unsym.)
    int AMD_NZDIAG             # number of entries on diagonal
    int AMD_NZ_A_PLUS_AT       # nz in A+A'
    int AMD_NDENSE             # number of "dense" rows/columns in A
    int AMD_MEMORY             # amount of memory used by AMD
    int AMD_NCMPA              # number of garbage collections in AMD
    int AMD_LNZ                # approx. nz in L, excluding the diagonal
    int AMD_NDIV               # number of fl. point divides for LU and LDL'
    int AMD_NMULTSUBS_LDL      # number of fl. point (*,-) pairs for LDL'
    int AMD_NMULTSUBS_LU       # number of fl. point (*,-) pairs for LU
    int AMD_DMAX               # max nz. in any column of L, incl. diagonal

    # return values of amd_order and amd_l_order
    int AMD_OUT_OF_MEMORY
    int AMD_INVALID

    # 32-bit AMD interface
    int amd_order(
        int32_t n,
        const int32_t Ap[],
        const int32_t Ai[],
        int32_t P[],
        double Control[],
        double Info[]
    )
    void amd_defaults(double Control[])

    # 64-bit AMD interface
    int amd_l_order(
        int64_t n,
        const int64_t Ap[],
        const int64_t Ai[],
        int64_t P[],
        double Control[],
        double Info[]
    )
