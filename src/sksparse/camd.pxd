# Cython CAMD header interface
#
# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: camd.pxd
#  Created: 2025-08-01 13:02
# =============================================================================
# distutils: language = c
# cython: language_level=3

from libc.stdint cimport int32_t, int64_t


cdef extern from "camd.h":
    # sizes of Control and Info
    int CAMD_CONTROL
    int CAMD_INFO

    # indices of Control
    int CAMD_DENSE
    int CAMD_AGGRESSIVE

    # indices of Info
    int CAMD_STATUS             # return value of camd_order and camd_l_order
    int CAMD_N                  # A is n-by-n
    int CAMD_NZ                 # number of nonzeros in A
    int CAMD_SYMMETRY           # symmetry of pattern ( is sym.,  is unsym.)
    int CAMD_NZDIAG             # number of entries on diagonal
    int CAMD_NZ_A_PLUS_AT       # nz in A+A'
    int CAMD_NDENSE             # number of "dense" rows/columns in A
    int CAMD_MEMORY             # amount of memory used by CAMD
    int CAMD_NCMPA              # number of garbage collections in CAMD
    int CAMD_LNZ                # approx. nz in L, excluding the diagonal
    int CAMD_NDIV               # number of fl. point divides for LU and LDL'
    int CAMD_NMULTSUBS_LDL      # number of fl. point (*,-) pairs for LDL'
    int CAMD_NMULTSUBS_LU       # number of fl. point (*,-) pairs for LU
    int CAMD_DMAX               # max nz. in any column of L, incl. diagonal

    # return values of camd_order and camd_l_order
    int CAMD_OUT_OF_MEMORY
    int CAMD_INVALID

    # 32-bit CAMD interface
    int camd_order(
        int32_t n,
        const int32_t Ap[],
        const int32_t Ai[],
        int32_t P[],
        double Control[],
        double Info[],
        const int32_t C[]
    )
    void camd_defaults(double Control[])

    # 64-bit CAMD interface
    int camd_l_order(
        int64_t n,
        const int64_t Ap[],
        const int64_t Ai[],
        int64_t P[],
        double Control[],
        double Info[],
        const int64_t C[]
    )
