# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: klu.pxd
#  Created: 2025-10-30 20:39
# =============================================================================
# distutils: language = c

from libc.stdint cimport int32_t, int64_t


cdef extern from "klu.h":
    # Get all #define constants
    enum:
        KLU_OK
        KLU_SINGULAR
        KLU_OUT_OF_MEMORY
        KLU_INVALID
        KLU_TOO_LARGE

    # ctypedef int32_t (*user_order_func) (int32_t, int32_t*, int32_t*, int32_t*, struct klu_common*)

    ctypedef struct klu_common:
        double tol
        double memgrow
        double initmem_amd
        double initmem
        double maxwork
        int btf
        int ordering
        int scale
        # user_order_func* user_order  # TODO
        void *user_data
        int halt_if_singular
        int status
        int nrealloc
        int32_t structural_rank
        int32_t numerical_rank
        int32_t singular_col
        int32_t noffdiag
        double flops
        double rcond
        double condest
        double rgrowth
        double work
        size_t memusage
        size_t mempeak

    ctypedef struct klu_l_common:
        double tol
        double memgrow
        double initmem_amd
        double initmem
        double maxwork
        int btf
        int ordering
        int scale
        # user_order_func* user_order  # TODO
        void *user_data
        int halt_if_singular
        int status
        int nrealloc
        int64_t structural_rank
        int64_t numerical_rank
        int64_t singular_col
        int64_t noffdiag
        double flops
        double rcond
        double condest
        double rgrowth
        double work
        size_t memusage
        size_t mempeak

    ctypedef struct klu_symbolic:
        double symmetry
        double est_flops
        double lnz
        double unz
        double *Lnz
        int32_t n
        int32_t nz
        int32_t *P
        int32_t *Q
        int32_t *R
        int32_t nzoff
        int32_t nblocks
        int32_t maxblock
        int32_t ordering
        int32_t do_btf
        int32_t structural_rank

    ctypedef struct klu_l_symbolic:
        double symmetry
        double est_flops
        double lnz
        double unz
        double *Lnz
        int64_t n
        int64_t nz
        int64_t *P
        int64_t *Q
        int64_t *R
        int64_t nzoff
        int64_t nblocks
        int64_t maxblock
        int64_t ordering
        int64_t do_btf
        int64_t structural_rank

    ctypedef struct klu_numeric:
        int32_t n
        int32_t nblocks
        int32_t lnz
        int32_t unz
        int32_t max_lnz_block
        int32_t max_unz_block
        int32_t *Pnum
        int32_t *Pinv
        int32_t *Lip
        int32_t *Uip
        int32_t *Llen
        int32_t *Ulen
        void **LUbx
        size_t *LUsize
        void *Udiag
        double *Rs
        size_t worksize
        void *Work
        void *Xwork
        int32_t *Iwork
        int32_t *Offp
        int32_t *Offi
        void *Offx
        int32_t nzoff

    ctypedef struct klu_l_numeric:
        int64_t n
        int64_t nblocks
        int64_t lnz
        int64_t unz
        int64_t max_lnz_block
        int64_t max_unz_block
        int64_t *Pnum
        int64_t *Pinv
        int64_t *Lip
        int64_t *Uip
        int64_t *Llen
        int64_t *Ulen
        void **LUbx
        size_t *LUsize
        void *Udiag
        double *Rs
        size_t worksize
        void *Work
        void *Xwork
        int64_t *Iwork
        int64_t *Offp
        int64_t *Offi
        void *Offx
        int64_t nzoff

    # ---------------------------------------------------------------------------------
    #         Functions
    # ---------------------------------------------------------------------------------
    int klu_defaults(klu_common *Common)
    int klu_l_defaults(klu_l_common *Common)

    klu_symbolic* klu_analyze(
        int32_t n,
        int32_t Ap[],
        int32_t Ai[],
        klu_common *Common
    )

    klu_l_symbolic* klu_l_analyze(
        int64_t n,
        int64_t Ap[],
        int64_t Ai[],
        klu_l_common *Common
    )

    # NOTE alias so we can use "def klu_factor(...)" externally
    klu_numeric* c_klu_factor "klu_factor"(
        int32_t Ap[],
        int32_t Ai[],
        double Ax[],
        klu_symbolic *Symbolic,
        klu_common *Common
    )

    klu_numeric *klu_z_factor(
        int32_t Ap[],
        int32_t Ai[],
        double Ax[],
        klu_symbolic *Symbolic,
        klu_common *Common
    )

    klu_l_numeric *klu_l_factor(
        int64_t Ap[],
        int64_t Ai[],
        double Ax[],
        klu_l_symbolic *Symbolic,
        klu_l_common *Common
    )

    klu_l_numeric *klu_zl_factor(
        int64_t Ap[],
        int64_t Ai[],
        double Ax[],
        klu_l_symbolic *Symbolic,
        klu_l_common *Common
    )


    int klu_refactor(
        int32_t Ap[],
        int32_t Ai[],
        double Ax[],
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_z_refactor(
        int32_t Ap[],
        int32_t Ai[],
        double Ax[],
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_l_refactor(
        int64_t Ap[],
        int64_t Ai[],
        double Ax[],
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_zl_refactor(
        int64_t Ap[],
        int64_t Ai[],
        double Ax[],
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int c_klu_solve "klu_solve"(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        int32_t ldim,
        int32_t nrhs,
        double B[],
        klu_common *Common
    )

    int klu_z_solve(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        int32_t ldim,
        int32_t nrhs,
        double B[],
        klu_common *Common
    )

    int klu_l_solve(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        int64_t ldim,
        int64_t nrhs,
        double B[],
        klu_l_common *Common
    )

    int klu_zl_solve(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        int64_t ldim,
        int64_t nrhs,
        double B[],
        klu_l_common *Common
    )

    int klu_sort(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_z_sort(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_l_sort(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_zl_sort(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_rcond(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_z_rcond(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_l_rcond(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_zl_rcond(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_flops(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_z_flops(
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_l_flops(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_zl_flops(
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_rgrowth(
        int32_t Ap[],
        int32_t Ai[],
        double Ax[],
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_z_rgrowth(
        int32_t Ap[],
        int32_t Ai[],
        double Ax[],
        klu_symbolic *Symbolic,
        klu_numeric *Numeric,
        klu_common *Common
    )

    int klu_l_rgrowth(
        int64_t Ap[],
        int64_t Ai[],
        double Ax[],
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_zl_rgrowth(
        int64_t Ap[],
        int64_t Ai[],
        double Ax[],
        klu_l_symbolic *Symbolic,
        klu_l_numeric *Numeric,
        klu_l_common *Common
    )

    int klu_extract(
        klu_numeric *Numeric,
        klu_symbolic *Symbolic,
        int32_t *Lp,
        int32_t *Li,
        double *Lx,
        int32_t *Up,
        int32_t *Ui,
        double *Ux,
        int32_t *Fp,
        int32_t *Fi,
        double *Fx,
        int32_t *P,
        int32_t *Q,
        double *Rs,
        int32_t *R,
        klu_common *Common
    )

    int klu_z_extract(
        klu_numeric *Numeric,
        klu_symbolic *Symbolic,
        int32_t *Lp,
        int32_t *Li,
        double *Lx,
        double *Lz,
        int32_t *Up,
        int32_t *Ui,
        double *Ux,
        double *Uz,
        int32_t *Fp,
        int32_t *Fi,
        double *Fx,
        double *Fz,
        int32_t *P,
        int32_t *Q,
        double *Rs,
        int32_t *R,
        klu_common *Common
    )

    int klu_l_extract(
        klu_l_numeric *Numeric,
        klu_l_symbolic *Symbolic,
        int64_t *Lp,
        int64_t *Li,
        double *Lx,
        int64_t *Up,
        int64_t *Ui,
        double *Ux,
        int64_t *Fp,
        int64_t *Fi,
        double *Fx,
        int64_t *P,
        int64_t *Q,
        double *Rs,
        int64_t *R,
        klu_l_common *Common
    )

    int klu_zl_extract(
        klu_l_numeric *Numeric,
        klu_l_symbolic *Symbolic,
        int64_t *Lp,
        int64_t *Li,
        double *Lx,
        double *Lz,
        int64_t *Up,
        int64_t *Ui,
        double *Ux,
        double *Uz,
        int64_t *Fp,
        int64_t *Fi,
        double *Fx,
        double *Fz,
        int64_t *P,
        int64_t *Q,
        double *Rs,
        int64_t *R,
        klu_l_common *Common
    )

    int klu_free_symbolic(klu_symbolic **Symbolic, klu_common *Common)
    int klu_l_free_symbolic(klu_l_symbolic **Symbolic, klu_l_common *Common)
    int klu_free_numeric(klu_numeric **Numeric, klu_common *Common)
    int klu_z_free_numeric (klu_numeric **Numeric, klu_common *Common)
    int klu_l_free_numeric (klu_l_numeric **Numeric, klu_l_common *Common)
    int klu_zl_free_numeric (klu_l_numeric **Numeric, klu_l_common *Common)
