# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: cholmod.pxd
#  Created: 2025-08-11 12:59
# =============================================================================
# distutils: language = c
# cython: language_level=3

from libc.stdlib cimport malloc
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy, memset
from numpy cimport float32_t, float64_t, complex64_t, complex128_t


cdef extern from "cholmod.h":
    # xtypes
    int CHOLMOD_PATTERN
    int CHOLMOD_REAL
    int CHOLMOD_COMPLEX
    # int CHOLMOD_ZOMPLEX  # only used in old MATLAB interface

    # itypes
    int CHOLMOD_INT
    int CHOLMOD_LONG

    # dtypes
    int CHOLMOD_SINGLE
    int CHOLMOD_DOUBLE

    # supernodal types
    int CHOLMOD_SIMPLICIAL
    int CHOLMOD_AUTO
    int CHOLMOD_SUPERNODAL

    # Ordering methods
    int CHOLMOD_MAXMETHODS
    int CHOLMOD_NATURAL
    int CHOLMOD_GIVEN
    int CHOLMOD_AMD
    int CHOLMOD_METIS
    int CHOLMOD_NESDIS
    int CHOLMOD_COLAMD
    int CHOLMOD_POSTORDERED

    # Output codes
    int CHOLMOD_OK
    int CHOLMOD_NOT_INSTALLED
    int CHOLMOD_OUT_OF_MEMORY
    int CHOLMOD_TOO_LARGE
    int CHOLMOD_INVALID
    int CHOLMOD_GPU_PROBLEM
    int CHOLMOD_NOT_POSDEF
    int CHOLMOD_DSMALL

    # Solve codes
    int CHOLMOD_A
    int CHOLMOD_LDLt
    int CHOLMOD_LD
    int CHOLMOD_DLt
    int CHOLMOD_L
    int CHOLMOD_Lt
    int CHOLMOD_D
    int CHOLMOD_P
    int CHOLMOD_Pt

    ctypedef struct cholmod_method_struct:
        int ordering
        size_t nd_small
        int nd_components
        double nd_oksep
        int nd_camd

    ctypedef struct cholmod_common:
        int supernodal
        int final_asis
        int final_super
        int final_ll
        int final_pack
        int final_monotonic
        int final_resymbol
        int quick_return_if_not_posdef
        int nmethods
        cholmod_method_struct method[]
        int postorder
        int status

    ctypedef struct cholmod_factor:
        size_t n
        size_t minor
        void *Perm
        void *ColCount
        size_t nzmax
        void *p
        void *i
        void *x
        void *z
        void *nz
        void *next
        void *prev
        int ordering
        int is_ll
        int is_super
        int is_monotonic
        int itype
        int xtype
        int dtype

    ctypedef struct cholmod_sparse:
        size_t nrow
        size_t ncol
        size_t nzmax
        void *p
        void *i
        void *x
        void *z
        int stype
        int itype
        int xtype
        int dtype
        int sorted
        int packed

    ctypedef struct cholmod_dense:
        size_t nrow
        size_t ncol
        size_t nzmax
        size_t d
        void *x
        void *z
        int xtype
        int dtype

    int cholmod_start(cholmod_common *Common)
    int cholmod_l_start(cholmod_common *Common)

    int cholmod_finish(cholmod_common *Common)
    int cholmod_l_finish(cholmod_common *Common)

    cholmod_factor* cholmod_analyze(cholmod_sparse *A, cholmod_common *Common)
    cholmod_factor* cholmod_l_analyze(cholmod_sparse *A, cholmod_common *Common)

    cholmod_factor* cholmod_analyze_p(
        cholmod_sparse *A,
        int32_t *UserPerm,
        int32_t *fset,
        size_t fsize,
        cholmod_common *Common
    )
    cholmod_factor* cholmod_l_analyze_p(
        cholmod_sparse *A,
        int64_t *UserPerm,
        int64_t *fset,
        size_t fsize,
        cholmod_common *Common
    )

    int cholmod_factorize(cholmod_sparse *A, cholmod_factor *L, cholmod_common *Common)
    int cholmod_l_factorize(cholmod_sparse *A, cholmod_factor *L, cholmod_common *Common)

    int cholmod_factorize_p(
        cholmod_sparse *A,
        double beta [2],
        int32_t *fset,
        size_t fsize,
        cholmod_factor *L,
        cholmod_common *Common
    )
    int cholmod_l_factorize_p(
        cholmod_sparse *A,
        double beta[2],
        int64_t *fset,
        size_t fsize,
        cholmod_factor *L,
        cholmod_common *Common
    )

    cholmod_sparse *cholmod_spsolve(
        int sys,
        cholmod_factor *L,
        cholmod_sparse *B,
        cholmod_common *Common
    )
    cholmod_sparse *cholmod_l_spsolve(
        int sys,
        cholmod_factor *L,
        cholmod_sparse *B,
        cholmod_common *Common
    )

    cholmod_dense *cholmod_solve(
        int sys,
        cholmod_factor *L,
        cholmod_dense *B,
        cholmod_common *Common
    )
    cholmod_dense *cholmod_l_solve(
        int sys,
        cholmod_factor *L,
        cholmod_dense *B,
        cholmod_common *Common
    )

    int cholmod_updown(
        int update,
        cholmod_sparse *C,
        cholmod_factor *L,
        cholmod_common *Common
    )
    int cholmod_l_updown(
        int update,
        cholmod_sparse *C,
        cholmod_factor *L,
        cholmod_common *Common
    )

    int cholmod_rowadd(
        size_t k,
        cholmod_sparse *R,
        cholmod_factor *L,
        cholmod_common *Common
    )
    int cholmod_l_rowadd(
        size_t k,
        cholmod_sparse *R,
        cholmod_factor *L,
        cholmod_common *Common
    )

    int cholmod_rowdel(
        size_t k,
        cholmod_sparse *R,
        cholmod_factor *L,
        cholmod_common *Common
    )
    int cholmod_l_rowdel(
        size_t k,
        cholmod_sparse *R,
        cholmod_factor *L,
        cholmod_common *Common
    )

    cholmod_sparse *cholmod_submatrix(
        cholmod_sparse *A,
        int32_t *rset,
        int64_t rsize,
        int32_t *cset,
        int64_t csize,
        int mode,
        int sorted,
        cholmod_common *Common
    )
    cholmod_sparse *cholmod_l_submatrix(
        cholmod_sparse *A,
        int64_t *rset,
        int64_t rsize,
        int64_t *cset,
        int64_t csize,
        int mode,
        int sorted,
        cholmod_common *Common
    )

    double cholmod_rcond(cholmod_factor *L, cholmod_common *Common)
    double cholmod_l_rcond(cholmod_factor *L, cholmod_common *Common)

    cholmod_sparse* cholmod_factor_to_sparse(cholmod_factor *L, cholmod_common *Common)
    cholmod_sparse* cholmod_l_factor_to_sparse(cholmod_factor *L, cholmod_common *Common)

    cholmod_sparse *cholmod_allocate_sparse(
        size_t nrow,
        size_t ncol,
        size_t nzmax,
        int sorted,
        int packed,
        int stype,
        int xdtype,
        cholmod_common *Common
    )
    cholmod_sparse *cholmod_l_allocate_sparse(
        size_t nrow,
        size_t ncol,
        size_t nzmax,
        int sorted,
        int packed,
        int stype,
        int xdtype,
        cholmod_common *Common
    )

    cholmod_factor* cholmod_allocate_factor(size_t n, cholmod_common *Common)
    cholmod_factor* cholmod_l_allocate_factor(size_t n, cholmod_common *Common)

    int cholmod_change_factor(
        int to_xtype,
        int to_ll,
        int to_super,
        int to_packed,
        int to_monotonic,
        cholmod_factor *L,
        cholmod_common *Common
    ) 
    int cholmod_l_change_factor(
        int to_xtype,
        int to_ll,
        int to_super,
        int to_packed,
        int to_monotonic,
        cholmod_factor *L,
        cholmod_common *Common
    ) 

    int cholmod_etree(
        cholmod_sparse *A,
        int32_t *Parent,
        cholmod_common *Common
    )
    int cholmod_l_etree(
        cholmod_sparse *A,
        int64_t *Parent,
        cholmod_common *Common
    )

    int32_t cholmod_postorder(
        int32_t *Parent,
        size_t n,
        int32_t *Weight,
        int32_t *Post,
        cholmod_common *Common
    )
    int64_t cholmod_l_postorder(
        int64_t *Parent,
        size_t n,
        int64_t *Weight,
        int64_t *Post,
        cholmod_common *Common
    )

    int cholmod_rowcolcounts(
        cholmod_sparse *A,
        int32_t *fset,
        size_t fsize,
        int32_t *Parent,
        int32_t *Post,
        int32_t *RowCount,
        int32_t *ColCount,
        int32_t *First,
        int32_t *Level,
        cholmod_common *Common
    )
    int cholmod_l_rowcolcounts(
        cholmod_sparse *A,
        int64_t *fset,
        size_t fsize,
        int64_t *Parent,
        int64_t *Post,
        int64_t *RowCount,
        int64_t *ColCount,
        int64_t *First,
        int64_t *Level,
        cholmod_common *Common
    )

    int cholmod_row_subtree(
        cholmod_sparse *A,
        cholmod_sparse *F,
        size_t krow,
        int32_t *Parent,
        cholmod_sparse *R,
        cholmod_common *Common
    )
    int cholmod_l_row_subtree(
        cholmod_sparse *A,
        cholmod_sparse *F,
        size_t krow,
        int64_t *Parent,
        cholmod_sparse *R,
        cholmod_common *Common
    )

    int cholmod_resymbol(
        cholmod_sparse *A,
        int *fset,
        size_t fsize,
        int pack,
        cholmod_factor *L,
        cholmod_common *Common
    )
    int cholmod_l_resymbol(
        cholmod_sparse *A,
        int *fset,
        size_t fsize,
        int pack,
        cholmod_factor *L,
        cholmod_common *Common
    )

    int cholmod_resymbol_noperm(
        cholmod_sparse *A,
        int *fset,
        size_t fsize,
        int pack,
        cholmod_factor *L,
        cholmod_common *Common
    )
    int cholmod_l_resymbol_noperm(
        cholmod_sparse *A,
        int *fset,
        size_t fsize,
        int pack,
        cholmod_factor *L,
        cholmod_common *Common
    )

    int64_t cholmod_bisect(
        cholmod_sparse *A,
        int32_t *fset,
        size_t fsize,
        int compress,
        int32_t *Partition,
        cholmod_common *Common
    )
    int64_t cholmod_l_bisect(
        cholmod_sparse *A,
        int64_t *fset,
        size_t fsize,
        int compress,
        int64_t *Partition,
        cholmod_common *Common
    )

    int64_t cholmod_nested_dissection(
        cholmod_sparse *A,
        int32_t *fset,
        size_t fsize,
        int32_t *Perm,
        int32_t *CParent,
        int32_t *Cmember,
        cholmod_common *Common
    )
    int64_t cholmod_l_nested_dissection(
        cholmod_sparse *A,
        int64_t *fset,
        size_t fsize,
        int64_t *Perm,
        int64_t *CParent,
        int64_t *Cmember,
        cholmod_common *Common
    )

    int cholmod_metis(
        cholmod_sparse *A,
        int32_t *fset,
        size_t fsize,
        int postorder,
        int32_t *Perm,
        cholmod_common *Common
    )
    int cholmod_l_metis(
        cholmod_sparse *A,
        int64_t *fset,
        size_t fsize,
        int postorder,
        int64_t *Perm,
        cholmod_common *Common
    )

    int64_t cholmod_collapse_septree(
        size_t n,
        size_t ncomponents,
        double nd_oksep,
        size_t nd_small,
        int32_t *CParent,
        int32_t *Cmember,
        cholmod_common *Common
    )
    int64_t cholmod_l_collapse_septree(
        size_t n,
        size_t ncomponents,
        double nd_oksep,
        size_t nd_small,
        int64_t *CParent,
        int64_t *Cmember,
        cholmod_common *Common
    )

    void *cholmod_free(
        size_t n,
        size_t size,
        void *p,
        cholmod_common *Common
    )
    void *cholmod_l_free(
        size_t n,
        size_t size,
        void *p,
        cholmod_common *Common
    )

    int cholmod_free_sparse(cholmod_sparse **A, cholmod_common *Common)
    int cholmod_l_free_sparse(cholmod_sparse **A, cholmod_common *Common)

    int cholmod_free_dense(cholmod_dense **A, cholmod_common *Common)
    int cholmod_l_free_dense(cholmod_dense **A, cholmod_common *Common)

    int cholmod_free_factor(cholmod_factor **L, cholmod_common *Common)
    int cholmod_l_free_factor(cholmod_factor **L, cholmod_common *Common)

    int cholmod_drop(double tol, cholmod_sparse *A, cholmod_common *Common)
    int cholmod_l_drop(double tol, cholmod_sparse *A, cholmod_common *Common)

    cholmod_sparse* cholmod_transpose(cholmod_sparse *A, int mode, cholmod_common *Common)
    cholmod_sparse* cholmod_l_transpose(cholmod_sparse *A, int mode, cholmod_common *Common)

    void *cholmod_malloc(size_t n, size_t size, cholmod_common *Common)
    void *cholmod_l_malloc(size_t n, size_t size, cholmod_common *Common)

    int cholmod_check_perm(int32_t *Perm, size_t len, size_t n, cholmod_common *Common)
    int cholmod_l_check_perm(int64_t *Perm, size_t len, size_t n, cholmod_common *Common)
