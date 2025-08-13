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

from libc.stdint cimport int32_t, int64_t
from libc.string cimport memset
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

    # Ordering methods
    int CHOLMOD_MAXMETHODS
    int CHOLMOD_NATURAL
    # int CHOLMOD_GIVEN
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

    ctypedef struct cholmod_method_struct:
        int ordering

    ctypedef struct cholmod_common:
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

    int cholmod_start(cholmod_common *Common)
    int cholmod_l_start(cholmod_common *Common)

    int cholmod_finish(cholmod_common *Common)
    int cholmod_l_finish(cholmod_common *Common)

    cholmod_factor* cholmod_analyze(cholmod_sparse *A, cholmod_common *Common)
    cholmod_factor* cholmod_l_analyze(cholmod_sparse *A, cholmod_common *Common)

    int cholmod_factorize(cholmod_sparse *A, cholmod_factor *L, cholmod_common *Common)
    int cholmod_l_factorize(cholmod_sparse *A, cholmod_factor *L, cholmod_common *Common)

    cholmod_sparse* cholmod_factor_to_sparse(cholmod_factor *L, cholmod_common *Common)
    cholmod_sparse* cholmod_l_factor_to_sparse(cholmod_factor *L, cholmod_common *Common)

    int cholmod_free_sparse(cholmod_sparse **A, cholmod_common *Common)
    int cholmod_l_free_sparse(cholmod_sparse **A, cholmod_common *Common)

    int cholmod_free_factor(cholmod_factor **L, cholmod_common *Common)
    int cholmod_l_free_factor(cholmod_factor **L, cholmod_common *Common)

    int cholmod_drop(double tol, cholmod_sparse *A, cholmod_common *Common)
    int cholmod_l_drop(double tol, cholmod_sparse *A, cholmod_common *Common)

    cholmod_sparse* cholmod_transpose(cholmod_sparse *A, int mode, cholmod_common *Common)
    cholmod_sparse* cholmod_l_transpose(cholmod_sparse *A, int mode, cholmod_common *Common)
