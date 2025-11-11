# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: spqr.pxd
#  Created: 2025-11-06 19:04
# =============================================================================
# distutils: language = c++

from libc.stdint cimport int32_t, int64_t, uintptr_t
from libc.stdlib cimport malloc
from libc.string cimport memcpy

from sksparse.cholmod cimport cholmod_common, cholmod_dense, cholmod_sparse


cdef extern from "SuiteSparseQR_definitions.h":
    # Get the #define'd constants
    enum:
        # Ordering methods
        SPQR_ORDERING_FIXED
        SPQR_ORDERING_NATURAL
        SPQR_ORDERING_COLAMD
        SPQR_ORDERING_GIVEN
        SPQR_ORDERING_CHOLMOD
        SPQR_ORDERING_AMD
        SPQR_ORDERING_METIS
        SPQR_ORDERING_DEFAULT
        SPQR_ORDERING_BEST
        SPQR_ORDERING_BESTAMD
        # tolerance options
        SPQR_DEFAULT_TOL
        SPQR_NO_TOL
        # qmult methods
        SPQR_QTX
        SPQR_QX
        SPQR_XQT
        SPQR_XQ
        # solution systems
        SPQR_RX_EQUALS_B
        SPQR_RETX_EQUALS_B
        SPQR_RTX_EQUALS_B
        SPQR_RTX_EQUALS_ETB


cdef extern from "SuiteSparseQR.hpp":
    cdef cppclass spqr_symbolic[Int]:
        Int m
        Int n
        Int anz

        Int *Sp
        Int *Sj
        Int *Qfill
        Int *PLinv
        Int *Sleft

        Int nf
        Int maxfn

        Int *Parent
        Int *Child
        Int *Childp

        Int *Super
        Int *Rp
        Int *Rj
        Int *Post
        Int rjsize

        Int do_rank_detection
        Int maxstack
        Int hisize
        Int keepH
        Int *Hip
        Int ntasks
        Int ns

        # These are for task parallelism (not yet supported)
        # Int *TaskChildp
        # Int *TaskChild
        # Int *TaskStack
        # Int *TaskFront
        # Int *TaskFrontp
        # Int *On_stack
        # Int *Stack_maxstack
        # Int *Fm
        # Int *Cm

        # size_t maxcsize
        # size_t maxesize
        # Int *ColCount

        # spqr_gpu_impl <Int> *QRgpu

    cdef cppclass spqr_numeric[Entry, Int]:
        Entry **Rblock
        Entry **Stacks
        Int *Stack_size
        Int hisize
        Int n
        Int m
        Int nf
        Int ntasks
        Int ns
        Int maxstack
        # rank detection
        char *Rdead
        Int rank
        Int rank1
        Int maxfrank
        double norm_E_fro
        # keeping Householder vectors
        Int keepH
        Int rjsize
        Int *HStair
        Entry *HTau
        Int *Hii
        Int *HPinv
        Int *Hm
        Int *Hr
        Int maxfm

    cdef cppclass SuiteSparseQR_factorization[Entry, Int]:
        double tol
        spqr_symbolic[Int] *QRsym
        spqr_numeric[Entry, Int] *QRnum
        Int *R1p
        Int *R1j
        Entry *R1x
        Int r1nz
        Int *Q1fill
        Int *P1inv
        Int *HP1inv
        Int *Rmap
        Int *RmapInv
        Int n1rows
        Int n1cols
        Int narows
        Int nacols
        Int bncols
        Int rank
        int allow_tol

    # "Expert" Functions
    SuiteSparseQR_factorization[Entry, Int] *SuiteSparseQR_factorize[Entry, Int](
        int ordering,
        double tol,
        cholmod_sparse *A,
        cholmod_common *cc
    )

    SuiteSparseQR_factorization[Entry, Int] *SuiteSparseQR_symbolic[Entry, Int](
        int ordering,
        int allow_tol,
        cholmod_sparse *A,
        cholmod_common *cc
    )

    int SuiteSparseQR_numeric[Entry, Int](
        double tol,
        cholmod_sparse *A,
        SuiteSparseQR_factorization[Entry, Int] *QR,
        cholmod_common *cc
    )

    cholmod_dense *SuiteSparseQR_qmult[Entry, Int](
        int method,
        SuiteSparseQR_factorization[Entry, Int] *QR,
        cholmod_dense *Xdense,
        cholmod_common *cc
    )

    cholmod_dense *SuiteSparseQR_solve[Entry, Int](
        int system,
        SuiteSparseQR_factorization[Entry, Int] *QR,
        cholmod_dense *B,
        cholmod_common *cc
    )

    int SuiteSparseQR_free[Entry, Int](
        SuiteSparseQR_factorization[Entry, Int] **QR,
        cholmod_common *cc
    )
