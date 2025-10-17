# Part of the scikit-sparse project.
# Copyright (C) 2025 the scikit-sparse developers. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: umfpack.pxd
#  Created: 2025-10-16 11:17
# =============================================================================
# distutils: language = c

from libc.stdint cimport int32_t, int64_t


cdef extern from "umfpack.h":
    # Get all #define constants
    enum:
        # length of Control and Info arrays
        UMFPACK_INFO
        UMFPACK_CONTROL

        # -------------------------------------------------------------------------
        #         Info Parameters
        # -------------------------------------------------------------------------
        # return status and Info for all routines
        UMFPACK_STATUS
        UMFPACK_NROW
        UMFPACK_NCOL
        UMFPACK_NZ

        # computed in _symbolic and _numeric
        UMFPACK_SIZE_OF_UNIT

        # computed in _symbolic
        UMFPACK_SIZE_OF_INT
        UMFPACK_SIZE_OF_LONG
        UMFPACK_SIZE_OF_POINTER
        UMFPACK_SIZE_OF_ENTRY
        UMFPACK_NDENSE_ROW
        UMFPACK_NEMPTY_ROW
        UMFPACK_NDENSE_COL
        UMFPACK_NEMPTY_COL
        UMFPACK_SYMBOLIC_DEFRAG
        UMFPACK_SYMBOLIC_PEAK_MEMORY
        UMFPACK_SYMBOLIC_SIZE
        UMFPACK_SYMBOLIC_TIME
        UMFPACK_SYMBOLIC_WALLTIME
        UMFPACK_STRATEGY_USED
        UMFPACK_ORDERING_USED
        UMFPACK_QFIXED
        UMFPACK_DIAG_PREFERRED
        UMFPACK_PATTERN_SYMMETRY
        UMFPACK_NZ_A_PLUS_AT
        UMFPACK_NZDIAG

        # AMD statistics
        UMFPACK_SYMMETRIC_LUNZ
        UMFPACK_SYMMETRIC_FLOPS
        UMFPACK_SYMMETRIC_NDENSE
        UMFPACK_SYMMETRIC_DMAX

        # singleton pruning
        UMFPACK_COL_SINGLETONS
        UMFPACK_ROW_SINGLETONS
        UMFPACK_N2
        UMFPACK_S_SYMMETRIC

        # estimates in _symbolic
        UMFPACK_NUMERIC_SIZE_ESTIMATE
        UMFPACK_PEAK_MEMORY_ESTIMATE
        UMFPACK_FLOPS_ESTIMATE
        UMFPACK_LNZ_ESTIMATE
        UMFPACK_UNZ_ESTIMATE
        UMFPACK_VARIABLE_INIT_ESTIMATE
        UMFPACK_VARIABLE_PEAK_ESTIMATE
        UMFPACK_VARIABLE_FINAL_ESTIMATE
        UMFPACK_MAX_FRONT_SIZE_ESTIMATE
        UMFPACK_MAX_FRONT_NROWS_ESTIMATE
        UMFPACK_MAX_FRONT_NCOLS_ESTIMATE

        # exact values in _numeric
        UMFPACK_NUMERIC_SIZE
        UMFPACK_PEAK_MEMORY
        UMFPACK_FLOPS
        UMFPACK_LNZ
        UMFPACK_UNZ
        UMFPACK_VARIABLE_INIT
        UMFPACK_VARIABLE_PEAK
        UMFPACK_VARIABLE_FINAL
        UMFPACK_MAX_FRONT_SIZE
        UMFPACK_MAX_FRONT_NROWS
        UMFPACK_MAX_FRONT_NCOLS

        # computed in _numeric
        UMFPACK_NUMERIC_DEFRAG
        UMFPACK_NUMERIC_REALLOC
        UMFPACK_NUMERIC_COSTLY_REALLOC
        UMFPACK_COMPRESSED_PATTERN
        UMFPACK_LU_ENTRIES
        UMFPACK_NUMERIC_TIME
        UMFPACK_UDIAG_NZ
        UMFPACK_RCOND
        UMFPACK_WAS_SCALED
        UMFPACK_RSMIN
        UMFPACK_RSMAX
        UMFPACK_UMIN
        UMFPACK_UMAX
        UMFPACK_ALLOC_INIT_USED
        UMFPACK_FORCED_UPDATES
        UMFPACK_NUMERIC_WALLTIME
        UMFPACK_NOFF_DIAG

        UMFPACK_ALL_LNZ
        UMFPACK_ALL_UNZ
        UMFPACK_NZDROPPED

        # computed in _solve
        UMFPACK_IR_TAKEN
        UMFPACK_IR_ATTEMPTED
        UMFPACK_OMEGA1
        UMFPACK_OMEGA2
        UMFPACK_SOLVE_FLOPS
        UMFPACK_SOLVE_TIME
        UMFPACK_SOLVE_WALLTIME

        # -------------------------------------------------------------------------
        #         Control parameters for all routines
        # -------------------------------------------------------------------------
        UMFPACK_PRL

        # used in _symbolic only
        UMFPACK_DENSE_ROW
        UMFPACK_DENSE_COL
        UMFPACK_BLOCK_SIZE
        UMFPACK_STRATEGY
        UMFPACK_ORDERING
        UMFPACK_FIXQ
        UMFPACK_AMD_DENSE
        UMFPACK_AGGRESSIVE
        UMFPACK_SINGLETONS

        # used in _numeric only
        UMFPACK_PIVOT_TOLERANCE
        UMFPACK_ALLOC_INIT
        UMFPACK_SYM_PIVOT_TOLERANCE
        UMFPACK_SCALE
        UMFPACK_FRONT_ALLOC_INIT
        UMFPACK_DROPTOL

        # used in _solve only
        UMFPACK_IRSTEP

        # compile-time
        UMFPACK_COMPILED_WITH_BLAS

        # new in v6.0.0
        UMFPACK_STRATEGY_THRESH_SYM
        UMFPACK_STRATEGY_THRESH_NNZDIAG

        # Control[UMFPACK_STRATEGY] values
        UMFPACK_STRATEGY_AUTO
        UMFPACK_STRATEGY_UNSYMMETRIC
        UMFPACK_STRATEGY_OBSOLETE
        UMFPACK_STRATEGY_SYMMETRIC

        UMFPACK_SCALE_NONE
        UMFPACK_SCALE_SUM
        UMFPACK_SCALE_MAX

        UMFPACK_ORDERING_CHOLMOD
        UMFPACK_ORDERING_AMD
        UMFPACK_ORDERING_GIVEN
        UMFPACK_ORDERING_METIS
        UMFPACK_ORDERING_BEST
        UMFPACK_ORDERING_NONE
        UMFPACK_ORDERING_USER
        UMFPACK_ORDERING_METIS_GUARD

        # Defaults
        UMFPACK_DEFAULT_PRL
        UMFPACK_DEFAULT_DENSE_ROW
        UMFPACK_DEFAULT_DENSE_COL
        UMFPACK_DEFAULT_PIVOT_TOLERANCE
        UMFPACK_DEFAULT_SYM_PIVOT_TOLERANCE
        UMFPACK_DEFAULT_BLOCK_SIZE
        UMFPACK_DEFAULT_ALLOC_INIT
        UMFPACK_DEFAULT_FRONT_ALLOC_INIT
        UMFPACK_DEFAULT_IRSTEP
        UMFPACK_DEFAULT_SCALE
        UMFPACK_DEFAULT_STRATEGY
        UMFPACK_DEFAULT_AMD_DENSE
        UMFPACK_DEFAULT_FIXQ
        UMFPACK_DEFAULT_AGGRESSIVE
        UMFPACK_DEFAULT_DROPTOL
        UMFPACK_DEFAULT_ORDERING
        UMFPACK_DEFAULT_SINGLETONS
        UMFPACK_DEFAULT_STRATEGY_THRESH_SYM
        UMFPACK_DEFAULT_STRATEGY_THRESH_NNZDIAG

        # Output status values
        UMFPACK_OK

        UMFPACK_WARNING_singular_matrix
        UMFPACK_WARNING_determinant_underflow
        UMFPACK_WARNING_determinant_overflow

        UMFPACK_ERROR_out_of_memory
        UMFPACK_ERROR_invalid_Numeric_object
        UMFPACK_ERROR_invalid_Symbolic_object
        UMFPACK_ERROR_argument_missing
        UMFPACK_ERROR_n_nonpositive
        UMFPACK_ERROR_invalid_matrix
        UMFPACK_ERROR_different_pattern
        UMFPACK_ERROR_invalid_system
        UMFPACK_ERROR_invalid_permutation
        UMFPACK_ERROR_internal_error
        UMFPACK_ERROR_file_IO
        UMFPACK_ERROR_ordering_failed
        UMFPACK_ERROR_invalid_blob

        # Solve system types
        UMFPACK_A
        UMFPACK_At
        UMFPACK_Aat

        UMFPACK_Pt_L
        UMFPACK_L
        UMFPACK_Lt_P
        UMFPACK_Lat_P
        UMFPACK_Lt
        UMFPACK_Lat

        UMFPACK_U_Qt
        UMFPACK_U
        UMFPACK_Q_Ut
        UMFPACK_Q_Uat
        UMFPACK_Ut
        UMFPACK_Uat

    # -------------------------------------------------------------------------
    #         Functions
    # -------------------------------------------------------------------------
    void umfpack_di_defaults(
        double Control[UMFPACK_CONTROL]
    )

    int umfpack_di_symbolic(
        int32_t n_row,
        int32_t n_col,
        const int32_t Ap[],
        const int32_t Ai[],
        const double Ax[],
        void **Symbolic,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_dl_symbolic(
        int64_t n_row,
        int64_t n_col,
        const int64_t Ap[],
        const int64_t Ai[],
        const double Ax[],
        void **Symbolic,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_zi_symbolic(
        int32_t n_row,
        int32_t n_col,
        const int32_t Ap[],
        const int32_t Ai[],
        const double Ax[],
        const double Az[],
        void **Symbolic,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_zl_symbolic(
        int64_t n_row,
        int64_t n_col,
        const int64_t Ap[],
        const int64_t Ai[],
        const double Ax[],
        const double Az[],
        void **Symbolic,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_di_numeric(
        const int32_t Ap[],
        const int32_t Ai[],
        const double Ax[],
        void *Symbolic,
        void **Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_dl_numeric(
        const int64_t Ap[],
        const int64_t Ai[],
        const double Ax[],
        void *Symbolic,
        void **Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_zi_numeric(
        const int32_t Ap[],
        const int32_t Ai[],
        const double Ax[],
        const double Az[],
        void *Symbolic,
        void **Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_zl_numeric(
        const int64_t Ap[],
        const int64_t Ai[],
        const double Ax[],
        const double Az[],
        void *Symbolic,
        void **Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )


    int umfpack_di_solve(
        int sys,
        const int32_t Ap[],
        const int32_t Ai[],
        const double Ax[],
        double X[],
        const double B[],
        void *Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_dl_solve(
        int sys,
        const int64_t Ap[],
        const int64_t Ai[],
        const double Ax[],
        double X[],
        const double B[],
        void *Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_zi_solve(
        int sys,
        const int32_t Ap[],
        const int32_t Ai[],
        const double Ax[],
        const double Az[],
        double Xx[],
        double Xz[],
        const double Bx[],
        const double Bz[],
        void *Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    int umfpack_zl_solve(
        int sys,
        const int64_t Ap[],
        const int64_t Ai[],
        const double Ax[],
        const double Az[],
        double Xx[],
        double Xz[],
        const double Bx[],
        const double Bz[],
        void *Numeric,
        const double Control[UMFPACK_CONTROL],
        double Info[UMFPACK_INFO]
    )

    void umfpack_di_free_symbolic(
        void **Symbolic
    )

    void umfpack_dl_free_symbolic(
        void **Symbolic
    )

    void umfpack_zi_free_symbolic(
        void **Symbolic
    )

    void umfpack_zl_free_symbolic(
        void **Symbolic
    )

    void umfpack_di_free_numeric(
        void **Numeric
    )

    void umfpack_dl_free_numeric(
        void **Numeric
    )

    void umfpack_zi_free_numeric(
        void **Numeric
    )

    void umfpack_zl_free_numeric(
        void **Numeric
    )

    # -------------------------------------------------------------------------
    #         Debugging
    # -------------------------------------------------------------------------
    void umfpack_di_report_control(
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_di_report_symbolic(
        void *Symbolic,
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_dl_report_symbolic(
        void *Symbolic,
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_zi_report_symbolic(
        void *Symbolic,
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_zl_report_symbolic(
        void *Symbolic,
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_di_report_numeric(
        void *Numeric,
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_dl_report_numeric(
        void *Numeric,
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_zi_report_numeric(
        void *Numeric,
        const double Control[UMFPACK_CONTROL]
    )

    int umfpack_zl_report_numeric(
        void *Numeric,
        const double Control[UMFPACK_CONTROL]
    )

