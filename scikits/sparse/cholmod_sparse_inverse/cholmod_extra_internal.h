/* ========================================================================== */
/* === cholmod_extra_internal.h ============================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Copyright (C) 2012 Jaakko Luttinen
 *
 * cholmod_extra_internal.h is licensed under Version 2 of the GNU
 * General Public License, or (at your option) any later version. See
 * LICENSE for a text of the license.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * This file is part of CHOLMOD Extra Module.
 *
 * CHOLDMOD Extra Module is free software: you can redistribute it
 * and/or modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 2 of
 * the License, or (at your option) any later version.
 *
 * CHOLMOD Extra Module is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CHOLMOD Extra Module.  If not, see
 * <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * CHOLMOD Extra Module. Routines for internal use.
 *
 * BLAS_dsymv             Symmetric matrix and vector multiplication
 * BLAS_ddot              Dot product of two vectors
 *
 * -------------------------------------------------------------------------- */

#ifndef CHOLMOD_EXTRA_INTERNAL_H
#define CHOLMOD_EXTRA_INTERNAL_H

#include <suitesparse/cholmod_internal.h>

#ifdef SUN64

#define BLAS_DSYR2K dsyr2k_64_
#define BLAS_DSYMM dsymm_64_
#define BLAS_DSYMV dsymv_64_
#define BLAS_DDOT ddot_64_

#elif defined (BLAS_NO_UNDERSCORE)

#define BLAS_DSYR2K dsyr2k
#define BLAS_DSYMM dsymm
#define BLAS_DSYMV dsymv
#define BLAS_DDOT ddot

#else

#define BLAS_DSYR2K dsyr2k_
#define BLAS_DSYMM dsymm_
#define BLAS_DSYMV dsymv_
#define BLAS_DDOT ddot_

#endif

// DSYR2K(UPLO,TRANS,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
void BLAS_DSYR2K (char *uplo, char *trans, BLAS_INT *n, BLAS_INT *k, double *alpha,
                 double *A, BLAS_INT *lda, double *B, BLAS_INT *ldb, double *beta,
                 double *C, BLAS_INT *ldc) ;
#define BLAS_dsyr2k(uplo,trans,m,k,alpha,A,lda,B,ldb,beta,C,ldc)      \
{ \
    BLAS_INT N = n, K = k, LDA = lda, LDB = ldb, LDC = ldc ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ(K,k) && EQ (LDA,lda) &&  \
        EQ (LDB,ldb) && EQ (LDC,ldc))) \
    { \
        Common->blas_ok = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || Common->blas_ok) \
    { \
        BLAS_DSYR2K (uplo, trans, &N, &K, alpha, A, &LDA, B, &LDB, beta, C, &LDC) ; \
    } \
}

// DSYMM(SIDE,UPLO,M,N,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
void BLAS_DSYMM (char *side, char *uplo, BLAS_INT *m, BLAS_INT *n, double *alpha,
                 double *A, BLAS_INT *lda, double *B, BLAS_INT *ldb, double *beta,
                 double *C, BLAS_INT *ldc) ;
#define BLAS_dsymm(side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc) \
{ \
    BLAS_INT M = m, N = n, LDA = lda, LDB = ldb, LDC = ldc ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ(N,n) && EQ (LDA,lda) &&  \
        EQ (LDB,ldb) && EQ (LDC,ldc))) \
    { \
        Common->blas_ok = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || Common->blas_ok) \
    { \
        BLAS_DSYMM (side, uplo, &M, &N, alpha, A, &LDA, B, &LDB, beta, C, &LDC) ; \
    } \
}


// DSYMV(UPLO,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
void BLAS_DSYMV (char *uplo, BLAS_INT *m, double *alpha,
	double *A, BLAS_INT *lda, double *X, BLAS_INT *incx, double *beta,
	double *Y, BLAS_INT *incy) ;
#define BLAS_dsymv(trans,m,alpha,A,lda,X,incx,beta,Y,incy) \
{ \
    BLAS_INT M = m, LDA = lda, INCX = incx, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (LDA,lda) && \
        EQ (INCX,incx) && EQ (INCY,incy))) \
    { \
        Common->blas_ok = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || Common->blas_ok) \
    { \
        BLAS_DSYMV (trans, &M, alpha, A, &LDA, X, &INCX, beta, Y, &INCY) ; \
    } \
}

// DDOT(N,DX,INCX,DY,INCY)
double BLAS_DDOT (BLAS_INT *n, double *X, BLAS_INT *incx, 
                  double *Y, BLAS_INT *incy) ;
#define BLAS_ddot(n,X,incx,Y,incy,z)             \
{ \
    BLAS_INT N = n, INCX = incx, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && \
        EQ (INCX,incx) && EQ (INCY,incy))) \
    { \
        Common->blas_ok = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || Common->blas_ok) \
    { \
        z = BLAS_DDOT (&N, X, &INCX, Y, &INCY) ; \
    } \
}

#endif
