/* ========================================================================== */
/* === cholmod_spinv ======================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Copyright (C) 2005,2006 Timothy A. Davis
 * Copyright (C) 2008,2009,2010 Jarno Vanhatalo
 * Copyright (C) 2012 Jaakko Luttinen
 *
 * cholmod_spinv.c is licensed under Version 3.0 of the GNU General 
 * Public License. See LICENSE for a text of the license.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * This file is part of CHOLMOD Extra Module.
 *
 * CHOLDMOD Extra Module is free software: you can redistribute it
 * and/or modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of
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
 *
 * Given an LL' or LDL' factorization of A, compute the sparse inverse
 * of A, that is, a matrix with the same sparsity as A but elements
 * from inv(A).  Note that, in general, inv(A) is dense but this
 * computes only some elements of it.  At the moment, only simplicial
 * LDL factorization and real xtypes are supported.
 *
 * References:
 * -------------------------------------------------------------------------- */

#include "cholmod_extra.h"
#include "cholmod_extra_internal.h"

//#include <suitesparse/cholmod_internal.h>
#include <suitesparse/cholmod_cholesky.h>

#define PERM(j) (Lperm != NULL ? Lperm[j] : j)

void CHOLMOD(spinv_block)
(
    double *L, 
    double *Z, 
    double *V, 
    Int m, 
    Int n,
    cholmod_common *Common
)
{
    double zero[2]      = {0.0, 0.0} ;
    double one[2]       = {1.0, 0.0} ;
    double minus_one[2] = {-1.0, 0.0} ;
    //double minus_half[2] = {-0.5, 0.0} ;
    double *L1, *L2 ;
    double *Z1, *Z2 ;
    Int i, j ;

    Int m1 = n ;      // rows of Z1/L1
    Int m2 = m - m1 ; // rows of Z2/L2
    Int ld = m ;      // leading dimension of Z1/Z2/L1/L2

    Z1 = Z ;      // pointer to Z1
    Z2 = Z + m1 ; // pointer to Z2
    L1 = L ;      // pointer to L1
    L2 = L + m1 ; // pointer to L2

    /*
     * Initialize Z1 to identity matrix
     */
    for (i = 0; i < m1; i++)
    {
        for (j = 0; j < m1; j++)
            Z1[i+j*ld] = ((i == j) ? 1.0 : 0.0) ;
    }

    if (m2 > 0) 
    {
        
        // Z2 = - V * L2
        // DSYMM(SIDE,UPLO,M,N,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
        BLAS_dsymm("L", // left multiply
                   "L", // in lower triangular form
                   m2, n, 
                   minus_one,
                   V, m2, 
                   L2, ld, 
                   zero, Z2, ld) ;

        // Z1 = -Z2'*L2 + Z1 = -L2'*V*L2 + I
        // DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
        BLAS_dgemm("T", // transpose Z2
                   "N", // no transpose for L2
                   m1, m1, m2,
                   minus_one,
                   Z2, ld,
                   L2, ld,
                   one, Z1, ld) ;

    }

    // Z1 = L1' \ Z1
    // DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
    BLAS_dtrsm("L", // divide from left
               "L", // lower triangular
               "T", // transpose
               "N", // not unit diagonal
               m1, m1,
               one,
               L1, ld,
               Z1, ld) ;

    // Z = Z / L1
    // DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
    BLAS_dtrsm("R", // divide from right 
               "L", // lower triangular
               "N", // no transpose
               "N", // not unit diagonal
               m, n,
               one,
               L1, ld,
               Z, ld) ;

}




/* ========================================================================== */
/* === cholmod_spinv_super ================================================== */
/* ========================================================================== */

cholmod_sparse *CHOLMOD(spinv_super)   /* returns the sparse inverse of X */
(
    /* ---- input ---- */
    cholmod_factor *L,	/* (supernodal) factorization to use */
    /* --------------- */
    cholmod_common *Common
    )
{

    Int s, i, j ;
    Int *Super, *Ls, *Lpi, *Lpx ;
    Int psi0, psi1, j0, j1 ;
//    Int ms, ns, ld, m1, m2, scol ;
    Int ms, ns, m1, m2, scol ;
    int xtype;
    cholmod_sparse *X ;
    double *Lx, *Xx, *Xz, *Z, *V ; //, *Z1, *Z2, *L1, *L2 ;
    Int  *Xp, *Xi;
    Int *perm, *Lperm, *ncol;
    Int n, il, jl, kl, ix, jx, kx, ip, jp;
    size_t nz, nsuper, maxsize;

    /* 
     * Compute the sparse inverse. 
     */

    // Dimensionality of the matrix
    n = L->n ;

    // Number of non-zero elements
    //nz = (L->xsize - n) / 2 + n;

    // Result has the same xtype as the factor
    xtype = L->xtype ;

    // Allocate the result X
    X = CHOLMOD(spzeros) (n, n, 0, xtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        CHOLMOD(free_sparse) (&X, Common) ;
        return (NULL) ;
    }

    // Shorthand notation
    Xp = X->p ;
    Xi = X->i ;
    Xx = X->x ;
    Xz = X->z ;
    Super = L->super ;
    Lpi = L->pi ;
    Lpx = L->px ;
    Ls = L->s ;
    Lperm = L->Perm ;
    Lx = L->x ;
    nsuper = L->nsuper ;


    /*
     * Compute the mapping to the permuted result:
     * X->x[perm[i]] ~ L->x[i]
     * Both X and L are lower triangular
     */

    /* Count non-zeros on columns */
    nz = 0;
    for (s = 0; s < nsuper; s++)
    {
        j0 = Super[s] ;   // first column of the supernode
        j1 = Super[s+1] ; // last column (+1)
        ns = j1 - j0;     // number of columns

        psi0 = Lpi[s] ;    // "pointer" to first row index
        psi1 = Lpi[s+1] ;  // "pointer" to last row index (+1)
        ms = psi1 - psi0;  // number of rows

        for (j = 0; j < ns; j++)
        {
            jp = PERM(j0+j) ; // permuted column index
            for (i = j; i < ms; i++)
            {
                ip = PERM(Ls[psi0+i]) ; // permuted row index
                // Increase the number of elements in the column
                jx = MIN(ip,jp) ;
                Xp[jx+1]++ ;
                // Increase the number of non-zero elements
                nz++ ;
            }
        }
        
    }
    Xi = realloc(Xi, nz*sizeof(Int)) ;
    Xx = realloc(Xx, nz*sizeof(double)) ;
    X->i = Xi ;
    X->x = Xx ;
    X->nzmax = nz ;
    
    /* Compute column pointers by computing cumulative sum */
    for (jx = 1; jx <= n; jx++)
        Xp[jx] += Xp[jx-1] ;

    /* 
     * Add row indices and compute permutation mapping 
     */
    ncol = (Int*)calloc(n, sizeof(Int)) ;
    perm = malloc(L->xsize*sizeof(Int)) ; // permutation mapping
    for (s = 0; s < nsuper; s++)
    {
        j0 = Super[s] ;   // first column of the supernode
        j1 = Super[s+1] ; // last column (+1)
        ns = j1 - j0;     // number of columns

        psi0 = Lpi[s] ;    // "pointer" to first row index
        psi1 = Lpi[s+1] ;  // "pointer" to last row index (+1)
        ms = psi1 - psi0;  // number of rows

        for (j = 0; j < ns; j++)
        {
            jp = PERM(j0+j) ; // permuted column index
            for (i = j; i < ms; i++)
            {
                ip = PERM(Ls[psi0+i]) ; // permuted row index
                // Increase the number of elements in the column
                jx = MIN(ip,jp) ;      // column of X
                ix = MAX(ip,jp) ;      // row of X
                kx = Xp[jx]+ncol[jx] ; // index of X
                // Increase the number of elements on the column
                ncol[jx]++ ; 
                // Add row index
                Xi[kx] = ix ;
                // Mapping X[perm[k]] ~ L[k]
                kl = Lpx[s] + i + j*ms ; // index of L
                //kl = Lpx[s] + j + i*ms ; // index of L
                perm[kl] = kx ;
            }
        }
        
    }
    free(ncol) ;
    X->sorted = FALSE ;

    /* 
     * Allocate workspace using the size of the largest supernode 
     */
    maxsize = 0 ;
    for (s = 0; s < L->nsuper; s++)
    {
        if (Lpx[s+1] - Lpx[s] > maxsize)
            maxsize = Lpx[s+1] - Lpx[s] ;
    }
    V = malloc(L->maxesize*L->maxesize*sizeof(double)) ;
    Z = malloc(maxsize*sizeof(double)) ;

    /*
     * Compute the sparse inverse
     */
    for (s = nsuper - 1; s >= 0; s--)
    {

        /*
         * Define some helpful variables for the active supernode
         */

        j0 = Super[s] ;   // first column of the supernode
        j1 = Super[s+1] ; // last column (+1)
        ns = j1 - j0 ;     // number of columns

        psi0 = Lpi[s] ;    // "pointer" to first row index
        psi1 = Lpi[s+1] ;  // "pointer" to last row index (+1)
        ms = psi1 - psi0 ; // number of rows

        // Z = [Z1; Z2] where Z1 is ns x ns and Z2 is (ms-ns) x ns
        // L = [L1; L2] where L1 is ns x ns and L2 is (ms-ns) x ns
        m1 = ns ;               // rows of Z1/L1
        m2 = ms - ns ;          // rows of Z2/L2

        /*
         * Collect V (symmetric in lower triangular form)
         */
        scol = s + 1 ; 
        for (j = 0; j < m2; j++)
        {
            // Row index of the j:th non-zero
            // element in L2
            // = relevant column index of X
            jx = Ls[psi0+m1+j] ;
            
            // Find supernode containing the column jx
            while (Super[scol+1]-1 < jx)
                scol++ ;
            jl = jx - Super[scol] ; // column of the supernode

            // Set lower triangular elements of column
            // jz (no need to set upper triangular
            // elements because of the symmetry).
            il = 0 ;
            for (i = j; i < m2; i++)
            {
                ix = Ls[psi0+m1+i] ;
                // Find L[ix,jx]
                while (Ls[Lpi[scol]+il] < ix)
                    il++ ;
                
                // To summarize the finding:
                // L[ix,jx] is the element (il,jl) in supernode scol
                kl = Lpx[scol] + il + jl*(Lpi[scol+1]-Lpi[scol]) ;

                // Use the permutation mapping to get the
                // index of the corresponding element in X
                kx = perm[kl] ;
                                
                // Set V[i,j] = X[ix,jx]
                V[i+j*m2] = Xx[kx] ;
            }
                            
        }

        /*
         * Compute the inverse of the supernode block
         */
        CHOLMOD(spinv_block) (Lx + Lpx[s], Z, V, ms, ns, Common) ;
        
        /*
         * Store the result Z = [Z1; Z2] in X
         */
        for (j = 0; j < ns; j++)
        {
            for (i = j; i < m1; i++)
            {
                // Index of the corresponding element L[kl] ~ Z[i,j]
                kl = Lpx[s] + i + j*ms ; 
                // Mapping X[perm[kl]] ~ L[kl]
                kx = perm[kl] ;
                // Set the value (try to stabilize by utilizing
                // symmetry)
                Xx[kx] = 0.5*(Z[i+j*ms]+Z[j+i*ms]) ;
            }
            for (i = m1; i < ms; i++)
            {
                // Index of the corresponding element L[kl] ~ Z[i,j]
                kl = Lpx[s] + i + j*ms ; 
                // Mapping X[perm[kl]] ~ L[kl]
                kx = perm[kl] ;
                // Set the value
                Xx[kx] = Z[i+j*ms] ;
            }
        }

    }
            
    // Free workspace
    free(Z) ;
    free(V) ;
    free(perm) ;

    // The result is symmetric but only the lower triangular part was
    // computed.
    X->stype = -1 ;

    // Sort columns (is it necessary?)
    CHOLMOD(sort) (X, Common) ;
    
    if (Common->status == CHOLMOD_OK)
    {
        return (X) ;
    }
    else
    {
        CHOLMOD(free_sparse) (&X, Common) ;
        return (NULL) ;
    }

}
/* ========================================================================== */
/* === cholmod_spinv_simplicial ============================================= */
/* ========================================================================== */

cholmod_sparse *CHOLMOD(spinv_simplicial)  /* returns the sparse solution X */
(
    /* ---- input ---- */
    cholmod_factor *L,	/* (simplicial) factorization to use */
    /* --------------- */
    cholmod_common *Common
    )
{

    int xtype ;
    cholmod_sparse *X ;
    double *Lx, *Lz, *Xx, *Xz, *V, *z, *Lxj ;
    double djj ;
    Int *Li, *Lp, *Xp, *Xi ;
    Int *perm, *Lperm, *ncol ;
    Int n, kmin, kmax, nj, iz, jz, il, jl, kl, ix, jx, kx, ip, jp ;
    size_t nz, maxsize ;
//*
    double minus_one[2], one[2], zero[2] ;
    minus_one[0] = -1.0 ;
    minus_one[1] = 0.0 ;
    one[0] = 1.0 ;
    one[1] = 0.0 ;
    zero[0] = 0.0 ;
    zero[1] = 0.0 ;
//*/


    // Dimensionality of the matrix
    n = L->n ;

    // Number of non-zero elements
    nz = L->nzmax ;

    // Result has the same xtype as the factor
    xtype = L->xtype ;

    // Allocate the result X
    X = CHOLMOD(spzeros) (n, n, nz, xtype, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
        CHOLMOD(free_sparse) (&X, Common) ;
        return (NULL) ;
    }

    // Shorthand notation
    Xp = X->p ;
    Xi = X->i ;
    Xx = X->x ;
    Xz = X->z ;
    Lp = L->p ;
    Li = L->i ;
    Lx = L->x ;
    Lz = L->z ;
    Lperm = L->Perm ;

    V = NULL ;
    z = NULL ;
    Lxj = NULL ;

    /*
     * Compute the mapping to the permuted result:
     * X->x[perm[i]] ~ L->x[i]
     * Both X and L are lower triangular
     */

    /* Count non-zeros on columns */
    for (jl = 0; jl < n; jl++)
    {
        jp = PERM(jl) ; // permuted column
        for (kl = Lp[jl]; kl < Lp[jl+1]; kl++)
        {
            il = Li[kl] ;    // row of L
            ip = PERM(il) ; // permuted row
            // Increase the number of elements in the column
            jx = MIN(ip,jp) ;
            Xp[jx+1]++ ;
        }
    }
    
    /* Compute column pointers by computing cumulative sum */
    maxsize = 0 ;
    for (jx = 1; jx <= n; jx++)
    {
        if (Xp[jx] > maxsize)
            maxsize = Xp[jx] - 1 ; // number of non-zeros (without diagonal)
        Xp[jx] += Xp[jx-1] ;
    }

    /* Add row indices */
    ncol = (Int*)calloc(n, sizeof(Int)) ;
    perm = malloc(nz*sizeof(Int)) ; // permutation mapping

    for (jl = 0; jl < n; jl++)
    {
        jp = PERM(jl) ; // permuted column
        for (kl = Lp[jl]; kl < Lp[jl+1]; kl++)
        {
            il = Li[kl] ;          // row of L
            ip = PERM(il) ;       // permuted row
            jx = MIN(ip,jp) ;      // column of X
            ix = MAX(ip,jp) ;      // row of X
            kx = Xp[jx]+ncol[jx] ; // index of X
            // Increase elements on the column
            ncol[jx]++ ; 
            // Add row index
            Xi[kx] = ix ;
            // Mapping X[perm[i]] ~ L[i]
            perm[kl] = kx ;
        }
    }
    free(ncol) ;
    X->sorted = FALSE ;

    // Allocate memory for a temporary matrix and vector
    z = malloc((maxsize+1)*sizeof(double)) ;
    V = malloc((maxsize*maxsize)*sizeof(double)) ;
                            
    if (L->is_ll)
    {

        switch (xtype)
        {
        case CHOLMOD_REAL:
            ERROR (CHOLMOD_INVALID,"Real xtype for L*L' not implemented.") ;
            break ;

        case CHOLMOD_COMPLEX:
            ERROR (CHOLMOD_INVALID,"Complex xtype for L*L' not implemented.") ;
            break ;

        case CHOLMOD_ZOMPLEX:
            ERROR (CHOLMOD_INVALID,"Zomplex xtype for L*L' not implemented.") ;
            break ;

        }
    }
    else
    {

        switch (xtype)
        {
        case CHOLMOD_REAL:

            for (jl = n-1; jl >= 0; jl--)
            {
                // Indices of non-zero elements in j-th column 
                kmin = Lp[jl];         // first index
                kmax = Lp[jl+1] - 1;   // last index
                nj = kmax - kmin; // number of non-zero elements (without diagonal)

                // Diagonal entry of D: D[j,j]
                djj = Lx[kmin] ;
                if (kmax > kmin)
                {
                    // j-th column vector of L (without the
                    // diagonal element and zeros)
                    Lxj = Lx + (kmin+1) ;

                    // Form Z
                    for (jz = 0; jz < nj; jz++)
                    {
                        // Row index of the (jz+1):th non-zero
                        // element on column j
                        // = relevant column index of X
                        jx = Li[kmin+1+jz] ;
                        // Index of the diagonal element on column
                        // jx
                        kx = Lp[jx] ;

                        // Set lower triangular elements of column
                        // jz (no need to set upper triangular
                        // elements because of the symmetry).
                        for (iz = jz; iz < nj; iz++)
                        {
                            ix = Li[kmin+1+iz] ;
                            // Find X[row,jx]
                            while (Li[kx] < ix)
                                kx++ ;
                                
                            // Set Z[iz,jz] = X[ix,jx]
                            V[iz+jz*nj] = Xx[perm[kx]] ;
                        }
                            
                    }

                    // DSYMV(UPLO,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
                    BLAS_dsymv("L", nj, one, V, nj, Lxj, 1, zero, z, 1) ;

                    // Copy the result to the lower part of X
                    for (iz = 0; iz < nj; iz++)
                    {
                        kx = kmin + 1 + iz ;
                        Xx[perm[kx]] = -z[iz] ;
                    }

                    // Compute the diagonal element X[j,j]
                    // DDOT(N,DX,INCX,DY,INCY)
                    BLAS_ddot(nj, z, 1, Lxj, 1, Xx[perm[kmin]]) ;
                    Xx[perm[kmin]] += 1.0/djj ;

                }
                else
                {
                    // Compute the diagonal element X[j,j]
                    Xx[perm[kmin]] = 1.0/djj ;
                }
            }
            break ;

        case CHOLMOD_COMPLEX:
            ERROR (CHOLMOD_INVALID,"Complex xtype for L*D*L' not implemented.") ;
            break ;

        case CHOLMOD_ZOMPLEX:
            ERROR (CHOLMOD_INVALID,"Zomplex xtype for L*D*L' not implemented.") ;
            break ;
        }

    }

    free(V) ;
    free(z) ;
    free(perm) ;

    // The result is symmetric but only the lower triangular part was
    // computed.
    X->stype = -1 ;

    // Sort columns (is it necessary?)
    CHOLMOD(sort) (X, Common) ;
    
    if (Common->status == CHOLMOD_OK)
    {
        return (X) ;
    }
    else
    {
        CHOLMOD(free_sparse) (&X, Common) ;
        return (NULL) ;
    }

}


/* ========================================================================== */
/* === cholmod_spinv ======================================================== */
/* ========================================================================== */

cholmod_sparse *CHOLMOD(spinv)    /* returns the sparse solution X */
(
    /* ---- input ---- */
    cholmod_factor *L,	/* factorization to use */
    /* --------------- */
    cholmod_common *Common
    )
{

    ASSERT (L->xtype != CHOLMOD_PATTERN) ;  /* L is not symbolic */

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (L, NULL) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, NULL) ;
    Common->status = CHOLMOD_OK ;

    /* 
     * Compute the sparse inverse. 
     */
    if (L->is_super)
    {
        return CHOLMOD(spinv_super) (L, Common) ;
    }
    else
    {
        return CHOLMOD(spinv_simplicial) (L, Common) ;
    }
}



