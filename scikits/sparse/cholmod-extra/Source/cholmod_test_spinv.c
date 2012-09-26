
#include "cholmod_extra.h"
#include <suitesparse/cholmod.h>
#include <suitesparse/cholmod_internal.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int uniform_rand(int l, int u)
{
    return (rand()%(u-l)) + l ;
}

double compute_error(cholmod_dense *A, cholmod_dense *B, cholmod_dense *M)
{
    double error, e ;
    int nrow, ncol, i, j ;
    double *Ax, *Bx, *Mx ;
    nrow = A->nrow ;
    ncol = A->ncol ;
    Ax = A->x ;
    Bx = B->x ;
    Mx = M->x ;
    error = 0 ;
    for (i = 0; i < nrow; i++)
    {
        for (j=0; j < ncol; j++)
        {
            if (Mx[i+j*nrow] != 0)
            {
                e = Ax[i+j*nrow] - Bx[i+j*nrow] ;
                error += e*e ;
            }
        }
    }
    return sqrt(error) ;
}

int main(void)
{
    int N = 1000 ;
    int i, j, n ;
    //Int *I, *J ;
    //double *X ;
    int nz = 0;
    double *Ax ;
    double x, error ;
    cholmod_dense *A, *invK, *spinvK, *I ;
    cholmod_sparse *K, *V ;
    cholmod_factor *L ;
    cholmod_common Common ;
    clock_t start, end;
     double cpu_time_used;
     
    // Start using CHOLMOD
    cholmod_start(&Common) ;

    /* SPARSE COVARIANCE MATRIX CONSTRUCTION */

    // Generate random symmetric positive (semi)definite matrix
    A = cholmod_zeros(N, N, CHOLMOD_REAL, &Common) ;
    Ax = A->x ;
    nz = 10*N ;
    for (n = 0 ; n < nz; n++)
    {
        i = uniform_rand(0,N) ;
        j = uniform_rand(0,N) ;
        x = uniform_rand(0,4) ; // just some random number
        Ax[i+j*N] += x ;
        Ax[j+i*N] += x ;
        Ax[i+i*N] += abs(x) ;
        Ax[j+j*N] += abs(x) ;
    }
    // Make positive-definite by adding something positive to the
    // diagonal
    for (n = 0; n < N; n++)
    {
        Ax[n+n*N] += 1 ;
    }

    // Make the matrix sparse
    K = cholmod_dense_to_sparse(A, TRUE, &Common) ;

    // Identity matrix
    I = cholmod_eye(N,N,CHOLMOD_REAL,&Common) ;

    /* SIMPLICIAL */

    // Factorize
    Common.supernodal = CHOLMOD_SIMPLICIAL ;
    L = cholmod_analyze(K, &Common) ;
    cholmod_factorize(K, L, &Common) ;

    // Compute the sparse inverse and the full inverse
    start = clock();
    V = cholmod_spinv(L, &Common) ;
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    invK = cholmod_solve(CHOLMOD_A, L, I, &Common) ;
    spinvK = cholmod_sparse_to_dense(V, &Common) ;

    // Compute error
    error = compute_error(invK, spinvK, A) ;
    printf("Error for simplicial: %g (CPU-time: %g)\n", error, cpu_time_used) ;

    /* SUPERNODAL */

    // Factorize
    Common.supernodal = CHOLMOD_SUPERNODAL ;
    L = cholmod_analyze(K, &Common) ;
    cholmod_factorize(K, L, &Common) ;

    // Compute the sparse inverse and the full inverse
    start = clock();
    V = cholmod_spinv(L, &Common) ;
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    invK = cholmod_solve(CHOLMOD_A, L, I, &Common) ;
    spinvK = cholmod_sparse_to_dense(V, &Common) ;

    // Compute error
    error = compute_error(invK, spinvK, A) ;
    printf("Error for supernodal: %g (CPU-time: %g)\n", error, cpu_time_used) ;

    /* CLEANUP */

    // Free memory
    cholmod_free_dense(&A, &Common) ;
    cholmod_free_dense(&invK, &Common) ;
    cholmod_free_dense(&spinvK, &Common) ;
    cholmod_free_dense(&I, &Common) ;
    //cholmod_free_triplet(&A, &Common) ;
    cholmod_free_sparse(&K, &Common) ;
    cholmod_free_sparse(&V, &Common) ;
    cholmod_finish(&Common) ;

    return 0 ;
}
