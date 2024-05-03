#ifndef _SCTL_BLAS_H_
#define _SCTL_BLAS_H_

extern "C" {
/**
 * DGEMM  performs one of the matrix-matrix operations
 *
 *    C := alpha*op( A )*op( B ) + beta*C,
 *
 * where  op( X ) is one of
 *
 *    op( X ) = X   or   op( X ) = X**T,
 *
 * alpha and beta are scalars, and A, B and C are matrices, with op( A )
 * an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 */
void sgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const float* ALPHA, const float* A, const int* LDA, const float* B, const int* LDB, const float* BETA, float* C, const int* LDC) noexcept;
void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const double* ALPHA, const double* A, const int* LDA, const double* B, const int* LDB, const double* BETA, double* C, const int* LDC) noexcept;
}

#endif
