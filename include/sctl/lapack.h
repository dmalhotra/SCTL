#ifndef _SCTL_LAPACK_H_
#define _SCTL_LAPACK_H_

extern "C" {
/**
 * DGESVD computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors. The SVD is written
 *
 *      A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 * V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order.  The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 */
void sgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N, float* A, const int* LDA, float* S, float* U, const int* LDU, float* VT, const int* LDVT, float* WORK, const int* LWORK, int* INFO) noexcept;
void dgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N, double* A, const int* LDA, double* S, double* U, const int* LDU, double* VT, const int* LDVT, double* WORK, const int* LWORK, int* INFO) noexcept;
}

#endif // _SCTL_LAPACK_H_
