#ifndef _SCTL_MAT_UTILS_HPP_
#define _SCTL_MAT_UTILS_HPP_

#include "sctl/common.hpp"  // for sctl

namespace sctl {
namespace mat {

template <class ValueType> void gemm(char TransA, char TransB, int M, int N, int K, ValueType alpha, Iterator<ValueType> A, int lda, Iterator<ValueType> B, int ldb, ValueType beta, Iterator<ValueType> C, int ldc);

template <class ValueType> void svd(char *JOBU, char *JOBVT, int *M, int *N, Iterator<ValueType> A, int *LDA, Iterator<ValueType> S, Iterator<ValueType> U, int *LDU, Iterator<ValueType> VT, int *LDVT, Iterator<ValueType> WORK, int *LWORK, int *INFO);

/**
 * Computes the pseudo inverse of matrix M(n1xn2) (in row major form)
 * and returns the output M_(n2xn1).
 */
template <class ValueType> void pinv(Iterator<ValueType> M, int n1, int n2, ValueType eps, Iterator<ValueType> M_);

}  // end namespace mat
}  // end namespace sctl

#endif // _SCTL_MAT_UTILS_HPP_
