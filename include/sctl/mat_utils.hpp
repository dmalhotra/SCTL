// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _SCTL_MAT_UTILS_HPP_
#define _SCTL_MAT_UTILS_HPP_

#include "sctl/common.hpp"  // for sctl

namespace sctl {
namespace mat {

template <class ValueType> void gemm(char TransA, char TransB, Long M, Long N, Long K, ValueType alpha, ConstIterator<ValueType> A, Long lda, ConstIterator<ValueType> B, Long ldb, ValueType beta, Iterator<ValueType> C, Long ldc);

template <class ValueType> void svd(char *JOBU, char *JOBVT, Long *M, Long *N, Iterator<ValueType> A, Long *LDA, Iterator<ValueType> S, Iterator<ValueType> U, Long *LDU, Iterator<ValueType> VT, Long *LDVT, Iterator<ValueType> WORK, Long *LWORK, Long *INFO);

/**
 * Computes the pseudo inverse of matrix M(n1xn2) (in row major form)
 * and returns the output M_(n2xn1).
 */
template <class ValueType> void pinv(Iterator<ValueType> M, Long n1, Long n2, ValueType eps, Iterator<ValueType> M_);

}  // end namespace mat
}  // end namespace sctl

#endif // _SCTL_MAT_UTILS_HPP_
