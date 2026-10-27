// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause-Open-MPI
//
// Declarations of reference BLAS routines, from the argument lists of BLAS/SRC/sgemm.f and
// dgemm.f in Reference-LAPACK (https://github.com/Reference-LAPACK/lapack), with inputs const.
// The description is quoted from dgemm.f, under the following license:
//
// Copyright (c) 1992-2025 The University of Tennessee and The University
//                         of Tennessee Research Foundation.  All rights
//                         reserved.
// Copyright (c) 2000-2025 The University of California Berkeley. All
//                         rights reserved.
// Copyright (c) 2006-2025 The University of Colorado Denver.  All rights
//                         reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer listed
//   in this license in the documentation and/or other materials
//   provided with the distribution.
//
// - Neither the name of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// The copyright holders provide no reassurances that the source code
// provided does not infringe any patent, copyright, or any other
// intellectual property rights of third parties.  The copyright holders
// disclaim any liability to any recipient for claims brought against
// recipient by any third party for infringement of that parties
// intellectual property rights.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _SCTL_BLAS_H_
#define _SCTL_BLAS_H_

#include <cstdint>

#include "sctl/common.hpp"  // for Long, SCTL_ASSERT_MSG

#ifndef SCTL_FORTRAN_SYMBOL
// Binds a declaration to the Fortran routine `name` without declaring `name`, whose declarations
// differ between libraries (return type, const, noexcept, hidden string lengths, integer width)
#define SCTL_FORTRAN_SYMBOL(name) __asm__(SCTL_FORTRAN_SYMBOL_STR_(__USER_LABEL_PREFIX__) #name)
#define SCTL_FORTRAN_SYMBOL_STR_(x) SCTL_FORTRAN_SYMBOL_STR__(x)
#define SCTL_FORTRAN_SYMBOL_STR__(x) #x
#endif

namespace sctl {
namespace detail_blas {
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
 *
 * SGEMM is the single precision version. The integer arguments point to integers of IntBytes() bytes.
 */
void sgemm(const char* TRANSA, const char* TRANSB, const void* M, const void* N, const void* K, const float* ALPHA, const float* A, const void* LDA, const float* B, const void* LDB, const float* BETA, float* C, const void* LDC) noexcept SCTL_FORTRAN_SYMBOL(sgemm_);
void dgemm(const char* TRANSA, const char* TRANSB, const void* M, const void* N, const void* K, const double* ALPHA, const double* A, const void* LDA, const double* B, const void* LDB, const double* BETA, double* C, const void* LDC) noexcept SCTL_FORTRAN_SYMBOL(dgemm_);

/**
 * ISAMAX finds the index of the first element having maximum absolute value (0 if N < 1); returns
 * the low 32 bits of it.
 */
int32_t isamax(const void* N, const float* SX, const void* INCX) noexcept SCTL_FORTRAN_SYMBOL(isamax_);

/**
 * Bytes of a BLAS and LAPACK integer: sizeof(SCTL_BLAS_INT) if it is defined, otherwise detected
 * from the BLAS on the first call: 4 (LP64) or 8 (ILP64).
 */
inline int IntBytes() noexcept {
#if defined(SCTL_BLAS_INT)
  static_assert(sizeof(SCTL_BLAS_INT) == 4 || sizeof(SCTL_BLAS_INT) == 8, "SCTL_BLAS_INT must be a 32-bit or 64-bit integer type");
  return (int)sizeof(SCTL_BLAS_INT);
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return 4;  // the detection below needs little-endian; define SCTL_BLAS_INT for ILP64
#else
  // N is 3 in its low 32 bits and negative in 64: an LP64 ISAMAX returns 2, an ILP64 one 0
  static const int bytes = [] {
    const int64_t n = (int64_t)0xffffffff00000003ULL, incx = 1;
    const float x[3] = {1, 3, 2};
    return isamax(&n, x, &incx) == 0 ? 8 : 4;
  }();
  return bytes;
#endif
}

/**
 * Debug builds (without NDEBUG): aborts unless every value fits in the integers of a 32-bit BLAS.
 */
template <class... T> inline void CheckInt32([[maybe_unused]] T... v) noexcept {
#ifndef NDEBUG
  SCTL_ASSERT_MSG(((v == (int32_t)v) && ...), "BLAS/LAPACK argument too large for the 32-bit integers of the linked BLAS; link an ILP64 BLAS");
#endif
}

/**
 * C := alpha*op(A)*op(B) + beta*C with SGEMM or DGEMM (fn), passing the integers as IntBytes() bytes.
 */
template <class Real, class Fn> inline void Gemm(Fn& fn, char TransA, char TransB, Long M, Long N, Long K, Real alpha, const Real* A, Long lda, const Real* B, Long ldb, Real beta, Real* C, Long ldc) noexcept {
  static_assert(sizeof(Long) == 8, "Long must be 64-bit");
  if (IntBytes() == 8) {  // copies in both branches: taking the parameters' addresses would spill them on every call
    const int64_t M_ = M, N_ = N, K_ = K, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    fn(&TransA, &TransB, &M_, &N_, &K_, &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_);
  } else {
    CheckInt32(M, N, K, lda, ldb, ldc);
    const int32_t M_ = (int32_t)M, N_ = (int32_t)N, K_ = (int32_t)K, lda_ = (int32_t)lda, ldb_ = (int32_t)ldb, ldc_ = (int32_t)ldc;
    fn(&TransA, &TransB, &M_, &N_, &K_, &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_);
  }
}
}  // end namespace detail_blas
}  // end namespace sctl

#endif // _SCTL_BLAS_H_
