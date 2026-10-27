// Copyright (C) 2017-2026 The Simons Foundation, Inc. and the SCTL authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause-Open-MPI
//
// Declarations of reference LAPACK routines, from the argument lists of SRC/sgesvd.f and
// dgesvd.f in Reference-LAPACK (https://github.com/Reference-LAPACK/lapack), with inputs const.
// The description is quoted from dgesvd.f, under the following license:
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

#ifndef _SCTL_LAPACK_H_
#define _SCTL_LAPACK_H_

#include "sctl/blas.h"  // SCTL_FORTRAN_SYMBOL, detail_blas::IntBytes()

namespace sctl {
namespace detail_lapack {
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
 *
 * Note that the routine returns V**T, not V.
 *
 * SGESVD is the single precision version. The integer arguments point to integers of
 * detail_blas::IntBytes() bytes.
 */
void sgesvd(const char* JOBU, const char* JOBVT, const void* M, const void* N, float* A, const void* LDA, float* S, float* U, const void* LDU, float* VT, const void* LDVT, float* WORK, const void* LWORK, void* INFO) noexcept SCTL_FORTRAN_SYMBOL(sgesvd_);
void dgesvd(const char* JOBU, const char* JOBVT, const void* M, const void* N, double* A, const void* LDA, double* S, double* U, const void* LDU, double* VT, const void* LDVT, double* WORK, const void* LWORK, void* INFO) noexcept SCTL_FORTRAN_SYMBOL(dgesvd_);

/**
 * SVD with SGESVD or DGESVD (fn), passing the integers as detail_blas::IntBytes() bytes.
 */
template <class Real, class Fn> inline void Gesvd(Fn& fn, const char* JOBU, const char* JOBVT, Long M, Long N, Real* A, Long LDA, Real* S, Real* U, Long LDU, Real* VT, Long LDVT, Real* WORK, Long LWORK, Long* INFO) noexcept {
  if (detail_blas::IntBytes() == 8) {  // copies in both branches, as in detail_blas::Gemm
    const int64_t M_ = M, N_ = N, LDA_ = LDA, LDU_ = LDU, LDVT_ = LDVT, LWORK_ = LWORK;
    int64_t INFO_ = 0;
    fn(JOBU, JOBVT, &M_, &N_, A, &LDA_, S, U, &LDU_, VT, &LDVT_, WORK, &LWORK_, &INFO_);
    *INFO = INFO_;
  } else {
    detail_blas::CheckInt32(M, N, LDA, LDU, LDVT, LWORK);
    const int32_t M_ = (int32_t)M, N_ = (int32_t)N, LDA_ = (int32_t)LDA, LDU_ = (int32_t)LDU, LDVT_ = (int32_t)LDVT, LWORK_ = (int32_t)LWORK;
    int32_t INFO_ = 0;
    fn(JOBU, JOBVT, &M_, &N_, A, &LDA_, S, U, &LDU_, VT, &LDVT_, WORK, &LWORK_, &INFO_);
    *INFO = INFO_;
  }
}
}  // end namespace detail_lapack
}  // end namespace sctl

#endif // _SCTL_LAPACK_H_
