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

#ifndef SCTL_FORTRAN_SYMBOL
// Binds a declaration to the Fortran routine `name` without declaring `name`, whose declarations
// differ between libraries (return type, const, noexcept, hidden string lengths)
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
 * SGEMM is the single precision version.
 */
void sgemm(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const float* ALPHA, const float* A, const int* LDA, const float* B, const int* LDB, const float* BETA, float* C, const int* LDC) noexcept SCTL_FORTRAN_SYMBOL(sgemm_);
void dgemm(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K, const double* ALPHA, const double* A, const int* LDA, const double* B, const int* LDB, const double* BETA, double* C, const int* LDC) noexcept SCTL_FORTRAN_SYMBOL(dgemm_);
}  // end namespace detail_blas
}  // end namespace sctl

#endif // _SCTL_BLAS_H_
