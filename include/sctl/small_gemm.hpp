#ifndef _SCTL_SMALL_GEMM_HPP_
#define _SCTL_SMALL_GEMM_HPP_

#include "sctl/common.hpp"    // for Long, sctl
#include "sctl/iterator.hpp"  // for Iterator, ConstIterator

#if defined(SCTL_HAVE_LIBXSMM)
#include <libxsmm.h>          // for libxsmm_gemmfunction
#endif

namespace sctl {

template <class ValueType> class Matrix;

/**
 * Template argument of SmallGEMM for a size given at run time.
 */
constexpr Long DynamicSize = -1;

/**
 * Product of small dense matrices, C = A B or C += A B, where A is m x k, B is k x n and C is
 * m x n, each stored row by row: contiguously, or with the rows of each a given number of values
 * apart (the row strides lda, ldb and ldc).
 *
 * With SCTL_HAVE_LIBXSMM defined, float and double products use a kernel that LIBXSMM generates
 * for the sizes, the row strides and the mode (overwrite or accumulate); it is looked up when the
 * object is constructed, through a per-thread table after the first time, and LIBXSMM keeps it
 * until the program ends. Otherwise float and double products use a register-blocked loop over
 * Vec. Products of std::complex<float> and std::complex<double> use such a loop also with LIBXSMM,
 * and other types (long double, QuadReal, std::complex<long double>) a scalar loop.
 *
 * Constructing an object takes a few ns, so it can be done where the product is needed. Applying
 * it does not change the object, so several threads can apply the same object at once.
 *
 * @tparam ValueType Element type.
 * @tparam M, N, K Sizes known at compile time, or DynamicSize for sizes given to the constructor;
 * the loop over Vec is specialized for the fixed ones.
 * @tparam LDA, LDB, LDC Row strides known at compile time, or DynamicSize for strides given to the
 * constructor; as for the sizes.
 *
 * Each product adds 2 m n k to the profiler's FLOP counter, or 8 m n k for std::complex (real
 * operations).
 */
template <class ValueType, Long M = DynamicSize, Long N = DynamicSize, Long K = DynamicSize, Long LDA = DynamicSize, Long LDB = DynamicSize, Long LDC = DynamicSize> class SmallGEMM {
  static_assert((LDA == DynamicSize || K == DynamicSize || LDA >= K) && (LDB == DynamicSize || N == DynamicSize || LDB >= N) && (LDC == DynamicSize || N == DynamicSize || LDC >= N), "a row stride shorter than its row");

 public:
  /**
   * @param accumulate If true, C += A B; otherwise C = A B, and C is not read.
   * @param m, n, k Sizes; each must equal its template argument unless that is DynamicSize.
   * The row strides are those of the template arguments, or of contiguous storage where these are
   * DynamicSize.
   */
  explicit SmallGEMM(bool accumulate = false, Long m = M, Long n = N, Long k = K);

  /**
   * @param accumulate, m, n, k As above.
   * @param lda, ldb, ldc Row strides: row i of A starts at A + i lda, and so on; at least k, n and n
   * (contiguous storage); each must equal its template argument unless that is DynamicSize.
   */
  SmallGEMM(bool accumulate, Long m, Long n, Long k, Long lda, Long ldb, Long ldc);

  /**
   * C = A B, or C += A B, with the row strides of the object.
   */
  void operator()(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B) const;

  /**
   * C = A B, or C += A B, for the leading m x k, k x n and m x n blocks of A, B and C; their numbers
   * of columns must equal the row strides of the object, and their numbers of rows be at least m, k
   * and m.
   */
  void operator()(Matrix<ValueType>& C, const Matrix<ValueType>& A, const Matrix<ValueType>& B) const;

 private:
  Long m_, n_, k_;
  Long lda_, ldb_, ldc_;
  bool accumulate_;
#if defined(SCTL_HAVE_LIBXSMM)
  libxsmm_gemmfunction kernel_; // null when the loop over Vec is used
#endif
};

}  // namespace sctl

#endif  // _SCTL_SMALL_GEMM_HPP_
