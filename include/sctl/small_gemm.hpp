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
 * m x n, each stored contiguously row by row.
 *
 * With SCTL_HAVE_LIBXSMM defined, float and double products use a kernel that LIBXSMM generates
 * for the sizes and the mode (overwrite or accumulate); it is looked up when the object is
 * constructed, through a per-thread table after the first time, and LIBXSMM keeps it until the
 * program ends. Otherwise float and double products use a register-blocked loop over Vec, and
 * other types (long double, QuadReal, complex) a loop over the entries of C.
 *
 * Constructing an object takes a few ns, so it can be done where the product is needed. Applying
 * it does not change the object, so several threads can apply the same object at once.
 *
 * @tparam ValueType Element type.
 * @tparam M, N, K Sizes known at compile time, or DynamicSize for sizes given to the constructor;
 * the loop over Vec is specialized for the fixed ones.
 */
template <class ValueType, Long M = DynamicSize, Long N = DynamicSize, Long K = DynamicSize> class SmallGEMM {
 public:
  /**
   * @param accumulate If true, C += A B; otherwise C = A B, and C is not read.
   * @param m, n, k Sizes; each must equal its template argument unless that is DynamicSize.
   */
  explicit SmallGEMM(bool accumulate = false, Long m = M, Long n = N, Long k = K);

  /**
   * C = A B, or C += A B, with each matrix stored contiguously row by row.
   */
  void operator()(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B) const;

  /**
   * C = A B, or C += A B; the sizes of the matrices must match those of the object.
   */
  void operator()(Matrix<ValueType>& C, const Matrix<ValueType>& A, const Matrix<ValueType>& B) const;

 private:
  Long m_, n_, k_;
  bool accumulate_;
#if defined(SCTL_HAVE_LIBXSMM)
  libxsmm_gemmfunction kernel_; // null when the loop over Vec is used
#endif
};

}  // namespace sctl

#endif  // _SCTL_SMALL_GEMM_HPP_
