#ifndef _SCTL_SMALL_GEMM_TXX_
#define _SCTL_SMALL_GEMM_TXX_

#include <type_traits>          // for integral_constant, is_same

#include "sctl/common.hpp"      // for Long, Integer, SCTL_ASSERT, sctl
#include "sctl/small_gemm.hpp"  // for SmallGEMM, DynamicSize
#include "sctl/iterator.hpp"    // for Iterator, ConstIterator
#include "sctl/iterator.txx"    // for Iterator::operator[]
#include "sctl/matrix.hpp"      // for Matrix
#include "sctl/vec.hpp"         // for Vec, FMA
#include "sctl/vec.txx"         // for Vec::Load, Vec::Store

namespace sctl {

namespace detail_small_gemm {

  /**
   * How VecProduct updates C: C = A B; C += A B; or C = alpha A B + beta C, where C is not read if
   * beta is 0.
   */
  enum class Update { Overwrite, Accumulate, AlphaBeta };

  /**
   * C (m x n) = A (m x k) B (k x n), updated as U says; all row-major, with row strides lda, ldb
   * and ldc, or k, n and n if Contiguous. A template size other than DynamicSize replaces the
   * argument, so the loops are specialized for it. Rows of C in blocks of 4, columns in tiles of 2
   * vectors and then 1, the leftover columns in narrower vectors and then one at a time. Not
   * inlined: unrolled for fixed sizes inside a caller's loop, the code was up to 1.9x slower.
   */
  template <class ValueType, Long M, Long N, Long K, Update U, bool Contiguous> [[gnu::noinline]] void VecProduct(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B, const Long m_, const Long n_, const Long k_, const Long lda_, const Long ldb_, const Long ldc_, const ValueType alpha, const ValueType beta) {
    const Long m = (M != DynamicSize ? M : m_);
    const Long n = (N != DynamicSize ? N : n_);
    const Long k = (K != DynamicSize ? K : k_);
    const Long lda = (Contiguous ? k : lda_);
    const Long ldb = (Contiguous ? n : ldb_);
    const Long ldc = (Contiguous ? n : ldc_);
    constexpr Integer VL = Vec<ValueType>::Size();
    using I1 = std::integral_constant<Integer, 1>;
    using I2 = std::integral_constant<Integer, 2>;
    using I4 = std::integral_constant<Integer, 4>;
    using IV = std::integral_constant<Integer, VL>;

    // Rows i0.. (MR of them) and columns j0.. (NV vectors of W) of C
    const auto tile = [A, B, C, k, lda, ldb, ldc, alpha, beta](const Long i0, const Long j0, auto mr, auto nv, auto w) {
      constexpr Integer MR = decltype(mr)::value;
      constexpr Integer NV = decltype(nv)::value;
      constexpr Integer W = decltype(w)::value;
      using V = Vec<ValueType, W>;
      V acc[MR][NV];
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) acc[r][v] = V((ValueType)0);
      }
      for (Long l = 0; l < k; l++) {
        V b[NV];
        for (Integer v = 0; v < NV; v++) b[v] = V::Load(&B[l * ldb + j0 + v * W]);
        for (Integer r = 0; r < MR; r++) {
          const V a(A[(i0 + r) * lda + l]);
          for (Integer v = 0; v < NV; v++) acc[r][v] = FMA(a, b[v], acc[r][v]);
        }
      }
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) {
          ValueType* Cv = &C[(i0 + r) * ldc + j0 + v * W];
          if constexpr (U == Update::Accumulate) {
            acc[r][v] = acc[r][v] + V::Load(Cv);
          } else if constexpr (U == Update::AlphaBeta) {
            acc[r][v] = acc[r][v] * V(alpha);
            if (beta != 0) acc[r][v] = FMA(V(beta), V::Load(Cv), acc[r][v]);
          }
          acc[r][v].Store(Cv);
        }
      }
    };

    const Long n_full = n - n % VL; // columns done in full-width vectors
    const Long n_pair = n_full - n_full % (2 * VL); // columns done in pairs of vectors
    const auto cols = [&tile, n_full, n_pair](const Long i0, auto mr) {
      for (Long j = 0; j < n_pair; j += 2 * VL) tile(i0, j, mr, I2{}, IV{});
      for (Long j = n_pair; j < n_full; j += VL) tile(i0, j, mr, I1{}, IV{});
    };
    const auto cols_tail = [&tile, A, B, C, n, k, lda, ldb, ldc, alpha, beta, n_full](const Long i0, auto mr) { // columns n_full..
      constexpr Integer MR = decltype(mr)::value;
      Long j = n_full;
      if constexpr (VL > 8) {
        for (; j + 8 <= n; j += 8) tile(i0, j, mr, I1{}, std::integral_constant<Integer, 8>{});
      }
      if constexpr (VL > 4) {
        for (; j + 4 <= n; j += 4) tile(i0, j, mr, I1{}, std::integral_constant<Integer, 4>{});
      }
      if constexpr (VL > 2) {
        for (; j + 2 <= n; j += 2) tile(i0, j, mr, I1{}, std::integral_constant<Integer, 2>{});
      }
      for (; j < n; j++) {
        for (Integer r = 0; r < MR; r++) {
          ValueType s = 0;
          for (Long l = 0; l < k; l++) s += A[(i0 + r) * lda + l] * B[l * ldb + j];
          ValueType& c = C[(i0 + r) * ldc + j];
          if constexpr (U == Update::Overwrite) {
            c = s;
          } else if constexpr (U == Update::Accumulate) {
            c += s;
          } else {
            c = alpha * s + (beta != 0 ? beta * c : ValueType(0));
          }
        }
      }
    };

    // The leftover columns in a second pass: inside the first loop they slowed down its tiles
    const Long m_quad = m - m % 4; // rows done in blocks of 4
    for (Long i = 0; i < m_quad; i += 4) cols(i, I4{});
    for (Long i = m_quad; i < m; i++) cols(i, I1{});
    if (n_full < n) {
      for (Long i = 0; i < m_quad; i += 4) cols_tail(i, I4{});
      for (Long i = m_quad; i < m; i++) cols_tail(i, I1{});
    }
  }

}  // namespace detail_small_gemm

template <class ValueType, Long M, Long N, Long K> inline SmallGEMM<ValueType, M, N, K>::SmallGEMM(const bool accumulate, const Long m, const Long n, const Long k) : m_(m), n_(n), k_(k), accumulate_(accumulate) {
  SCTL_ASSERT(m >= 0 && n >= 0 && k >= 0);
  SCTL_ASSERT((M == DynamicSize || m == M) && (N == DynamicSize || n == N) && (K == DynamicSize || k == K));
#if defined(SCTL_HAVE_LIBXSMM)
  kernel_ = nullptr;
  if constexpr (std::is_same<ValueType, double>::value || std::is_same<ValueType, float>::value) {
    if (m * n * k > 0) {
      // A per-thread table in front of LIBXSMM's lookup, which takes about 40 ns, as long as the
      // smallest products
      struct Entry {
        Long m = -1, n = -1, k = -1;
        bool accumulate = false;
        libxsmm_gemmfunction kernel = nullptr;
      };
      thread_local Entry table[64];
      Entry& e = table[(m * 31 + n * 17 + k * 7 + (accumulate ? 1 : 0)) % 64];
      if (e.m != m || e.n != n || e.k != k || e.accumulate != accumulate) {
        constexpr libxsmm_datatype T = (std::is_same<ValueType, double>::value ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32);
        // Row-major C = A B is column-major C^T (n x m) = B^T (n x k) A^T (k x m)
        const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape((libxsmm_blasint)n, (libxsmm_blasint)m, (libxsmm_blasint)k, (libxsmm_blasint)n, (libxsmm_blasint)k, (libxsmm_blasint)n, T, T, T, T);
        e.kernel = libxsmm_dispatch_gemm(shape, (libxsmm_bitfield)(accumulate ? 0 : LIBXSMM_GEMM_FLAG_BETA_0), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
        e.m = m;
        e.n = n;
        e.k = k;
        e.accumulate = accumulate;
      }
      kernel_ = e.kernel;
    }
  }
#endif
}

template <class ValueType, Long M, Long N, Long K> inline void SmallGEMM<ValueType, M, N, K>::operator()(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B) const {
#if defined(SCTL_HAVE_LIBXSMM)
  if (kernel_) { // operands exchanged, as in the constructor
    libxsmm_gemm_param prm;
    prm.a.primary = (void*)&B[0];
    prm.b.primary = (void*)&A[0];
    prm.c.primary = (void*)&C[0];
    kernel_(&prm);
    return;
  }
#endif
  using detail_small_gemm::Update;
  if (accumulate_) {
    detail_small_gemm::VecProduct<ValueType, M, N, K, Update::Accumulate, true>(C, A, B, m_, n_, k_, k_, n_, n_, (ValueType)1, (ValueType)1);
  } else {
    detail_small_gemm::VecProduct<ValueType, M, N, K, Update::Overwrite, true>(C, A, B, m_, n_, k_, k_, n_, n_, (ValueType)1, (ValueType)0);
  }
}

template <class ValueType, Long M, Long N, Long K> inline void SmallGEMM<ValueType, M, N, K>::operator()(Matrix<ValueType>& C, const Matrix<ValueType>& A, const Matrix<ValueType>& B) const {
  SCTL_ASSERT(A.Dim(0) == m_ && A.Dim(1) == k_ && B.Dim(0) == k_ && B.Dim(1) == n_ && C.Dim(0) == m_ && C.Dim(1) == n_);
  (*this)(C.begin(), A.begin(), B.begin());
}

}  // namespace sctl

#endif  // _SCTL_SMALL_GEMM_TXX_
