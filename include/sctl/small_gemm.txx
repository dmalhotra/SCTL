#ifndef _SCTL_SMALL_GEMM_TXX_
#define _SCTL_SMALL_GEMM_TXX_

#include <complex>              // for complex
#include <type_traits>          // for integral_constant, is_same

#include "sctl/common.hpp"      // for Long, Integer, SCTL_ASSERT, sctl
#include "sctl/small_gemm.hpp"  // for SmallGEMM, DynamicSize
#include "sctl/iterator.hpp"    // for Iterator, ConstIterator
#include "sctl/iterator.txx"    // for Iterator::operator[]
#include "sctl/math_utils.hpp"  // for IsComplex
#include "sctl/matrix.hpp"      // for Matrix
#include "sctl/profile.hpp"     // for Profile, ProfileCounter, FlopCount
#include "sctl/profile.txx"     // for Profile::IncrementCounter
#include "sctl/vec.hpp"         // for Vec, FMA, swap_pairs
#include "sctl/vec.txx"         // for Vec::Load, Vec::Store

namespace sctl {

namespace detail_small_gemm {

  /**
   * How a product updates C: C = A B; C += A B; or C = alpha A B + beta C, where C is not read if
   * beta is 0.
   */
  enum class Update { Overwrite, Accumulate, AlphaBeta };

  /**
   * True for float and double, whose products VecProduct computes; other types have no Vec
   * arithmetic (complex types, see ComplexVecTiles, and user types), or a slower one than a scalar
   * loop (long double, 1.8-2.8x).
   */
  template <class ValueType> constexpr bool VecTiles = std::is_same<ValueType, float>::value || std::is_same<ValueType, double>::value;

  /**
   * True for std::complex<float> and std::complex<double> when a vector holds at least one entry
   * (two reals); ComplexVecProduct computes their products.
   */
  template <class ValueType> constexpr bool ComplexVecTiles = false;
  template <class Real> constexpr bool ComplexVecTiles<std::complex<Real>> = VecTiles<Real> && (Vec<Real>::Size() >= 2);

  /**
   * C (m x n) = A (m x k) B (k x n), updated as U says; all row-major, with row strides lda, ldb
   * and ldc. Each entry of C sums over k in order, starting from its first term. Real types in
   * blocks of 2 x 2 entries, whose sums proceed independently (1.5-1.8x faster than one entry at a
   * time for long double; larger blocks do not fit in its 8 x87 registers); std::complex one entry
   * at a time, with the products written out in real arithmetic, without the recovery of infinite
   * results that std::complex multiplication does (1.6x faster for std::complex<long double>).
   */
  template <class ValueType, Update U> void ScalarProduct(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B, const Long m, const Long n, const Long k, const Long lda, const Long ldb, const Long ldc, const ValueType alpha, const ValueType beta) {
    if constexpr (U == Update::AlphaBeta) { // alpha = 1 with beta = 0 or 1 in the other modes, which do not multiply by alpha and beta
      if (alpha == ValueType(1) && beta == ValueType(0)) return ScalarProduct<ValueType, Update::Overwrite>(C, A, B, m, n, k, lda, ldb, ldc, alpha, beta);
      if (alpha == ValueType(1) && beta == ValueType(1)) return ScalarProduct<ValueType, Update::Accumulate>(C, A, B, m, n, k, lda, ldb, ldc, alpha, beta);
    }
    constexpr Integer BS = (IsComplex<ValueType> ? 1 : 2); // rows and columns per block
    using I1 = std::integral_constant<Integer, 1>;
    using IB = std::integral_constant<Integer, BS>;
    const bool beta_zero = (beta == ValueType(0)); // once: a function call for QuadReal

    const auto mul = [](const ValueType& a, const ValueType& b) -> ValueType {
      if constexpr (IsComplex<ValueType>) {
        return ValueType(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
      } else {
        return a * b;
      }
    };

    // Rows i0.. (MR of them) and columns j0.. (NR of them) of C
    const auto tile = [&mul, A, B, C, k, lda, ldb, ldc, alpha, beta, beta_zero](const Long i0, const Long j0, auto mr, auto nr) {
      constexpr Integer MR = decltype(mr)::value;
      constexpr Integer NR = decltype(nr)::value;
      ValueType s[MR][NR];
      for (Integer r = 0; r < MR; r++) {
        for (Integer c = 0; c < NR; c++) s[r][c] = (k > 0 ? mul(A[(i0 + r) * lda], B[j0 + c]) : ValueType(0));
      }
      for (Long l = 1; l < k; l++) {
        for (Integer r = 0; r < MR; r++) {
          for (Integer c = 0; c < NR; c++) s[r][c] += mul(A[(i0 + r) * lda + l], B[l * ldb + j0 + c]);
        }
      }
      for (Integer r = 0; r < MR; r++) {
        for (Integer c = 0; c < NR; c++) {
          ValueType& x = C[(i0 + r) * ldc + j0 + c];
          if constexpr (U == Update::Overwrite) {
            x = s[r][c];
          } else if constexpr (U == Update::Accumulate) {
            x += s[r][c];
          } else {
            x = alpha * s[r][c] + (beta_zero ? ValueType(0) : beta * x);
          }
        }
      }
    };

    const Long m_blk = m - m % BS, n_blk = n - n % BS; // rows and columns done in blocks
    const auto row = [&tile, n, n_blk](const Long i0, auto mr) {
      for (Long j = 0; j < n_blk; j += BS) tile(i0, j, mr, IB{});
      for (Long j = n_blk; j < n; j++) tile(i0, j, mr, I1{});
    };
    for (Long i = 0; i < m_blk; i += BS) row(i, IB{});
    for (Long i = m_blk; i < m; i++) row(i, I1{});
  }

  /**
   * C (m x n) = A (m x k) B (k x n), updated as U says; all row-major, with row strides lda, ldb
   * and ldc, or k, n and n if Contiguous. A template size or stride other than DynamicSize replaces
   * the argument, so the loops are specialized for it. Tiles of 8 rows of C by 1 vector for AMD Zen 4,
   * of 8 rows by 3 vectors and then 1 for other AVX-512 CPUs, and of 4 rows by 2 vectors and then 1
   * otherwise. For k = 8: on an AMD EPYC 9474F, 4 x 2 was 1.6x slower than 8 x 1 for double and
   * 2.2x for float, and 8 x 3 1.1x slower; on an Intel Xeon Platinum 8362 and a w5-3435X, 8 x 1 was
   * up to 1.2x slower than 8 x 3 for 174 x 94 x 8 and up to 1.8x for 96 columns; with AVX2, 8 x 1
   * was 1.2x slower than 4 x 2. The leftover rows in blocks of 4 (with AVX-512), 2 and 1. The
   * leftover columns of C = A B in one full vector that ends at column n and writes some columns
   * again, with the same values; otherwise in an SSE or AVX vector of half or a quarter of the width
   * if they fill it (masked loads and stores of the full width were up to 1.4x slower), or else in
   * one vector of which only the first n % VL lanes are loaded and stored (LoadPartial,
   * StorePartial). For 16 x n x 8 on a w5-3435X, up to 5.9x faster than narrower vectors and then one
   * column at a time, and at most 1.15x slower. Not inlined: unrolled for fixed sizes inside a
   * caller's loop, the code was up to 1.9x slower.
   */
  template <class ValueType, Long M, Long N, Long K, Update U, bool Contiguous, Long LDA = DynamicSize, Long LDB = DynamicSize, Long LDC = DynamicSize> [[gnu::noinline]] void VecProduct(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B, const Long m_, const Long n_, const Long k_, const Long lda_, const Long ldb_, const Long ldc_, const ValueType alpha, const ValueType beta) {
    const Long m = (M != DynamicSize ? M : m_);
    const Long n = (N != DynamicSize ? N : n_);
    const Long k = (K != DynamicSize ? K : k_);
    const Long lda = (LDA != DynamicSize ? LDA : Contiguous ? k : lda_);
    const Long ldb = (LDB != DynamicSize ? LDB : Contiguous ? n : ldb_);
    const Long ldc = (LDC != DynamicSize ? LDC : Contiguous ? n : ldc_);
    constexpr Integer VL = Vec<ValueType>::Size();
    using I1 = std::integral_constant<Integer, 1>;
    using I2 = std::integral_constant<Integer, 2>;
    using I4 = std::integral_constant<Integer, 4>;
    using IV = std::integral_constant<Integer, VL>;
    using IH = std::integral_constant<Integer, VL / 2>;
    using IQ = std::integral_constant<Integer, VL / 4>;
#if defined(__AVX512F__) && defined(__znver4__)
    using IR = std::integral_constant<Integer, 8>; // rows per block
    using IT = std::integral_constant<Integer, 1>; // vectors per tile
#elif defined(__AVX512F__)
    using IR = std::integral_constant<Integer, 8>;
    using IT = std::integral_constant<Integer, 3>;
#else
    using IR = std::integral_constant<Integer, 4>;
    using IT = std::integral_constant<Integer, 2>;
#endif

    // Rows i0.. (MR of them) and columns j0.. (NV vectors of W) of C; if Partial, only the first nc
    // columns of the one vector. Always inlined, as are cols and cols_tail: GCC left some out of
    // line, and kept acc on the stack there
    const auto tile = [A, B, C, k, lda, ldb, ldc, alpha, beta](const Long i0, const Long j0, auto mr, auto nv, auto w, auto partial, const Integer nc) __attribute__((always_inline)) {
      constexpr Integer MR = decltype(mr)::value;
      constexpr Integer NV = decltype(nv)::value;
      constexpr Integer W = decltype(w)::value;
      constexpr bool Partial = decltype(partial)::value;
      using V = Vec<ValueType, W>;
      const auto load = [nc](const ValueType* p) {
        if constexpr (Partial) {
          return V::LoadPartial(p, nc);
        } else {
          return V::Load(p);
        }
      };
      V acc[MR][NV];
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) acc[r][v] = V((ValueType)0);
      }
      for (Long l = 0; l < k; l++) {
        V b[NV];
        for (Integer v = 0; v < NV; v++) b[v] = load(&B[l * ldb + j0 + v * W]);
        for (Integer r = 0; r < MR; r++) {
          const V a(A[(i0 + r) * lda + l]);
          for (Integer v = 0; v < NV; v++) acc[r][v] = FMA(a, b[v], acc[r][v]);
        }
      }
      // All rows of C read before any is written: a partial load of a row shorter than a vector
      // overlaps the partial store of the row before, and waits for it. Both loops unrolled first:
      // otherwise GCC 11-14 keep acc on the stack and, with masked loads in the l loop, store it
      // there in every iteration
#pragma GCC unroll 8
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) {
          const ValueType* Cv = &C[(i0 + r) * ldc + j0 + v * W];
          if constexpr (U == Update::Accumulate) {
            acc[r][v] = acc[r][v] + load(Cv);
          } else if constexpr (U == Update::AlphaBeta) {
            acc[r][v] = acc[r][v] * V(alpha);
            if (beta != 0) acc[r][v] = FMA(V(beta), load(Cv), acc[r][v]);
          }
        }
      }
#pragma GCC unroll 8
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) {
          ValueType* Cv = &C[(i0 + r) * ldc + j0 + v * W];
          if constexpr (Partial) {
            acc[r][v].StorePartial(Cv, nc);
          } else {
            acc[r][v].Store(Cv);
          }
        }
      }
    };

    const Long n_full = n - n % VL; // columns done in full-width vectors
    const Long n_tile = n_full - n_full % (IT::value * VL); // columns done in tiles of IT vectors
    const auto cols = [&tile, n_full, n_tile](const Long i0, auto mr) __attribute__((always_inline)) {
      for (Long j = 0; j < n_tile; j += IT::value * VL) tile(i0, j, mr, IT{}, IV{}, std::false_type{}, VL);
      for (Long j = n_tile; j < n_full; j += VL) tile(i0, j, mr, I1{}, IV{}, std::false_type{}, VL);
    };
    const auto cols_tail = [&tile, n, n_full](const Long i0, auto mr) __attribute__((always_inline)) { // columns n_full..
      if constexpr (U == Update::Overwrite) {
        if (n >= VL) {
          tile(i0, n - VL, mr, I1{}, IV{}, std::false_type{}, VL);
          return;
        }
      }
      const auto fills = [&tile, n, n_full, i0, mr](auto w) __attribute__((always_inline)) { // the columns in one SSE or AVX vector of W, if they fill it
        constexpr Integer W = decltype(w)::value;
        if constexpr (W * sizeof(ValueType) >= 16) {
          if (n - n_full == W) {
            tile(i0, n_full, mr, I1{}, w, std::false_type{}, W);
            return true;
          }
        }
        return false;
      };
      if (fills(IH{}) || fills(IQ{})) return;
      tile(i0, n_full, mr, I1{}, IV{}, std::true_type{}, (Integer)(n - n_full));
    };

    // Rows in blocks of IR, then of 4, 2 and 1 below IR; the leftover columns in a second pass:
    // inside the first loop they slowed down its tiles. Through a lambda over the row blocks that
    // takes cols or cols_tail, the code was up to 1.5x slower
    const Long m_blk = m - m % IR::value;
    const Long m_4 = (IR::value > 4 ? m - m % 4 : m_blk);
    const Long m_2 = (IR::value > 2 ? m - m % 2 : m_4);
    for (Long i = 0; i < m_blk; i += IR::value) cols(i, IR{});
    for (Long i = m_blk; i < m_4; i += 4) cols(i, I4{});
    for (Long i = m_4; i < m_2; i += 2) cols(i, I2{});
    for (Long i = m_2; i < m; i++) cols(i, I1{});
    if (n_full < n) {
      for (Long i = 0; i < m_blk; i += IR::value) cols_tail(i, IR{});
      for (Long i = m_blk; i < m_4; i += 4) cols_tail(i, I4{});
      for (Long i = m_4; i < m_2; i += 2) cols_tail(i, I2{});
      for (Long i = m_2; i < m; i++) cols_tail(i, I1{});
    }
  }

  /**
   * As VecProduct, for std::complex<float> and std::complex<double>, with the same row strides and
   * template sizes and strides: a row of B or C is a vector of interleaved real and imaginary parts.
   * A tile sums re(A) B and im(A) B separately and combines them once, at the end, with the parts of
   * each entry exchanged by swap_pairs. Rows of C in blocks of 4 with the 32 vector registers of
   * AVX-512 and of 2 otherwise (16 or 8 accumulators; blocks of 4 with 16 registers were 30-50%
   * slower), columns in tiles of 2 vectors and then 1, the leftover columns as in VecProduct: in a
   * vector of half or a quarter of the width if they fill it, or else in one vector of which only
   * their reals are loaded and stored.
   */
  template <class ValueType, Long M, Long N, Long K, Update U, bool Contiguous, Long LDA = DynamicSize, Long LDB = DynamicSize, Long LDC = DynamicSize> [[gnu::noinline]] void ComplexVecProduct(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B, const Long m_, const Long n_, const Long k_, const Long lda_, const Long ldb_, const Long ldc_, const ValueType alpha, const ValueType beta) {
    using Real = typename ValueType::value_type;
    const Long m = (M != DynamicSize ? M : m_);
    const Long n = (N != DynamicSize ? N : n_);
    const Long k = (K != DynamicSize ? K : k_);
    const Long lda = (LDA != DynamicSize ? LDA : Contiguous ? k : lda_);
    const Long ldb = (LDB != DynamicSize ? LDB : Contiguous ? n : ldb_);
    const Long ldc = (LDC != DynamicSize ? LDC : Contiguous ? n : ldc_);
    constexpr Integer VL = Vec<Real>::Size();
    constexpr Integer VE = VL / 2; // entries per vector
    using I1 = std::integral_constant<Integer, 1>;
    using I2 = std::integral_constant<Integer, 2>;
#if defined(__AVX512F__)
    using IR = std::integral_constant<Integer, 4>; // rows per block
#else
    using IR = std::integral_constant<Integer, 2>;
#endif
    using IV = std::integral_constant<Integer, VL>;
    using IH = std::integral_constant<Integer, VL / 2>;
    using IQ = std::integral_constant<Integer, VL / 4>;
    const bool alpha_one = (alpha == ValueType(1)), beta_zero = (beta == ValueType(0));

    // Rows i0.. (MR of them) and entries j0.. (NV vectors of W reals) of C; if Partial, only the first
    // nc reals of the one vector. Inlined, as in VecProduct
    const auto tile = [A, B, C, k, lda, ldb, ldc, alpha, beta, alpha_one, beta_zero](const Long i0, const Long j0, auto mr, auto nv, auto w, auto partial, const Integer nc) __attribute__((always_inline)) {
      constexpr Integer MR = decltype(mr)::value;
      constexpr Integer NV = decltype(nv)::value;
      constexpr Integer W = decltype(w)::value;
      constexpr bool Partial = decltype(partial)::value;
      using V = Vec<Real, W>;
      const auto load = [nc](const Real* p) {
        if constexpr (Partial) {
          return V::LoadPartial(p, nc);
        } else {
          return V::Load(p);
        }
      };
      V re[MR][NV], im[MR][NV]; // sums of re(A) B and im(A) B
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) {
          re[r][v] = V((Real)0);
          im[r][v] = V((Real)0);
        }
      }
      for (Long l = 0; l < k; l++) {
        const Real* Bl = reinterpret_cast<const Real*>(&B[l * ldb + j0]);
        V b[NV];
        for (Integer v = 0; v < NV; v++) b[v] = load(Bl + v * W);
        for (Integer r = 0; r < MR; r++) {
          const ValueType a = A[(i0 + r) * lda + l];
          const V a_re(a.real()), a_im(a.imag());
          for (Integer v = 0; v < NV; v++) {
            re[r][v] = FMA(a_re, b[v], re[r][v]);
            im[r][v] = FMA(a_im, b[v], im[r][v]);
          }
        }
      }

      Real sign_[W]; // -1 for the real parts, 1 for the imaginary ones
      for (Integer i = 0; i < W; i++) sign_[i] = (Real)(i % 2 ? 1 : -1);
      const V sign = V::Load(sign_);
      const auto scale = [&sign](const ValueType z, const V& x) { return FMA(V(z.real()), x, swap_pairs(x) * (sign * V(z.imag()))); }; // z times each entry of x
      // As in VecProduct, all rows of C read before any is written, and both loops unrolled first; re
      // holds the result
#pragma GCC unroll 8
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) {
          const Real* Cv = reinterpret_cast<const Real*>(&C[(i0 + r) * ldc + j0]) + v * W;
          re[r][v] = FMA(sign, swap_pairs(im[r][v]), re[r][v]);
          if constexpr (U == Update::Accumulate) {
            re[r][v] = re[r][v] + load(Cv);
          } else if constexpr (U == Update::AlphaBeta) {
            if (!alpha_one) re[r][v] = scale(alpha, re[r][v]);
            if (!beta_zero) re[r][v] = re[r][v] + scale(beta, load(Cv));
          }
        }
      }
#pragma GCC unroll 8
      for (Integer r = 0; r < MR; r++) {
        for (Integer v = 0; v < NV; v++) {
          Real* Cv = reinterpret_cast<Real*>(&C[(i0 + r) * ldc + j0]) + v * W;
          if constexpr (Partial) {
            re[r][v].StorePartial(Cv, nc);
          } else {
            re[r][v].Store(Cv);
          }
        }
      }
    };

    const Long n_full = n - n % VE; // entries done in full-width vectors
    const Long n_pair = n_full - n_full % (2 * VE); // entries done in pairs of vectors
    const auto cols = [&tile, n_full, n_pair](const Long i0, auto mr) __attribute__((always_inline)) {
      for (Long j = 0; j < n_pair; j += 2 * VE) tile(i0, j, mr, I2{}, IV{}, std::false_type{}, VL);
      for (Long j = n_pair; j < n_full; j += VE) tile(i0, j, mr, I1{}, IV{}, std::false_type{}, VL);
    };
    const auto cols_tail = [&tile, n, n_full](const Long i0, auto mr) __attribute__((always_inline)) { // entries n_full..
      const auto fills = [&tile, n, n_full, i0, mr](auto w) __attribute__((always_inline)) { // as in VecProduct, for W reals
        constexpr Integer W = decltype(w)::value;
        if constexpr (W * sizeof(Real) >= 16) {
          if (2 * (n - n_full) == W) {
            tile(i0, n_full, mr, I1{}, w, std::false_type{}, W);
            return true;
          }
        }
        return false;
      };
      if (fills(IH{}) || fills(IQ{})) return;
      tile(i0, n_full, mr, I1{}, IV{}, std::true_type{}, (Integer)(2 * (n - n_full)));
    };

    // The leftover columns in a second pass, as in VecProduct
    const Long m_blk = m - m % IR::value; // rows done in blocks
    for (Long i = 0; i < m_blk; i += IR::value) cols(i, IR{});
    for (Long i = m_blk; i < m; i++) cols(i, I1{});
    if (n_full < n) {
      for (Long i = 0; i < m_blk; i += IR::value) cols_tail(i, IR{});
      for (Long i = m_blk; i < m; i++) cols_tail(i, I1{});
    }
  }

}  // namespace detail_small_gemm

template <class ValueType, Long M, Long N, Long K, Long LDA, Long LDB, Long LDC> inline SmallGEMM<ValueType, M, N, K, LDA, LDB, LDC>::SmallGEMM(const bool accumulate, const Long m, const Long n, const Long k) : SmallGEMM(accumulate, m, n, k, (LDA != DynamicSize ? LDA : k), (LDB != DynamicSize ? LDB : n), (LDC != DynamicSize ? LDC : n)) {}

template <class ValueType, Long M, Long N, Long K, Long LDA, Long LDB, Long LDC> inline SmallGEMM<ValueType, M, N, K, LDA, LDB, LDC>::SmallGEMM(const bool accumulate, const Long m, const Long n, const Long k, const Long lda, const Long ldb, const Long ldc) : m_(m), n_(n), k_(k), lda_(lda), ldb_(ldb), ldc_(ldc), accumulate_(accumulate) {
  SCTL_ASSERT(m >= 0 && n >= 0 && k >= 0);
  SCTL_ASSERT((M == DynamicSize || m == M) && (N == DynamicSize || n == N) && (K == DynamicSize || k == K));
  SCTL_ASSERT(lda >= k && ldb >= n && ldc >= n);
  SCTL_ASSERT((LDA == DynamicSize || lda == LDA) && (LDB == DynamicSize || ldb == LDB) && (LDC == DynamicSize || ldc == LDC));
#if defined(SCTL_HAVE_LIBXSMM)
  kernel_ = nullptr;
  if constexpr (std::is_same<ValueType, double>::value || std::is_same<ValueType, float>::value) {
    if (m * n * k > 0) {
      // A per-thread table in front of LIBXSMM's lookup, which takes about 40 ns, as long as the
      // smallest products
      struct Entry {
        Long m = -1, n = -1, k = -1, lda = -1, ldb = -1, ldc = -1;
        bool accumulate = false;
        libxsmm_gemmfunction kernel = nullptr;
      };
      thread_local Entry table[64];
      Entry& e = table[(m * 31 + n * 17 + k * 7 + (lda - k) * 13 + (ldb - n) * 11 + (ldc - n) * 5 + (accumulate ? 1 : 0)) % 64];
      if (e.m != m || e.n != n || e.k != k || e.lda != lda || e.ldb != ldb || e.ldc != ldc || e.accumulate != accumulate) {
        constexpr libxsmm_datatype T = (std::is_same<ValueType, double>::value ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32);
        // Row-major C = A B is column-major C^T (n x m) = B^T (n x k) A^T (k x m)
        const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape((libxsmm_blasint)n, (libxsmm_blasint)m, (libxsmm_blasint)k, (libxsmm_blasint)ldb, (libxsmm_blasint)lda, (libxsmm_blasint)ldc, T, T, T, T);
        e.kernel = libxsmm_dispatch_gemm(shape, (libxsmm_bitfield)(accumulate ? 0 : LIBXSMM_GEMM_FLAG_BETA_0), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
        e.m = m;
        e.n = n;
        e.k = k;
        e.lda = lda;
        e.ldb = ldb;
        e.ldc = ldc;
        e.accumulate = accumulate;
      }
      kernel_ = e.kernel;
    }
  }
#endif
}

template <class ValueType, Long M, Long N, Long K, Long LDA, Long LDB, Long LDC> inline void SmallGEMM<ValueType, M, N, K, LDA, LDB, LDC>::operator()(Iterator<ValueType> C, ConstIterator<ValueType> A, ConstIterator<ValueType> B) const {
  Profile::IncrementCounter(ProfileCounter::FLOP, FlopCount<ValueType>::MulAdd * m_ * n_ * k_);
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
  if constexpr (detail_small_gemm::VecTiles<ValueType>) {
    if constexpr (LDA == DynamicSize && LDB == DynamicSize && LDC == DynamicSize) { // with a fixed stride, only the strided loop
      if (lda_ == k_ && ldb_ == n_ && ldc_ == n_) { // contiguous: strides from the sizes, so fixed with them
        if (accumulate_) {
          detail_small_gemm::VecProduct<ValueType, M, N, K, Update::Accumulate, true>(C, A, B, m_, n_, k_, k_, n_, n_, (ValueType)1, (ValueType)1);
        } else {
          detail_small_gemm::VecProduct<ValueType, M, N, K, Update::Overwrite, true>(C, A, B, m_, n_, k_, k_, n_, n_, (ValueType)1, (ValueType)0);
        }
        return;
      }
    }
    if (accumulate_) {
      detail_small_gemm::VecProduct<ValueType, M, N, K, Update::Accumulate, false, LDA, LDB, LDC>(C, A, B, m_, n_, k_, lda_, ldb_, ldc_, (ValueType)1, (ValueType)1);
    } else {
      detail_small_gemm::VecProduct<ValueType, M, N, K, Update::Overwrite, false, LDA, LDB, LDC>(C, A, B, m_, n_, k_, lda_, ldb_, ldc_, (ValueType)1, (ValueType)0);
    }
  } else if constexpr (detail_small_gemm::ComplexVecTiles<ValueType>) {
    if constexpr (LDA == DynamicSize && LDB == DynamicSize && LDC == DynamicSize) { // as for VecTiles
      if (lda_ == k_ && ldb_ == n_ && ldc_ == n_) {
        if (accumulate_) {
          detail_small_gemm::ComplexVecProduct<ValueType, M, N, K, Update::Accumulate, true>(C, A, B, m_, n_, k_, k_, n_, n_, (ValueType)1, (ValueType)1);
        } else {
          detail_small_gemm::ComplexVecProduct<ValueType, M, N, K, Update::Overwrite, true>(C, A, B, m_, n_, k_, k_, n_, n_, (ValueType)1, (ValueType)0);
        }
        return;
      }
    }
    if (accumulate_) {
      detail_small_gemm::ComplexVecProduct<ValueType, M, N, K, Update::Accumulate, false, LDA, LDB, LDC>(C, A, B, m_, n_, k_, lda_, ldb_, ldc_, (ValueType)1, (ValueType)1);
    } else {
      detail_small_gemm::ComplexVecProduct<ValueType, M, N, K, Update::Overwrite, false, LDA, LDB, LDC>(C, A, B, m_, n_, k_, lda_, ldb_, ldc_, (ValueType)1, (ValueType)0);
    }
  } else {
    if (accumulate_) {
      detail_small_gemm::ScalarProduct<ValueType, Update::Accumulate>(C, A, B, m_, n_, k_, lda_, ldb_, ldc_, (ValueType)1, (ValueType)1);
    } else {
      detail_small_gemm::ScalarProduct<ValueType, Update::Overwrite>(C, A, B, m_, n_, k_, lda_, ldb_, ldc_, (ValueType)1, (ValueType)0);
    }
  }
}

template <class ValueType, Long M, Long N, Long K, Long LDA, Long LDB, Long LDC> inline void SmallGEMM<ValueType, M, N, K, LDA, LDB, LDC>::operator()(Matrix<ValueType>& C, const Matrix<ValueType>& A, const Matrix<ValueType>& B) const {
  SCTL_ASSERT(A.Dim(1) == lda_ && B.Dim(1) == ldb_ && C.Dim(1) == ldc_);
  SCTL_ASSERT(A.Dim(0) >= m_ && B.Dim(0) >= k_ && C.Dim(0) >= m_);
  (*this)(C.begin(), A.begin(), B.begin());
}

}  // namespace sctl

#endif  // _SCTL_SMALL_GEMM_TXX_
