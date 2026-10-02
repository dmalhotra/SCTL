// Tests for sctl/small_gemm.{hpp,txx}.
//
// SmallGEMM against a direct triple loop, with the sizes fixed at compile time, given at run time,
// or mixed, in both modes (C = A B and C += A B), for float, double, long double and QuadReal, with
// contiguous rows and with row strides beyond the row lengths, given at run time or fixed at compile
// time; with one object applied by several threads at once; and through the Matrix overload, on whole
// matrices and on leading blocks. In the overwrite mode C starts as NaN, which must not be read; so
// are the values between the rows of A and B, and those of C must not change.

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "sctl.hpp"

#include "test-utils.hpp"

using sctl::DynamicSize;
using sctl::Long;
using sctl::SmallGEMM;

// One product of random m x k and k x n matrices, whose rows (and those of C) are pa, pb and pc
// values longer than the matrices (as the fixed strides LDA, LDB and LDC say, if any); true if every
// entry of C is within (4 k + 8) eps of the triple loop, relative to the sum of the absolute values
// of its terms, and the values between the rows of C are unchanged
template <class T, Long M, Long N, Long K, Long LDA = DynamicSize, Long LDB = DynamicSize, Long LDC = DynamicSize> static bool check(const bool accumulate, const Long m, const Long n, const Long k, const Long pa = 0, const Long pb = 0, const Long pc = 0) {
  const T nan = T(std::numeric_limits<double>::quiet_NaN()); // numeric_limits<QuadReal> has no NaN
  const T fill = T(7);
  const Long lda = k + pa, ldb = n + pb, ldc = n + pc;
  std::vector<T> A(m * lda, nan), B(k * ldb, nan), C(m * ldc, fill);
  for (Long i = 0; i < m; i++) {
    for (Long l = 0; l < k; l++) A[i * lda + l] = T(std::rand()) / T(RAND_MAX) - T(0.5);
  }
  for (Long l = 0; l < k; l++) {
    for (Long j = 0; j < n; j++) B[l * ldb + j] = T(std::rand()) / T(RAND_MAX) - T(0.5);
  }
  for (Long i = 0; i < m; i++) {
    for (Long j = 0; j < n; j++) C[i * ldc + j] = (accumulate ? T(std::rand()) / T(RAND_MAX) : nan);
  }
  const std::vector<T> C0 = C;

  // The constructor with strides when the template arguments do not give them all
  using GEMM = SmallGEMM<T, M, N, K, LDA, LDB, LDC>;
  const bool given = ((LDA == DynamicSize && pa) || (LDB == DynamicSize && pb) || (LDC == DynamicSize && pc));
  const GEMM gemm = (given ? GEMM(accumulate, m, n, k, lda, ldb, ldc) : GEMM(accumulate, m, n, k));
  gemm(sctl::Ptr2Itr<T>(C.data(), (Long)C.size()), sctl::Ptr2ConstItr<T>(A.data(), (Long)A.size()), sctl::Ptr2ConstItr<T>(B.data(), (Long)B.size()));

  bool ok = true;
  for (Long i = 0; i < m; i++) {
    for (Long j = 0; j < n; j++) {
      T ref = (accumulate ? C0[i * ldc + j] : T(0)), mag = sctl::fabs<T>(ref);
      for (Long l = 0; l < k; l++) {
        ref += A[i * lda + l] * B[l * ldb + j];
        mag += sctl::fabs<T>(A[i * lda + l] * B[l * ldb + j]);
      }
      const T c = C[i * ldc + j];
      ok = ok && (c == c) && sctl::fabs<T>(c - ref) <= T(4 * k + 8) * sctl::machine_eps<T>() * mag;
    }
    for (Long j = n; j < ldc; j++) ok = ok && (C[i * ldc + j] == fill);
  }
  if (!ok) std::printf("  failed: m=%ld n=%ld k=%ld lda=%ld ldb=%ld ldc=%ld accumulate=%d\n", (long)m, (long)n, (long)k, (long)lda, (long)ldb, (long)ldc, (int)accumulate);
  return ok;
}

int main() {
  std::srand(1);
  const auto test_type = [](auto zero, const char* name) {
    using T = decltype(zero);
    std::printf("%s :\n", name);
    for (const bool acc : {false, true}) {
      // Sizes fixed at compile time: every tile width, leftover rows and columns, and a single row
      CHECK(check<T, 6, 24, 12>(acc, 6, 24, 12));
      CHECK(check<T, 36, 12, 12>(acc, 36, 12, 12));
      CHECK(check<T, 1, 12, 12>(acc, 1, 12, 12));
      CHECK(check<T, 13, 23, 11>(acc, 13, 23, 11));
      CHECK(check<T, 108, 23, 12>(acc, 108, 23, 12));
      CHECK(check<T, 36, 24, 12>(acc, 36, 24, 12));
      CHECK(check<T, 4, 16, 8>(acc, 4, 16, 8));
      CHECK(check<T, 5, 7, 3>(acc, 5, 7, 3));
      CHECK(check<T, 3, 1, 5>(acc, 3, 1, 5));
      CHECK(check<T, 2, 3, 0>(acc, 2, 3, 0));

      // Some sizes fixed, some given at run time
      int nbad = 0;
      for (int trial = 0; trial < 20; trial++) {
        const Long m = std::rand() % 41, n = std::rand() % 41;
        nbad += !check<T, DynamicSize, DynamicSize, 12>(acc, m, n, 12);
        nbad += !check<T, DynamicSize, 24, DynamicSize>(acc, m, 24, n);
        nbad += !check<T, 6, DynamicSize, DynamicSize>(acc, 6, m, n);
      }
      CHECK(nbad == 0);

      // All sizes given at run time, zero included
      nbad = 0;
      for (int trial = 0; trial < 200; trial++) {
        nbad += !check<T, DynamicSize, DynamicSize, DynamicSize>(acc, std::rand() % 41, std::rand() % 41, std::rand() % 41);
      }
      CHECK(nbad == 0);

      // Row strides beyond the row lengths, with the sizes given at run time or some fixed
      nbad = 0;
      for (int trial = 0; trial < 100; trial++) {
        const Long m = std::rand() % 41, n = std::rand() % 41, k = std::rand() % 41;
        const Long pa = std::rand() % 9, pb = std::rand() % 9, pc = std::rand() % 9;
        nbad += !check<T, DynamicSize, DynamicSize, DynamicSize>(acc, m, n, k, pa, pb, pc);
        nbad += !check<T, DynamicSize, 24, 12>(acc, m, 24, 12, pa, pb, pc);
        nbad += !check<T, 6, DynamicSize, 8>(acc, 6, n, 8, pa, pb, pc);
      }
      CHECK(nbad == 0);

      // Row strides fixed at compile time, all or some
      nbad = 0;
      for (int trial = 0; trial < 100; trial++) {
        const Long m = std::rand() % 41, n = std::rand() % 41, k = std::rand() % 41;
        const Long pa = std::rand() % 9, pb = std::rand() % 9, pc = std::rand() % 9;
        nbad += !check<T, DynamicSize, 24, 12, 20, 30, 31>(acc, m, 24, 12, 8, 6, 7);
        nbad += !check<T, DynamicSize, DynamicSize, DynamicSize, 48, DynamicSize, DynamicSize>(acc, m, n, k, 48 - k, pb, pc);
        nbad += !check<T, 6, DynamicSize, 8, DynamicSize, DynamicSize, 45>(acc, 6, n, 8, pa, pb, 45 - n);
        nbad += !check<T, 8, 8, DynamicSize, DynamicSize, 72, 8>(acc, 8, 8, k, pa, 64, 0);
      }
      CHECK(nbad == 0);
    }
  };
  test_type(float(0), "float");
  test_type(double(0), "double");
  test_type((long double)0, "long double");
#ifdef SCTL_QUAD_T
  test_type(sctl::QuadReal(0), "QuadReal");
#endif

  std::printf("one object, several threads :\n");
  {
    constexpr Long n = 12;
    const SmallGEMM<double, n, n, n> gemm;
    int nbad = 0;
    #pragma omp parallel for schedule(static) reduction(+:nbad) num_threads(8)
    for (int t = 0; t < 64; t++) {
      std::vector<double> A(n * n), B(n * n), C(n * n);
      for (Long i = 0; i < n * n; i++) {
        A[i] = std::sin(1.0 + (double)(i + 17 * t));
        B[i] = std::cos(2.0 + (double)(i + 29 * t));
      }
      gemm(sctl::Ptr2Itr<double>(C.data(), n * n), sctl::Ptr2ConstItr<double>(A.data(), n * n), sctl::Ptr2ConstItr<double>(B.data(), n * n));
      for (Long i = 0; i < n; i++) {
        for (Long j = 0; j < n; j++) {
          double ref = 0;
          for (Long l = 0; l < n; l++) ref += A[i * n + l] * B[l * n + j];
          nbad += (std::fabs(C[i * n + j] - ref) > 1e-13 * n);
        }
      }
    }
    CHECK(nbad == 0);
  }

  std::printf("std::complex<double> (no Vec arithmetic) :\n");
  {
    using Z = std::complex<double>;
    constexpr Long m = 5, n = 7, k = 3;
    std::vector<Z> A(m * k), B(k * n), C0(m * n);
    for (Long i = 0; i < m * k; i++) A[i] = Z(std::sin(1.0 + (double)i), std::cos(2.0 + (double)i));
    for (Long i = 0; i < k * n; i++) B[i] = Z(std::cos(3.0 + (double)i), std::sin(4.0 + (double)i));
    for (Long i = 0; i < m * n; i++) C0[i] = Z(0.5, std::sin(5.0 + (double)i));
    for (const bool acc : {false, true}) {
      const SmallGEMM<Z, m, n, k> fixed(acc);
      const SmallGEMM<Z> dynamic(acc, m, n, k);
      std::vector<Z> C1 = C0, C2 = C0;
      fixed(sctl::Ptr2Itr<Z>(C1.data(), m * n), sctl::Ptr2ConstItr<Z>(A.data(), m * k), sctl::Ptr2ConstItr<Z>(B.data(), k * n));
      dynamic(sctl::Ptr2Itr<Z>(C2.data(), m * n), sctl::Ptr2ConstItr<Z>(A.data(), m * k), sctl::Ptr2ConstItr<Z>(B.data(), k * n));
      int nbad = 0;
      for (Long i = 0; i < m; i++) {
        for (Long j = 0; j < n; j++) {
          Z ref = (acc ? C0[i * n + j] : Z(0));
          for (Long l = 0; l < k; l++) ref += A[i * k + l] * B[l * n + j];
          nbad += !(std::abs(C1[i * n + j] - ref) <= 1e-14) + !(std::abs(C2[i * n + j] - ref) <= 1e-14);
        }
      }
      CHECK(nbad == 0);
    }
  }

  std::printf("Matrix overload :\n");
  {
    sctl::Matrix<double> A(7, 5), B(5, 9), C(7, 9);
    for (Long i = 0; i < 7 * 5; i++) A[0][i] = (double)i;
    for (Long i = 0; i < 5 * 9; i++) B[0][i] = 1.0 / (double)(i + 1);
    const SmallGEMM<double> gemm(false, 7, 9, 5);
    gemm(C, A, B);
    double err = 0;
    for (Long i = 0; i < 7; i++) {
      for (Long j = 0; j < 9; j++) {
        double ref = 0;
        for (Long l = 0; l < 5; l++) ref += A[i][l] * B[l][j];
        err = std::max(err, std::fabs(C[i][j] - ref));
      }
    }
    CHECK(err <= 1e-13);
  }

  std::printf("Matrix overload, leading blocks :\n");
  {
    // C += A B for the leading 5 x 4, 4 x 7 and 5 x 7 blocks of 6 x 6, 5 x 9 and 7 x 8 matrices
    constexpr Long m = 5, n = 7, k = 4;
    sctl::Matrix<double> A(6, 6), B(5, 9), C(7, 8);
    for (Long i = 0; i < A.Dim(0) * A.Dim(1); i++) A[0][i] = std::sin(1.0 + (double)i);
    for (Long i = 0; i < B.Dim(0) * B.Dim(1); i++) B[0][i] = std::cos(2.0 + (double)i);
    const SmallGEMM<double> dynamic(true, m, n, k, A.Dim(1), B.Dim(1), C.Dim(1));
    const SmallGEMM<double, m, n, k, 6, 9, 8> fixed(true); // the strides from the template arguments
    for (const bool use_fixed : {false, true}) {
      for (Long i = 0; i < C.Dim(0) * C.Dim(1); i++) C[0][i] = 7.0;
      if (use_fixed) fixed(C, A, B);
      else dynamic(C, A, B);
      double err = 0;
      int nbad = 0;
      for (Long i = 0; i < C.Dim(0); i++) {
        for (Long j = 0; j < C.Dim(1); j++) {
          if (i < m && j < n) {
            double ref = 7.0;
            for (Long l = 0; l < k; l++) ref += A[i][l] * B[l][j];
            err = std::max(err, std::fabs(C[i][j] - ref));
          } else {
            nbad += (C[i][j] != 7.0);
          }
        }
      }
      CHECK(err <= 1e-13 && nbad == 0);
    }
  }

  TEST_SUMMARY_RETURN();
}
