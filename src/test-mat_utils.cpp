// Per-function tests for sctl/mat_utils.{hpp,txx}.
//
// The public free functions in `mat_utils` are low-level BLAS/LAPACK wrappers:
//   - gemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
//   - svd(JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO)
//   - pinv(M, n1, n2, eps, M_)
//
// gemm is verified against hand-computed reference products, and for random
// shapes against a direct triple loop; svd via
// reconstruction A = U * diag(S) * V^T; pinv via the Moore-Penrose
// identity A * pinv(A) * A = A.

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/mat_utils.hpp"
#include "sctl/mat_utils.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Iterator;
using sctl::ConstIterator;

// Type of the parts of a std::complex, or T itself
template <class T> struct RealOf {
  using type = T;
};
template <class T> struct RealOf<std::complex<T>> {
  using type = T;
};

int main() {
  // --- gemm: C := alpha*A*B + beta*C, BLAS column-major convention ---
  // A: 3x2 column-major (LDA=3); columns are [1,2,3], [4,5,6].
  // B: 2x3 column-major (LDB=2); columns are [7,8], [9,10], [11,12].
  // Then A*B is 3x3 with expected column-major layout below.
  std::printf("gemm no-transpose :\n");
  {
    double A[6] = {1, 2, 3,   4, 5, 6};               // col0=[1,2,3], col1=[4,5,6]
    double B[6] = {7, 8,   9, 10,   11, 12};          // col0=[7,8], col1=[9,10], col2=[11,12]
    double C[9] = {0};
    sctl::mat::gemm<double>('N', 'N', /*M=*/3, /*N=*/3, /*K=*/2,
                            1.0, sctl::Ptr2ConstItr<double>(A, 6), 3,
                                 sctl::Ptr2ConstItr<double>(B, 6), 2,
                            0.0, sctl::Ptr2Itr     <double>(C, 9), 3);
    // Expected (3x3 col-major):
    //   col 0 = [39, 54, 69]
    //   col 1 = [49, 68, 87]
    //   col 2 = [59, 82, 105]
    const double expected[9] = {39, 54, 69,   49, 68, 87,   59, 82, 105};
    for (int i = 0; i < 9; ++i) CHECK(test_utils::approx_eq(C[i], expected[i]));

    // Accumulate: C := A*B + 1.0*C, should double C.
    sctl::mat::gemm<double>('N', 'N', 3, 3, 2,
                            1.0, sctl::Ptr2ConstItr<double>(A, 6), 3,
                                 sctl::Ptr2ConstItr<double>(B, 6), 2,
                            1.0, sctl::Ptr2Itr     <double>(C, 9), 3);
    for (int i = 0; i < 9; ++i) CHECK(test_utils::approx_eq(C[i], 2 * expected[i]));

    // alpha = -1, beta = 0
    sctl::mat::gemm<double>('N', 'N', 3, 3, 2,
                           -1.0, sctl::Ptr2ConstItr<double>(A, 6), 3,
                                 sctl::Ptr2ConstItr<double>(B, 6), 2,
                            0.0, sctl::Ptr2Itr     <double>(C, 9), 3);
    for (int i = 0; i < 9; ++i) CHECK(test_utils::approx_eq(C[i], -expected[i]));
  }

  // --- gemm no-transpose, random shapes, against a direct triple loop ---
  // Sizes up to 40 reach every tile and leftover width of the fallback, except for float the widest
  // tile on AVX-512 CPUs other than AMD Zen 4 (48 rows of C, computed as C^T), which sizes up to 100
  // reach; 200 reaches its threaded path. Leading dimensions exceed the sizes, alpha is 1 or -0.5,
  // beta is 0, 1 or 0.7, each with an imaginary part for complex types. With beta = 0, C starts as
  // NaN, which must not be read; rows M..ldc-1 of C must not be written.
  std::printf("gemm no-transpose, random shapes :\n");
  {
    const auto check_random = [](auto zero, const int ntrial, const int maxdim) {
      using T = decltype(zero);
      using R = typename RealOf<T>::type;
      const auto value = [](const double re, const double im) { // im is dropped for real types
        if constexpr (std::is_same<T, R>::value) {
          return T(re);
        } else {
          return T(R(re), R(im));
        }
      };
      const auto random_value = [&value]() { // each part in [-0.5, 0.5]
        const double re = (double)std::rand() / RAND_MAX - 0.5;
        return value(re, (double)std::rand() / RAND_MAX - 0.5);
      };
      const auto abs_value = [](const T& x) {
        if constexpr (std::is_same<T, R>::value) {
          return sctl::fabs<T>(x);
        } else {
          return std::abs(x);
        }
      };
      const T pad = value(12345, 0);
      const T nan = value(std::numeric_limits<double>::quiet_NaN(), 0); // numeric_limits<QuadReal> has no NaN
      int nbad = 0;
      for (int trial = 0; trial < ntrial; ++trial) {
        const int M = 1 + std::rand() % maxdim, N = 1 + std::rand() % maxdim, K = 1 + std::rand() % maxdim;
        const int lda = M + std::rand() % 3, ldb = K + std::rand() % 3, ldc = M + std::rand() % 3;
        const T alpha = (trial % 2 ? value(1, 0) : value(-0.5, 0.75));
        const T beta = (trial % 3 == 0 ? value(0, 0) : (trial % 3 == 1 ? value(1, 0) : value(0.7, -0.3)));
        std::vector<T> A((size_t)lda * K), B((size_t)ldb * N), C((size_t)ldc * N);
        for (auto& x : A) x = random_value();
        for (auto& x : B) x = random_value();
        for (int n = 0; n < N; ++n) {
          for (int m = 0; m < ldc; ++m) C[m + (size_t)ldc * n] = (m >= M ? pad : (beta == T(0) ? nan : random_value()));
        }
        const std::vector<T> C0 = C;
        sctl::mat::gemm<T>('N', 'N', M, N, K, alpha, sctl::Ptr2ConstItr<T>(A.data(), (Long)A.size()), lda,
                           sctl::Ptr2ConstItr<T>(B.data(), (Long)B.size()), ldb, beta, sctl::Ptr2Itr<T>(C.data(), (Long)C.size()), ldc);
        bool ok = true;
        for (int n = 0; n < N; ++n) {
          for (int m = 0; m < ldc; ++m) {
            const T c = C[m + (size_t)ldc * n];
            if (m >= M) {
              ok = ok && (c == pad);
              continue;
            }
            T ref = 0;
            R mag = 0;
            for (int k = 0; k < K; ++k) {
              ref += A[m + (size_t)lda * k] * B[k + (size_t)ldb * n];
              mag += abs_value(A[m + (size_t)lda * k] * B[k + (size_t)ldb * n]);
            }
            ref = alpha * ref + (beta == T(0) ? T(0) : beta * C0[m + (size_t)ldc * n]);
            mag = abs_value(alpha) * mag + (beta == T(0) ? R(0) : abs_value(beta * C0[m + (size_t)ldc * n]));
            ok = ok && (c == c) && abs_value(c - ref) <= R(4 * K + 8) * sctl::machine_eps<R>() * mag;
          }
        }
        if (!ok && !nbad++) std::printf("  first failure: M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n", M, N, K, lda, ldb, ldc);
      }
      return nbad;
    };
    std::srand(1);
    CHECK(check_random(float(0), 300, 40) == 0);
    CHECK(check_random(float(0), 30, 100) == 0);
    CHECK(check_random(double(0), 300, 40) == 0);
    CHECK(check_random((long double)0, 100, 40) == 0);
#ifdef SCTL_QUAD_T
    CHECK(check_random(sctl::QuadReal(0), 30, 16) == 0);
#endif
    CHECK(check_random(double(0), 4, 200) == 0);
    CHECK(check_random(std::complex<float>(0), 300, 40) == 0);
    CHECK(check_random(std::complex<double>(0), 300, 40) == 0);
    CHECK(check_random(std::complex<long double>(0), 100, 40) == 0);
    CHECK(check_random(std::complex<double>(0), 4, 200) == 0);
  }

  // --- pinv: on a diagonal matrix, pinv is the reciprocal-diagonal. ---
  // Layout-agnostic for diagonal matrices.
  std::printf("pinv (diagonal) :\n");
  {
    double A[4] = {3, 0,
                   0, 5};
    double Ap[4] = {0};
    sctl::mat::pinv<double>(sctl::Ptr2Itr<double>(A, 4),
                            /*n1=*/2, /*n2=*/2, /*eps=*/-1.0,
                            sctl::Ptr2Itr<double>(Ap, 4));
    CHECK(test_utils::approx_eq(Ap[0], 1.0 / 3.0, 1e-9));
    CHECK(test_utils::approx_eq(Ap[1], 0.0,        1e-9));
    CHECK(test_utils::approx_eq(Ap[2], 0.0,        1e-9));
    CHECK(test_utils::approx_eq(Ap[3], 1.0 / 5.0, 1e-9));
  }

  // --- svd: singular values of a diagonal matrix are |diagonal entries|, sorted ---
  std::printf("svd (diagonal) :\n");
  {
    // A = diag(5, 3, 1) (3x3 column-major). LAPACK SVD should produce S = (5, 3, 1).
    int M = 3, N = 3;
    double A[9]  = {5, 0, 0,   0, 3, 0,   0, 0, 1};  // col-major diag
    double S[3]  = {0};
    double U[9]  = {0};
    double VT[9] = {0};
    int LDA = M, LDU = M, LDVT = 3;
    int LWORK = -1;
    double wkopt = 0;
    int INFO = 0;
    char jobu = 'S', jobvt = 'S';
    sctl::mat::svd<double>(&jobu, &jobvt, &M, &N,
                           sctl::Ptr2Itr<double>(A,  9), &LDA,
                           sctl::Ptr2Itr<double>(S,  3),
                           sctl::Ptr2Itr<double>(U,  9), &LDU,
                           sctl::Ptr2Itr<double>(VT, 9), &LDVT,
                           sctl::Ptr2Itr<double>(&wkopt, 1), &LWORK, &INFO);
    LWORK = (int)wkopt;
    std::vector<double> WORK(LWORK);
    sctl::mat::svd<double>(&jobu, &jobvt, &M, &N,
                           sctl::Ptr2Itr<double>(A,  9), &LDA,
                           sctl::Ptr2Itr<double>(S,  3),
                           sctl::Ptr2Itr<double>(U,  9), &LDU,
                           sctl::Ptr2Itr<double>(VT, 9), &LDVT,
                           sctl::Ptr2Itr<double>(WORK.data(), LWORK), &LWORK, &INFO);
    CHECK(INFO == 0);
    CHECK(test_utils::approx_eq(S[0], 5.0, 1e-9));
    CHECK(test_utils::approx_eq(S[1], 3.0, 1e-9));
    CHECK(test_utils::approx_eq(S[2], 1.0, 1e-9));
  }

  TEST_SUMMARY_RETURN();
}
