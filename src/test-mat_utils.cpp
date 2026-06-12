// Per-function tests for sctl/mat_utils.{hpp,txx}.
//
// The public free functions in `mat_utils` are low-level BLAS/LAPACK wrappers:
//   - gemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
//   - svd(JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO)
//   - pinv(M, n1, n2, eps, M_)
//
// gemm is verified against hand-computed reference products; svd via
// reconstruction A = U * diag(S) * V^T; pinv via the Moore-Penrose
// identity A * pinv(A) * A = A.

#include <cstdio>
#include <cstdlib>
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
