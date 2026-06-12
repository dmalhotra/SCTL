// Per-function tests for sctl/matrix.{hpp,txx}.
//
// Covers every public member of sctl::Matrix<T>: constructors, Swap, ReInit,
// Dim, SetZero, copy/move assignment, elementwise +/- and scalar
// arithmetic, matrix multiplication (operator* and GEMM), element access
// (operator() and operator[]), Transpose, SVD, pinv, Write/Read.

#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/matrix.hpp"
#include "sctl/matrix.txx"
#include "sctl/permutation.hpp"
#include "sctl/permutation.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Matrix;
using sctl::Permutation;

template <class T> static inline bool mat_approx(const Matrix<T>& A, const Matrix<T>& B, T tol = T(1e-10)) {
  if (A.Dim(0) != B.Dim(0) || A.Dim(1) != B.Dim(1)) return false;
  for (Long i = 0; i < A.Dim(0); ++i)
    for (Long j = 0; j < A.Dim(1); ++j)
      if (!test_utils::approx_eq(A(i, j), B(i, j), tol)) return false;
  return true;
}

int main() {
  // --- default ctor ---
  std::printf("default ctor :\n");
  {
    Matrix<double> M;
    CHECK(M.Dim(0) == 0 && M.Dim(1) == 0);
  }

  // --- sized ctor + Dim + operator() ---
  std::printf("sized ctor / Dim / operator() :\n");
  {
    Matrix<double> M(3, 4);
    CHECK(M.Dim(0) == 3 && M.Dim(1) == 4);
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 4; ++j)
        M(i, j) = (double)(i * 10 + j);
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 4; ++j)
        CHECK(M(i, j) == (double)(i * 10 + j));
  }

  // --- operator[] (row pointer) ---
  std::printf("operator[] :\n");
  {
    Matrix<int> M(2, 3);
    M(0, 0) = 1; M(0, 1) = 2; M(0, 2) = 3;
    M(1, 0) = 4; M(1, 1) = 5; M(1, 2) = 6;
    CHECK(M[0][0] == 1 && M[0][2] == 3);
    CHECK(M[1][0] == 4 && M[1][2] == 6);
    M[1][1] = 50;
    CHECK(M(1, 1) == 50);
  }

  // --- copy + move ctor / assignment ---
  std::printf("copy / move :\n");
  {
    Matrix<int> A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Matrix<int> B(A);
    Matrix<int> C;
    C = A;
    CHECK(B(0,1) == 2 && C(1,0) == 3);
    A(0,0) = 99;
    CHECK(B(0,0) == 1 && C(0,0) == 1);  // deep

    Matrix<int> D(std::move(A));
    CHECK(D(0,0) == 99);
    CHECK(A.Dim(0) == 0 && A.Dim(1) == 0);  // moved-from empty
  }

  // --- Swap ---
  std::printf("Swap :\n");
  {
    Matrix<int> A(2, 2); A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Matrix<int> B(1, 3); B(0,0)=10; B(0,1)=20; B(0,2)=30;
    A.Swap(B);
    CHECK(A.Dim(0)==1 && A.Dim(1)==3 && A(0,2)==30);
    CHECK(B.Dim(0)==2 && B.Dim(1)==2 && B(1,1)==4);
  }

  // --- ReInit ---
  std::printf("ReInit :\n");
  {
    Matrix<double> M(2, 2);
    M.ReInit(5, 6);
    CHECK(M.Dim(0) == 5 && M.Dim(1) == 6);
  }

  // --- SetZero ---
  std::printf("SetZero :\n");
  {
    Matrix<double> M(3, 3);
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 3; ++j) M(i, j) = 1.0;
    M.SetZero();
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 3; ++j) CHECK(M(i, j) == 0.0);
  }

  // --- matrix + - matrix ---
  std::printf("elementwise + - :\n");
  {
    Matrix<int> A(2,2); A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Matrix<int> B(2,2); B(0,0)=10;B(0,1)=10;B(1,0)=10;B(1,1)=10;
    Matrix<int> S = A + B;
    Matrix<int> D = B - A;
    CHECK(S(0,0)==11 && S(1,1)==14);
    CHECK(D(0,0)== 9 && D(1,1)== 6);

    Matrix<int> C = A;
    C += B; CHECK(C(0,0)==11 && C(1,1)==14);
    C -= B; CHECK(C(0,0)==1  && C(1,1)==4);
  }

  // --- matrix * matrix ---
  std::printf("matrix * matrix :\n");
  {
    // [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50]
    Matrix<double> A(2,2); A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Matrix<double> B(2,2); B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;
    Matrix<double> C = A * B;
    CHECK(C.Dim(0) == 2 && C.Dim(1) == 2);
    CHECK(C(0,0) == 19 && C(0,1) == 22);
    CHECK(C(1,0) == 43 && C(1,1) == 50);
  }

  // --- GEMM (matrix * matrix into output) ---
  std::printf("GEMM :\n");
  {
    Matrix<double> A(2,3); for (Long i = 0; i < 6; ++i) A[0][i] = (double)i;
    Matrix<double> B(3,2); for (Long i = 0; i < 6; ++i) B[0][i] = (double)i;
    Matrix<double> C(2,2); C.SetZero();
    Matrix<double>::GEMM(C, A, B);  // beta = 0 (default)
    Matrix<double> R = A * B;
    CHECK(mat_approx(C, R));
    // accumulate: C := A*B + 1.0*C  (so call again with beta=1)
    Matrix<double>::GEMM(C, A, B, 1.0);
    Matrix<double> R2 = R + R;
    CHECK(mat_approx(C, R2));
  }

  // --- scalar ops ---
  std::printf("scalar ops :\n");
  {
    Matrix<double> A(2,2); A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Matrix<double> B = A * 2.0;
    Matrix<double> C = A + 1.0;
    CHECK(B(0,0)==2 && B(1,1)==8);
    CHECK(C(0,0)==2 && C(1,1)==5);
    A *= 3.0; CHECK(A(0,0)==3 && A(1,1)==12);
    A /= 3.0; CHECK(A(0,0)==1 && A(1,1)==4);
    A = 7.0;  for (Long i=0;i<2;++i) for (Long j=0;j<2;++j) CHECK(A(i,j)==7.0);

    // free-fn  scalar OP M
    Matrix<double> M(2,2); M(0,0)=1; M(0,1)=2; M(1,0)=3; M(1,1)=4;
    Matrix<double> P = 10.0 + M;
    CHECK(P(0,0)==11 && P(1,1)==14);
  }

  // --- Transpose (static) ---
  std::printf("Transpose :\n");
  {
    Matrix<double> A(2,3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    Matrix<double> T;
    Matrix<double>::Transpose(T, A);
    CHECK(T.Dim(0) == 3 && T.Dim(1) == 2);
    for (Long i = 0; i < 2; ++i)
      for (Long j = 0; j < 3; ++j)
        CHECK(T(j, i) == A(i, j));
    // transpose-of-transpose = identity
    Matrix<double> TT;
    Matrix<double>::Transpose(TT, T);
    CHECK(mat_approx(TT, A));
  }

  // --- RowPerm / ColPerm ---
  std::printf("RowPerm / ColPerm :\n");
  {
    Matrix<int> A(3,3);
    int k = 1;
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 3; ++j) A(i,j) = k++;
    // identity permutation should be a no-op
    Permutation<int> P(3);
    // RandPerm-free identity: by default Permutation<T>(n) is identity?
    // Construct identity explicitly:
    for (Long i = 0; i < 3; ++i) { P.perm[i] = (Long)i; P.scal[i] = 1; }
    Matrix<int> A1 = A;
    A1.RowPerm(P);
    A1.ColPerm(P);
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 3; ++j) CHECK(A1(i,j) == A(i,j));
  }

  // --- SVD: A == U * S * V^T residual ---
  std::printf("SVD :\n");
  {
    // 3x2 matrix with known structure
    Matrix<double> A(3, 2);
    A(0,0)=1; A(0,1)=0;
    A(1,0)=0; A(1,1)=2;
    A(2,0)=3; A(2,1)=0;

    Matrix<double> U, S, VT;
    Matrix<double> Acopy = A;
    Acopy.SVD(U, S, VT);
    // S returned as a matrix of singular values along the diagonal.
    // Reconstruct A = U * S * V^T and compare.
    Matrix<double> tmp = U * S;
    Matrix<double> R = tmp * VT;
    CHECK(mat_approx(R, A, 1e-9));
  }

  // --- pinv: A * pinv(A) * A == A ---
  std::printf("pinv :\n");
  {
    Matrix<double> A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    Matrix<double> Acopy = A;
    Matrix<double> Ap = Acopy.pinv();
    CHECK(Ap.Dim(0) == 3 && Ap.Dim(1) == 2);
    Matrix<double> AAp  = A * Ap;
    Matrix<double> AApA = AAp * A;
    CHECK(mat_approx(AApA, A, 1e-9));
  }

  // --- operator<< (smoke) ---
  std::printf("operator<< :\n");
  {
    Matrix<int> A(2,2); A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    std::ostringstream os;
    os << A;
    CHECK(!os.str().empty());
  }

  // --- Write / Read ---
  std::printf("Write / Read :\n");
  {
    Matrix<double> A(2, 3);
    for (Long i = 0; i < 2; ++i)
      for (Long j = 0; j < 3; ++j) A(i, j) = (double)(i + j * 7);
    const char* fname = "/tmp/sctl-test-matrix.bin";
    A.Write(fname);
    Matrix<double> B;
    B.Read(fname);
    CHECK(mat_approx(A, B));
    std::remove(fname);
  }

  TEST_SUMMARY_RETURN();
}
