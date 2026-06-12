// Per-function tests for sctl/permutation.{hpp,txx}.
//
// Covers every public member of sctl::Permutation<T>: constructors, RandPerm,
// GetMatrix, Dim, Transpose, scalar arithmetic, composition (P * Q),
// permutation applied to Matrix (P * M, M * P), operator<<.

#include <cstdio>
#include <set>
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

int main() {
  // --- default ctor ---
  std::printf("default ctor :\n");
  {
    Permutation<double> P;
    CHECK(P.Dim() == 0);
  }

  // --- sized ctor : identity permutation, unit scaling ---
  std::printf("sized ctor (identity) :\n");
  {
    Permutation<int> P(5);
    CHECK(P.Dim() == 5);
    for (Long i = 0; i < 5; ++i) {
      CHECK(P.perm[i] == i);
      CHECK(P.scal[i] == 1);
    }
  }

  // --- RandPerm : permutation property ---
  std::printf("RandPerm :\n");
  {
    Permutation<double> P = Permutation<double>::RandPerm(8);
    CHECK(P.Dim() == 8);
    std::set<Long> seen;
    for (Long i = 0; i < 8; ++i) {
      CHECK(P.perm[i] >= 0 && P.perm[i] < 8);
      seen.insert(P.perm[i]);
    }
    CHECK((Long)seen.size() == 8);  // every index in [0,8) appears exactly once
  }

  // --- GetMatrix : recovers an n x n matrix ---
  std::printf("GetMatrix :\n");
  {
    Permutation<double> P(3);
    P.perm[0] = 2; P.perm[1] = 0; P.perm[2] = 1;
    P.scal[0] = 1; P.scal[1] = 2; P.scal[2] = -1;
    Matrix<double> M = P.GetMatrix();
    CHECK(M.Dim(0) == 3 && M.Dim(1) == 3);
    // Column-major layout: P(i, j) = scal[j] if perm[j] == i, else 0
    // (matches the doc: P = [e(p_1)*s_1 ... e(p_n)*s_n]).
    for (Long i = 0; i < 3; ++i) {
      for (Long j = 0; j < 3; ++j) {
        const double expected = (P.perm[j] == i) ? P.scal[j] : 0.0;
        CHECK(M(i, j) == expected);
      }
    }
  }

  // --- Transpose : inverse of an identity-scaled permutation ---
  // For perm-only (scal == 1) permutations, P^T = P^-1; i.e. P * P^T = I.
  std::printf("Transpose :\n");
  {
    Permutation<double> P = Permutation<double>::RandPerm(6);
    Permutation<double> PT = P.Transpose();
    Permutation<double> PPT = P * PT;  // should be identity in perm
    for (Long i = 0; i < 6; ++i) {
      CHECK(PPT.perm[i] == i);
    }
  }

  // --- scalar *= /= : scales scal[] entries ---
  std::printf("scalar ops :\n");
  {
    Permutation<double> P(3);
    P.scal[0] = 1; P.scal[1] = 2; P.scal[2] = 3;
    P *= 10.0;
    CHECK(P.scal[0] == 10 && P.scal[1] == 20 && P.scal[2] == 30);
    P /= 2.0;
    CHECK(P.scal[0] == 5  && P.scal[1] == 10 && P.scal[2] == 15);
    Permutation<double> Q = P * 2.0;
    CHECK(Q.scal[0] == 10 && Q.scal[1] == 20 && Q.scal[2] == 30);
    Permutation<double> R = P / 5.0;
    CHECK(R.scal[0] == 1  && R.scal[1] == 2  && R.scal[2] == 3);
  }

  // --- composition P * Q : matches matrix product of their matrix forms ---
  std::printf("composition P*Q :\n");
  {
    Permutation<double> P(4); P.perm[0]=1; P.perm[1]=0; P.perm[2]=3; P.perm[3]=2;
    Permutation<double> Q(4); Q.perm[0]=2; Q.perm[1]=3; Q.perm[2]=0; Q.perm[3]=1;
    // Set non-unit scaling too.
    P.scal[0]=2; P.scal[1]=3; P.scal[2]=5; P.scal[3]=7;
    Q.scal[0]=1; Q.scal[1]=1; Q.scal[2]=1; Q.scal[3]=1;

    Permutation<double> PQ = P * Q;
    // Compare to matrix product.
    Matrix<double> Pm = P.GetMatrix();
    Matrix<double> Qm = Q.GetMatrix();
    Matrix<double> Rm = Pm * Qm;
    Matrix<double> PQm = PQ.GetMatrix();
    for (Long i = 0; i < 4; ++i)
      for (Long j = 0; j < 4; ++j)
        CHECK(test_utils::approx_eq(PQm(i,j), Rm(i,j), 1e-12));
  }

  // --- P * M and M * P : agree with explicit matrix multiplication ---
  std::printf("P * M and M * P :\n");
  {
    Permutation<double> P(3); P.perm[0]=2; P.perm[1]=0; P.perm[2]=1; P.scal[0]=1; P.scal[1]=2; P.scal[2]=-3;
    Matrix<double> M(3, 4);
    int k = 1;
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 4; ++j) M(i, j) = (double)(k++);
    Matrix<double> PM = P * M;
    Matrix<double> R = P.GetMatrix() * M;
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 4; ++j)
        CHECK(test_utils::approx_eq(PM(i,j), R(i,j), 1e-12));

    // M * Q (matching shapes) -- M is 3x4, Q must be 4x4
    Permutation<double> Q(4); Q.perm[0]=3; Q.perm[1]=2; Q.perm[2]=1; Q.perm[3]=0;
    Q.scal[0]=1; Q.scal[1]=2; Q.scal[2]=3; Q.scal[3]=4;
    Matrix<double> MQ = M * Q;
    Matrix<double> R2 = M * Q.GetMatrix();
    for (Long i = 0; i < 3; ++i)
      for (Long j = 0; j < 4; ++j)
        CHECK(test_utils::approx_eq(MQ(i,j), R2(i,j), 1e-12));
  }

  // --- operator<< (smoke) ---
  std::printf("operator<< :\n");
  {
    Permutation<int> P(3);
    std::ostringstream os;
    os << P;
    CHECK(!os.str().empty());
  }

  TEST_SUMMARY_RETURN();
}
