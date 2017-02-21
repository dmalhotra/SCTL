#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <pvfmm/mat_utils.hpp>
#include <pvfmm/mem_mgr.hpp>
#include <pvfmm/profile.hpp>

namespace pvfmm {

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Matrix<ValueType>& M) {
  std::ios::fmtflags f(std::cout.flags());
  output << std::fixed << std::setprecision(4) << std::setiosflags(std::ios::left);
  for (Long i = 0; i < M.Dim(0); i++) {
    for (Long j = 0; j < M.Dim(1); j++) {
      float f = ((float)M(i, j));
      if (pvfmm::fabs<ValueType>(f) < 1e-25) f = 0;
      output << std::setw(10) << ((double)f) << ' ';
    }
    output << ";\n";
  }
  std::cout.flags(f);
  return output;
}

template <class ValueType> Matrix<ValueType>::Matrix() {
  dim[0] = 0;
  dim[1] = 0;
  own_data = true;
  data_ptr = NULL;
}

template <class ValueType> Matrix<ValueType>::Matrix(Long dim1, Long dim2, Iterator<ValueType> data_, bool own_data_) {
  dim[0] = dim1;
  dim[1] = dim2;
  own_data = own_data_;
  if (own_data) {
    if (dim[0] * dim[1] > 0) {
      data_ptr = aligned_new<ValueType>(dim[0] * dim[1]);
      if (data_ != NULL) {
        memcopy(data_ptr, data_, dim[0] * dim[1]);
      }
    } else
      data_ptr = NULL;
  } else
    data_ptr = data_;
}

template <class ValueType> Matrix<ValueType>::Matrix(const Matrix<ValueType>& M) {
  dim[0] = M.dim[0];
  dim[1] = M.dim[1];
  own_data = true;
  if (dim[0] * dim[1] > 0) {
    data_ptr = aligned_new<ValueType>(dim[0] * dim[1]);
    memcopy(data_ptr, M.data_ptr, dim[0] * dim[1]);
  } else
    data_ptr = NULL;
}

template <class ValueType> Matrix<ValueType>::~Matrix() {
  if (own_data) {
    if (data_ptr != NULL) {
      aligned_delete(data_ptr);
    }
  }
  data_ptr = NULL;
  dim[0] = 0;
  dim[1] = 0;
}

template <class ValueType> void Matrix<ValueType>::Swap(Matrix<ValueType>& M) {
  StaticArray<Long, 2> dim_;
  dim_[0] = dim[0];
  dim_[1] = dim[1];
  Iterator<ValueType> data_ptr_ = data_ptr;
  bool own_data_ = own_data;

  dim[0] = M.dim[0];
  dim[1] = M.dim[1];
  data_ptr = M.data_ptr;
  own_data = M.own_data;

  M.dim[0] = dim_[0];
  M.dim[1] = dim_[1];
  M.data_ptr = data_ptr_;
  M.own_data = own_data_;
}

template <class ValueType> void Matrix<ValueType>::ReInit(Long dim1, Long dim2, Iterator<ValueType> data_, bool own_data_) {
  if (own_data_ && own_data && dim[0] * dim[1] >= dim1 * dim2) {
    dim[0] = dim1;
    dim[1] = dim2;
    if (data_ != NULL) {
      memcopy(data_ptr, data_, dim[0] * dim[1]);
    }
  } else {
    Matrix<ValueType> tmp(dim1, dim2, data_, own_data_);
    this->Swap(tmp);
  }
}

template <class ValueType> void Matrix<ValueType>::Write(const char* fname) const {
  FILE* f1 = fopen(fname, "wb+");
  if (f1 == NULL) {
    std::cout << "Unable to open file for writing:" << fname << '\n';
    return;
  }
  StaticArray<uint64_t, 2> dim_;
  dim_[0] = (uint64_t)dim[0];
  dim_[1] = (uint64_t)dim[1];
  fwrite(&dim_[0], sizeof(uint64_t), 2, f1);
  fwrite(data_ptr, sizeof(ValueType), dim[0] * dim[1], f1);
  fclose(f1);
}

template <class ValueType> void Matrix<ValueType>::Read(const char* fname) {
  FILE* f1 = fopen(fname, "r");
  if (f1 == NULL) {
    std::cout << "Unable to open file for reading:" << fname << '\n';
    return;
  }
  StaticArray<uint64_t, 2> dim_;
  Long readlen = fread(&dim_[0], sizeof(uint64_t), 2, f1);
  assert(readlen == 2);

  ReInit(dim_[0], dim_[1]);
  readlen = fread(data_ptr, sizeof(ValueType), dim[0] * dim[1], f1);
  assert(readlen == dim[0] * dim[1]);
  fclose(f1);
}

template <class ValueType> Long Matrix<ValueType>::Dim(Long i) const { return dim[i]; }

template <class ValueType> void Matrix<ValueType>::SetZero() {
  if (dim[0] * dim[1]) pvfmm::memset(data_ptr, 0, dim[0] * dim[1]);
}

template <class ValueType> Iterator<ValueType> Matrix<ValueType>::Begin() { return data_ptr; }

template <class ValueType> ConstIterator<ValueType> Matrix<ValueType>::Begin() const { return data_ptr; }

template <class ValueType> Matrix<ValueType>& Matrix<ValueType>::operator=(const Matrix<ValueType>& M) {
  if (this != &M) {
    if (dim[0] * dim[1] < M.dim[0] * M.dim[1]) {
      ReInit(M.dim[0], M.dim[1]);
    }
    dim[0] = M.dim[0];
    dim[1] = M.dim[1];
    memcopy(data_ptr, M.data_ptr, dim[0] * dim[1]);
  }
  return *this;
}

template <class ValueType> Matrix<ValueType>& Matrix<ValueType>::operator+=(const Matrix<ValueType>& M) {
  assert(M.Dim(0) == Dim(0) && M.Dim(1) == Dim(1));
  Profile::Add_FLOP(dim[0] * dim[1]);

  for (Long i = 0; i < M.Dim(0) * M.Dim(1); i++) data_ptr[i] += M.data_ptr[i];
  return *this;
}

template <class ValueType> Matrix<ValueType>& Matrix<ValueType>::operator-=(const Matrix<ValueType>& M) {
  assert(M.Dim(0) == Dim(0) && M.Dim(1) == Dim(1));
  Profile::Add_FLOP(dim[0] * dim[1]);

  for (Long i = 0; i < M.Dim(0) * M.Dim(1); i++) data_ptr[i] -= M.data_ptr[i];
  return *this;
}

template <class ValueType> Matrix<ValueType> Matrix<ValueType>::operator+(const Matrix<ValueType>& M2) const {
  const Matrix<ValueType>& M1 = *this;
  assert(M2.Dim(0) == M1.Dim(0) && M2.Dim(1) == M1.Dim(1));
  Profile::Add_FLOP(dim[0] * dim[1]);

  Matrix<ValueType> M_r(M1.Dim(0), M1.Dim(1), NULL);
  for (Long i = 0; i < M1.Dim(0) * M1.Dim(1); i++) M_r[0][i] = M1[0][i] + M2[0][i];
  return M_r;
}

template <class ValueType> Matrix<ValueType> Matrix<ValueType>::operator-(const Matrix<ValueType>& M2) const {
  const Matrix<ValueType>& M1 = *this;
  assert(M2.Dim(0) == M1.Dim(0) && M2.Dim(1) == M1.Dim(1));
  Profile::Add_FLOP(dim[0] * dim[1]);

  Matrix<ValueType> M_r(M1.Dim(0), M1.Dim(1), NULL);
  for (Long i = 0; i < M1.Dim(0) * M1.Dim(1); i++) M_r[0][i] = M1[0][i] - M2[0][i];
  return M_r;
}

template <class ValueType> inline ValueType& Matrix<ValueType>::operator()(Long i, Long j) {
  assert(i < dim[0] && j < dim[1]);
  return data_ptr[i * dim[1] + j];
}

template <class ValueType> inline const ValueType& Matrix<ValueType>::operator()(Long i, Long j) const {
  assert(i < dim[0] && j < dim[1]);
  return data_ptr[i * dim[1] + j];
}

template <class ValueType> inline Iterator<ValueType> Matrix<ValueType>::operator[](Long i) {
  assert(i < dim[0]);
  return data_ptr + i * dim[1];
}

template <class ValueType> inline ConstIterator<ValueType> Matrix<ValueType>::operator[](Long i) const {
  assert(i < dim[0]);
  return (data_ptr + i * dim[1]);
}

template <class ValueType> Matrix<ValueType> Matrix<ValueType>::operator*(const Matrix<ValueType>& M) const {
  assert(dim[1] == M.dim[0]);
  Profile::Add_FLOP(2 * (((Long)dim[0]) * dim[1]) * M.dim[1]);

  Matrix<ValueType> M_r(dim[0], M.dim[1], NULL);
  if (M.Dim(0) * M.Dim(1) == 0 || this->Dim(0) * this->Dim(1) == 0) return M_r;
  mat::gemm<ValueType>('N', 'N', M.dim[1], dim[0], dim[1], 1.0, M.data_ptr, M.dim[1], data_ptr, dim[1], 0.0, M_r.data_ptr, M_r.dim[1]);
  return M_r;
}

template <class ValueType> void Matrix<ValueType>::GEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& A, const Matrix<ValueType>& B, ValueType beta) {
  assert(A.dim[1] == B.dim[0]);
  assert(M_r.dim[0] == A.dim[0]);
  assert(M_r.dim[1] == B.dim[1]);
  if (A.Dim(0) * A.Dim(1) == 0 || B.Dim(0) * B.Dim(1) == 0) return;
  Profile::Add_FLOP(2 * (((Long)A.dim[0]) * A.dim[1]) * B.dim[1]);
  mat::gemm<ValueType>('N', 'N', B.dim[1], A.dim[0], A.dim[1], 1.0, B.data_ptr, B.dim[1], A.data_ptr, A.dim[1], beta, M_r.data_ptr, M_r.dim[1]);
}

// cublasgemm wrapper
#if defined(PVFMM_HAVE_CUDA)
template <class ValueType> void Matrix<ValueType>::CUBLASGEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& A, const Matrix<ValueType>& B, ValueType beta) {
  if (A.Dim(0) * A.Dim(1) == 0 || B.Dim(0) * B.Dim(1) == 0) return;
  assert(A.dim[1] == B.dim[0]);
  assert(M_r.dim[0] == A.dim[0]);
  assert(M_r.dim[1] == B.dim[1]);
  Profile::Add_FLOP(2 * (((Long)A.dim[0]) * A.dim[1]) * B.dim[1]);
  mat::cublasgemm('N', 'N', B.dim[1], A.dim[0], A.dim[1], (ValueType)1.0, B.data_ptr, B.dim[1], A.data_ptr, A.dim[1], beta, M_r.data_ptr, M_r.dim[1]);
}
#endif

#define myswap(t, a, b) \
  {                     \
    t c = a;            \
    a = b;              \
    b = c;              \
  }
template <class ValueType> void Matrix<ValueType>::RowPerm(const Permutation<ValueType>& P) {
  Matrix<ValueType>& M = *this;
  if (P.Dim() == 0) return;
  assert(M.Dim(0) == P.Dim());
  Long d0 = M.Dim(0);
  Long d1 = M.Dim(1);

#pragma omp parallel for
  for (Long i = 0; i < d0; i++) {
    Iterator<ValueType> M_ = M[i];
    const ValueType s = P.scal[i];
    for (Long j = 0; j < d1; j++) M_[j] *= s;
  }

  Permutation<ValueType> P_ = P;
  for (Long i = 0; i < d0; i++)
    while (P_.perm[i] != i) {
      Long a = P_.perm[i];
      Long b = i;
      Iterator<ValueType> M_a = M[a];
      Iterator<ValueType> M_b = M[b];
      myswap(Long, P_.perm[a], P_.perm[b]);
      for (Long j = 0; j < d1; j++) myswap(ValueType, M_a[j], M_b[j]);
    }
}

template <class ValueType> void Matrix<ValueType>::ColPerm(const Permutation<ValueType>& P) {
  Matrix<ValueType>& M = *this;
  if (P.Dim() == 0) return;
  assert(M.Dim(1) == P.Dim());
  Long d0 = M.Dim(0);
  Long d1 = M.Dim(1);

  Integer omp_p = omp_get_max_threads();
  Matrix<ValueType> M_buff(omp_p, d1);

  ConstIterator<Long> perm_ = P.perm.Begin();
  ConstIterator<ValueType> scal_ = P.scal.Begin();
#pragma omp parallel for
  for (Long i = 0; i < d0; i++) {
    Integer pid = omp_get_thread_num();
    Iterator<ValueType> buff = M_buff[pid];
    Iterator<ValueType> M_ = M[i];
    for (Long j = 0; j < d1; j++) buff[j] = M_[j];
    for (Long j = 0; j < d1; j++) {
      M_[j] = buff[perm_[j]] * scal_[j];
    }
  }
}
#undef myswap

#define B1 128
#define B2 32
template <class ValueType> Matrix<ValueType> Matrix<ValueType>::Transpose() const {
  const Matrix<ValueType>& M = *this;
  Long d0 = M.dim[0];
  Long d1 = M.dim[1];
  Matrix<ValueType> M_r(d1, d0);

  const Long blk0 = ((d0 + B1 - 1) / B1);
  const Long blk1 = ((d1 + B1 - 1) / B1);
  const Long blks = blk0 * blk1;
#pragma omp parallel for
  for (Long k = 0; k < blks; k++) {
    Long i = (k % blk0) * B1;
    Long j = (k / blk0) * B1;
    Long d0_ = i + B1;
    if (d0_ >= d0) d0_ = d0;
    Long d1_ = j + B1;
    if (d1_ >= d1) d1_ = d1;
    for (Long ii = i; ii < d0_; ii += B2)
      for (Long jj = j; jj < d1_; jj += B2) {
        Long d0__ = ii + B2;
        if (d0__ >= d0) d0__ = d0;
        Long d1__ = jj + B2;
        if (d1__ >= d1) d1__ = d1;
        for (Long iii = ii; iii < d0__; iii++)
          for (Long jjj = jj; jjj < d1__; jjj++) {
            M_r[jjj][iii] = M[iii][jjj];
          }
      }
  }
  return M_r;
}

template <class ValueType> void Matrix<ValueType>::Transpose(Matrix<ValueType>& M_r, const Matrix<ValueType>& M) {
  Long d0 = M.dim[0];
  Long d1 = M.dim[1];
  if (M_r.dim[0] != d1 || M_r.dim[1] != d0) M_r.ReInit(d1, d0);

  const Long blk0 = ((d0 + B1 - 1) / B1);
  const Long blk1 = ((d1 + B1 - 1) / B1);
  const Long blks = blk0 * blk1;
#pragma omp parallel for
  for (Long k = 0; k < blks; k++) {
    Long i = (k % blk0) * B1;
    Long j = (k / blk0) * B1;
    Long d0_ = i + B1;
    if (d0_ >= d0) d0_ = d0;
    Long d1_ = j + B1;
    if (d1_ >= d1) d1_ = d1;
    for (Long ii = i; ii < d0_; ii += B2)
      for (Long jj = j; jj < d1_; jj += B2) {
        Long d0__ = ii + B2;
        if (d0__ >= d0) d0__ = d0;
        Long d1__ = jj + B2;
        if (d1__ >= d1) d1__ = d1;
        for (Long iii = ii; iii < d0__; iii++)
          for (Long jjj = jj; jjj < d1__; jjj++) {
            M_r[jjj][iii] = M[iii][jjj];
          }
      }
  }
}
#undef B2
#undef B1

template <class ValueType> void Matrix<ValueType>::SVD(Matrix<ValueType>& tU, Matrix<ValueType>& tS, Matrix<ValueType>& tVT) {
  pvfmm::Matrix<ValueType>& M = *this;
  pvfmm::Matrix<ValueType> M_ = M;
  int n = M.Dim(0);
  int m = M.Dim(1);

  int k = (m < n ? m : n);
  if (tU.Dim(0) != n || tU.Dim(1) != k) tU.ReInit(n, k);
  tU.SetZero();
  if (tS.Dim(0) != k || tS.Dim(1) != k) tS.ReInit(k, k);
  tS.SetZero();
  if (tVT.Dim(0) != k || tVT.Dim(1) != m) tVT.ReInit(k, m);
  tVT.SetZero();

  // SVD
  int INFO = 0;
  char JOBU = 'S';
  char JOBVT = 'S';

  int wssize = 3 * (m < n ? m : n) + (m > n ? m : n);
  int wssize1 = 5 * (m < n ? m : n);
  wssize = (wssize > wssize1 ? wssize : wssize1);

  Iterator<ValueType> wsbuf = aligned_new<ValueType>(wssize);
  pvfmm::mat::svd(&JOBU, &JOBVT, &m, &n, M.Begin(), &m, tS.Begin(), tVT.Begin(), &m, tU.Begin(), &k, wsbuf, &wssize, &INFO);
  aligned_delete<ValueType>(wsbuf);

  if (INFO != 0) std::cout << INFO << '\n';
  assert(INFO == 0);

  for (Long i = 1; i < k; i++) {
    tS[i][i] = tS[0][i];
    tS[0][i] = 0;
  }
  // std::cout<<tU*tS*tVT-M_<<'\n';
}

template <class ValueType> Matrix<ValueType> Matrix<ValueType>::pinv(ValueType eps) {
  if (eps < 0) {
    eps = 1.0;
    while (eps + (ValueType)1.0 > 1.0) eps *= 0.5;
    eps = pvfmm::sqrt<ValueType>(eps);
  }
  Matrix<ValueType> M_r(dim[1], dim[0]);
  mat::pinv(data_ptr, dim[0], dim[1], eps, M_r.data_ptr);
  this->ReInit(0, 0);
  return M_r;
}

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Permutation<ValueType>& P) {
  output << std::setprecision(4) << std::setiosflags(std::ios::left);
  Long size = P.perm.Dim();
  for (Long i = 0; i < size; i++) output << std::setw(10) << P.perm[i] << ' ';
  output << ";\n";
  for (Long i = 0; i < size; i++) output << std::setw(10) << P.scal[i] << ' ';
  output << ";\n";
  return output;
}

template <class ValueType> Permutation<ValueType>::Permutation(Long size) {
  perm.ReInit(size);
  scal.ReInit(size);
  for (Long i = 0; i < size; i++) {
    perm[i] = i;
    scal[i] = 1.0;
  }
}

template <class ValueType> Permutation<ValueType> Permutation<ValueType>::RandPerm(Long size) {
  Permutation<ValueType> P(size);
  for (Long i = 0; i < size; i++) {
    P.perm[i] = rand() % size;
    for (Long j = 0; j < i; j++)
      if (P.perm[i] == P.perm[j]) {
        i--;
        break;
      }
    P.scal[i] = ((ValueType)rand()) / RAND_MAX;
  }
  return P;
}

template <class ValueType> Matrix<ValueType> Permutation<ValueType>::GetMatrix() const {
  Long size = perm.Dim();
  Matrix<ValueType> M_r(size, size, NULL);
  for (Long i = 0; i < size; i++)
    for (Long j = 0; j < size; j++) M_r[i][j] = (perm[j] == i ? scal[j] : 0.0);
  return M_r;
}

template <class ValueType> Long Permutation<ValueType>::Dim() const { return perm.Dim(); }

template <class ValueType> Permutation<ValueType> Permutation<ValueType>::Transpose() {
  Long size = perm.Dim();
  Permutation<ValueType> P_r(size);

  Vector<Long>& perm_r = P_r.perm;
  Vector<ValueType>& scal_r = P_r.scal;
  for (Long i = 0; i < size; i++) {
    perm_r[perm[i]] = i;
    scal_r[perm[i]] = scal[i];
  }
  return P_r;
}

template <class ValueType> Permutation<ValueType> Permutation<ValueType>::operator*(const Permutation<ValueType>& P) {
  Long size = perm.Dim();
  assert(P.Dim() == size);

  Permutation<ValueType> P_r(size);
  Vector<Long>& perm_r = P_r.perm;
  Vector<ValueType>& scal_r = P_r.scal;
  for (Long i = 0; i < size; i++) {
    perm_r[i] = perm[P.perm[i]];
    scal_r[i] = scal[P.perm[i]] * P.scal[i];
  }
  return P_r;
}

template <class ValueType> Matrix<ValueType> Permutation<ValueType>::operator*(const Matrix<ValueType>& M) {
  if (Dim() == 0) return M;
  assert(M.Dim(0) == Dim());
  Long d0 = M.Dim(0);
  Long d1 = M.Dim(1);

  Matrix<ValueType> M_r(d0, d1, NULL);
  for (Long i = 0; i < d0; i++) {
    const ValueType s = scal[i];
    ConstIterator<ValueType> M_ = M[i];
    Iterator<ValueType> M_r_ = M_r[perm[i]];
    for (Long j = 0; j < d1; j++) M_r_[j] = M_[j] * s;
  }
  return M_r;
}

template <class ValueType> Matrix<ValueType> operator*(const Matrix<ValueType>& M, const Permutation<ValueType>& P) {
  if (P.Dim() == 0) return M;
  assert(M.Dim(1) == P.Dim());
  Long d0 = M.Dim(0);
  Long d1 = M.Dim(1);

  Matrix<ValueType> M_r(d0, d1, NULL);
  for (Long i = 0; i < d0; i++) {
    ConstIterator<Long> perm_ = P.perm.Begin();
    ConstIterator<ValueType> scal_ = P.scal.Begin();
    ConstIterator<ValueType> M_ = M[i];
    Iterator<ValueType> M_r_ = M_r[i];
    for (Long j = 0; j < d1; j++) M_r_[j] = M_[perm_[j]] * scal_[j];
  }
  return M_r;
}

}  // end namespace
