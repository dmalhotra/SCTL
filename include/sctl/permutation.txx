#ifndef _SCTL_PERMUTATION_TXX_
#define _SCTL_PERMUTATION_TXX_

#include <ios>                   // for ios
#include <ostream>               // for ostream
#include <stdlib.h>              // for rand, RAND_MAX
#include <iomanip>               // for operator<<, setw, setiosflags, setpr...
#include <ostream>               // for basic_ostream, char_traits, operator<<

#include "sctl/common.hpp"       // for Long, SCTL_ASSERT, SCTL_NAMESPACE
#include SCTL_INCLUDE(permutation.hpp)  // for Permutation, operator*, operator<<
#include SCTL_INCLUDE(iterator.hpp)     // for ConstIterator, Iterator
#include SCTL_INCLUDE(iterator.txx)     // for ConstIterator::operator[]
#include SCTL_INCLUDE(matrix.hpp)       // for Matrix
#include SCTL_INCLUDE(vector.hpp)       // for Vector
#include SCTL_INCLUDE(vector.txx)       // for Vector::operator[], Vector::Dim, Vec...

namespace SCTL_NAMESPACE {

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
  Matrix<ValueType> M_r(size, size);
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

template <class ValueType> Permutation<ValueType>& Permutation<ValueType>::operator*=(ValueType s) {
  for (Long i = 0; i < scal.Dim(); i++) scal[i] *= s;
  return *this;
}

template <class ValueType> Permutation<ValueType>& Permutation<ValueType>::operator/=(ValueType s) {
  for (Long i = 0; i < scal.Dim(); i++) scal[i] /= s;
  return *this;
}

template <class ValueType> Permutation<ValueType> Permutation<ValueType>::operator*(ValueType s) const {
  Permutation<ValueType> P = *this;
  P *= s;
  return P;
}

template <class ValueType> Permutation<ValueType> Permutation<ValueType>::operator/(ValueType s) const {
  Permutation<ValueType> P = *this;
  P /= s;
  return P;
}

template <class ValueType> Permutation<ValueType> Permutation<ValueType>::operator*(const Permutation<ValueType>& P) const {
  Long size = perm.Dim();
  SCTL_ASSERT(P.Dim() == size);

  Permutation<ValueType> P_r(size);
  Vector<Long>& perm_r = P_r.perm;
  Vector<ValueType>& scal_r = P_r.scal;
  for (Long i = 0; i < size; i++) {
    perm_r[i] = perm[P.perm[i]];
    scal_r[i] = scal[P.perm[i]] * P.scal[i];
  }
  return P_r;
}

template <class ValueType> Matrix<ValueType> Permutation<ValueType>::operator*(const Matrix<ValueType>& M) const {
  if (Dim() == 0) return M;
  SCTL_ASSERT(M.Dim(0) == Dim());
  Long d0 = M.Dim(0);
  Long d1 = M.Dim(1);

  Matrix<ValueType> M_r(d0, d1);
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
  SCTL_ASSERT(M.Dim(1) == P.Dim());
  Long d0 = M.Dim(0);
  Long d1 = M.Dim(1);

  Matrix<ValueType> M_r(d0, d1);
  for (Long i = 0; i < d0; i++) {
    ConstIterator<Long> perm_ = P.perm.begin();
    ConstIterator<ValueType> scal_ = P.scal.begin();
    ConstIterator<ValueType> M_ = M[i];
    Iterator<ValueType> M_r_ = M_r[i];
    for (Long j = 0; j < d1; j++) M_r_[j] = M_[perm_[j]] * scal_[j];
  }
  return M_r;
}

}  // end namespace

#endif // _SCTL_PERMUTATION_TXX_
