#ifndef _PVFMM_MATRIX_HPP_
#define _PVFMM_MATRIX_HPP_

#include <cstdint>
#include <cstdlib>

#include <pvfmm/common.hpp>

namespace pvfmm {

template <class ValueType> class Vector;
template <class ValueType> class Permutation;

template <class ValueType> class Matrix {

 public:
  Matrix();

  Matrix(Long dim1, Long dim2, Iterator<ValueType> data_ = NULL, bool own_data_ = true);

  Matrix(const Matrix<ValueType>& M);

  ~Matrix();

  void Swap(Matrix<ValueType>& M);

  void ReInit(Long dim1, Long dim2, Iterator<ValueType> data_ = NULL, bool own_data_ = true);

  void Write(const char* fname) const;

  void Read(const char* fname);

  Long Dim(Long i) const;

  void SetZero();

  Iterator<ValueType> Begin();

  ConstIterator<ValueType> Begin() const;

  Matrix<ValueType>& operator=(const Matrix<ValueType>& M);

  Matrix<ValueType>& operator+=(const Matrix<ValueType>& M);

  Matrix<ValueType>& operator-=(const Matrix<ValueType>& M);

  Matrix<ValueType> operator+(const Matrix<ValueType>& M2) const;

  Matrix<ValueType> operator-(const Matrix<ValueType>& M2) const;

  ValueType& operator()(Long i, Long j);

  const ValueType& operator()(Long i, Long j) const;

  Iterator<ValueType> operator[](Long i);

  ConstIterator<ValueType> operator[](Long i) const;

  Matrix<ValueType> operator*(const Matrix<ValueType>& M) const;

  static void GEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& A, const Matrix<ValueType>& B, ValueType beta = 0.0);

  // cublasgemm wrapper
  static void CUBLASGEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& A, const Matrix<ValueType>& B, ValueType beta = 0.0);

  void RowPerm(const Permutation<ValueType>& P);
  void ColPerm(const Permutation<ValueType>& P);

  Matrix<ValueType> Transpose() const;

  static void Transpose(Matrix<ValueType>& M_r, const Matrix<ValueType>& M);

  // Original matrix is destroyed.
  void SVD(Matrix<ValueType>& tU, Matrix<ValueType>& tS, Matrix<ValueType>& tVT);

  // Original matrix is destroyed.
  Matrix<ValueType> pinv(ValueType eps = -1);

 private:
  StaticArray<Long, 2> dim;
  Iterator<ValueType> data_ptr;
  bool own_data;
};

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Matrix<ValueType>& M);

/**
 * /brief P=[e(p1)*s1 e(p2)*s2 ... e(pn)*sn],
 * where e(k) is the kth unit vector,
 * perm := [p1 p2 ... pn] is the permutation vector,
 * scal := [s1 s2 ... sn] is the scaling vector.
 */
template <class ValueType> class Permutation {

 public:
  Permutation() {}

  Permutation(Long size);

  static Permutation<ValueType> RandPerm(Long size);

  Matrix<ValueType> GetMatrix() const;

  Long Dim() const;

  Permutation<ValueType> Transpose();

  Permutation<ValueType> operator*(const Permutation<ValueType>& P);

  Matrix<ValueType> operator*(const Matrix<ValueType>& M);

  Vector<Long> perm;
  Vector<ValueType> scal;
};

template <class ValueType> Matrix<ValueType> operator*(const Matrix<ValueType>& M, const Permutation<ValueType>& P);

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Permutation<ValueType>& P);

}  // end namespace

#include <pvfmm/matrix.txx>

#endif  //_PVFMM_MATRIX_HPP_
