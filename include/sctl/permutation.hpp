#ifndef _SCTL_PERMUTATION_HPP_
#define _SCTL_PERMUTATION_HPP_

#include <ostream>          // for ostream

#include "sctl/common.hpp"  // for Long, sctl
#include "sctl/vector.hpp"  // for Vector

namespace sctl {

template <class ValueType> class Matrix;

/**
 * Represents a permutation matrix P=[e(p1)*s1 e(p2)*s2 ... e(pn)*sn].
 * Where e(k) is the kth unit vector, perm := [p1 p2 ... pn] is the permutation vector,
 * and scal := [s1 s2 ... sn] is the scaling vector.
 *
 * @tparam ValueType The value type of the permutation matrix elements.
 */
template <class ValueType> class Permutation {

 public:
  /**
   * Default constructor.
   */
  Permutation() {}

  /**
   * Constructs a permutation matrix with the given size.
   *
   * @param size The size of the permutation matrix.
   */
  explicit Permutation(Long size);

  /**
   * Generates a random permutation matrix of the given size.
   *
   * @param size The size of the permutation matrix.
   * @return A random permutation matrix.
   */
  static Permutation<ValueType> RandPerm(Long size);

  /**
   * Retrieves the permutation matrix as a regular matrix.
   *
   * @return The permutation matrix represented as a regular matrix.
   */
  Matrix<ValueType> GetMatrix() const;

  /**
   * Returns the dimension of the permutation matrix.
   *
   * @return The dimension of the permutation matrix.
   */
  Long Dim() const;

  /**
   * Computes the transpose of the permutation matrix.
   *
   * @return The transpose of the permutation matrix.
   */
  Permutation<ValueType> Transpose();

  /**
   * Multiplies the permutation matrix by a scalar value.
   *
   * @param s The scalar value to multiply by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType>& operator*=(ValueType s);

  /**
   * Divides the permutation matrix by a scalar value.
   *
   * @param s The scalar value to divide by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType>& operator/=(ValueType s);

  /**
   * Multiplies the permutation matrix by a scalar value.
   *
   * @param s The scalar value to multiply by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType> operator*(ValueType s) const;

  /**
   * Divides the permutation matrix by a scalar value.
   *
   * @param s The scalar value to divide by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType> operator/(ValueType s) const;

  /**
   * Multiplies two permutation matrices together.
   *
   * @param P The permutation matrix to multiply with.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType> operator*(const Permutation<ValueType>& P) const;

  /**
   * Multiplies a matrix by the permutation matrix.
   *
   * @param M The matrix to multiply with.
   * @return The resulting matrix after permutation.
   */
  Matrix<ValueType> operator*(const Matrix<ValueType>& M) const;

  /**
   * The permutation vector.
   */
  Vector<Long> perm;

  /**
   * The scaling vector.
   */
  Vector<ValueType> scal;
};

/**
 * Overloaded multiplication operator to multiply a scalar value with each element of the permutation matrix.
 *
 * @param s The scalar value to multiply.
 * @param P The permutation matrix.
 * @return The resulting permutation matrix.
 */
template <class ValueType> Permutation<ValueType> operator*(ValueType s, const Permutation<ValueType>& P) { return P * s; }

/**
 * Multiplies a matrix by the permutation matrix.
 *
 * @param M The matrix to multiply with.
 * @param P The permutation matrix.
 * @return The resulting matrix after permutation.
 */
template <class ValueType> Matrix<ValueType> operator*(const Matrix<ValueType>& M, const Permutation<ValueType>& P);

/**
 * Overloaded stream insertion operator to output the permutation matrix to the specified output stream.
 *
 * @param output The output stream to write the permutation matrix to.
 * @param P The permutation matrix to output.
 * @return The output stream after writing the permutation matrix.
 */
template <class ValueType> std::ostream& operator<<(std::ostream& output, const Permutation<ValueType>& P);

}  // end namespace

#endif // _SCTL_PERMUTATION_HPP_
