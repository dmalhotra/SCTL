#ifndef _SCTL_PERMUTATION_HPP_
#define _SCTL_PERMUTATION_HPP_

#include <cstdint>
#include <cstdlib>

#include <sctl/common.hpp>
#include SCTL_INCLUDE(vector.hpp)

namespace SCTL_NAMESPACE {

template <class ValueType> class Matrix;

/**
 * @brief Represents a permutation matrix P=[e(p1)*s1 e(p2)*s2 ... e(pn)*sn].
 *
 * Where e(k) is the kth unit vector, perm := [p1 p2 ... pn] is the permutation vector,
 * and scal := [s1 s2 ... sn] is the scaling vector.
 *
 * @tparam ValueType The value type of the permutation matrix elements.
 */
template <class ValueType> class Permutation {

 public:
  /**
   * @brief Default constructor.
   */
  Permutation() {}

  /**
   * @brief Constructs a permutation matrix with the given size.
   *
   * @param size The size of the permutation matrix.
   */
  explicit Permutation(Long size);

  /**
   * @brief Generates a random permutation matrix of the given size.
   *
   * @param size The size of the permutation matrix.
   * @return A random permutation matrix.
   */
  static Permutation<ValueType> RandPerm(Long size);

  /**
   * @brief Retrieves the permutation matrix as a regular matrix.
   *
   * @return The permutation matrix represented as a regular matrix.
   */
  Matrix<ValueType> GetMatrix() const;

  /**
   * @brief Returns the dimension of the permutation matrix.
   *
   * @return The dimension of the permutation matrix.
   */
  Long Dim() const;

  /**
   * @brief Computes the transpose of the permutation matrix.
   *
   * @return The transpose of the permutation matrix.
   */
  Permutation<ValueType> Transpose();

  /**
   * @brief Multiplies the permutation matrix by a scalar value.
   *
   * @param s The scalar value to multiply by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType>& operator*=(ValueType s);

  /**
   * @brief Divides the permutation matrix by a scalar value.
   *
   * @param s The scalar value to divide by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType>& operator/=(ValueType s);

  /**
   * @brief Multiplies the permutation matrix by a scalar value.
   *
   * @param s The scalar value to multiply by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType> operator*(ValueType s) const;

  /**
   * @brief Divides the permutation matrix by a scalar value.
   *
   * @param s The scalar value to divide by.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType> operator/(ValueType s) const;

  /**
   * @brief Multiplies two permutation matrices together.
   *
   * @param P The permutation matrix to multiply with.
   * @return The resulting permutation matrix.
   */
  Permutation<ValueType> operator*(const Permutation<ValueType>& P) const;

  /**
   * @brief Multiplies a matrix by the permutation matrix.
   *
   * @param M The matrix to multiply with.
   * @return The resulting matrix after permutation.
   */
  Matrix<ValueType> operator*(const Matrix<ValueType>& M) const;

  /**
   * @brief The permutation vector.
   */
  Vector<Long> perm;

  /**
   * @brief The scaling vector.
   */
  Vector<ValueType> scal;
};

/**
 * @brief Overloaded multiplication operator to multiply a scalar value with each element of the permutation matrix.
 *
 * @param s The scalar value to multiply.
 * @param P The permutation matrix.
 * @return The resulting permutation matrix.
 */
template <class ValueType> Permutation<ValueType> operator*(ValueType s, const Permutation<ValueType>& P) { return P * s; }

/**
 * @brief Multiplies a matrix by the permutation matrix.
 *
 * @param M The matrix to multiply with.
 * @param P The permutation matrix.
 * @return The resulting matrix after permutation.
 */
template <class ValueType> Matrix<ValueType> operator*(const Matrix<ValueType>& M, const Permutation<ValueType>& P);

/**
 * @brief Overloaded stream insertion operator to output the permutation matrix to the specified output stream.
 *
 * @param output The output stream to write the permutation matrix to.
 * @param P The permutation matrix to output.
 * @return The output stream after writing the permutation matrix.
 */
template <class ValueType> std::ostream& operator<<(std::ostream& output, const Permutation<ValueType>& P);

}  // end namespace

#include SCTL_INCLUDE(permutation.txx)

#endif  //_SCTL_PERMUTATION_HPP_
