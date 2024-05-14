#ifndef _SCTL_MATRIX_HPP_
#define _SCTL_MATRIX_HPP_

#include <cstdint>
#include <cstdlib>

#include <sctl/common.hpp>
#include SCTL_INCLUDE(mem_mgr.hpp)

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;
template <class ValueType> class Permutation;

/**
 * @brief Class representing a matrix.
 *
 * @tparam ValueType The type of elements stored in the matrix.
 */
template <class ValueType> class Matrix {
 public:
  typedef ValueType value_type; ///< Type of elements stored in the matrix.
  typedef ValueType& reference; ///< Reference to a value in the matrix.
  typedef const ValueType& const_reference; ///< Const reference to a value in the matrix.
  typedef Iterator<ValueType> iterator; ///< Iterator over the elements of the matrix.
  typedef ConstIterator<ValueType> const_iterator; ///< Const iterator over the elements of the matrix.
  typedef Long difference_type; ///< Type representing the difference between two iterators.
  typedef Long size_type; ///< Type representing the size of the matrix.

  /**
   * @brief Default constructor. Constructs an empty matrix.
   */
  Matrix();

  /**
   * @brief Constructor to create a matrix with specified dimensions and optional initial data.
   *
   * @param dim1 Number of rows in the matrix.
   * @param dim2 Number of columns in the matrix.
   * @param data_ Pointer to the initial data (optional).
   * @param own_data_ Flag indicating ownership of the data (optional).
   */
  Matrix(Long dim1, Long dim2, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  /**
   * @brief Copy constructor.
   *
   * @param M Matrix to be copied.
   */
  Matrix(const Matrix<ValueType>& M);

  /**
   * @brief Destructor.
   */
  ~Matrix();

  /**
   * @brief Swaps the contents of two matrices.
   *
   * @param M Matrix to be swapped with.
   */
  void Swap(Matrix<ValueType>& M);

  /**
   * @brief Reinitializes the matrix with new dimensions and optional initial data.
   *
   * @param dim1 New number of rows.
   * @param dim2 New number of columns.
   * @param data_ Pointer to the initial data (optional).
   * @param own_data_ Flag indicating ownership of the data (optional).
   */
  void ReInit(Long dim1, Long dim2, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  /**
   * @brief Writes the matrix to a file.
   *
   * @param fname Filename to write the matrix data.
   */
  void Write(const char* fname) const;

  /**
   * @brief Writes the matrix to a file with specified type.
   *
   * @tparam Type Type of data to write.
   * @param fname Filename to write the matrix data.
   */
  template <class Type> void Write(const char* fname) const;

  /**
   * @brief Reads the matrix data from a file.
   *
   * @param fname Filename from which to read the matrix data.
   */
  void Read(const char* fname);

  /**
   * @brief Reads the matrix data from a file with specified type.
   *
   * @tparam Type Type of data to read.
   * @param fname Filename from which to read the matrix data.
   */
  template <class Type> void Read(const char* fname);

  /**
   * @brief Returns the size of the matrix along the specified dimension.
   *
   * @param i Dimension index (0 for rows, 1 for columns).
   * @return Size of the matrix along the specified dimension.
   */
  Long Dim(Long i) const;

  /**
   * @brief Sets all elements of the matrix to zero.
   */
  void SetZero();

  /**
   * @brief Returns an iterator to the beginning of the matrix.
   *
   * @return Iterator to the beginning of the matrix.
   */
  Iterator<ValueType> begin();

  /**
   * @brief Returns a const iterator to the beginning of the matrix.
   *
   * @return Const iterator to the beginning of the matrix.
   */
  ConstIterator<ValueType> begin() const;

  /**
   * @brief Returns an iterator to the end of the matrix.
   *
   * @return Iterator to the end of the matrix.
   */
  Iterator<ValueType> end();

  /**
   * @brief Returns a const iterator to the end of the matrix.
   *
   * @return Const iterator to the end of the matrix.
   */
  ConstIterator<ValueType> end() const;

  // Matrix-Matrix operations

  /**
   * @brief Assigns the contents of another matrix to this matrix.
   *
   * @param M Matrix to be assigned.
   * @return Reference to this matrix after assignment.
   */
  Matrix<ValueType>& operator=(const Matrix<ValueType>& M);

  /**
   * @brief Adds another matrix to this matrix element-wise.
   *
   * @param M Matrix to be added.
   * @return Reference to this matrix after addition.
   */
  Matrix<ValueType>& operator+=(const Matrix<ValueType>& M);

  /**
   * @brief Subtracts another matrix from this matrix element-wise.
   *
   * @param M Matrix to be subtracted.
   * @return Reference to this matrix after subtraction.
   */
  Matrix<ValueType>& operator-=(const Matrix<ValueType>& M);

  /**
   * @brief Adds another matrix to this matrix element-wise and returns the result.
   *
   * @param M2 Matrix to be added.
   * @return New matrix resulting from the addition.
   */
  Matrix<ValueType> operator+(const Matrix<ValueType>& M2) const;

  /**
   * @brief Subtracts another matrix from this matrix element-wise and returns the result.
   *
   * @param M2 Matrix to be subtracted.
   * @return New matrix resulting from the subtraction.
   */
  Matrix<ValueType> operator-(const Matrix<ValueType>& M2) const;

  /**
   * @brief Multiplies this matrix with another matrix.
   *
   * @param M Matrix to be multiplied with.
   * @return New matrix resulting from the multiplication.
   */
  Matrix<ValueType> operator*(const Matrix<ValueType>& M) const;

  /**
   * @brief Computes the matrix-matrix multiplication M_r = alpha * A * B + beta * M_r.
   *
   * @param M_r Result matrix.
   * @param A First matrix.
   * @param B Second matrix.
   * @param beta Coefficient for the existing values of M_r (default is 0.0).
   */
  static void GEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& A, const Matrix<ValueType>& B, ValueType beta = 0.0);

  /**
   * @brief Computes the matrix-matrix multiplication M_r = alpha * P * M + beta * M_r.
   *
   * @param M_r Result matrix.
   * @param P Permutation matrix.
   * @param M Matrix.
   * @param beta Coefficient for the existing values of M_r (default is 0.0).
   */
  static void GEMM(Matrix<ValueType>& M_r, const Permutation<ValueType>& P, const Matrix<ValueType>& M, ValueType beta = 0.0);

  /**
   * @brief Computes the matrix-matrix multiplication M_r = alpha * M * P + beta * M_r.
   *
   * @param M_r Result matrix.
   * @param M Matrix.
   * @param P Permutation matrix.
   * @param beta Coefficient for the existing values of M_r (default is 0.0).
   */
  static void GEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& M, const Permutation<ValueType>& P, ValueType beta = 0.0);

  // Matrix-Scalar operations

  /**
   * @brief Assigns a scalar value to all elements of the matrix.
   *
   * @param s Scalar value to be assigned.
   * @return Reference to this matrix after assignment.
   */
  Matrix<ValueType>& operator=(ValueType s);

  /**
   * @brief Adds a scalar value to each element of the matrix.
   *
   * @param s The scalar value to add.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator+=(ValueType s);

  /**
   * @brief Subtracts a scalar value from each element of the matrix.
   *
   * @param s The scalar value to subtract.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator-=(ValueType s);

  /**
   * @brief Multiplies each element of the matrix by a scalar value.
   *
   * @param s The scalar value to multiply by.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator*=(ValueType s);

  /**
   * @brief Divides each element of the matrix by a scalar value.
   *
   * @param s The scalar value to divide by.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator/=(ValueType s);

  /**
   * @brief Adds a scalar value to each element of the matrix, returning a new matrix.
   *
   * @param s The scalar value to add.
   * @return A new matrix with the scalar added to each element.
   */
  Matrix<ValueType> operator+(ValueType s) const;

  /**
   * @brief Subtracts a scalar value from each element of the matrix, returning a new matrix.
   *
   * @param s The scalar value to subtract.
   * @return A new matrix with the scalar subtracted from each element.
   */
  Matrix<ValueType> operator-(ValueType s) const;

  /**
   * @brief Multiplies each element of the matrix by a scalar value, returning a new matrix.
   *
   * @param s The scalar value to multiply by.
   * @return A new matrix with each element multiplied by the scalar.
   */
  Matrix<ValueType> operator*(ValueType s) const;

  /**
   * @brief Divides each element of the matrix by a scalar value, returning a new matrix.
   *
   * @param s The scalar value to divide by.
   * @return A new matrix with each element divided by the scalar.
   */
  Matrix<ValueType> operator/(ValueType s) const;

  // Element access

  /**
   * @brief Provides mutable access to the element at the specified row and column.
   *
   * @param i The row index.
   * @param j The column index.
   * @return A reference to the element at the specified position.
   */
  ValueType& operator()(Long i, Long j);

  /**
   * @brief Provides constant access to the element at the specified row and column.
   *
   * @param i The row index.
   * @param j The column index.
   * @return A constant reference to the element at the specified position.
   */
  const ValueType& operator()(Long i, Long j) const;

  /**
   * @brief Provides mutable access to a row of the matrix.
   *
   * @param i The row index.
   * @return An iterator pointing to the beginning of the specified row.
   */
  Iterator<ValueType> operator[](Long i);

  /**
   * @brief Provides constant access to a row of the matrix.
   *
   * @param i The row index.
   * @return A constant iterator pointing to the beginning of the specified row.
   */
  ConstIterator<ValueType> operator[](Long i) const;

  /**
   * @brief Permutes the rows of the matrix according to the given permutation.
   *
   * @param P The permutation to apply to the rows.
   */
  void RowPerm(const Permutation<ValueType>& P);

  /**
   * @brief Permutes the columns of the matrix according to the given permutation.
   *
   * @param P The permutation to apply to the columns.
   */
  void ColPerm(const Permutation<ValueType>& P);

  /**
   * @brief Computes the transpose of the matrix.
   *
   * @return The transpose of the matrix.
   */
  Matrix<ValueType> Transpose() const;

  /**
   * @brief Computes the transpose of the given matrix and stores the result in another matrix.
   *
   * @param M_r The matrix to store the transpose in.
   * @param M The matrix to transpose.
   */
  static void Transpose(Matrix<ValueType>& M_r, const Matrix<ValueType>& M);

  /**
   * @brief Computes the Singular Value Decomposition (SVD) of the matrix.
   *
   * @param tU The matrix containing the left singular vectors.
   * @param tS The matrix containing the singular values.
   * @param tVT The matrix containing the right singular vectors.
   *
   * @note Original matrix is destroyed.
   */
  void SVD(Matrix<ValueType>& tU, Matrix<ValueType>& tS, Matrix<ValueType>& tVT);

  /**
   * @brief Computes the Moore-Penrose pseudo-inverse of the matrix.
   *
   * @param eps The tolerance value for singular values close to zero. Defaults to -1.
   * @return The pseudo-inverse of the matrix.
   *
   * @note Original matrix is destroyed.
   */
  Matrix<ValueType> pinv(ValueType eps = -1);

 private:
  void Init(Long dim1, Long dim2, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  StaticArray<Long, 2> dim; ///< Dimensions of the matrix.
  Iterator<ValueType> data_ptr; ///< Pointer to the data of the matrix.
  bool own_data; ///< Flag indicating ownership of the data.
};

/**
 * @brief Overloaded stream insertion operator to output the matrix to the specified output stream.
 *
 * @param output The output stream to write the matrix to.
 * @param M The matrix to output.
 * @return The output stream after writing the matrix.
 */
template <class ValueType> std::ostream& operator<<(std::ostream& output, const Matrix<ValueType>& M);

/**
 * @brief Overloaded addition operator to add a scalar value to each element of the matrix.
 *
 * @param s The scalar value to add.
 * @param M The matrix to add the scalar to.
 * @return The resulting matrix after adding the scalar.
 */
template <class ValueType> Matrix<ValueType> operator+(ValueType s, const Matrix<ValueType>& M) { return M + s; }

/**
 * @brief Overloaded subtraction operator to subtract a matrix from a scalar value.
 *
 * @param s The scalar value to subtract from.
 * @param M The matrix to subtract from the scalar.
 * @return The resulting matrix after subtracting the scalar.
 */
template <class ValueType> Matrix<ValueType> operator-(ValueType s, const Matrix<ValueType>& M) { return s + (M * -1.0); }

/**
 * @brief Overloaded multiplication operator to multiply a scalar value with each element of the matrix.
 *
 * @param s The scalar value to multiply.
 * @param M The matrix to multiply the scalar with.
 * @return The resulting matrix after multiplying the scalar.
 */
template <class ValueType> Matrix<ValueType> operator*(ValueType s, const Matrix<ValueType>& M) { return M * s; }

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

#include SCTL_INCLUDE(matrix.txx)

#endif  //_SCTL_MATRIX_HPP_
