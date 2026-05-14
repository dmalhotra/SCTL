#ifndef _SCTL_MATRIX_HPP_
#define _SCTL_MATRIX_HPP_

#include <ostream>                // for ostream

#include "sctl/common.hpp"        // for Long, sctl
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator
#include "sctl/iterator.txx"      // for NullIterator
#include "sctl/static-array.hpp"  // for StaticArray

namespace sctl {

template <class ValueType> class Permutation;

/**
 * Class representing a matrix. The data is stored in row-major order.  It can optionally make use
 * the **BLAS** and **LAPACK** libraries when available by defining the macros `SCTL_HAVE_BLAS` and
 * `SCTL_HAVE_LAPACK` respectively.
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
   * Default constructor. Constructs an empty matrix.
   */
  Matrix();

  /**
   * Constructor to create a matrix with specified dimensions and optional initial data.
   *
   * @param dim1 Number of rows in the matrix.
   * @param dim2 Number of columns in the matrix.
   * @param data_ Pointer to the initial data (optional).
   * @param own_data_ Flag indicating ownership of the data (optional).
   */
  Matrix(Long dim1, Long dim2, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  /**
   * Copy constructor.
   *
   * @param M Matrix to be copied.
   */
  Matrix(const Matrix<ValueType>& M);

  /**
   * Move constructor. Steals ownership from `M`; leaves `M` empty
   * (`Dim(0) == 0`, `Dim(1) == 0`) and in a valid destructible state.
   */
  Matrix(Matrix<ValueType>&& M) noexcept;

  /**
   * Destructor.
   */
  ~Matrix();

  /**
   * Swap the contents of two matrices. O(1) — no elements are copied. Ownership
   * travels with the buffer, so it is safe to swap an owning matrix with a
   * non-owning view.
   *
   * @param M Matrix to be swapped with.
   */
  void Swap(Matrix<ValueType>& M);

  /**
   * Reinitializes the matrix with new dimensions and optional initial data.
   *
   * @param dim1 New number of rows.
   * @param dim2 New number of columns.
   * @param data_ Pointer to the initial data (optional).
   * @param own_data_ Flag indicating ownership of the data (optional).
   */
  void ReInit(Long dim1, Long dim2, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  /**
   * Writes the matrix to a file.
   *
   * @param fname Filename to write the matrix data.
   */
  void Write(const char* fname) const;

  /**
   * Writes the matrix to a file with specified type.
   *
   * @tparam Type Type of data to write.
   * @param fname Filename to write the matrix data.
   */
  template <class Type> void Write(const char* fname) const;

  /**
   * Reads the matrix data from a file.
   *
   * @param fname Filename from which to read the matrix data.
   */
  void Read(const char* fname);

  /**
   * Reads the matrix data from a file with specified type.
   *
   * @tparam Type Type of data to read.
   * @param fname Filename from which to read the matrix data.
   */
  template <class Type> void Read(const char* fname);

  /**
   * Returns the size of the matrix along the specified dimension.
   *
   * @param i Dimension index (0 for rows, 1 for columns).
   * @return Size of the matrix along the specified dimension.
   */
  [[nodiscard]] Long Dim(Long i) const noexcept;

  /**
   * Sets all elements of the matrix to zero.
   */
  void SetZero();

  /**
   * Returns an iterator to the beginning of the matrix.
   *
   * @return Iterator to the beginning of the matrix.
   */
  [[nodiscard]] Iterator<ValueType> begin();

  /**
   * Returns a const iterator to the beginning of the matrix.
   *
   * @return Const iterator to the beginning of the matrix.
   */
  [[nodiscard]] ConstIterator<ValueType> begin() const;

  /**
   * Returns an iterator to the end of the matrix.
   *
   * @return Iterator to the end of the matrix.
   */
  [[nodiscard]] Iterator<ValueType> end();

  /**
   * Returns a const iterator to the end of the matrix.
   *
   * @return Const iterator to the end of the matrix.
   */
  [[nodiscard]] ConstIterator<ValueType> end() const;

  // Matrix-Matrix operations

  /**
   * Assigns the contents of another matrix to this matrix.
   *
   * @param M Matrix to be assigned.
   * @return Reference to this matrix after assignment.
   */
  Matrix<ValueType>& operator=(const Matrix<ValueType>& M);

  /**
   * Move assignment. Swaps state with `M` when both sides own their buffers;
   * otherwise copies `M`'s contents into `*this` (resizing as needed).
   *
   * @note If `*this` is a non-owning view and `M`'s shape differs, the view
   *       binding is lost — `*this` becomes an owning matrix with a fresh
   *       buffer. Same applies to copy-assignment.
   */
  Matrix<ValueType>& operator=(Matrix<ValueType>&& M) noexcept;

  /**
   * Adds another matrix to this matrix element-wise.
   *
   * @param M Matrix to be added.
   * @return Reference to this matrix after addition.
   */
  Matrix<ValueType>& operator+=(const Matrix<ValueType>& M);

  /**
   * Subtracts another matrix from this matrix element-wise.
   *
   * @param M Matrix to be subtracted.
   * @return Reference to this matrix after subtraction.
   */
  Matrix<ValueType>& operator-=(const Matrix<ValueType>& M);

  /**
   * Adds another matrix to this matrix element-wise and returns the result.
   *
   * @param M2 Matrix to be added.
   * @return New matrix resulting from the addition.
   */
  [[nodiscard]] Matrix<ValueType> operator+(const Matrix<ValueType>& M2) const;

  /**
   * Subtracts another matrix from this matrix element-wise and returns the result.
   *
   * @param M2 Matrix to be subtracted.
   * @return New matrix resulting from the subtraction.
   */
  [[nodiscard]] Matrix<ValueType> operator-(const Matrix<ValueType>& M2) const;

  /**
   * Multiplies this matrix with another matrix.
   *
   * @param M Matrix to be multiplied with.
   * @return New matrix resulting from the multiplication.
   */
  [[nodiscard]] Matrix<ValueType> operator*(const Matrix<ValueType>& M) const;

  /**
   * Computes the matrix-matrix multiplication M_r = A * B + beta * M_r.
   *
   * @param M_r Result matrix.
   * @param A First matrix.
   * @param B Second matrix.
   * @param beta Coefficient for the existing values of M_r (default is 0.0).
   */
  static void GEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& A, const Matrix<ValueType>& B, ValueType beta = 0.0);

  /**
   * Computes the matrix-matrix multiplication M_r = P * M + beta * M_r.
   *
   * @param M_r Result matrix.
   * @param P Permutation matrix.
   * @param M Matrix.
   * @param beta Coefficient for the existing values of M_r (default is 0.0).
   */
  static void GEMM(Matrix<ValueType>& M_r, const Permutation<ValueType>& P, const Matrix<ValueType>& M, ValueType beta = 0.0);

  /**
   * Computes the matrix-matrix multiplication M_r = M * P + beta * M_r.
   *
   * @param M_r Result matrix.
   * @param M Matrix.
   * @param P Permutation matrix.
   * @param beta Coefficient for the existing values of M_r (default is 0.0).
   */
  static void GEMM(Matrix<ValueType>& M_r, const Matrix<ValueType>& M, const Permutation<ValueType>& P, ValueType beta = 0.0);

  // Matrix-Scalar operations

  /**
   * Assigns a scalar value to all elements of the matrix.
   *
   * @param s Scalar value to be assigned.
   * @return Reference to this matrix after assignment.
   */
  Matrix<ValueType>& operator=(ValueType s);

  /**
   * Adds a scalar value to each element of the matrix.
   *
   * @param s The scalar value to add.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator+=(ValueType s);

  /**
   * Subtracts a scalar value from each element of the matrix.
   *
   * @param s The scalar value to subtract.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator-=(ValueType s);

  /**
   * Multiplies each element of the matrix by a scalar value.
   *
   * @param s The scalar value to multiply by.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator*=(ValueType s);

  /**
   * Divides each element of the matrix by a scalar value.
   *
   * @param s The scalar value to divide by.
   * @return A reference to the modified matrix.
   */
  Matrix<ValueType>& operator/=(ValueType s);

  /**
   * Adds a scalar value to each element of the matrix, returning a new matrix.
   *
   * @param s The scalar value to add.
   * @return A new matrix with the scalar added to each element.
   */
  [[nodiscard]] Matrix<ValueType> operator+(ValueType s) const;

  /**
   * Subtracts a scalar value from each element of the matrix, returning a new matrix.
   *
   * @param s The scalar value to subtract.
   * @return A new matrix with the scalar subtracted from each element.
   */
  [[nodiscard]] Matrix<ValueType> operator-(ValueType s) const;

  /**
   * Multiplies each element of the matrix by a scalar value, returning a new matrix.
   *
   * @param s The scalar value to multiply by.
   * @return A new matrix with each element multiplied by the scalar.
   */
  [[nodiscard]] Matrix<ValueType> operator*(ValueType s) const;

  /**
   * Divides each element of the matrix by a scalar value, returning a new matrix.
   *
   * @param s The scalar value to divide by.
   * @return A new matrix with each element divided by the scalar.
   */
  [[nodiscard]] Matrix<ValueType> operator/(ValueType s) const;

  // Element access

  /**
   * Provides mutable access to the element at the specified row and column.
   *
   * @param i The row index.
   * @param j The column index.
   * @return A reference to the element at the specified position.
   */
  ValueType& operator()(Long i, Long j);

  /**
   * Provides constant access to the element at the specified row and column.
   *
   * @param i The row index.
   * @param j The column index.
   * @return A constant reference to the element at the specified position.
   */
  const ValueType& operator()(Long i, Long j) const;

  /**
   * Provides mutable access to a row of the matrix.
   *
   * @param i The row index.
   * @return An iterator pointing to the beginning of the specified row.
   */
  Iterator<ValueType> operator[](Long i);

  /**
   * Provides constant access to a row of the matrix.
   *
   * @param i The row index.
   * @return A constant iterator pointing to the beginning of the specified row.
   */
  ConstIterator<ValueType> operator[](Long i) const;

  /**
   * Permutes the rows of the matrix according to the given permutation.
   *
   * @param P The permutation to apply to the rows.
   */
  void RowPerm(const Permutation<ValueType>& P);

  /**
   * Permutes the columns of the matrix according to the given permutation.
   *
   * @param P The permutation to apply to the columns.
   */
  void ColPerm(const Permutation<ValueType>& P);

  /**
   * Computes the transpose of the matrix.
   *
   * @return The transpose of the matrix.
   */
  [[nodiscard]] Matrix<ValueType> Transpose() const;

  /**
   * Computes the transpose of the given matrix and stores the result in another matrix.
   *
   * @param M_r The matrix to store the transpose in.
   * @param M The matrix to transpose.
   */
  static void Transpose(Matrix<ValueType>& M_r, const Matrix<ValueType>& M);

  /**
   * Computes the Singular Value Decomposition (SVD) of the matrix.
   *
   * @param tU The matrix containing the left singular vectors.
   * @param tS The matrix containing the singular values.
   * @param tVT The matrix containing the right singular vectors.
   *
   * @warning Original matrix is destroyed.
   */
  void SVD(Matrix<ValueType>& tU, Matrix<ValueType>& tS, Matrix<ValueType>& tVT);

  /**
   * Computes the Moore-Penrose pseudo-inverse of the matrix.
   *
   * @param eps Relative threshold on singular values: any `sigma_i` with
   * `sigma_i < eps * sigma_max` is treated as zero (its reciprocal is set to
   * zero rather than `1/sigma_i`). The default value of `-1` is a sentinel
   * for "auto-pick" and is replaced internally by `sqrt(machine_eps<ValueType>())`.
   * Pass an explicit non-negative value to override.
   * @return The pseudo-inverse of the matrix.
   *
   * @warning Original matrix is destroyed.
   */
  [[nodiscard]] Matrix<ValueType> pinv(ValueType eps = -1);

 private:
  void Init(Long dim1, Long dim2, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  StaticArray<Long, 2> dim; ///< Dimensions of the matrix.
  Long capacity; /**< Capacity of the matrix. */
  Iterator<ValueType> data_ptr; ///< Pointer to the data of the matrix.
  bool own_data; ///< Flag indicating ownership of the data.
};

/**
 * Overloaded stream insertion operator to output the matrix to the specified output stream.
 *
 * @param output The output stream to write the matrix to.
 * @param M The matrix to output.
 * @return The output stream after writing the matrix.
 */
template <class ValueType> std::ostream& operator<<(std::ostream& output, const Matrix<ValueType>& M);

/**
 * Overloaded addition operator to add a scalar value to each element of the matrix.
 *
 * @param s The scalar value to add.
 * @param M The matrix to add the scalar to.
 * @return The resulting matrix after adding the scalar.
 */
template <class ValueType> [[nodiscard]] Matrix<ValueType> operator+(ValueType s, const Matrix<ValueType>& M) { return M + s; }

/**
 * Overloaded subtraction operator to subtract a matrix from a scalar value.
 *
 * @param s The scalar value to subtract from.
 * @param M The matrix to subtract from the scalar.
 * @return The resulting matrix after subtracting the scalar.
 */
template <class ValueType> [[nodiscard]] Matrix<ValueType> operator-(ValueType s, const Matrix<ValueType>& M) { return s + (M * -1.0); }

/**
 * Overloaded multiplication operator to multiply a scalar value with each element of the matrix.
 *
 * @param s The scalar value to multiply.
 * @param M The matrix to multiply the scalar with.
 * @return The resulting matrix after multiplying the scalar.
 */
template <class ValueType> [[nodiscard]] Matrix<ValueType> operator*(ValueType s, const Matrix<ValueType>& M) { return M * s; }

}  // end namespace

#endif // _SCTL_MATRIX_HPP_
