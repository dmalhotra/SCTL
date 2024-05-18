#ifndef _SCTL_VECTOR_HPP_
#define _SCTL_VECTOR_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(mem_mgr.hpp)

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>

namespace SCTL_NAMESPACE {

// forward declaration
template <class ValueType> Iterator<ValueType> NullIterator();

/**
 * @brief A contiguous array of elements. The elements can be accesses with a non-negatvie index.  The vector can be the
 * owner of the memory allocated on the heap (automatically aligned to `SCTL_MEM_ALIGN` bytes for SIMD vectorization) or
 * it may be constructed from a user provided memory location (using Iterator<ValueType>).
 *
 * @tparam ValueType Type of the elements stored in the vector.
 */
template <class ValueType> class Vector {
 public:
  typedef ValueType value_type; /**< Type of the elements stored in the vector. */
  typedef ValueType& reference; /**< Reference to an element in the vector. */
  typedef const ValueType& const_reference; /**< Const reference to an element in the vector. */
  typedef Iterator<ValueType> iterator; /**< Iterator for traversing the vector. */
  typedef ConstIterator<ValueType> const_iterator; /**< Const iterator for traversing the vector. */
  typedef Long difference_type; /**< Integer type representing the difference between two iterators. */
  typedef Long size_type; /**< Integer type representing the size of the vector. */

  /**
   * @brief Default constructor.
   */
  Vector();

  /**
   * @brief Constructor with dimension and data pointer.
   *
   * @param dim Dimension of the vector.
   * @param data Pointer to the data.
   * @param own_data Flag indicating ownership of data.
   */
  explicit Vector(Long dim, Iterator<ValueType> data = NullIterator<ValueType>(), bool own_data = true);

  /**
   * @brief Copy constructor.
   *
   * @param V Another vector to copy from.
   */
  Vector(const Vector& V);

  /**
   * @brief Constructor from std::vector.
   *
   * @param V std::vector to construct from.
   */
  explicit Vector(const std::vector<ValueType>& V);

  /**
   * @brief Constructor from initializer list.
   *
   * @param V Initializer list to construct from.
   */
  explicit Vector(std::initializer_list<ValueType> V);

  /**
   * @brief Destructor.
   */
  ~Vector();

  /**
   * @brief Swap the contents of two vectors.
   *
   * @param v1 Vector to swap with.
   */
  void Swap(Vector<ValueType>& v1);

  /**
   * @brief Reinitialize the vector.
   *
   * @param dim New dimension of the vector.
   * @param data New data pointer.
   * @param own_data Flag indicating ownership of new data.
   */
  void ReInit(Long dim, Iterator<ValueType> data = NullIterator<ValueType>(), bool own_data = true);

  /**
   * @brief Write the vector to a file.
   *
   * @param fname File name to write to.
   */
  void Write(const char* fname) const;

  /**
   * @brief Write the vector to a file with a different data type.
   *
   * @tparam Type Type of data to write.
   * @param fname File name to write to.
   */
  template <class Type> void Write(const char* fname) const;

  /**
   * @brief Read the vector from a file.
   *
   * @param fname File name to read from.
   */
  void Read(const char* fname);

  /**
   * @brief Read the vector from a file with a different data type.
   *
   * @tparam Type Type of data to read.
   * @param fname File name to read from.
   */
  template <class Type> void Read(const char* fname);

  /**
   * @brief Get the dimension of the vector.
   *
   * @return Long Dimension of the vector.
   */
  Long Dim() const;

  //Long Capacity() const;

  /**
   * @brief Set all elements of the vector to zero.
   */
  void SetZero();

  /**
   * @brief Get an iterator pointing to the beginning of the vector.
   *
   * @return Iterator<ValueType> Iterator pointing to the beginning of the vector.
   */
  Iterator<ValueType> begin();

  /**
   * @brief Get a const iterator pointing to the beginning of the vector.
   *
   * @return ConstIterator<ValueType> Const iterator pointing to the beginning of the vector.
   */
  ConstIterator<ValueType> begin() const;

  /**
   * @brief Get an iterator pointing to the end of the vector.
   *
   * @return Iterator<ValueType> Iterator pointing to the end of the vector.
   */
  Iterator<ValueType> end();

  /**
   * @brief Get a const iterator pointing to the end of the vector.
   *
   * @return ConstIterator<ValueType> Const iterator pointing to the end of the vector.
   */
  ConstIterator<ValueType> end() const;

  /**
   * @brief Add an element to the end of the vector.
   *
   * @param x Element to be added.
   */
  void PushBack(const ValueType& x);

  // Element access

  /**
   * @brief Access an element of the vector.
   *
   * @param j Index of the element to access.
   * @return ValueType& Reference to the accessed element.
   */
  ValueType& operator[](Long j);

  /**
   * @brief Access a const element of the vector.
   *
   * @param j Index of the element to access.
   * @return const ValueType& Const reference to the accessed element.
   */
  const ValueType& operator[](Long j) const;

  // Vector-Vector operations

  /**
   * @brief Assignment operator from std::vector.
   *
   * @param V std::vector to assign from.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator=(const std::vector<ValueType>& V);

  /**
   * @brief Assignment operator.
   *
   * @param V Vector to assign from.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator=(const Vector& V);

  /**
   * @brief Addition assignment operator.
   *
   * @param V Vector to add.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator+=(const Vector& V);

  /**
   * @brief Subtraction assignment operator.
   *
   * @param V Vector to subtract.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator-=(const Vector& V);

  /**
   * @brief Multiplication assignment operator.
   *
   * @param V Vector to multiply.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator*=(const Vector& V);

  /**
   * @brief Division assignment operator.
   *
   * @param V Vector to divide.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator/=(const Vector& V);

  /**
   * @brief Addition operator.
   *
   * @param V Vector to add.
   * @return Vector Resultant vector after addition.
   */
  Vector operator+(const Vector& V) const;

  /**
   * @brief Subtraction operator.
   *
   * @param V Vector to subtract.
   * @return Vector Resultant vector after subtraction.
   */
  Vector operator-(const Vector& V) const;

  /**
   * @brief Multiplication operator.
   *
   * @param V Vector to multiply.
   * @return Vector Resultant vector after multiplication.
   */
  Vector operator*(const Vector& V) const;

  /**
   * @brief Division operator.
   *
   * @param V Vector to divide.
   * @return Vector Resultant vector after division.
   */
  Vector operator/(const Vector& V) const;

  /**
   * @brief Negation operator.
   *
   * @return Vector Negated vector.
   */
  Vector operator-() const ;

  // Vector-Scalar operations

  /**
   * @brief Assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to assign.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator=(VType s);

  /**
   * @brief Addition assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to add.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator+=(VType s);

  /**
   * @brief Subtraction assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to subtract.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator-=(VType s);

  /**
   * @brief Multiplication assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to multiply.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator*=(VType s);

  /**
   * @brief Division assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to divide.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator/=(VType s);

  /**
   * @brief Addition operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to add.
   * @return Vector Resultant vector after addition.
   */
  template <class VType> Vector operator+(VType s) const;

  /**
   * @brief Subtraction operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to subtract.
   * @return Vector Resultant vector after subtraction.
   */
  template <class VType> Vector operator-(VType s) const;

  /**
   * @brief Multiplication operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to multiply.
   * @return Vector Resultant vector after multiplication.
   */
  template <class VType> Vector operator*(VType s) const;

  /**
   * @brief Division operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to divide.
   * @return Vector Resultant vector after division.
   */
  template <class VType> Vector operator/(VType s) const;

 private:
  /**
   * @brief Initialize the vector.
   *
   * @param dim Dimension of the vector.
   * @param data Pointer to the data.
   * @param own_data Flag indicating ownership of data.
   */
  void Init(Long dim, Iterator<ValueType> data = NullIterator<ValueType>(), bool own_data = true);

  Long dim; /**< Dimension of the vector. */
  Long capacity; /**< Capacity of the vector. */
  Iterator<ValueType> data_ptr; /**< Pointer to the data. */
  bool own_data; /**< Flag indicating ownership of the data. */
};

// Function template declarations for vector-scalar operations...

template <class VType, class ValueType> Vector<ValueType> operator+(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> Vector<ValueType> operator-(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> Vector<ValueType> operator*(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> Vector<ValueType> operator/(VType s, const Vector<ValueType>& V);

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Vector<ValueType>& V);

}  // end namespace

#include SCTL_INCLUDE(vector.txx)

#endif  //_SCTL_VECTOR_HPP_
