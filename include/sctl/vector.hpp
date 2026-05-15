#ifndef _SCTL_VECTOR_HPP_
#define _SCTL_VECTOR_HPP_

#include <ostream>            // for ostream
#include <initializer_list>   // for initializer_list
#include <vector>             // for vector

#include "sctl/common.hpp"    // for Long, sctl
#include "sctl/iterator.hpp"  // for Iterator, ConstIterator

namespace sctl {

// forward declarations
template <class ValueType> Iterator<ValueType> NullIterator();
template <class ValueType> class ScratchBuf;

/**
 * A contiguous array of elements. The elements can be accesses with a non-negative index.  The vector can be the
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
   * Default constructor.
   */
  Vector();

  /**
   * Constructor with dimension and data pointer.
   *
   * @param dim Dimension of the vector.
   * @param data Pointer to the data.
   * @param own_data Flag indicating ownership of data.
   */
  explicit Vector(Long dim, Iterator<ValueType> data = NullIterator<ValueType>(), bool own_data = true);

  /**
   * Copy constructor.
   *
   * @param V Another vector to copy from.
   */
  Vector(const Vector& V);

  /**
   * Move constructor. Steals ownership from `V`; leaves `V` empty
   * (`Dim() == 0`) and in a valid destructible state.
   */
  Vector(Vector&& V) noexcept;

  /**
   * Constructor from std::vector.
   *
   * @param V std::vector to construct from.
   */
  explicit Vector(const std::vector<ValueType>& V);

  /**
   * Constructor from initializer list.
   *
   * @param V Initializer list to construct from.
   */
  explicit Vector(std::initializer_list<ValueType> V);

  /**
   * Construct a non-owning view of a ScratchBuf. The Vector aliases the
   * buffer's storage and does not free it; the ScratchBuf owns the lifetime.
   *
   * The ctor is `explicit` to prevent silent copies on assignment:
   * `existing = scratch_buf;` won't compile, because `operator=(Vector&&)`
   * cannot reach this ctor through copy-initialization. Use direct-init
   * (`Vector<T> v(buf);`) for a view.
   */
  explicit Vector(ScratchBuf<ValueType>& buf);

  /**
   * Destructor.
   */
  ~Vector();

  /**
   * Swap the contents of two vectors. O(1) — no elements are copied. Ownership
   * travels with the buffer, so it is safe to swap an owning vector with a
   * non-owning view.
   *
   * @param v1 Vector to swap with.
   */
  void Swap(Vector<ValueType>& v1);

  /**
   * Reinitialize the vector.
   *
   * @param dim New dimension of the vector.
   * @param data New data pointer.
   * @param own_data Flag indicating ownership of new data.
   */
  void ReInit(Long dim, Iterator<ValueType> data = NullIterator<ValueType>(), bool own_data = true);

  /**
   * Write the vector to a file.
   *
   * @param fname File name to write to.
   */
  void Write(const char* fname) const;

  /**
   * Write the vector to a file with a different data type.
   *
   * @tparam Type Type of data to write.
   * @param fname File name to write to.
   */
  template <class Type> void Write(const char* fname) const;

  /**
   * Read the vector from a file.
   *
   * @param fname File name to read from.
   */
  void Read(const char* fname);

  /**
   * Read the vector from a file with a different data type.
   *
   * @tparam Type Type of data to read.
   * @param fname File name to read from.
   */
  template <class Type> void Read(const char* fname);

  /**
   * Get the dimension of the vector.
   *
   * @return Long Dimension of the vector.
   */
  [[nodiscard]] Long Dim() const noexcept;

  /**
   * Set all elements of the vector to zero.
   */
  void SetZero();

  /**
   * Get an iterator pointing to the beginning of the vector.
   *
   * @return Iterator<ValueType> Iterator pointing to the beginning of the vector.
   */
  [[nodiscard]] Iterator<ValueType> begin();

  /**
   * Get a const iterator pointing to the beginning of the vector.
   *
   * @return ConstIterator<ValueType> Const iterator pointing to the beginning of the vector.
   */
  [[nodiscard]] ConstIterator<ValueType> begin() const;

  /**
   * Get an iterator pointing to the end of the vector.
   *
   * @return Iterator<ValueType> Iterator pointing to the end of the vector.
   */
  [[nodiscard]] Iterator<ValueType> end();

  /**
   * Get a const iterator pointing to the end of the vector.
   *
   * @return ConstIterator<ValueType> Const iterator pointing to the end of the vector.
   */
  [[nodiscard]] ConstIterator<ValueType> end() const;

  /**
   * Add an element to the end of the vector.
   *
   * @param x Element to be added.
   */
  void PushBack(const ValueType& x);

  // Element access

  /**
   * Access an element of the vector.
   *
   * @param j Index of the element to access.
   * @return ValueType& Reference to the accessed element.
   */
  ValueType& operator[](Long j);

  /**
   * Access a const element of the vector.
   *
   * @param j Index of the element to access.
   * @return const ValueType& Const reference to the accessed element.
   */
  const ValueType& operator[](Long j) const;

  // Vector-Vector operations

  /**
   * Assignment operator from std::vector.
   *
   * @param V std::vector to assign from.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator=(const std::vector<ValueType>& V);

  /**
   * Assignment operator.
   *
   * @param V Vector to assign from.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator=(const Vector& V);

  /**
   * Move assignment. Swaps state with `V` when both sides own their buffers;
   * otherwise copies `V`'s contents into `*this` (resizing as needed).
   *
   * @note If `*this` is a non-owning view and `V`'s size differs, the view
   *       binding is lost — `*this` becomes an owning vector with a fresh
   *       buffer. Same applies to copy-assignment.
   */
  Vector& operator=(Vector&& V) noexcept;

  /**
   * Addition assignment operator.
   *
   * @param V Vector to add.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator+=(const Vector& V);

  /**
   * Subtraction assignment operator.
   *
   * @param V Vector to subtract.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator-=(const Vector& V);

  /**
   * Multiplication assignment operator.
   *
   * @param V Vector to multiply.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator*=(const Vector& V);

  /**
   * Division assignment operator.
   *
   * @param V Vector to divide.
   * @return Vector& Reference to the modified vector.
   */
  Vector& operator/=(const Vector& V);

  /**
   * Addition operator.
   *
   * @param V Vector to add.
   * @return Vector Resultant vector after addition.
   */
  [[nodiscard]] Vector operator+(const Vector& V) const;

  /**
   * Subtraction operator.
   *
   * @param V Vector to subtract.
   * @return Vector Resultant vector after subtraction.
   */
  [[nodiscard]] Vector operator-(const Vector& V) const;

  /**
   * Multiplication operator.
   *
   * @param V Vector to multiply.
   * @return Vector Resultant vector after multiplication.
   */
  [[nodiscard]] Vector operator*(const Vector& V) const;

  /**
   * Division operator.
   *
   * @param V Vector to divide.
   * @return Vector Resultant vector after division.
   */
  [[nodiscard]] Vector operator/(const Vector& V) const;

  /**
   * Negation operator.
   *
   * @return Vector Negated vector.
   */
  [[nodiscard]] Vector operator-() const ;

  // Vector-Scalar operations

  /**
   * Assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to assign.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator=(VType s);

  /**
   * Addition assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to add.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator+=(VType s);

  /**
   * Subtraction assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to subtract.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator-=(VType s);

  /**
   * Multiplication assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to multiply.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator*=(VType s);

  /**
   * Division assignment operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to divide.
   * @return Vector& Reference to the modified vector.
   */
  template <class VType> Vector& operator/=(VType s);

  /**
   * Addition operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to add.
   * @return Vector Resultant vector after addition.
   */
  template <class VType> [[nodiscard]] Vector operator+(VType s) const;

  /**
   * Subtraction operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to subtract.
   * @return Vector Resultant vector after subtraction.
   */
  template <class VType> [[nodiscard]] Vector operator-(VType s) const;

  /**
   * Multiplication operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to multiply.
   * @return Vector Resultant vector after multiplication.
   */
  template <class VType> [[nodiscard]] Vector operator*(VType s) const;

  /**
   * Division operator with a scalar.
   *
   * @tparam VType Type of the scalar.
   * @param s Scalar value to divide.
   * @return Vector Resultant vector after division.
   */
  template <class VType> [[nodiscard]] Vector operator/(VType s) const;

 private:
  /**
   * Initialize the vector.
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

template <class VType, class ValueType> [[nodiscard]] Vector<ValueType> operator+(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> [[nodiscard]] Vector<ValueType> operator-(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> [[nodiscard]] Vector<ValueType> operator*(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> [[nodiscard]] Vector<ValueType> operator/(VType s, const Vector<ValueType>& V);

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Vector<ValueType>& V);

}  // end namespace

#endif // _SCTL_VECTOR_HPP_
