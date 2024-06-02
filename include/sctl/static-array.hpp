#ifndef _SCTL_STATIC_ARRAY_HPP_
#define _SCTL_STATIC_ARRAY_HPP_

#include <initializer_list>  // for initializer_list

#include "sctl/common.hpp"   // for Long, sctl

namespace sctl {

#ifdef SCTL_MEMDEBUG

/**
 * A fixed-size array class with additional functionalities for memory debugging.
 * This class provides a wrapper around a fixed-size array with custom iterators and
 * various operators to facilitate element access and manipulation. It is used when
 * `SCTL_MEMDEBUG` is defined. In a regular build, `StaticArray` is an alias for a
 * C-style fixed-size array.
 *
 * @tparam ValueType The type of elements stored in the array.
 * @tparam DIM The number of elements in the array.
 */
template <class ValueType, Long DIM> class StaticArray {
  typedef Long difference_type;

 public:
  /**
   * Default constructor.
   */
  StaticArray() = default;

  /**
   * Copy constructor.
   *
   * @param other The `StaticArray` to copy from.
   */
  StaticArray(const StaticArray&) = default;

  /**
   * Copy assignment operator.
   *
   * @param other The `StaticArray` to copy from.
   * @return A reference to the updated `StaticArray`.
   */
  StaticArray& operator=(const StaticArray&) = default;

  /**
   * Constructor with initializer list.
   *
   * @param arr_ An initializer list to initialize the array.
   */
  explicit StaticArray(std::initializer_list<ValueType> arr_);

  /**
   * Default destructor.
   */
  ~StaticArray() = default;

  /**
   * Dereference operator (const).
   *
   * @return A const reference to the element at the current position.
   */
  const ValueType& operator*() const;

  /**
   * Dereference operator.
   *
   * @return A reference to the element at the current position.
   */
  ValueType& operator*();

  /**
   * Member access operator (const).
   *
   * @return A const pointer to the element at the current position.
   */
  const ValueType* operator->() const;

  /**
   * Member access operator.
   *
   * @return A pointer to the element at the current position.
   */
  ValueType* operator->();

  /**
   * Subscript operator (const).
   *
   * @param off The offset from the beginning of the array.
   * @return A const reference to the element at the specified offset.
   */
  const ValueType& operator[](difference_type off) const;

  /**
   * Subscript operator.
   *
   * @param off The offset from the beginning of the array.
   * @return A reference to the element at the specified offset.
   */
  ValueType& operator[](difference_type off);

  /**
   * Conversion to constant iterator.
   *
   * @return A constant iterator pointing to the beginning of the array.
   */
  operator ConstIterator<ValueType>() const;

  /**
   * Conversion to iterator.
   *
   * @return An iterator pointing to the beginning of the array.
   */
  operator Iterator<ValueType>();

  /**
   * Addition operator (const).
   *
   * @param i The offset to add.
   * @return A constant iterator pointing to the new position.
   */
  ConstIterator<ValueType> operator+(difference_type i) const;

  /**
   * Addition operator.
   *
   * @param i The offset to add.
   * @return An iterator pointing to the new position.
   */
  Iterator<ValueType> operator+(difference_type i);

  /**
   * Addition operator for constant iterator.
   *
   * @param i The offset to add.
   * @param right The `StaticArray` to operate on.
   * @return A constant iterator pointing to the new position.
   */
  template <class T, Long d> friend ConstIterator<T> operator+(typename StaticArray<T,d>::difference_type i, const StaticArray<T,d>& right);

  /**
   * Addition operator for iterator.
   *
   * @param i The offset to add.
   * @param right The `StaticArray` to operate on.
   * @return An iterator pointing to the new position.
   */
  template <class T, Long d> friend Iterator<T> operator+(typename StaticArray<T,d>::difference_type i, StaticArray<T,d>& right);

  /**
   * Subtraction operator (const).
   *
   * @param i The offset to subtract.
   * @return A constant iterator pointing to the new position.
   */
  ConstIterator<ValueType> operator-(difference_type i) const;

  /**
   * Subtraction operator.
   *
   * @param i The offset to subtract.
   * @return An iterator pointing to the new position.
   */
  Iterator<ValueType> operator-(difference_type i);

  /**
   * Difference operator.
   *
   * @param I The constant iterator to subtract.
   * @return The difference between the current position and the iterator.
   */
  difference_type operator-(const ConstIterator<ValueType>& I) const;

  /**
   * Equality comparison operator.
   *
   * @param I The constant iterator to compare with.
   * @return `true` if the iterators are equal, `false` otherwise.
   */
  bool operator==(const ConstIterator<ValueType>& I) const;

  /**
   * Inequality comparison operator.
   *
   * @param I The constant iterator to compare with.
   * @return `true` if the iterators are not equal, `false` otherwise.
   */
  bool operator!=(const ConstIterator<ValueType>& I) const;

  /**
   * Less-than comparison operator.
   *
   * @param I The constant iterator to compare with.
   * @return `true` if the current iterator is less than the given iterator, `false` otherwise.
   */
  bool operator< (const ConstIterator<ValueType>& I) const;

  /**
   * Less-than-or-equal comparison operator.
   *
   * @param I The constant iterator to compare with.
   * @return `true` if the current iterator is less than or equal to the given iterator, `false` otherwise.
   */
  bool operator<=(const ConstIterator<ValueType>& I) const;

  /**
   * Greater-than comparison operator.
   *
   * @param I The constant iterator to compare with.
   * @return `true` if the current iterator is greater than the given iterator, `false` otherwise.
   */
  bool operator> (const ConstIterator<ValueType>& I) const;

  /**
   * Greater-than-or-equal comparison operator.
   *
   * @param I The constant iterator to compare with.
   * @return `true` if the current iterator is greater than or equal to the given iterator, `false` otherwise.
   */
  bool operator>=(const ConstIterator<ValueType>& I) const;

 private:
  ValueType arr_[DIM]; ///< The array of elements.
};

#endif

}  // end namespace sctl

#endif // _SCTL_STATIC_ARRAY_HPP_
