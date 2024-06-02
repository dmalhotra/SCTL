#ifndef _SCTL_ITERATOR_HPP_
#define _SCTL_ITERATOR_HPP_

#include <ostream>          // for ostream
#include <iterator>         // for random_access_iterator_tag
#include <ostream>          // for operator<<, basic_ostream

#include "sctl/common.hpp"  // for Long, sctl

namespace sctl {

#ifdef SCTL_MEMDEBUG

/**
 * A constant iterator with additional functionalities for memory debugging.  It is enabled by
 * defining the macro `SCTL_MEMDEBUG`, otherwise it is an alias for a raw pointer.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 */
template <class ValueType> class ConstIterator {

  template <typename T> friend class ConstIterator;

  template <typename T> friend class Iterator;

  void IteratorAssertChecks(Long j = 0) const;

 public:
  typedef Long difference_type;
  typedef ValueType value_type;
  typedef const ValueType* pointer;
  typedef const ValueType& reference;
  typedef std::random_access_iterator_tag iterator_category;

 protected:
  char* base;                 ///< Base pointer of the array.
  difference_type len;        ///< Length of the array.
  difference_type offset;     ///< Offset from the base pointer.
  Long alloc_ctr;             ///< Allocation counter for memory management.
  void* mem_head;             ///< Pointer to the head of the memory block.
  static const Long ValueSize = sizeof(ValueType);  ///< Size of each element.

 public:
  /**
   * Default constructor.
   */
  ConstIterator();

  /**
   * Constructor with base pointer and length.
   *
   * @param base_ Base pointer of the array.
   * @param len_ Length of the array.
   * @param dynamic_alloc If true, dynamic allocation is used.
   */
  explicit ConstIterator(pointer base_, difference_type len_, bool dynamic_alloc = false);

  /**
   * Template copy constructor.
   *
   * @tparam AnotherType The type of elements in the other iterator.
   * @param I The `ConstIterator` to copy from.
   */
  template <class AnotherType> explicit ConstIterator(const ConstIterator<AnotherType>& I);

  // value_type* like operators
  /**
   * Dereference operator.
   *
   * @return A reference to the element at the current position.
   */
  reference operator*() const;

  /**
   * Member access operator.
   *
   * @return A pointer to the element at the current position.
   */
  pointer operator->() const;

  /**
   * Subscript operator.
   *
   * @param off The offset from the current position.
   * @return A reference to the element at the specified offset.
   */
  reference operator[](difference_type off) const;

  // Increment / Decrement
  /**
   * Pre-increment operator.
   *
   * @return A reference to the updated iterator.
   */
  ConstIterator& operator++();

  /**
   * Post-increment operator.
   *
   * @return A copy of the iterator before incrementing.
   */
  ConstIterator operator++(int);

  /**
   * Pre-decrement operator.
   *
   * @return A reference to the updated iterator.
   */
  ConstIterator& operator--();

  /**
   * Post-decrement operator.
   *
   * @return A copy of the iterator before decrementing.
   */
  ConstIterator operator--(int);

  // Arithmetic
  /**
   * Addition assignment operator.
   *
   * @param i The offset to add.
   * @return A reference to the updated iterator.
   */
  ConstIterator& operator+=(difference_type i);

  /**
   * Addition operator.
   *
   * @param i The offset to add.
   * @return A new iterator pointing to the new position.
   */
  ConstIterator operator+(difference_type i) const;

  /**
   * Addition operator for constant iterator.
   *
   * @tparam T The type of elements pointed to by the iterator.
   * @param i The offset to add.
   * @param right The `ConstIterator` to add the offset to.
   * @return A new iterator pointing to the new position.
   */
  template <class T> friend ConstIterator<T> operator+(typename ConstIterator<T>::difference_type i, const ConstIterator<T>& right);

  /**
   * Subtraction assignment operator.
   *
   * @param i The offset to subtract.
   * @return A reference to the updated iterator.
   */
  ConstIterator& operator-=(difference_type i);

  /**
   * Subtraction operator.
   *
   * @param i The offset to subtract.
   * @return A new iterator pointing to the new position.
   */
  ConstIterator operator-(difference_type i) const;

  /**
   * Difference operator.
   *
   * @param I The iterator to subtract.
   * @return The difference between the current position and the given iterator.
   */
  difference_type operator-(const ConstIterator& I) const;

  // Comparison operators
  /**
   * Equality comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the iterators are equal, `false` otherwise.
   */
  bool operator==(const ConstIterator& I) const;

  /**
   * Inequality comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the iterators are not equal, `false` otherwise.
   */
  bool operator!=(const ConstIterator& I) const;

  /**
   * Less-than comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is less than the given iterator, `false` otherwise.
   */
  bool operator<(const ConstIterator& I) const;

  /**
   * Less-than-or-equal comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is less than or equal to the given iterator, `false` otherwise.
   */
  bool operator<=(const ConstIterator& I) const;

  /**
   * Greater-than comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is greater than the given iterator, `false` otherwise.
   */
  bool operator>(const ConstIterator& I) const;

  /**
   * Greater-than-or-equal comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is greater than or equal to the given iterator, `false` otherwise.
   */
  bool operator>=(const ConstIterator& I) const;

  /**
   * Output stream operator.
   *
   * @param out The output stream.
   * @param I The iterator to output.
   * @return The output stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const ConstIterator& I) {
    out << "(" << (long long)I.base << "+" << I.offset << ":" << I.len << ")";
    return out;
  }
};

/**
 * An iterator with additional functionalities for memory debugging.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 */
template <class ValueType> class Iterator : public ConstIterator<ValueType> {

 public:
  typedef Long difference_type;
  typedef ValueType value_type;
  typedef ValueType* pointer;
  typedef ValueType& reference;
  typedef std::random_access_iterator_tag iterator_category;

 public:
  /**
   * Default constructor.
   */
  Iterator();

  /**
   * Constructor with base pointer and length.
   *
   * @param base_ Base pointer of the array.
   * @param len_ Length of the array.
   * @param dynamic_alloc If true, dynamic allocation is used.
   */
  explicit Iterator(pointer base_, difference_type len_, bool dynamic_alloc = false);

  /**
   * Template copy constructor.
   *
   * @tparam AnotherType The type of elements in the other iterator.
   * @param I The `ConstIterator` to copy from.
   */
  template <class AnotherType> explicit Iterator(const ConstIterator<AnotherType>& I);

  // value_type* like operators
  /**
   * Dereference operator.
   *
   * @return A reference to the element at the current position.
   */
  reference operator*() const;

  /**
   * Member access operator.
   *
   * @return A pointer to the element at the current position.
   */
  value_type* operator->() const;

  /**
   * Subscript operator.
   *
   * @param off The offset from the current position.
   * @return A reference to the element at the specified offset.
   */
  reference operator[](difference_type off) const;

  // Increment / Decrement
  /**
   * Pre-increment operator.
   *
   * @return A reference to the updated iterator.
   */
  Iterator& operator++();

  /**
   * Post-increment operator.
   *
   * @return A copy of the iterator before incrementing.
   */
  Iterator operator++(int);

  /**
   * Pre-decrement operator.
   *
   * @return A reference to the updated iterator.
   */
  Iterator& operator--();

  /**
   * Post-decrement operator.
   *
   * @return A copy of the iterator before decrementing.
   */
  Iterator operator--(int);

  // Arithmetic
  /**
   * Addition assignment operator.
   *
   * @param i The offset to add.
   * @return A reference to the updated iterator.
   */
  Iterator& operator+=(difference_type i);

  /**
   * Addition operator.
   *
   * @param i The offset to add.
   * @return A new iterator pointing to the new position.
   */
  Iterator operator+(difference_type i) const;

  /**
   * Addition operator for iterator.
   *
   * @tparam T The type of elements pointed to by the iterator.
   * @param i The offset to add.
   * @param right The `Iterator` to add the offset to.
   * @return A new iterator pointing to the new position.
   */
  template <class T> friend Iterator<T> operator+(typename Iterator<T>::difference_type i, const Iterator<T>& right);

  /**
   * Subtraction assignment operator.
   *
   * @param i The offset to subtract.
   * @return A reference to the updated iterator.
   */
  Iterator& operator-=(difference_type i);

  /**
   * Subtraction operator.
   *
   * @param i The offset to subtract.
   * @return A new iterator pointing to the new position.
   */
  Iterator operator-(difference_type i) const;

  /**
   * Difference operator.
   *
   * @param I The constant iterator to subtract.
   * @return The difference between the current position and the given iterator.
   */
  difference_type operator-(const ConstIterator<ValueType>& I) const;
};

#endif

/**
 * Returns a null iterator.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 * @return An iterator pointing to null.
 */
template <class ValueType> Iterator<ValueType> NullIterator();

/**
 * Converts a pointer to an iterator.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 * @param ptr The pointer to convert.
 * @param len The length of the array.
 * @return An iterator pointing to the given pointer.
 */
template <class ValueType> Iterator<ValueType> Ptr2Itr(void* ptr, Long len);

/**
 * Converts a const pointer to a const iterator.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 * @param ptr The const pointer to convert.
 * @param len The length of the array.
 * @return A const iterator pointing to the given pointer.
 */
template <class ValueType> ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len);

/**
 * Wrapper for memcpy. Also checks if source and destination pointers are the same.
 *
 * @tparam ValueType The type of elements to copy.
 * @param destination The iterator pointing to the destination.
 * @param source The const iterator pointing to the source.
 * @param num The number of elements to copy.
 * @return An iterator pointing to the destination after copying.
 */
template <class ValueType> Iterator<ValueType> memcopy(Iterator<ValueType> destination, ConstIterator<ValueType> source, Long num);

/**
 * Wrapper for memset.
 *
 * @tparam ValueType The type of elements to set.
 * @param ptr The iterator pointing to the memory block.
 * @param value The value to set.
 * @param num The number of elements to set.
 * @return An iterator pointing to the memory block after setting.
 */
template <class ValueType> Iterator<ValueType> memset(Iterator<ValueType> ptr, int value, Long num);

}  // end namespace sctl

#endif // _SCTL_ITERATOR_HPP_
