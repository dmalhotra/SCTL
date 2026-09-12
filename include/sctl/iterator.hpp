#ifndef _SCTL_ITERATOR_HPP_
#define _SCTL_ITERATOR_HPP_

#include <iterator>         // for random_access_iterator_tag
#include <ostream>          // for operator<<, basic_ostream
#include <type_traits>      // for conditional, enable_if, is_const, is_convertible, remove_const

#include "sctl/common.hpp"  // for Long, sctl

namespace sctl {

#ifdef SCTL_MEMDEBUG

/**
 * An iterator with additional functionalities for memory debugging. It is enabled by defining the
 * macro `SCTL_MEMDEBUG`, otherwise it is an alias for a raw pointer. `ValueType` carries the
 * constness, as it does for a pointer: an `Iterator<const T>` reads but does not write, and
 * `ConstIterator<T>` names that. An iterator over mutable elements converts to one over const
 * elements, and not the reverse.
 *
 * @tparam ValueType The type of elements pointed to by the iterator, possibly const-qualified.
 */
template <class ValueType> class Iterator {

  template <typename T> friend class Iterator;

  void IteratorAssertChecks(Long j = 0) const;

  /** Whether `U*` converts to `ValueType*`: the same type, or the same type with const added. */
  template <class U> using QualificationConversion = typename std::enable_if<std::is_convertible<U*, ValueType*>::value, int>::type;

  /** Whether it does not, so the conversion reinterprets and has to be asked for. */
  template <class U> using ReinterpretingConversion = typename std::enable_if<!std::is_convertible<U*, ValueType*>::value, int>::type;

 public:
  typedef Long difference_type;
  typedef typename std::remove_const<ValueType>::type value_type;
  typedef ValueType* pointer;
  typedef ValueType& reference;
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
   * Converting constructor, for a conversion a pointer would also make implicitly: the same
   * element type, or the same type with const added.
   *
   * @tparam AnotherType The type of elements in the other iterator.
   * @param I The iterator to convert from.
   */
  template <class AnotherType, QualificationConversion<AnotherType> = 0> Iterator(const Iterator<AnotherType>& I)
      : base(I.base), len(I.len), offset(I.offset), alloc_ctr(I.alloc_ctr), mem_head(I.mem_head) {}

  /**
   * Reinterpreting constructor, for a conversion a pointer would need a cast for. The alignment the
   * new element type requires is checked, as a cast cannot be.
   *
   * @tparam AnotherType The type of elements in the other iterator.
   * @param I The iterator to reinterpret.
   */
  template <class AnotherType, ReinterpretingConversion<AnotherType> = 0> explicit Iterator(const Iterator<AnotherType>& I)
      : base(I.base), len(I.len), offset(I.offset), alloc_ctr(I.alloc_ctr), mem_head(I.mem_head) {
    SCTL_ASSERT_MSG((uintptr_t)(this->base + this->offset) % alignof(ValueType) == 0, "invalid alignment during pointer type conversion.");
  }

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
   * Difference operator. Takes an iterator over the same element type whether or not it is const,
   * so that a mutable and a const iterator into one array can be subtracted.
   *
   * @param I The iterator to subtract.
   * @return The difference between the current position and the given iterator.
   */
  template <class AnotherType> difference_type operator-(const Iterator<AnotherType>& I) const;

  // Comparison operators
  /**
   * Equality comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the iterators are equal, `false` otherwise.
   */
  template <class AnotherType> bool operator==(const Iterator<AnotherType>& I) const;

  /**
   * Inequality comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the iterators are not equal, `false` otherwise.
   */
  template <class AnotherType> bool operator!=(const Iterator<AnotherType>& I) const;

  /**
   * Less-than comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is less than the given iterator, `false` otherwise.
   */
  template <class AnotherType> bool operator<(const Iterator<AnotherType>& I) const;

  /**
   * Less-than-or-equal comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is less than or equal to the given iterator, `false` otherwise.
   */
  template <class AnotherType> bool operator<=(const Iterator<AnotherType>& I) const;

  /**
   * Greater-than comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is greater than the given iterator, `false` otherwise.
   */
  template <class AnotherType> bool operator>(const Iterator<AnotherType>& I) const;

  /**
   * Greater-than-or-equal comparison operator.
   *
   * @param I The iterator to compare with.
   * @return `true` if the current iterator is greater than or equal to the given iterator, `false` otherwise.
   */
  template <class AnotherType> bool operator>=(const Iterator<AnotherType>& I) const;

  /**
   * Output stream operator.
   *
   * @param out The output stream.
   * @param I The iterator to output.
   * @return The output stream.
   */
  friend std::ostream& operator<<(std::ostream& out, const Iterator& I) {
    out << "(" << (long long)I.base << "+" << I.offset << ":" << I.len << ")";
    return out;
  }
};

#endif

/**
 * Returns a null iterator.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 * @return An iterator pointing to null.
 */
template <class ValueType> Iterator<ValueType> NullIterator();

/** `void*`, or `const void*` when `ValueType` is const, so `Ptr2Itr` cannot drop a const. */
template <class ValueType> using VoidPtr = typename std::conditional<std::is_const<ValueType>::value, const void*, void*>::type;

/**
 * Converts a pointer to an iterator.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 * @param ptr The pointer to convert.
 * @param len The number of `ValueType` elements in the array (not bytes).
 * Used to bounds-check accesses through the returned iterator when
 * `SCTL_MEMDEBUG` is defined; ignored in release builds.
 * @return An iterator pointing to the given pointer.
 */
template <class ValueType> Iterator<ValueType> Ptr2Itr(VoidPtr<ValueType> ptr, Long len);

/**
 * Converts a const pointer to a const iterator.
 *
 * @tparam ValueType The type of elements pointed to by the iterator.
 * @param ptr The const pointer to convert.
 * @param len The number of `ValueType` elements in the array (not bytes).
 * Used to bounds-check accesses through the returned iterator when
 * `SCTL_MEMDEBUG` is defined; ignored in release builds.
 * @return A const iterator pointing to the given pointer.
 */
template <class ValueType> ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len);

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
