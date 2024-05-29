#ifndef _SCTL_ITERATOR_HPP_
#define _SCTL_ITERATOR_HPP_

#include <ostream>          // for ostream
#include <iterator>         // for random_access_iterator_tag
#include <ostream>          // for operator<<, basic_ostream

#include "sctl/common.hpp"  // for Long, sctl

namespace sctl {

#ifdef SCTL_MEMDEBUG

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
  char* base;
  difference_type len, offset;
  Long alloc_ctr;
  void* mem_head;
  static const Long ValueSize = sizeof(ValueType);

 public:
  ConstIterator();

  explicit ConstIterator(pointer base_, difference_type len_, bool dynamic_alloc = false);

  template <class AnotherType> explicit ConstIterator(const ConstIterator<AnotherType>& I);

  // value_type* like operators
  reference operator*() const;

  pointer operator->() const;

  reference operator[](difference_type off) const;

  // Increment / Decrement
  ConstIterator& operator++();

  ConstIterator operator++(int);

  ConstIterator& operator--();

  ConstIterator operator--(int);

  // Arithmetic
  ConstIterator& operator+=(difference_type i);

  ConstIterator operator+(difference_type i) const;

  template <class T> friend ConstIterator<T> operator+(typename ConstIterator<T>::difference_type i, const ConstIterator<T>& right);

  ConstIterator& operator-=(difference_type i);

  ConstIterator operator-(difference_type i) const;

  difference_type operator-(const ConstIterator& I) const;

  // Comparison operators
  bool operator==(const ConstIterator& I) const;

  bool operator!=(const ConstIterator& I) const;

  bool operator<(const ConstIterator& I) const;

  bool operator<=(const ConstIterator& I) const;

  bool operator>(const ConstIterator& I) const;

  bool operator>=(const ConstIterator& I) const;

  friend std::ostream& operator<<(std::ostream& out, const ConstIterator& I) {
    out << "(" << (long long)I.base << "+" << I.offset << ":" << I.len << ")";
    return out;
  }
};

template <class ValueType> class Iterator : public ConstIterator<ValueType> {

 public:
  typedef Long difference_type;
  typedef ValueType value_type;
  typedef ValueType* pointer;
  typedef ValueType& reference;
  typedef std::random_access_iterator_tag iterator_category;

 public:
  Iterator();

  explicit Iterator(pointer base_, difference_type len_, bool dynamic_alloc = false);

  template <class AnotherType> explicit Iterator(const ConstIterator<AnotherType>& I);

  // value_type* like operators
  reference operator*() const;

  value_type* operator->() const;

  reference operator[](difference_type off) const;

  // Increment / Decrement
  Iterator& operator++();

  Iterator operator++(int);

  Iterator& operator--();

  Iterator operator--(int);

  // Arithmetic
  Iterator& operator+=(difference_type i);

  Iterator operator+(difference_type i) const;

  template <class T> friend Iterator<T> operator+(typename Iterator<T>::difference_type i, const Iterator<T>& right);

  Iterator& operator-=(difference_type i);

  Iterator operator-(difference_type i) const;

  difference_type operator-(const ConstIterator<ValueType>& I) const;
};

#endif

template <class ValueType> Iterator<ValueType> NullIterator();
template <class ValueType> Iterator<ValueType> Ptr2Itr(void* ptr, Long len);
template <class ValueType> ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len);

/**
 * Wrapper to memcpy. Also checks if source and destination pointers are the same.
 */
template <class ValueType> Iterator<ValueType> memcopy(Iterator<ValueType> destination, ConstIterator<ValueType> source, Long num);

template <class ValueType> Iterator<ValueType> memset(Iterator<ValueType> ptr, int value, Long num);

}  // end namespace sctl

#endif // _SCTL_ITERATOR_HPP_
