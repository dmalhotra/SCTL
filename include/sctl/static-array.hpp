#ifndef _SCTL_STATIC_ARRAY_HPP_
#define _SCTL_STATIC_ARRAY_HPP_

#include <initializer_list>  // for initializer_list

#include "sctl/common.hpp"   // for Long, sctl

namespace sctl {

#ifdef SCTL_MEMDEBUG

template <class ValueType, Long DIM> class StaticArray {
  typedef Long difference_type;

 public:
  StaticArray() = default;

  StaticArray(const StaticArray&) = default;

  StaticArray& operator=(const StaticArray&) = default;

  explicit StaticArray(std::initializer_list<ValueType> arr_);

  ~StaticArray() = default;

  // value_type* like operators
  const ValueType& operator*() const;

  ValueType& operator*();

  const ValueType* operator->() const;

  ValueType* operator->();

  const ValueType& operator[](difference_type off) const;

  ValueType& operator[](difference_type off);

  operator ConstIterator<ValueType>() const;

  operator Iterator<ValueType>();

  // Arithmetic
  ConstIterator<ValueType> operator+(difference_type i) const;

  Iterator<ValueType> operator+(difference_type i);

  template <class T, Long d> friend ConstIterator<T> operator+(typename StaticArray<T,d>::difference_type i, const StaticArray<T,d>& right);

  template <class T, Long d> friend Iterator<T> operator+(typename StaticArray<T,d>::difference_type i, StaticArray<T,d>& right);

  ConstIterator<ValueType> operator-(difference_type i) const;

  Iterator<ValueType> operator-(difference_type i);

  difference_type operator-(const ConstIterator<ValueType>& I) const;

  // Comparison operators
  bool operator==(const ConstIterator<ValueType>& I) const;

  bool operator!=(const ConstIterator<ValueType>& I) const;

  bool operator< (const ConstIterator<ValueType>& I) const;

  bool operator<=(const ConstIterator<ValueType>& I) const;

  bool operator> (const ConstIterator<ValueType>& I) const;

  bool operator>=(const ConstIterator<ValueType>& I) const;

 private:

  ValueType arr_[DIM];
};

#endif

}  // end namespace sctl

#endif // _SCTL_STATIC_ARRAY_HPP_
