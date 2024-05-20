#ifndef _SCTL_STATIC_ARRAY_TXX_
#define _SCTL_STATIC_ARRAY_TXX_

#include <initializer_list>       // for initializer_list

#include "sctl/common.hpp"        // for Long, SCTL_ASSERT_MSG, SCTL_NAMESPACE
#include SCTL_INCLUDE(static-array.hpp)  // for StaticArray, operator+
#include SCTL_INCLUDE(iterator.hpp)      // for ConstIterator, Iterator

namespace SCTL_NAMESPACE {

#ifdef SCTL_MEMDEBUG

template <class ValueType, Long DIM> inline StaticArray<ValueType,DIM>::StaticArray(std::initializer_list<ValueType> arr_) : StaticArray() {
  // static_assert(arr_.size() <= DIM, "too many initializer values"); // allowed in C++14
  SCTL_ASSERT_MSG(arr_.size() <= DIM, "too many initializer values");
  for (Long i = 0; i < (Long)arr_.size(); i++) this->arr_[i] = arr_.begin()[i];
}

template <class ValueType, Long DIM> inline const ValueType& StaticArray<ValueType,DIM>::operator*() const {
  return (*this)[0];
}

template <class ValueType, Long DIM> inline ValueType& StaticArray<ValueType,DIM>::operator*() {
  return (*this)[0];
}

template <class ValueType, Long DIM> inline const ValueType* StaticArray<ValueType,DIM>::operator->() const {
  return (ConstIterator<ValueType>)*this;
}

template <class ValueType, Long DIM> inline ValueType* StaticArray<ValueType,DIM>::operator->() {
  return (Iterator<ValueType>)*this;
}

template <class ValueType, Long DIM> inline const ValueType& StaticArray<ValueType,DIM>::operator[](difference_type off) const {
  return ((ConstIterator<ValueType>)*this)[off];
}

template <class ValueType, Long DIM> inline ValueType& StaticArray<ValueType,DIM>::operator[](difference_type off) {
  return ((Iterator<ValueType>)*this)[off];
}

template <class ValueType, Long DIM> inline StaticArray<ValueType,DIM>::operator ConstIterator<ValueType>() const {
  return ConstIterator<ValueType>(arr_, DIM);
}

template <class ValueType, Long DIM> inline StaticArray<ValueType,DIM>::operator Iterator<ValueType>() {
  return Iterator<ValueType>(arr_, DIM);
}

template <class ValueType, Long DIM> inline ConstIterator<ValueType> StaticArray<ValueType,DIM>::operator+(difference_type i) const {
  return (ConstIterator<ValueType>)*this + i;
}

template <class ValueType, Long DIM> inline Iterator<ValueType> StaticArray<ValueType,DIM>::operator+(difference_type i) {
  return (Iterator<ValueType>)*this + i;
}

template <class T, Long d> inline ConstIterator<T> operator+(typename StaticArray<T,d>::difference_type i, const StaticArray<T,d>& right) {
  return i + (ConstIterator<T>)right;
}

template <class T, Long d> inline Iterator<T> operator+(typename StaticArray<T,d>::difference_type i, StaticArray<T,d>& right) {
  return i + (Iterator<T>)right;
}

template <class ValueType, Long DIM> inline ConstIterator<ValueType> StaticArray<ValueType,DIM>::operator-(difference_type i) const {
  return (ConstIterator<ValueType>)*this - i;
}

template <class ValueType, Long DIM> inline Iterator<ValueType> StaticArray<ValueType,DIM>::operator-(difference_type i) {
  return (Iterator<ValueType>)*this - i;
}

template <class ValueType, Long DIM> inline typename StaticArray<ValueType,DIM>::difference_type StaticArray<ValueType,DIM>::operator-(const ConstIterator<ValueType>& I) const {
  return (ConstIterator<ValueType>)*this - (ConstIterator<ValueType>)I;
}

template <class ValueType, Long DIM> inline bool StaticArray<ValueType,DIM>::operator==(const ConstIterator<ValueType>& I) const {
  return (ConstIterator<ValueType>)*this == I;
}

template <class ValueType, Long DIM> inline bool StaticArray<ValueType,DIM>::operator!=(const ConstIterator<ValueType>& I) const {
  return (ConstIterator<ValueType>)*this != I;
}

template <class ValueType, Long DIM> inline bool StaticArray<ValueType,DIM>::operator< (const ConstIterator<ValueType>& I) const {
  return (ConstIterator<ValueType>)*this <  I;
}

template <class ValueType, Long DIM> inline bool StaticArray<ValueType,DIM>::operator<=(const ConstIterator<ValueType>& I) const {
  return (ConstIterator<ValueType>)*this <= I;
}

template <class ValueType, Long DIM> inline bool StaticArray<ValueType,DIM>::operator> (const ConstIterator<ValueType>& I) const {
  return (ConstIterator<ValueType>)*this >  I;
}

template <class ValueType, Long DIM> inline bool StaticArray<ValueType,DIM>::operator>=(const ConstIterator<ValueType>& I) const {
  return (ConstIterator<ValueType>)*this >= I;
}

#endif

}  // end namespace

#endif // _SCTL_STATIC_ARRAY_TXX_
