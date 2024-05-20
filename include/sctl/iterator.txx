#ifndef _SCTL_ITERATOR_TXX_
#define _SCTL_ITERATOR_TXX_

#include <stdint.h>           // for uintptr_t
#include <cstring>            // for memcpy, memset
#include <type_traits>        // for is_trivially_copyable

#include "sctl/common.hpp"    // for Long, SCTL_UNUSED, SCTL_ASSERT_MSG, SCT...
#include SCTL_INCLUDE(iterator.hpp)  // for ConstIterator, Iterator, operator+, Nul...
#include SCTL_INCLUDE(mem_mgr.hpp)   // for MemoryManager
#include SCTL_INCLUDE(mem_mgr.txx)   // for MemoryManager::CheckMemHead, MemoryMana...
#include SCTL_INCLUDE(vector.hpp)    // for NullIterator

namespace SCTL_NAMESPACE {

#ifdef SCTL_MEMDEBUG

template <class ValueType> ConstIterator<ValueType>::ConstIterator() : base(nullptr), len(0), offset(0), alloc_ctr(0), mem_head(nullptr) {}

template <class ValueType> inline ConstIterator<ValueType>::ConstIterator(const ValueType* base_, difference_type len_, bool dynamic_alloc) {
  this->base = (char*)base_;
  this->len = len_ * (Long)sizeof(ValueType);
  this->offset = 0;
  SCTL_ASSERT_MSG((uintptr_t)(this->base + this->offset) % alignof(ValueType) == 0, "invalid alignment during pointer type conversion.");
  if (dynamic_alloc) {
    MemoryManager::MemHead& mh = *&MemoryManager::GetMemHead((char*)this->base);
    MemoryManager::CheckMemHead(mh);
    alloc_ctr = mh.alloc_ctr;
    mem_head = &mh;
  } else
    mem_head = nullptr;
}

template <class ValueType> template <class AnotherType> inline ConstIterator<ValueType>::ConstIterator(const ConstIterator<AnotherType>& I) {
  this->base = I.base;
  this->len = I.len;
  this->offset = I.offset;
  SCTL_ASSERT_MSG((uintptr_t)(this->base + this->offset) % alignof(ValueType) == 0, "invalid alignment during pointer type conversion.");
  this->alloc_ctr = I.alloc_ctr;
  this->mem_head = I.mem_head;
}

template <class ValueType> inline void ConstIterator<ValueType>::IteratorAssertChecks(Long j) const {
  //const auto& base = this->base;
  const auto& offset = this->offset + j * (Long)sizeof(ValueType);
  const auto& len = this->len;
  const auto& mem_head = this->mem_head;
  const auto& alloc_ctr = this->alloc_ctr;

  if (*this == NullIterator<ValueType>()) SCTL_WARN("dereferencing a nullptr is undefined.");
  SCTL_ASSERT_MSG(offset >= 0 && offset + (Long)sizeof(ValueType) <= len, "access to pointer [B" << (offset < 0 ? "" : "+") << offset << ",B" << (offset + (Long)sizeof(ValueType) < 0 ? "" : "+") << offset + (Long)sizeof(ValueType) << ") is outside of the range [B,B+" << len << ").");
  if (mem_head) {
    MemoryManager::MemHead& mh = *(MemoryManager::MemHead*)(mem_head);
    SCTL_ASSERT_MSG(mh.alloc_ctr == alloc_ctr, "invalid memory address or corrupted memory.");
  }
}

template <class ValueType> inline typename ConstIterator<ValueType>::reference ConstIterator<ValueType>::operator*() const {
  this->IteratorAssertChecks();
  return *(ValueType*)(base + offset);
}

template <class ValueType> inline typename ConstIterator<ValueType>::pointer ConstIterator<ValueType>::operator->() const {
  this->IteratorAssertChecks();
  return (ValueType*)(base + offset);
}

template <class ValueType> inline typename ConstIterator<ValueType>::reference ConstIterator<ValueType>::operator[](difference_type j) const {
  this->IteratorAssertChecks(j);
  return *(ValueType*)(base + offset + j * (Long)sizeof(ValueType));
}

template <class ValueType> inline ConstIterator<ValueType>& ConstIterator<ValueType>::operator++() {
  offset += (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline ConstIterator<ValueType> ConstIterator<ValueType>::operator++(int) {
  ConstIterator<ValueType> tmp(*this);
  ++*this;
  return tmp;
}

template <class ValueType> inline ConstIterator<ValueType>& ConstIterator<ValueType>::operator--() {
  offset -= (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline ConstIterator<ValueType> ConstIterator<ValueType>::operator--(int) {
  ConstIterator<ValueType> tmp(*this);
  --*this;
  return tmp;
}

template <class ValueType> inline ConstIterator<ValueType>& ConstIterator<ValueType>::operator+=(difference_type i) {
  offset += i * (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline ConstIterator<ValueType> ConstIterator<ValueType>::operator+(difference_type i) const {
  ConstIterator<ValueType> tmp(*this);
  tmp.offset += i * (Long)sizeof(ValueType);
  return tmp;
}

template <class T> inline ConstIterator<T> operator+(typename ConstIterator<T>::difference_type i, const ConstIterator<T>& right) {
  return (right + i);
}

template <class ValueType> inline ConstIterator<ValueType>& ConstIterator<ValueType>::operator-=(difference_type i) {
  offset -= i * (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline ConstIterator<ValueType> ConstIterator<ValueType>::operator-(difference_type i) const {
  ConstIterator<ValueType> tmp(*this);
  tmp.offset -= i * (Long)sizeof(ValueType);
  return tmp;
}

template <class ValueType> inline typename ConstIterator<ValueType>::difference_type ConstIterator<ValueType>::operator-(const ConstIterator& I) const {
  // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
  Long diff = ((pointer)(base + offset)) - ((pointer)(I.base + I.offset));
  SCTL_ASSERT_MSG(I.base + I.offset + diff * (Long)sizeof(ValueType) == base + offset, "invalid memory address alignment.");
  return diff;
}

template <class ValueType> inline bool ConstIterator<ValueType>::operator==(const ConstIterator& I) const {
  return (base + offset == I.base + I.offset);
}

template <class ValueType> inline bool ConstIterator<ValueType>::operator!=(const ConstIterator& I) const {
  return !(*this == I);
}

template <class ValueType> inline bool ConstIterator<ValueType>::operator<(const ConstIterator& I) const {
  // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
  return (base + offset) < (I.base + I.offset);
}

template <class ValueType> inline bool ConstIterator<ValueType>::operator<=(const ConstIterator& I) const {
  // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
  return (base + offset) <= (I.base + I.offset);
}

template <class ValueType> inline bool ConstIterator<ValueType>::operator>(const ConstIterator& I) const {
  // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
  return (base + offset) > (I.base + I.offset);
}

template <class ValueType> inline bool ConstIterator<ValueType>::operator>=(const ConstIterator& I) const {
  // if (base != I.base) SCTL_WARN("comparing two unrelated memory addresses.");
  return (base + offset) >= (I.base + I.offset);
}



template <class ValueType> inline Iterator<ValueType>::Iterator() : ConstIterator<ValueType>() {}

template <class ValueType> inline Iterator<ValueType>::Iterator(pointer base_, difference_type len_, bool dynamic_alloc) : ConstIterator<ValueType>(base_, len_, dynamic_alloc) {}

template <class ValueType> template <class AnotherType> inline Iterator<ValueType>::Iterator(const ConstIterator<AnotherType>& I) : ConstIterator<ValueType>(I) {}

template <class ValueType> inline typename Iterator<ValueType>::reference Iterator<ValueType>::operator*() const {
  this->IteratorAssertChecks();
  return *(ValueType*)(this->base + this->offset);
}

template <class ValueType> inline typename Iterator<ValueType>::value_type* Iterator<ValueType>::operator->() const {
  this->IteratorAssertChecks();
  return (ValueType*)(this->base + this->offset);
}

template <class ValueType> inline typename Iterator<ValueType>::reference Iterator<ValueType>::operator[](difference_type j) const {
  this->IteratorAssertChecks(j);
  return *(ValueType*)(this->base + this->offset + j * (Long)sizeof(ValueType));
}

template <class ValueType> inline Iterator<ValueType>& Iterator<ValueType>::operator++() {
  this->offset += (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline Iterator<ValueType> Iterator<ValueType>::operator++(int) {
  Iterator<ValueType> tmp(*this);
  ++*this;
  return tmp;
}

template <class ValueType> inline Iterator<ValueType>& Iterator<ValueType>::operator--() {
  this->offset -= (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline Iterator<ValueType> Iterator<ValueType>::operator--(int) {
  Iterator<ValueType> tmp(*this);
  --*this;
  return tmp;
}

template <class ValueType> inline Iterator<ValueType>& Iterator<ValueType>::operator+=(difference_type i) {
  this->offset += i * (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline Iterator<ValueType> Iterator<ValueType>::operator+(difference_type i) const {
  Iterator<ValueType> tmp(*this);
  tmp.offset += i * (Long)sizeof(ValueType);
  return tmp;
}

template <class T> inline Iterator<T> operator+(typename Iterator<T>::difference_type i, const Iterator<T>& right) {
  return (right + i);
}

template <class ValueType> inline Iterator<ValueType>& Iterator<ValueType>::operator-=(difference_type i) {
  this->offset -= i * (Long)sizeof(ValueType);
  return *this;
}

template <class ValueType> inline Iterator<ValueType> Iterator<ValueType>::operator-(difference_type i) const {
  Iterator<ValueType> tmp(*this);
  tmp.offset -= i * (Long)sizeof(ValueType);
  return tmp;
}

template <class ValueType> inline typename Iterator<ValueType>::difference_type Iterator<ValueType>::operator-(const ConstIterator<ValueType>& I) const {
  return static_cast<const ConstIterator<ValueType>&>(*this) - I;
}



template <class ValueType> inline Iterator<ValueType> Ptr2Itr(void* ptr, Long len) {
  return Iterator<ValueType>((ValueType*)ptr, len);
}
template <class ValueType> inline ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len) {
  return ConstIterator<ValueType>((ValueType*)ptr, len);
}

#else

template <class ValueType> inline Iterator<ValueType> Ptr2Itr(void* ptr, Long len) {
  return (Iterator<ValueType>) ptr;
}
template <class ValueType> inline ConstIterator<ValueType> Ptr2ConstItr(const void* ptr, Long len) {
  return (ConstIterator<ValueType>) ptr;
}

#endif

template <class ValueType> inline Iterator<ValueType> NullIterator() {
  return Ptr2Itr<ValueType>(nullptr, 0);
}

template <class ValueType> inline Iterator<ValueType> memcopy(Iterator<ValueType> destination, ConstIterator<ValueType> source, Long num) {
  if (destination != source && num > 0) {
#ifdef SCTL_MEMDEBUG
    SCTL_UNUSED(source[0]);
    SCTL_UNUSED(source[num - 1]);
    SCTL_UNUSED(destination[0]);
    SCTL_UNUSED(destination[num - 1]);
#endif
    if (std::is_trivially_copyable<ValueType>::value) {
      memcpy((void*)&destination[0], (const void*)&source[0], num * sizeof(ValueType));
    } else {
      for (Long i = 0; i < num; i++) destination[i] = source[i];
    }
  }
  return destination;
}

template <class ValueType> inline Iterator<ValueType> memset(Iterator<ValueType> ptr, int value, Long num) {
  if (num) {
#ifdef SCTL_MEMDEBUG
    SCTL_UNUSED(ptr[0]      );
    SCTL_UNUSED(ptr[num - 1]);
#endif
    SCTL_ASSERT(std::is_trivially_copyable<ValueType>::value);
    ::memset((void*)&ptr[0], value, num * sizeof(ValueType));
  }
  return ptr;
}

}  // end namespace

#endif // _SCTL_ITERATOR_TXX_
