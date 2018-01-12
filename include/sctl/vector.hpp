#ifndef _SCTL_VECTOR_HPP_
#define _SCTL_VECTOR_HPP_

#include SCTL_INCLUDE(common.hpp)

#include <vector>
#include <cstdlib>
#include <cstdint>

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector {

 public:
  typedef ValueType ValType;

  Vector();

  Vector(Long dim_, Iterator<ValueType> data_ = Iterator<ValueType>(NULL), bool own_data_ = true);

  Vector(const Vector& V);

  Vector(const std::vector<ValueType>& V);

  ~Vector();

  void Swap(Vector<ValueType>& v1);

  void ReInit(Long dim_, Iterator<ValueType> data_ = NULL, bool own_data_ = true);

  void Write(const char* fname) const;

  void Read(const char* fname);

  Long Dim() const;

  Long Capacity() const;

  void SetZero();

  Iterator<ValueType> Begin();

  ConstIterator<ValueType> Begin() const;

  void PushBack(const ValueType& x);

  Vector& operator=(const Vector& V);

  Vector& operator=(const std::vector<ValueType>& V);

  ValueType& operator[](Long j);

  const ValueType& operator[](Long j) const;

 private:
  Long dim;
  Long capacity;
  Iterator<ValueType> data_ptr;
  bool own_data;
};

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Vector<ValueType>& V);

}  // end namespace

#include SCTL_INCLUDE(vector.txx)

#endif  //_SCTL_VECTOR_HPP_
