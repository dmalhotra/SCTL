#ifndef _PVFMM_VECTOR_HPP_
#define _PVFMM_VECTOR_HPP_

#include <pvfmm/common.hpp>

#include <vector>
#include <cstdlib>
#include <stdint.h>

namespace pvfmm {

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

#include <pvfmm/vector.txx>

#endif  //_PVFMM_VECTOR_HPP_
