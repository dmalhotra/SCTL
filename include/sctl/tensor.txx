#ifndef _SCTL_TENSOR_TXX_
#define _SCTL_TENSOR_TXX_

#include <ios>                    // for ios
#include <ostream>                // for ostream
#include <stdlib.h>               // for drand48
#include <iomanip>                // for operator<<, setiosflags, setprecision
#include <iostream>               // for basic_ostream, operator<<, cout

#include "sctl/common.hpp"        // for Long, Integer, SCTL_UNUSED, SCTL_NA...
#include "sctl/tensor.hpp"        // for Tensor, TensorArgExtract, operator<<
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator
#include "sctl/iterator.txx"      // for NullIterator, memcopy, Ptr2Itr
#include "sctl/math_utils.hpp"    // for fabs
#include "sctl/static-array.hpp"  // for StaticArray

namespace sctl {

  template <class ValueType, bool own_data, Long... Args> void Tensor<ValueType, own_data, Args...>::test() {
    // Define a tensor with dimensions 2x3
    Tensor<ValueType, true, 2, 3> tensor;

    // Initialize tensor elements with random values
    for (auto& x : tensor) x = drand48();

    // Output tensor multiplied by 10
    std::cout << "Tensor multiplied by 10:\n" << tensor * 10 << std::endl;

    // Output tensor multiplied by its right rotation
    std::cout << "Tensor multiplied by its right rotation:\n" << tensor * tensor.RotateRight() << std::endl;

    // Output tensor multiplied by its left rotation and then added 5
    std::cout << "Tensor multiplied by its left rotation and then added 5:\n" << tensor * tensor.RotateLeft() + 5 << std::endl;

    // Output tensor properties
    std::cout << "Tensor Order: " << tensor.Order() << '\n';
    std::cout << "Tensor Size: " << tensor.Size() << '\n';
    std::cout << "Dimension 0: " << tensor.template Dim<0>() << '\n';
    std::cout << "Dimension 1: " << tensor.template Dim<1>() << '\n';
  }

  template <Long k> static constexpr Long TensorArgExtract() {
    return 1;
  }
  template <Long k, Long d0, Long... dd> static constexpr Long TensorArgExtract() {
    return k==0 ? d0 : TensorArgExtract<k-1,dd...>();
  }

  template <typename T> static constexpr Long TensorArgCount() {
    return 0;
  }
  template <typename T, Long d, Long... dd> static constexpr Long TensorArgCount() {
    return 1 + TensorArgCount<void, dd...>();
  }

  template <Long k> static constexpr Long TensorArgProduct() {
    return 1;
  }
  template <Long k, Long d, Long... dd> static constexpr Long TensorArgProduct() {
    return (k >= 0 ? d : 1) * TensorArgProduct<k+1, dd...>();
  }

  template <Long k, class ValueType, bool own_data_, Long... dd> struct TensorRotateType {
    using Value = Tensor<ValueType,own_data_,dd...>;
  };
  template <class ValueType, bool own_data_, Long d, Long... dd> struct TensorRotateType<0,ValueType,own_data_,d,dd...> {
    using Value = Tensor<ValueType,own_data_,d,dd...>;
  };
  template <Long k, class ValueType, bool own_data_, Long d, Long... dd> struct TensorRotateType<k,ValueType,own_data_,d,dd...> {
    using Value = typename TensorRotateType<k-1,ValueType,own_data_,dd...,d>::Value;
  };
  template <class ValueType, bool own_data, Long... dd> struct TensorRotateLeftType {
    using Value = typename TensorRotateType<1, ValueType, own_data, dd...>::Value;
  };
  template <class ValueType, bool own_data, Long... dd> struct TensorRotateRightType {
    using Value = typename TensorRotateType<TensorArgCount<void, dd...>()-1, ValueType, own_data, dd...>::Value;
  };



  template <class ValueType, bool own_data, Long... Args> constexpr Long Tensor<ValueType, own_data, Args...>::Order() {
    return TensorArgCount<void, Args...>();
  }

  template <class ValueType, bool own_data, Long... Args> constexpr Long Tensor<ValueType, own_data, Args...>::Size() {
    return TensorArgProduct<0, Args...>();
  }

  template <class ValueType, bool own_data, Long... Args> template <Long k> constexpr Long Tensor<ValueType, own_data, Args...>::Dim() {
    return TensorArgExtract<k, Args...>();
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, own_data, Args...>::Tensor() {
    Init(NullIterator<ValueType>());
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, own_data, Args...>::Tensor(Iterator<ValueType> src_iter) {
    Init(src_iter);
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, own_data, Args...>::Tensor(ConstIterator<ValueType> src_iter) {
    static_assert(own_data || !Size(), "Cannot use ConstIterator as storage for Tensor.");
    Init((Iterator<ValueType>)src_iter);
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, own_data, Args...>::Tensor(const Tensor &M) {
    Init((Iterator<ValueType>)M.begin());
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, own_data, Args...>::Tensor(const ValueType& v) {
    static_assert(own_data || Size() == 0, "Memory pointer must be provided to initialize Tensor types with own_data=false");
    Init(NullIterator<ValueType>());
    for (auto& x : *this) x = v;
  }

  template <class ValueType, bool own_data, Long... Args> template <bool own_data_> Tensor<ValueType, own_data, Args...>::Tensor(const Tensor<ValueType,own_data_,Args...> &M) {
    static_assert(Size() <= M.Size(), "Initializer must be at least the size of the object.");
    Init((Iterator<ValueType>)M.begin());
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, own_data, Args...>& Tensor<ValueType, own_data, Args...>::operator=(const Tensor &M) {
    memcopy(begin(), M.begin(), Size());
    return *this;
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, own_data, Args...>& Tensor<ValueType, own_data, Args...>::operator=(const ValueType& v) {
    for (auto& x : *this) x = v;
    return *this;
  }


  template <class ValueType, bool own_data, Long... Args> Iterator<ValueType> Tensor<ValueType, own_data, Args...>::begin() {
    return own_data ? (Iterator<ValueType>)buff : iter_[0];
  }

  template <class ValueType, bool own_data, Long... Args> ConstIterator<ValueType> Tensor<ValueType, own_data, Args...>::begin() const {
    return own_data ? (ConstIterator<ValueType>)buff : (ConstIterator<ValueType>)iter_[0];
  }

  template <class ValueType, bool own_data, Long... Args> Iterator<ValueType> Tensor<ValueType, own_data, Args...>::end() {
    return begin() + Size();
  }

  template <class ValueType, bool own_data, Long... Args> ConstIterator<ValueType> Tensor<ValueType, own_data, Args...>::end() const {
    return begin() + Size();
  }


  template <class ValueType, bool own_data, Long... Args> template <class ...PackedLong> ValueType& Tensor<ValueType, own_data, Args...>::operator()(PackedLong... ii) {
    return begin()[offset<0>(ii...)];
  }

  template <class ValueType, bool own_data, Long... Args> template <class ...PackedLong> ValueType Tensor<ValueType, own_data, Args...>::operator()(PackedLong... ii) const {
    return begin()[offset<0>(ii...)];
  }


  template <class ValueType, bool own_data, Long... Args> typename TensorRotateLeftType<ValueType,true,Args...>::Value Tensor<ValueType, own_data, Args...>::RotateLeft() const {
    typename TensorRotateLeftType<ValueType,true,Args...>::Value Tr;
    const auto& T = *this;

    constexpr Long N0 = Dim<0>();
    constexpr Long N1 = Size() / N0;
    for (Long i = 0; i < N0; i++) {
      for (Long j = 0; j < N1; j++) {
        Tr.begin()[j*N0+i] = T.begin()[i*N1+j];
      }
    }
    return Tr;
  }

  template <class ValueType, bool own_data, Long... Args> typename TensorRotateRightType<ValueType,true,Args...>::Value Tensor<ValueType, own_data, Args...>::RotateRight() const {
    typename TensorRotateRightType<ValueType,true,Args...>::Value Tr;
    const auto& T = *this;

    constexpr Long N0 = Dim<Order()-1>();
    constexpr Long N1 = Size() / N0;
    for (Long i = 0; i < N0; i++) {
      for (Long j = 0; j < N1; j++) {
        Tr.begin()[i*N1+j] = T.begin()[j*N0+i];
      }
    }
    return Tr;
  }


  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator+() const {
    return *this;
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator-() const {
    Tensor<ValueType, true, Args...> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Size(); i++) {
      M0.begin()[i] = -M1.begin()[i];
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator+(const ValueType &s) const {
    Tensor<ValueType, true, Args...> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Size(); i++) {
      M0.begin()[i] = M1.begin()[i] + s;
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator-(const ValueType &s) const {
    Tensor<ValueType, true, Args...> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Size(); i++) {
      M0.begin()[i] = M1.begin()[i] - s;
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator*(const ValueType &s) const {
    Tensor<ValueType, true, Args...> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Size(); i++) {
      M0.begin()[i] = M1.begin()[i]*s;
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator/(const ValueType &s) const {
    Tensor<ValueType, true, Args...> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Size(); i++) {
      M0.begin()[i] = M1.begin()[i]/s;
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> template <bool own_data_> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator+(const Tensor<ValueType, own_data_, Args...> &M2) const {
    Tensor<ValueType, true, Args...> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Size(); i++) {
      M0.begin()[i] = M1.begin()[i] + M2.begin()[i];
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> template <bool own_data_> Tensor<ValueType, true, Args...> Tensor<ValueType, own_data, Args...>::operator-(const Tensor<ValueType, own_data_, Args...> &M2) const {
    Tensor<ValueType, true, Args...> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Size(); i++) {
      M0.begin()[i] = M1.begin()[i] - M2.begin()[i];
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> template <bool own_data_, Long N1, Long N2> Tensor<ValueType, true, TensorArgExtract<0, Args...>(), N2> Tensor<ValueType, own_data, Args...>::operator*(const Tensor<ValueType, own_data_, N1, N2> &M2) const {
    static_assert(Order() == 2, "Multiplication is only defined for tensors of order two.");
    static_assert(Dim<1>() == N1, "Tensor dimensions don't match for multiplication.");
    Tensor<ValueType, true, Dim<0>(), N2> M0;
    const auto &M1 = *this;

    for (Long i = 0; i < Dim<0>(); i++) {
      for (Long j = 0; j < N2; j++) {
        ValueType Mij = 0;
        for (Long k = 0; k < N1; k++) {
          Mij += M1(i,k)*M2(k,j);
        }
        M0(i,j) = Mij;
      }
    }
    return M0;
  }

  template <class ValueType, bool own_data, Long... Args> template <Integer k> Long Tensor<ValueType, own_data, Args...>::offset() {
    return 0;
  }
  template <class ValueType, bool own_data, Long... Args> template <Integer k, class ...PackedLong> Long Tensor<ValueType, own_data, Args...>::offset(Long i, PackedLong... ii) {
    return i * TensorArgProduct<-(k+1),Args...>() + offset<k+1>(ii...);
  }

  template <class ValueType, bool own_data, Long... Args> void Tensor<ValueType, own_data, Args...>::Init(Iterator<ValueType> src_iter) {
    if (own_data) {
      if (src_iter != NullIterator<ValueType>()) {
        memcopy((Iterator<ValueType>)buff, src_iter, Size());
      }
    } else {
      if (Size()) {
        SCTL_UNUSED(src_iter[0]);
        SCTL_UNUSED(src_iter[Size()-1]);
        iter_[0] = Ptr2Itr<ValueType>(&src_iter[0], Size());
      } else {
        iter_[0] = NullIterator<ValueType>();
      }
    }
  }

  template <class ValueType, bool own_data, Long N1, Long N2> std::ostream& operator<<(std::ostream &output, const Tensor<ValueType, own_data, N1, N2> &M) {
    std::ios::fmtflags f(std::cout.flags());
    output << std::fixed << std::setprecision(4) << std::setiosflags(std::ios::left);
    for (Long i = 0; i < N1; i++) {
      for (Long j = 0; j < N2; j++) {
        float f = ((float)M(i,j));
        if (sctl::fabs<float>(f) < 1e-25) f = 0;
        output << std::setw(10) << ((double)f) << ' ';
      }
      output << ";\n";
    }
    std::cout.flags(f);
    return output;
  }

}  // end namespace

#endif // _SCTL_TENSOR_TXX_
