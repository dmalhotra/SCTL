#ifndef _SCTL_VEC_HPP_
#define _SCTL_VEC_HPP_

#include <ostream>                  // for ostream

#include "sctl/common.hpp"          // for Integer, sctl
#include "sctl/intrin-wrapper.hpp"  // for Mask, VecData

namespace sctl {

  /**
   * Returns the default SIMD vector length for the given scalar type.
   */
  template <class ScalarType> constexpr Integer DefaultVecLen();

  /**
   * This class template provides functionality for working with SIMD vectors, enabling
   * efficient parallelization of computations on multiple data elements simultaneously.
   * It can optionally make use of the **Intel SVML** (by defining the macro
   * `SCTL_HAVE_SVML`) and **libmvec** (by defining the macro `SCTL_HAVE_LIBMVEC`)
   * libraries when they are available.
   *
   * @tparam ValueType Data type of the vector elements.
   * @tparam N Number of elements in the vector. Defaults to DefaultVecLen<ValueType>().
   */
  template <class ValueType, Integer N = DefaultVecLen<ValueType>()> class alignas(sizeof(ValueType) * N) Vec {
    public:
      /**
       * Type alias for the scalar type of the vector elements.
       */
      using ScalarType = ValueType;

      /**
       * Type alias for the internal data representation of the vector.
       */
      using VData = VecData<ScalarType,N>;

      /**
       * Type alias for the mask type associated with the vector.
       */
      using MaskType = Mask<VData>;

      /**
       * Get the size of the vector.
       *
       * @return The size of the vector.
       */
      static constexpr Integer Size();

      /**
       * Create a vector initialized with all elements set to zero.
       *
       * @return Zero-initialized vector.
       */
      static inline Vec Zero();

      /**
       * Load a scalar value into all elements of the vector.
       *
       * @param p Pointer to the scalar value.
       * @return Vector with all elements loaded with the scalar value.
       */
      static inline Vec Load1(ScalarType const* p);

      /**
       * Load a vector of scalar values from unaligned memory.
       *
       * @param p Pointer to the scalar values.
       * @return Vector loaded with the scalar values.
       */
      static inline Vec Load(ScalarType const* p);

      /**
       * Load a vector of scalar values from aligned memory.
       *
       * @param p Pointer to the scalar values.
       * @return Vector loaded with the scalar values from aligned memory.
       */
      static inline Vec LoadAligned(ScalarType const* p);

      /**
       * Default constructor.
       */
      Vec() = default;

      /**
       * Copy constructor.
       *
       * @param v_ Vector to copy from.
       */
      Vec(const Vec&) = default;

      /**
       * Copy assignment operator.
       *
       * @param v_ Vector to copy from.
       * @return Reference to the assigned vector.
       */
      Vec& operator=(const Vec&) = default;

      /**
       * Destructor.
       */
      ~Vec() = default;

      /**
       * Constructor initializing vector with given data.
       *
       * @param v_ Vector data.
       */
      inline Vec(const VData& v_);

      /**
       * Constructor initializing vector with a scalar value.
       *
       * @param a Scalar value to initialize vector elements.
       */
      inline Vec(const ScalarType& a);

      /**
       * Constructor initializing vector with multiple scalar values.
       *
       * @tparam T Data type of scalar values.
       * @tparam T1 Variadic template parameter pack for scalar values.
       * @param x First scalar value.
       * @param args Remaining scalar values.
       */
      template <class T,class ...T1> inline Vec(T x, T1... args);

      /**
       * Store the vector data into unaligned memory.
       *
       * @param p Pointer to the memory location to store the data.
       */
      inline void Store(ScalarType* p) const;

      /**
       * Store the vector data into aligned memory.
       *
       * @param p Pointer to the memory location to store the data.
       */
      inline void StoreAligned(ScalarType* p) const;

      // Element access

      /**
       * Access individual elements of the vector.
       *
       * @param i Index of the element to access.
       * @return Value of the element at the specified index.
       */
      inline ScalarType operator[](Integer i) const;

      /**
       * Insert a value at the specified index in the vector.
       *
       * @param i Index at which to insert the value.
       * @param value Value to insert.
       */
      inline void insert(Integer i, ScalarType value);

      // Arithmetic operators

      /**
       * Unary plus operator.
       *
       * @return The vector with all elements unchanged.
       */
      inline Vec operator+() const;

      /**
       * Unary minus operator.
       *
       * @return The negated vector.
       */
      inline Vec operator-() const;

      // Bitwise operators

      /**
       * Bitwise NOT operator.
       *
       * @return The bitwise complement of the vector.
       */
      inline Vec operator~() const;

      // Assignment operators

      /**
       * Assignment operator with a scalar value.
       *
       * @param a Scalar value to assign to all elements of the vector.
       * @return Reference to the modified vector.
       */
      inline Vec& operator=(const ScalarType& a);

      /**
       * Multiplication assignment operator with another vector.
       *
       * @param rhs Vector to multiply with.
       * @return Reference to the modified vector.
       */
      inline Vec& operator*=(const Vec& rhs);

      /**
       * Division assignment operator with another vector.
       *
       * @param rhs Vector to divide by.
       * @return Reference to the modified vector.
       */
      inline Vec& operator/=(const Vec& rhs);

      /**
       * Addition assignment operator with another vector.
       *
       * @param rhs Vector to add.
       * @return Reference to the modified vector.
       */
      inline Vec& operator+=(const Vec& rhs);

      /**
       * Subtraction assignment operator with another vector.
       *
       * @param rhs Vector to subtract.
       * @return Reference to the modified vector.
       */
      inline Vec& operator-=(const Vec& rhs);

      /**
       * Bitwise AND assignment operator with another vector.
       *
       * @param rhs Vector for bitwise AND operation.
       * @return Reference to the modified vector.
       */
      inline Vec& operator&=(const Vec& rhs);

      /**
       * Bitwise XOR assignment operator with another vector.
       *
       * @param rhs Vector for bitwise XOR operation.
       * @return Reference to the modified vector.
       */
      inline Vec& operator^=(const Vec& rhs);

      /**
       * Bitwise OR assignment operator with another vector.
       *
       * @param rhs Vector for bitwise OR operation.
       * @return Reference to the modified vector.
       */
      inline Vec& operator|=(const Vec& rhs);

      /**
       * Set the vector data.
       *
       * @param v_ Vector data to set.
       */
      inline void set(const VData& v_);

      /**
       * Get the vector data.
       *
       * @return Reference to the vector data.
       */
      inline const VData& get() const;

      /**
       * Get the vector data.
       *
       * @return Reference to the vector data.
       */
      inline VData& get();

    private:
      /**
       * Helper struct for initializing vectors with multiple scalar values.
       */
      template <class T, class... T2> struct InitVec;

      /**
       * Internal data representation of the vector.
       */
      VData v;
  };

  // Conversion operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType convert2mask(const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> RoundReal2Real(const Vec<ValueType,N>& x);
  template <class RealVec, class IntVec> inline RealVec ConvertInt2Real(const IntVec& x);
  template <class IntVec, class RealVec> inline IntVec RoundReal2Int(const RealVec& x);
  template <class MaskType> inline Vec<typename MaskType::ScalarType,MaskType::Size> convert2vec(const MaskType& a);
  //template <class Vec1, class Vec2> friend Vec1 reinterpret(const Vec2& x);


  // Arithmetic operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> FMA(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b, const Vec<ValueType,N>& c);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const ValueType& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const ValueType& a, const Vec<ValueType,N>& b);


  // Comparison operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const ValueType& b);

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const ValueType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const ValueType& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const ValueType& a, const Vec<ValueType,N>& b);


  // Bitwise operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const ValueType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const ValueType& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const ValueType& b, const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const ValueType& b, const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const ValueType& b, const Vec<ValueType,N>& a);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const ValueType& b, const Vec<ValueType,N>& a);


  // Bitshift
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator<<(const Vec<ValueType,N>& lhs, const Integer& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator>>(const Vec<ValueType,N>& lhs, const Integer& rhs);


  // Other operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const ValueType& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const ValueType& rhs);

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const ValueType& lhs, const Vec<ValueType,N>& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const ValueType& lhs, const Vec<ValueType,N>& rhs);


  // Special functions
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m);

  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m);

  template <class ValueType, Integer N> inline void sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline void approx_sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x);

  template <class ValueType, Integer N> inline Vec<ValueType,N> exp(const Vec<ValueType,N>& x);
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_exp(const Vec<ValueType,N>& x);

  #if defined(SCTL_HAVE_SVML) || defined(SCTL_HAVE_LIBMVEC)
  template <class ValueType, Integer N> inline Vec<ValueType,N> log(const Vec<ValueType,N>& x);
  #endif


  // Print
  template <class ValueType, Integer N> inline std::ostream& operator<<(std::ostream& os, const Vec<ValueType,N>& in);


  // Other operators
  template <class ValueType> inline void printb(const ValueType& x);

}

#endif // _SCTL_VEC_HPP_
