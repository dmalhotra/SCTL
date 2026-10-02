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
      [[nodiscard]] static constexpr Integer Size();

      /**
       * Create a vector initialized with all elements set to zero.
       *
       * @return Zero-initialized vector.
       */
      [[nodiscard]] static inline Vec Zero() noexcept;

      /**
       * Load a scalar value into all elements of the vector.
       *
       * @param p Pointer to the scalar value.
       * @return Vector with all elements loaded with the scalar value.
       */
      [[nodiscard]] static inline Vec Load1(ScalarType const* p);

      /**
       * Load a vector of scalar values from unaligned memory.
       *
       * @param p Pointer to the scalar values.
       * @return Vector loaded with the scalar values.
       */
      [[nodiscard]] static inline Vec Load(ScalarType const* p);

      /**
       * Load a vector of scalar values from aligned memory.
       *
       * @param p Pointer to the scalar values.
       * @return Vector loaded with the scalar values from aligned memory.
       */
      [[nodiscard]] static inline Vec LoadAligned(ScalarType const* p);

      /**
       * Load the elements selected by a mask from unaligned memory. The other
       * elements are zero, and their memory is not read.
       *
       * @param p Pointer to the scalar values.
       * @param m Mask selecting the elements to load.
       * @return Vector loaded with the selected scalar values.
       */
      [[nodiscard]] static inline Vec Load(ScalarType const* p, const MaskType& m);

      /**
       * Load the first n elements from unaligned memory. The other elements
       * are zero, and their memory is not read.
       *
       * @param p Pointer to the scalar values.
       * @param n Number of elements to load, n >= 0; all Size() elements if n >= Size().
       * @return Vector loaded with the first n scalar values.
       */
      [[nodiscard]] static inline Vec Load(ScalarType const* p, Integer n);

      /** Load element i from p[idx[i]]. */
      template <class IndexType> [[nodiscard]] static inline Vec Gather(ScalarType const* p, const Vec<IndexType,N>& idx);

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
       * Constructor initializing each of the N elements with its own scalar
       * value; exactly N values must be given (N >= 2).
       *
       * @tparam T0 Data type of the first value.
       * @tparam T1 Data type of the second value.
       * @tparam T2 Data types of the remaining values.
       * @param x0 First value.
       * @param x1 Second value.
       * @param args Remaining values.
       */
      template <class T0, class T1, class ...T2> inline Vec(T0 x0, T1 x1, T2... args);

      /**
       * Constructor joining two vectors of N/2 elements.
       *
       * @param lo Elements 0, ..., N/2-1.
       * @param hi Elements N/2, ..., N-1.
       */
      inline Vec(const Vec<ValueType,N/2>& lo, const Vec<ValueType,N/2>& hi);

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

      /**
       * Store the elements selected by a mask into unaligned memory. The
       * memory of the other elements is not written.
       *
       * @param p Pointer to the memory location to store the data.
       * @param m Mask selecting the elements to store.
       */
      inline void Store(ScalarType* p, const MaskType& m) const;

      /**
       * Store the first n elements into unaligned memory. The memory of the
       * other elements is not written.
       *
       * @param p Pointer to the memory location to store the data.
       * @param n Number of elements to store, n >= 0; all Size() elements if n >= Size().
       */
      inline void Store(ScalarType* p, Integer n) const;

      /** Store element i into p[idx[i]], in order of i: of equal indices, the last element is stored. */
      template <class IndexType> inline void Scatter(ScalarType* p, const Vec<IndexType,N>& idx) const;

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
       * Remainder assignment operator with another vector, for integer types.
       *
       * @param rhs Vector to divide by.
       * @return Reference to the modified vector.
       */
      inline Vec& operator%=(const Vec& rhs);

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

      /**
       * Get the low half of the vector.
       *
       * @return Vector of the elements 0, ..., N/2-1.
       */
      [[nodiscard]] inline Vec<ValueType,N/2> get_low() const;

      /**
       * Get the high half of the vector.
       *
       * @return Vector of the elements N/2, ..., N-1.
       */
      [[nodiscard]] inline Vec<ValueType,N/2> get_high() const;

    private:
      /**
       * Internal data representation of the vector.
       */
      VData v;
  };

  // Conversion operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType convert2mask(const Vec<ValueType,N>& a);
  /** Nearest integer, halves to even (std::rint); exact at native SSE, AVX, AVX-512 widths, elsewhere for |x| < 2^(SigBits-1). */
  template <class ValueType, Integer N> inline Vec<ValueType,N> RoundReal2Real(const Vec<ValueType,N>& x);
  template <class RealVec, class IntVec> inline RealVec ConvertInt2Real(const IntVec& x);
  /** RoundReal2Real as an integer, for |x| < 2^(SigBits-1); double in 2 or 4 lanes without AVX-512DQ: |x| < 2^31. */
  template <class IntVec, class RealVec> inline IntVec RoundReal2Int(const RealVec& x);
  template <class MaskType> inline Vec<typename MaskType::ScalarType,MaskType::Size> convert2vec(const MaskType& a);

  /**
   * Convert each element to the scalar type of VecTo as static_cast does; a
   * real value converted to an integer type is truncated toward zero.
   *
   * @tparam VecTo The Vec type of the result, with the same number of elements.
   * @param x The vector to convert.
   * @return The converted vector.
   */
  template <class VecTo, class ValueType, Integer N> inline VecTo Convert(const Vec<ValueType,N>& x);

  /**
   * Convert a mask to the mask type of VecTo, selecting the same elements.
   *
   * @tparam VecTo The Vec type whose mask type is the result, with the same number of elements.
   * @param m The mask to convert.
   * @return The converted mask.
   */
  template <class VecTo, class MaskType> inline typename VecTo::MaskType ConvertMask(const MaskType& m);


  // Arithmetic operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> FMA(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b, const Vec<ValueType,N>& c);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator%(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b); // integer types

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator%(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator*(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator/(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator+(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator-(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator%(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);


  // Comparison operators
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);

  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator< (const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator<=(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator>=(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator> (const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator==(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType operator!=(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> select(const typename Vec<ValueType,N>::MaskType& m, const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);


  // Bitwise operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const Vec<ValueType,N>& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const Vec<ValueType,N>& a, const typename Vec<ValueType,N>::ScalarType& b);

  template <class ValueType, Integer N> inline Vec<ValueType,N> operator&(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator^(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator|(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);
  template <class ValueType, Integer N> inline Vec<ValueType,N> AndNot(const typename Vec<ValueType,N>::ScalarType& a, const Vec<ValueType,N>& b);


  // Bitshift
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator<<(const Vec<ValueType,N>& lhs, const Integer& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator>>(const Vec<ValueType,N>& lhs, const Integer& rhs);

  /**
   * Bit shift each element of an integer vector left by the count in the same
   * element of rhs, 0 <= count < bit width of ValueType.
   */
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator<<(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);

  /**
   * Bit shift each element of an integer vector right by the count in the same
   * element of rhs, 0 <= count < bit width of ValueType, filling with the sign bit.
   */
  template <class ValueType, Integer N> inline Vec<ValueType,N> operator>>(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);


  // Other operators
  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const Vec<ValueType,N>& rhs);

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const Vec<ValueType,N>& lhs, const typename Vec<ValueType,N>::ScalarType& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const Vec<ValueType,N>& lhs, const typename Vec<ValueType,N>::ScalarType& rhs);

  template <class ValueType, Integer N> inline Vec<ValueType,N> max(const typename Vec<ValueType,N>::ScalarType& lhs, const Vec<ValueType,N>& rhs);
  template <class ValueType, Integer N> inline Vec<ValueType,N> min(const typename Vec<ValueType,N>::ScalarType& lhs, const Vec<ValueType,N>& rhs);

  /**
   * Transpose an NxN matrix of scalars held in N vectors, in-place. On return,
   * element j of v[i] is the original element i of v[j].
   *
   * @param v The N vectors forming the rows of the matrix.
   */
  template <class ValueType, Integer N> inline void transpose(Vec<ValueType,N> (&v)[N]);

  /**
   * Transpose an NxN matrix of scalars held in N vectors, in-place. Same as the
   * array overload, for vectors not held in an array.
   *
   * @param v0 The first row. Exactly N vectors must be given in total.
   * @param vs The remaining N-1 rows.
   */
  template <class ValueType, Integer N, class ...T> inline void transpose(Vec<ValueType,N>& v0, T&... vs);

  /**
   * Exchange the elements of each pair of adjacent lanes: (x0, x1, x2, x3, ...) becomes
   * (x1, x0, x3, x2, ...). N must be even.
   *
   * @param x The input vector.
   * @return The vector with the elements of each pair exchanged.
   */
  template <class ValueType, Integer N> inline Vec<ValueType,N> swap_pairs(const Vec<ValueType,N>& x);


  // Reductions

  /**
   * Sum of the elements. The two halves are added element by element until one
   * element is left, so the order of the additions, and the rounding of a real
   * result, is the same on every instruction set.
   *
   * @param v The vector to reduce.
   * @return The sum of the elements of v.
   */
  template <class ValueType, Integer N> inline ValueType reduce(const Vec<ValueType,N>& v);

  /**
   * Smallest element, as the min of the two halves taken until one element is left.
   *
   * @param v The vector to reduce.
   * @return The smallest element of v.
   */
  template <class ValueType, Integer N> inline ValueType reduce_min(const Vec<ValueType,N>& v);

  /**
   * Largest element, as the max of the two halves taken until one element is left.
   *
   * @param v The vector to reduce.
   * @return The largest element of v.
   */
  template <class ValueType, Integer N> inline ValueType reduce_max(const Vec<ValueType,N>& v);

  /**
   * Number of elements selected by a mask.
   *
   * @param m The mask.
   * @return The number of selected elements.
   */
  template <class VData> inline Integer reduce_count(const Mask<VData>& m);

  /**
   * Whether a mask selects every element.
   */
  template <class VData> inline bool all_of(const Mask<VData>& m);

  /**
   * Whether a mask selects at least one element.
   */
  template <class VData> inline bool any_of(const Mask<VData>& m);

  /**
   * Whether a mask selects no element.
   */
  template <class VData> inline bool none_of(const Mask<VData>& m);


  // Special functions

  /** 1/sqrt(x) to the given digits (-1: full precision, within a few ulp), for 1e-300 <= x <= 1e300. */
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x);

  /** As approx_rsqrt(x), with zero in the elements not in m; for x that can be zero. */
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_rsqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m);

  /** sqrt(x) as x * approx_rsqrt(x), to the given digits, for 1e-300 <= x <= 1e300. */
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x);

  /** As approx_sqrt(x), with zero in the elements not in m; for x that can be zero. */
  template <Integer digits, class ValueType, Integer N> inline Vec<ValueType,N> approx_sqrt(const Vec<ValueType,N>& x, const typename Vec<ValueType,N>::MaskType& m);

  /** Sine and cosine. FullRange = false: faster at native float, double widths without SVML, with an error that grows like |x| eps. */
  template <bool FullRange = true, class ValueType, Integer N> inline void sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x);
  /** Sine and cosine to the given number of digits; -1 for full precision. FullRange = false: faster, with an error that grows like |x| eps. */
  template <Integer digits, bool FullRange = true, class ValueType, Integer N> inline void approx_sincos(Vec<ValueType,N>& sinx, Vec<ValueType,N>& cosx, const Vec<ValueType,N>& x);

  /** e^x. Native float, double widths without SVML: inf from x = 709.44 (float: 88.38), and 0 for results below the smallest normal. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> exp(const Vec<ValueType,N>& x);
  /** e^x to the given digits (-1: exp), with the limits of exp. Without RangeCheck, x beyond the range of the result gives wrong values. */
  template <Integer digits, bool RangeCheck = true, class ValueType, Integer N> inline Vec<ValueType,N> approx_exp(const Vec<ValueType,N>& x);

  /** Natural logarithm; float, double: within about 1.2 ulp, vectorized at native widths. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> log(const Vec<ValueType,N>& x);

  /**
   * Square root, correctly rounded. Cycles per call, dependent / independent calls, Sapphire Rapids:
   * double x8 23 / 24, x4 13 / 12; approx_sqrt<-1> 39 / 5.9, 61 / 7.8.
   */
  template <class ValueType, Integer N> inline Vec<ValueType,N> sqrt(const Vec<ValueType,N>& x);

  /**
   * 1/sqrt(x) with correctly rounded sqrt. Cycles per call, dependent / independent calls, Sapphire Rapids:
   * double x8 45 / 40, x4 26 / 20; approx_rsqrt<-1> 36 / 5.2, 57 / 7.3.
   */
  template <class ValueType, Integer N> inline Vec<ValueType,N> rsqrt(const Vec<ValueType,N>& x);

  /** Absolute value. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> fabs(const Vec<ValueType,N>& x);

  /** Largest integer value not greater than x. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> floor(const Vec<ValueType,N>& x);

  /** Smallest integer value not less than x. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> ceil(const Vec<ValueType,N>& x);

  /** Magnitude of x with the sign of y. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> copysign(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y);

  /** Mask of the elements that are NaN. */
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType isnan(const Vec<ValueType,N>& x);

  /** Sine. FullRange = false: faster at native float, double widths without SVML, with an error that grows like |x| eps. */
  template <bool FullRange = true, class ValueType, Integer N> inline Vec<ValueType,N> sin(const Vec<ValueType,N>& x);

  /** Cosine. FullRange = false: faster at native float, double widths without SVML, with an error that grows like |x| eps. */
  template <bool FullRange = true, class ValueType, Integer N> inline Vec<ValueType,N> cos(const Vec<ValueType,N>& x);

  /** Tangent. FullRange = false: faster at native float, double widths without SVML, with an error that grows like |x| eps. */
  template <bool FullRange = true, class ValueType, Integer N> inline Vec<ValueType,N> tan(const Vec<ValueType,N>& x);

  /** Sine to the given number of digits; -1 for full precision. FullRange = false: faster, with an error that grows like |x| eps. */
  template <Integer digits, bool FullRange = true, class ValueType, Integer N> inline Vec<ValueType,N> approx_sin(const Vec<ValueType,N>& x);

  /** Cosine to the given number of digits; -1 for full precision. FullRange = false: faster, with an error that grows like |x| eps. */
  template <Integer digits, bool FullRange = true, class ValueType, Integer N> inline Vec<ValueType,N> approx_cos(const Vec<ValueType,N>& x);

  /** Tangent to the given number of digits; -1 for full precision. FullRange = false: faster, with an error that grows like |x| eps. */
  template <Integer digits, bool FullRange = true, class ValueType, Integer N> inline Vec<ValueType,N> approx_tan(const Vec<ValueType,N>& x);

  /** Angle of the point (x, y), in [-pi, pi], as std::atan2. SpecialValues = false: faster, but x, y both infinite or both zero give NaN. */
  template <bool SpecialValues = true, class ValueType, Integer N> inline Vec<ValueType,N> atan2(const Vec<ValueType,N>& y, const Vec<ValueType,N>& x);

  /** x to the power y; vectorized with SVML, libmvec, or AVX2 with FMA and AVX-512 (about 3 to 5 ulp), else one element at a time. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> pow(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y);

  /** Integer value nearest x, not larger in magnitude. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> trunc(const Vec<ValueType,N>& x);

  /** Integer value nearest x, halfway cases away from zero. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> round(const Vec<ValueType,N>& x);

  /** Mask of the elements that are infinite. */
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType isinf(const Vec<ValueType,N>& x);

  /** Mask of the elements that are neither infinite nor NaN. */
  template <class ValueType, Integer N> inline typename Vec<ValueType,N>::MaskType isfinite(const Vec<ValueType,N>& x);

  /** Arc tangent. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> atan(const Vec<ValueType,N>& x);

  /** Arc sine. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> asin(const Vec<ValueType,N>& x);

  /** Arc cosine. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> acos(const Vec<ValueType,N>& x);

  /** sqrt(x^2 + y^2) without overflow or underflow in between, and inf if x or y is inf, as std::hypot. AvoidOverflow = false: sqrt(x x + y y), faster. */
  template <bool AvoidOverflow = true, class ValueType, Integer N> inline Vec<ValueType,N> hypot(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y);

  /** 2 to the power x. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> exp2(const Vec<ValueType,N>& x);

  /** Base-2 logarithm; vectorized where log is. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> log2(const Vec<ValueType,N>& x);

  /** Base-10 logarithm; vectorized where log is. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> log10(const Vec<ValueType,N>& x);

  /** Cube root, one element at a time. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> cbrt(const Vec<ValueType,N>& x);

  /** Hyperbolic sine. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> sinh(const Vec<ValueType,N>& x);

  /** Hyperbolic cosine. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> cosh(const Vec<ValueType,N>& x);

  /** Hyperbolic tangent. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> tanh(const Vec<ValueType,N>& x);

  /** Remainder of x/y with the sign of x, as std::fmod, one element at a time. */
  template <class ValueType, Integer N> inline Vec<ValueType,N> fmod(const Vec<ValueType,N>& x, const Vec<ValueType,N>& y);


  // Print
  template <class ValueType, Integer N> inline std::ostream& operator<<(std::ostream& os, const Vec<ValueType,N>& in);


  // Other operators
  template <class ValueType> inline void printb(const ValueType& x);

}

#endif // _SCTL_VEC_HPP_
