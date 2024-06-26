#ifndef _SCTL_COMPLEX_HPP_
#define _SCTL_COMPLEX_HPP_

#include <ostream>          // for ostream

#include "sctl/common.hpp"  // for sctl

namespace sctl {

  /**
   * A template class for representing complex numbers.  This class provides functionalities for performing arithmetic
   * operations on complex numbers, including addition, subtraction, multiplication, and division.
   *
   * @tparam ValueType The type of the real and imaginary parts of the complex number.
   */
  template <class ValueType> class Complex {
    public:
      /**
       * Constructs a complex number with both real and imaginary parts initialized to zero.
       *
       * @param r The real part of the complex number.
       * @param i The imaginary part of the complex number.
       */
      Complex(ValueType r = 0, ValueType i = 0);

      /**
       * Unary negation operator.
       *
       * @return The negation of the complex number.
       */
      Complex<ValueType> operator-() const;

      /**
       * Conjugate of the complex number.
       *
       * @return The conjugate of the complex number.
       */
      Complex<ValueType> conj() const;

      /**
       * Checks if two complex numbers are equal.
       *
       * @param x The complex number to compare.
       * @return True if both complex numbers are equal, false otherwise.
       */
      bool operator==(const Complex<ValueType>& x) const;

      /**
       * Checks if two complex numbers are not equal.
       *
       * @param x The complex number to compare.
       * @return True if both complex numbers are not equal, false otherwise.
       */
      bool operator!=(const Complex<ValueType>& x) const;

      /**
       * Adds another complex number to this complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to add.
       */
      template <class ScalarType> void operator+=(const Complex<ScalarType>& x);

      /**
       * Subtracts another complex number from this complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to subtract.
       */
      template <class ScalarType> void operator-=(const Complex<ScalarType>& x);

      /**
       * Multiplies another complex number with this complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to multiply with.
       */
      template <class ScalarType> void operator*=(const Complex<ScalarType>& x);

      /**
       * Divides this complex number by another complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to divide by.
       */
      template <class ScalarType> void operator/=(const Complex<ScalarType>& x);

      /**
       * Adds a scalar value to this complex number.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param x The scalar value to add.
       * @return The result of the addition.
       */
      template <class ScalarType> Complex<ValueType> operator+(const ScalarType& x) const;

      /**
       * Subtracts a scalar value from this complex number.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param x The scalar value to subtract.
       * @return The result of the subtraction.
       */
      template <class ScalarType> Complex<ValueType> operator-(const ScalarType& x) const;

      /**
       * Multiplies this complex number by a scalar value.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param x The scalar value to multiply by.
       * @return The result of the multiplication.
       */
      template <class ScalarType> Complex<ValueType> operator*(const ScalarType& x) const;

      /**
       * Divides this complex number by a scalar value.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param y The scalar value to divide by.
       * @return The result of the division.
       */
      template <class ScalarType> Complex<ValueType> operator/(const ScalarType& y) const;

      /**
       * Adds another complex number to this complex number.
       *
       * @param x The other complex number to add.
       * @return The result of the addition.
       */
      Complex<ValueType> operator+(const Complex<ValueType>& x) const;

      /**
       * Subtracts another complex number from this complex number.
       *
       * @param x The other complex number to subtract.
       * @return The result of the subtraction.
       */
      Complex<ValueType> operator-(const Complex<ValueType>& x) const;

      /**
       * Multiplies another complex number with this complex number.
       *
       * @param x The other complex number to multiply with.
       * @return The result of the multiplication.
       */
      Complex<ValueType> operator*(const Complex<ValueType>& x) const;

      /**
       * Divides this complex number by another complex number.
       *
       * @param y The other complex number to divide by.
       * @return The result of the division.
       */
      Complex<ValueType> operator/(const Complex<ValueType>& y) const;

      ValueType real; ///< The real part of the complex number.
      ValueType imag; ///< The imaginary part of the complex number.
  };

  template <class ValueType> std::ostream& operator<<(std::ostream& output, const Complex<ValueType>& V);

}  // end namespace

#endif // _SCTL_COMPLEX_HPP_
