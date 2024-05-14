#ifndef _SCTL_COMPLEX_
#define _SCTL_COMPLEX_

#include <iostream>

#include <sctl/common.hpp>

namespace SCTL_NAMESPACE {

  /**
   * @brief A template class for representing complex numbers.
   *
   * This class provides functionalities for performing arithmetic operations
   * on complex numbers, including addition, subtraction, multiplication, and division.
   *
   * @tparam ValueType The type of the real and imaginary parts of the complex number.
   */
  template <class ValueType> class Complex {
    public:
      /**
       * @brief Default constructor.
       *
       * Constructs a complex number with both real and imaginary parts initialized to zero.
       *
       * @param r The real part of the complex number.
       * @param i The imaginary part of the complex number.
       */
      Complex<ValueType>(ValueType r = 0, ValueType i = 0);

      /**
       * @brief Unary negation operator.
       *
       * Returns the negation of the complex number.
       *
       * @return The negation of the complex number.
       */
      Complex<ValueType> operator-() const;

      /**
       * @brief Conjugate of the complex number.
       *
       * Returns the conjugate of the complex number.
       *
       * @return The conjugate of the complex number.
       */
      Complex<ValueType> conj() const;

      /**
       * @brief Equality comparison operator.
       *
       * Checks if two complex numbers are equal.
       *
       * @param x The complex number to compare.
       * @return True if both complex numbers are equal, false otherwise.
       */
      bool operator==(const Complex<ValueType>& x) const;

      /**
       * @brief Inequality comparison operator.
       *
       * Checks if two complex numbers are not equal.
       *
       * @param x The complex number to compare.
       * @return True if both complex numbers are not equal, false otherwise.
       */
      bool operator!=(const Complex<ValueType>& x) const;

      /**
       * @brief Addition assignment operator with another complex number.
       *
       * Adds another complex number to this complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to add.
       */
      template <class ScalarType> void operator+=(const Complex<ScalarType>& x);

      /**
       * @brief Subtraction assignment operator with another complex number.
       *
       * Subtracts another complex number from this complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to subtract.
       */
      template <class ScalarType> void operator-=(const Complex<ScalarType>& x);

      /**
       * @brief Multiplication assignment operator with another complex number.
       *
       * Multiplies another complex number with this complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to multiply with.
       */
      template <class ScalarType> void operator*=(const Complex<ScalarType>& x);

      /**
       * @brief Division assignment operator with another complex number.
       *
       * Divides this complex number by another complex number.
       *
       * @tparam ScalarType The type of the other complex number.
       * @param x The other complex number to divide by.
       */
      template <class ScalarType> void operator/=(const Complex<ScalarType>& x);

      /**
       * @brief Addition operator with a scalar value.
       *
       * Adds a scalar value to this complex number.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param x The scalar value to add.
       * @return The result of the addition.
       */
      template <class ScalarType> Complex<ValueType> operator+(const ScalarType& x) const;

      /**
       * @brief Subtraction operator with a scalar value.
       *
       * Subtracts a scalar value from this complex number.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param x The scalar value to subtract.
       * @return The result of the subtraction.
       */
      template <class ScalarType> Complex<ValueType> operator-(const ScalarType& x) const;

      /**
       * @brief Multiplication operator with a scalar value.
       *
       * Multiplies this complex number by a scalar value.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param x The scalar value to multiply by.
       * @return The result of the multiplication.
       */
      template <class ScalarType> Complex<ValueType> operator*(const ScalarType& x) const;

      /**
       * @brief Division operator with a scalar value.
       *
       * Divides this complex number by a scalar value.
       *
       * @tparam ScalarType The type of the scalar value.
       * @param y The scalar value to divide by.
       * @return The result of the division.
       */
      template <class ScalarType> Complex<ValueType> operator/(const ScalarType& y) const;

      /**
       * @brief Addition operator with another complex number.
       *
       * Adds another complex number to this complex number.
       *
       * @param x The other complex number to add.
       * @return The result of the addition.
       */
      Complex<ValueType> operator+(const Complex<ValueType>& x) const;

      /**
       * @brief Subtraction operator with another complex number.
       *
       * Subtracts another complex number from this complex number.
       *
       * @param x The other complex number to subtract.
       * @return The result of the subtraction.
       */
      Complex<ValueType> operator-(const Complex<ValueType>& x) const;

      /**
       * @brief Multiplication operator with another complex number.
       *
       * Multiplies another complex number with this complex number.
       *
       * @param x The other complex number to multiply with.
       * @return The result of the multiplication.
       */
      Complex<ValueType> operator*(const Complex<ValueType>& x) const;

      /**
       * @brief Division operator with another complex number.
       *
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

#include SCTL_INCLUDE(complex.txx)

#endif  //_SCTL_COMPLEX_
