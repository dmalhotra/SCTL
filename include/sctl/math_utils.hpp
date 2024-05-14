#ifndef _SCTL_MATH_UTILS_
#define _SCTL_MATH_UTILS_

#include <sctl/common.hpp>

#include <cmath>
#include <ostream>

namespace SCTL_NAMESPACE {

/**
 * @brief Returns the number of significant bits in the representation of the template type.
 *
 * @tparam Real The template type.
 * @return constexpr Integer The number of significant bits.
 */
template <class Real> inline constexpr Integer significant_bits();

/**
 * @brief Returns the machine epsilon (the difference between 1 and the smallest value greater than 1 that is representable).
 *
 * @tparam Real The template type.
 * @return constexpr Real The machine epsilon.
 */
template <class Real> inline constexpr Real machine_eps();

/**
 * @brief Converts a string to a real number.
 *
 * @tparam Real The template type.
 * @param str The input string.
 * @return Real The real number parsed from the string.
 */
template <class Real> Real atoreal(const char* str);

/**
 * @brief Returns the mathematical constant pi.
 *
 * @tparam Real The template type.
 * @return constexpr Real The value of pi.
 */
template <class Real> inline constexpr Real const_pi() { return (Real)3.1415926535897932384626433832795028841L; }

/**
 * @brief Returns the mathematical constant e.
 *
 * @tparam Real The template type.
 * @return constexpr Real The value of e.
 */
template <class Real> inline constexpr Real const_e() { return (Real)2.7182818284590452353602874713526624977L; }

/**
 * @brief Returns the absolute value of the input.
 *
 * @tparam Real The template type.
 * @param a The input value.
 * @return Real The absolute value of the input.
 */
template <class Real> inline Real fabs(const Real a) { return (Real)std::fabs(a); }

/**
 * @brief Truncates the input real number to the nearest integer towards zero.
 *
 * @tparam Real The template type.
 * @param a The input real number.
 * @return Real The truncated real number.
 */
template <class Real> inline Real trunc(const Real a) { return (Real)std::trunc(a); }

/**
 * @brief Rounds the input real number to the nearest integer.
 *
 * @tparam Real The template type.
 * @param a The input real number.
 * @return Real The rounded real number.
 */
template <class Real> inline Real round(const Real a) { return (Real)std::round(a); }

/**
 * @brief Rounds the input real number down to the nearest integer.
 *
 * @tparam Real The template type.
 * @param a The input real number.
 * @return Real The floor of the real number.
 */
template <class Real> inline Real floor(const Real a) { return (Real)std::floor(a); }

/**
 * @brief Rounds the input real number up to the nearest integer.
 *
 * @tparam Real The template type.
 * @param a The input real number.
 * @return Real The ceil of the real number.
 */
template <class Real> inline Real ceil(const Real a) { return (Real)std::ceil(a); }

/**
 * @brief Computes the square root of the input real number.
 *
 * @tparam Real The template type.
 * @param a The input real number.
 * @return Real The square root of the real number.
 */
template <class Real> inline Real sqrt(const Real a) { return (Real)std::sqrt(a); }

/**
 * @brief Computes the sine of the input angle.
 *
 * @tparam Real The template type.
 * @param a The input angle in radians.
 * @return Real The sine of the angle.
 */
template <class Real> inline Real sin(const Real a) { return (Real)std::sin(a); }

/**
 * @brief Computes the cosine of the input angle.
 *
 * @tparam Real The template type.
 * @param a The input angle in radians.
 * @return Real The cosine of the angle.
 */
template <class Real> inline Real cos(const Real a) { return (Real)std::cos(a); }

/**
 * @brief Computes the tangent of the input angle.
 *
 * @tparam Real The template type.
 * @param a The input angle in radians.
 * @return Real The tangent of the angle.
 */
template <class Real> inline Real tan(const Real a) { return (Real)std::tan(a); }

/**
 * @brief Computes the arcsine of the input value.
 *
 * @tparam Real The template type.
 * @param a The input value.
 * @return Real The arcsine in radians.
 */
template <class Real> inline Real asin(const Real a) { return (Real)std::asin(a); }

/**
 * @brief Computes the arccosine of the input value.
 *
 * @tparam Real The template type.
 * @param a The input value.
 * @return Real The arccosine in radians.
 */
template <class Real> inline Real acos(const Real a) { return (Real)std::acos(a); }

/**
 * @brief Computes the arctangent of the input value.
 *
 * @tparam Real The template type.
 * @param a The input value.
 * @return Real The arctangent in radians.
 */
template <class Real> inline Real atan(const Real a) { return (Real)std::atan(a); }

/**
 * @brief Computes the arctangent of the ratio of two input values.
 *
 * @tparam Real The template type.
 * @param a The numerator.
 * @param b The denominator.
 * @return Real The arctangent in radians.
 */
template <class Real> inline Real atan2(const Real a, const Real b) { return (Real)std::atan2(a, b); }

/**
 * @brief Computes the remainder of the division of two input values.
 *
 * @tparam Real The template type.
 * @param a The dividend.
 * @param b The divisor.
 * @return Real The remainder.
 */
template <class Real> inline Real fmod(const Real a, const Real b) { return (Real)std::fmod(a, b); }

/**
 * @brief Computes the exponential function of the input value.
 *
 * @tparam Real The template type.
 * @param a The input value.
 * @return Real The exponential value.
 */
template <class Real> inline Real exp(const Real a) { return (Real)std::exp(a); }

/**
 * @brief Computes the natural logarithm of the input value.
 *
 * @tparam Real The template type.
 * @param a The input value.
 * @return Real The natural logarithm.
 */
template <class Real> inline Real log(const Real a) { return (Real)std::log(a); }

/**
 * @brief Computes the base-2 logarithm of the input value.
 *
 * @tparam Real The template type.
 * @param a The input value.
 * @return Real The base-2 logarithm.
 */
template <class Real> inline Real log2(const Real a) { return (Real)std::log2(a); }

/**
 * @brief Computes the power of a base raised to an exponent.
 *
 * @tparam Real The template type of the base.
 * @tparam ExpType The template type of the exponent.
 * @param b The base.
 * @param e The exponent.
 * @return Real The result of the operation.
 */
template <class Real, class ExpType> inline Real pow(const Real b, const ExpType e);

/**
 * @brief Computes the power of a base raised to a compile-time constant exponent.
 *
 * @tparam e The compile-time constant exponent.
 * @tparam ValueType The template type of the base.
 * @param b The base.
 * @return constexpr ValueType The result of the operation.
 */
template <Long e, class ValueType> inline constexpr ValueType pow(ValueType b);


#ifdef SCTL_QUAD_T
/**
 * @brief Class representing a quadruple precision floating-point number.
 */
class QuadReal {
  typedef SCTL_QUAD_T QuadRealType;

  public:
    /**
     * @brief Default constructor.
     */
    QuadReal() = default;

    /**
     * @brief Copy constructor.
     *
     * @param v The value to copy.
     */
    constexpr QuadReal(const QuadReal& v) = default;

    /**
     * @brief Copy assignment operator.
     *
     * @param The value to copy.
     * @return QuadReal& The reference to the copied object.
     */
    QuadReal& operator=(const QuadReal&) = default;

    /**
     * @brief Destructor.
     */
    ~QuadReal() = default;

    /**
     * @brief Constructor with explicit conversion from another type.
     *
     * @tparam ValueType The template type of the value.
     * @param v The value to convert.
     */
    template <class ValueType> constexpr QuadReal(ValueType v) : val((QuadRealType)v) {}

    /**
     * @brief Explicit conversion operator to another type.
     *
     * @tparam ValueType The template type of the value.
     * @return constexpr ValueType The converted value.
     */
    template <class ValueType> explicit constexpr operator ValueType() const { return (ValueType)val; }

    /**
     * @brief Addition assignment operator.
     *
     * @param x The value to add.
     * @return QuadReal& The reference to the modified object.
     */
    QuadReal& operator+=(const QuadReal& x) { val += x.val; return *this; }

    /**
     * @brief Subtraction assignment operator.
     *
     * @param x The value to subtract.
     * @return QuadReal& The reference to the modified object.
     */
    QuadReal& operator-=(const QuadReal& x) { val -= x.val; return *this; }

    /**
     * @brief Multiplication assignment operator.
     *
     * @param x The value to multiply by.
     * @return QuadReal& The reference to the modified object.
     */
    QuadReal& operator*=(const QuadReal& x) { val *= x.val; return *this; }

    /**
     * @brief Division assignment operator.
     *
     * @param x The value to divide by.
     * @return QuadReal& The reference to the modified object.
     */
    QuadReal& operator/=(const QuadReal& x) { val /= x.val; return *this; }

    /**
     * @brief Addition operator.
     *
     * @param x The value to add.
     * @return constexpr QuadReal The result of the addition.
     */
    constexpr QuadReal operator+(const QuadReal& x) const { return QuadReal(val + x.val); }

    /**
     * @brief Subtraction operator.
     *
     * @param x The value to subtract.
     * @return constexpr QuadReal The result of the subtraction.
     */
    constexpr QuadReal operator-(const QuadReal& x) const { return QuadReal(val - x.val); }

    /**
     * @brief Multiplication operator.
     *
     * @param x The value to multiply by.
     * @return constexpr QuadReal The result of the multiplication.
     */
    constexpr QuadReal operator*(const QuadReal& x) const { return QuadReal(val * x.val); }

    /**
     * @brief Division operator.
     *
     * @param x The value to divide by.
     * @return constexpr QuadReal The result of the division.
     */
    constexpr QuadReal operator/(const QuadReal& x) const { return QuadReal(val / x.val); }

    /**
     * @brief Unary negation operator.
     *
     * @return constexpr QuadReal The negated value.
     */
    constexpr QuadReal operator-() const { return QuadReal(-val); }

    /**
     * @brief Less than comparison operator.
     *
     * @param x The value to compare with.
     * @return constexpr bool True if less than, otherwise false.
     */
    constexpr bool operator< (const QuadReal& x) const { return val <  x.val; }

    /**
     * @brief Greater than comparison operator.
     *
     * @param x The value to compare with.
     * @return constexpr bool True if greater than, otherwise false.
     */
    constexpr bool operator> (const QuadReal& x) const { return val >  x.val; }

    /**
     * @brief Not equal comparison operator.
     *
     * @param x The value to compare with.
     * @return constexpr bool True if not equal, otherwise false.
     */
    constexpr bool operator!=(const QuadReal& x) const { return val != x.val; }

    /**
     * @brief Equal comparison operator.
     *
     * @param x The value to compare with.
     * @return constexpr bool True if equal, otherwise false.
     */
    constexpr bool operator==(const QuadReal& x) const { return val == x.val; }

    /**
     * @brief Less than or equal comparison operator.
     *
     * @param x The value to compare with.
     * @return constexpr bool True if less than or equal, otherwise false.
     */
    constexpr bool operator<=(const QuadReal& x) const { return val <= x.val; }

    /**
     * @brief Greater than or equal comparison operator.
     *
     * @param x The value to compare with.
     * @return constexpr bool True if greater than or equal, otherwise false.
     */
    constexpr bool operator>=(const QuadReal& x) const { return val >= x.val; }

    /**
     * @brief Friend addition operator for QuadRealType and QuadReal.
     *
     * @param a The value to add.
     * @param b The value to add.
     * @return constexpr QuadReal The result of the addition.
     */
    constexpr friend QuadReal operator+(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) + b; }

    /**
     * @brief Friend subtraction operator for QuadRealType and QuadReal.
     *
     * @param a The value to subtract.
     * @param b The value to subtract.
     * @return constexpr QuadReal The result of the subtraction.
     */
    constexpr friend QuadReal operator-(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) - b; }

    /**
     * @brief Friend multiplication operator for QuadRealType and QuadReal.
     *
     * @param a The value to multiply by.
     * @param b The value to multiply.
     * @return constexpr QuadReal The result of the multiplication.
     */
    constexpr friend QuadReal operator*(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) * b; }

    /**
     * @brief Friend division operator for QuadRealType and QuadReal.
     *
     * @param a The value to divide by.
     * @param b The value to divide.
     * @return constexpr QuadReal The result of the division.
     */
    constexpr friend QuadReal operator/(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) / b; }

    /**
     * @brief Friend less than comparison operator for QuadRealType and QuadReal.
     *
     * @param a The value to compare with.
     * @param b The value to compare.
     * @return constexpr bool True if less than, otherwise false.
     */
    constexpr friend bool operator< (const QuadRealType& a, const QuadReal& b) { return QuadReal(a) <  b; }

    /**
     * @brief Friend greater than comparison operator for QuadRealType and QuadReal.
     *
     * @param a The value to compare with.
     * @param b The value to compare.
     * @return constexpr bool True if greater than, otherwise false.
     */
    constexpr friend bool operator> (const QuadRealType& a, const QuadReal& b) { return QuadReal(a) >  b; }

    /**
     * @brief Friend not equal comparison operator for QuadRealType and QuadReal.
     *
     * @param a The value to compare with.
     * @param b The value to compare.
     * @return constexpr bool True if not equal, otherwise false.
     */
    constexpr friend bool operator!=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) != b; }

    /**
     * @brief Friend equal comparison operator for QuadRealType and QuadReal.
     *
     * @param a The value to compare with.
     * @param b The value to compare.
     * @return constexpr bool True if equal, otherwise false.
     */
    constexpr friend bool operator==(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) == b; }

    /**
     * @brief Friend less than or equal comparison operator for QuadRealType and QuadReal.
     *
     * @param a The value to compare with.
     * @param b The value to compare.
     * @return constexpr bool True if less than or equal, otherwise false.
     */
    constexpr friend bool operator<=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) <= b; }

    /**
     * @brief Friend greater than or equal comparison operator for QuadRealType and QuadReal.
     *
     * @param a The value to compare with.
     * @param b The value to compare.
     * @return constexpr bool True if greater than or equal, otherwise false.
     */
    constexpr friend bool operator>=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) >= b; }

    /**
     * @brief Friend function for truncating a QuadReal.
     *
     * @param x The QuadReal to truncate.
     * @return QuadReal The truncated QuadReal.
     */
    friend QuadReal trunc<QuadReal>(const QuadReal);

  private:
    QuadRealType val; ///< The value of the QuadReal.
};

/**
 * @brief Overloads the output stream operator for QuadReal objects.
 *
 * @param output The output stream.
 * @param x The QuadReal object to be output.
 * @return std::ostream& The modified output stream.
 */
inline std::ostream& operator<<(std::ostream& output, const QuadReal& x);

/**
 * @brief Overloads the input stream operator for QuadReal objects.
 *
 * @param inputstream The input stream.
 * @param x The QuadReal object to store the input.
 * @return std::istream& The modified input stream.
 */
inline std::istream& operator>>(std::istream& inputstream, QuadReal& x);

#endif

}  // end namespace

#include SCTL_INCLUDE(math_utils.txx)

#endif  //_SCTL_MATH_UTILS_HPP_
