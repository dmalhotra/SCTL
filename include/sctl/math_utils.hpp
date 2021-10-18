#ifndef _SCTL_MATH_UTILS_
#define _SCTL_MATH_UTILS_

#include SCTL_INCLUDE(common.hpp)

#include <cmath>
#include <ostream>

namespace SCTL_NAMESPACE {

template <class Real> Real machine_eps();

template <class Real> static Real atoreal(const char* str);

template <class Real> inline constexpr Real const_pi() { return (Real)3.1415926535897932384626433832795028841L; }

template <class Real> inline constexpr Real const_e() { return (Real)2.7182818284590452353602874713526624977L; }

template <class Real> inline Real fabs(const Real a) { return (Real)::fabs(a); }

template <class Real> inline Real round(const Real a) { return (Real)::round(a); }

template <class Real> inline Real sqrt(const Real a) { return (Real)::sqrt(a); }

template <class Real> inline Real sin(const Real a) { return (Real)::sin(a); }

template <class Real> inline Real cos(const Real a) { return (Real)::cos(a); }

template <class Real> inline Real acos(const Real a) { return (Real)::acos(a); }

template <class Real> inline Real exp(const Real a) { return (Real)::exp(a); }

template <class Real> inline Real log(const Real a) { return (Real)::log(a); }

template <class Real, class ExpType> inline Real pow(const Real b, const ExpType e);

template <Long e, class ValueType> inline constexpr ValueType pow(ValueType b);


#ifdef SCTL_QUAD_T
class QuadReal {
  typedef SCTL_QUAD_T QuadRealType;
  public:

    QuadReal() = default;
    constexpr QuadReal(const QuadReal& v) = default;
    QuadReal& operator=(const QuadReal&) = default;
    ~QuadReal() = default;

    template <class ValueType> constexpr QuadReal(ValueType v) : val((QuadRealType)v) {}
    template <class ValueType> explicit constexpr operator ValueType() const { return (ValueType)val; }


    QuadReal& operator+=(const QuadReal& x) { val += x.val; return *this; }
    QuadReal& operator-=(const QuadReal& x) { val -= x.val; return *this; }
    QuadReal& operator*=(const QuadReal& x) { val *= x.val; return *this; }
    QuadReal& operator/=(const QuadReal& x) { val /= x.val; return *this; }

    constexpr QuadReal operator+(const QuadReal& x) const { return QuadReal(val + x.val); }
    constexpr QuadReal operator-(const QuadReal& x) const { return QuadReal(val - x.val); }
    constexpr QuadReal operator*(const QuadReal& x) const { return QuadReal(val * x.val); }
    constexpr QuadReal operator/(const QuadReal& x) const { return QuadReal(val / x.val); }

    constexpr QuadReal operator-() const { return QuadReal(-val); }

    constexpr bool operator< (const QuadReal& x) const { return val <  x.val; }
    constexpr bool operator> (const QuadReal& x) const { return val >  x.val; }
    constexpr bool operator!=(const QuadReal& x) const { return val != x.val; }
    constexpr bool operator==(const QuadReal& x) const { return val == x.val; }
    constexpr bool operator<=(const QuadReal& x) const { return val <= x.val; }
    constexpr bool operator>=(const QuadReal& x) const { return val >= x.val; }


    constexpr friend QuadReal operator+(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) + b; }
    constexpr friend QuadReal operator-(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) - b; }
    constexpr friend QuadReal operator*(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) * b; }
    constexpr friend QuadReal operator/(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) / b; }

    constexpr friend bool operator< (const QuadRealType& a, const QuadReal& b) { return QuadReal(a) <  b; }
    constexpr friend bool operator> (const QuadRealType& a, const QuadReal& b) { return QuadReal(a) >  b; }
    constexpr friend bool operator!=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) != b; }
    constexpr friend bool operator==(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) == b; }
    constexpr friend bool operator<=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) <= b; }
    constexpr friend bool operator>=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) >= b; }

    friend QuadReal round(const QuadReal& x) {
      #ifdef __SIZEOF_INT128__
      return (__int128)((x+(0.5-(x<0))).val);
      #else
      return (int64_t)((x+(0.5-(x<0))).val);
      #endif
    }

  private:
    QuadRealType val;
};

inline std::ostream& operator<<(std::ostream& output, const QuadReal& x);
inline std::istream& operator>>(std::istream& inputstream, QuadReal& x);
#endif

}  // end namespace

#include SCTL_INCLUDE(math_utils.txx)

#endif  //_SCTL_MATH_UTILS_HPP_
