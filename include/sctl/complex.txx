
namespace SCTL_NAMESPACE {

  template <class ValueType> Complex<ValueType>::Complex(ValueType r, ValueType i) : real(r), imag(i) {}

  template <class ValueType> Complex<ValueType> Complex<ValueType>::operator-() const {
    Complex<ValueType> z;
    z.real = -real;
    z.imag = -imag;
    return z;
  }

  template <class ValueType> Complex<ValueType> Complex<ValueType>::conj() const {
    Complex<ValueType> z;
    z.real = real;
    z.imag = -imag;
    return z;
  }


  template <class ValueType> bool Complex<ValueType>::operator==(const Complex<ValueType>& x) const {
    return real == x.real && imag == x.imag;
  }

  template <class ValueType> bool Complex<ValueType>::operator!=(const Complex<ValueType>& x) const {
    return !((*this) == x);
  }


  template <class ValueType> template <class ScalarType> void Complex<ValueType>::operator+=(const Complex<ScalarType>& x) {
    (*this) = (*this) + x;
  }

  template <class ValueType> template <class ScalarType> void Complex<ValueType>::operator-=(const Complex<ScalarType>& x) {
    (*this) = (*this) - x;
  }

  template <class ValueType> template <class ScalarType> void Complex<ValueType>::operator*=(const Complex<ScalarType>& x) {
    (*this) = (*this) * x;
  }

  template <class ValueType> template <class ScalarType> void Complex<ValueType>::operator/=(const Complex<ScalarType>& x) {
    (*this) = (*this) / x;
  }


  template <class ValueType> template <class ScalarType> Complex<ValueType> Complex<ValueType>::operator+(const ScalarType& x) const {
    Complex<ValueType> z;
    z.real = real + x;
    z.imag = imag;
    return z;
  }

  template <class ValueType> template <class ScalarType> Complex<ValueType> Complex<ValueType>::operator-(const ScalarType& x) const {
    Complex<ValueType> z;
    z.real = real - x;
    z.imag = imag;
    return z;
  }

  template <class ValueType> template <class ScalarType> Complex<ValueType> Complex<ValueType>::operator*(const ScalarType& x) const {
    Complex<ValueType> z;
    z.real = real * x;
    z.imag = imag * x;
    return z;
  }

  template <class ValueType> template <class ScalarType> Complex<ValueType> Complex<ValueType>::operator/(const ScalarType& y) const {
    Complex<ValueType> z;
    z.real = real / y;
    z.imag = imag / y;
    return z;
  }


  template <class ValueType> Complex<ValueType> Complex<ValueType>::operator+(const Complex<ValueType>& x) const {
    Complex<ValueType> z;
    z.real = real + x.real;
    z.imag = imag + x.imag;
    return z;
  }

  template <class ValueType> Complex<ValueType> Complex<ValueType>::operator-(const Complex<ValueType>& x) const {
    Complex<ValueType> z;
    z.real = real - x.real;
    z.imag = imag - x.imag;
    return z;
  }

  template <class ValueType> Complex<ValueType> Complex<ValueType>::operator*(const Complex<ValueType>& x) const {
    Complex<ValueType> z;
    z.real = real * x.real - imag * x.imag;
    z.imag = imag * x.real + real * x.imag;
    return z;
  }

  template <class ValueType> Complex<ValueType> Complex<ValueType>::operator/(const Complex<ValueType>& y) const {
    Complex<ValueType> z;
    ValueType y_inv = 1 / (y.real * y.real + y.imag * y.imag);
    z.real = (y.real * real + y.imag * imag) * y_inv;
    z.imag = (y.real * imag - y.imag * real) * y_inv;
    return z;
  }

  template <class ScalarType, class ValueType> Complex<ValueType> operator*(const ScalarType& x, const Complex<ValueType>& y) {
    Complex<ValueType> z;
    z.real = y.real * x;
    z.imag = y.imag * x;
    return z;
  }

  template <class ScalarType, class ValueType> Complex<ValueType> operator+(const ScalarType& x, const Complex<ValueType>& y) {
    Complex<ValueType> z;
    z.real = y.real + x;
    z.imag = y.imag;
    return z;
  }

  template <class ScalarType, class ValueType> Complex<ValueType> operator-(const ScalarType& x, const Complex<ValueType>& y) {
    Complex<ValueType> z;
    z.real = y.real - x;
    z.imag = y.imag;
    return z;
  }

  template <class ScalarType, class ValueType> Complex<ValueType> operator/(const ScalarType& x, const Complex<ValueType>& y) {
    Complex<ValueType> z;
    ValueType y_inv = 1 / (y.real * y.real + y.imag * y.imag);
    z.real =  (y.real * x) * y_inv;
    z.imag = -(y.imag * x) * y_inv;
    return z;
  }

  template <class ValueType> std::ostream& operator<<(std::ostream& output, const Complex<ValueType>& V) {
    output << "(" << V.real <<"," << V.imag << ")";
    return output;
  }

}  // end namespace

