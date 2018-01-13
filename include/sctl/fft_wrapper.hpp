#ifndef _SCTL_FFT_WRAPPER_
#define _SCTL_FFT_WRAPPER_

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <vector>

#include SCTL_INCLUDE(common.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)
#include SCTL_INCLUDE(matrix.hpp)

namespace SCTL_NAMESPACE {


template <class ValueType> class Complex {
  public:

    Complex<ValueType> operator*(const Complex<ValueType>& x){
      Complex<ValueType> z;
      z.real = real * x.real - imag * x.imag;
      z.imag = imag * x.real - real * x.imag;
      return z;
    }

    Complex<ValueType> operator*(const ValueType& x){
      Complex<ValueType> z;
      z.real = real * x;
      z.imag = imag * x;
      return z;
    }

    Complex<ValueType> operator+(const Complex<ValueType>& x){
      Complex<ValueType> z;
      z.real = real + x.real;
      z.imag = imag + x.imag;
      return z;
    }

    Complex<ValueType> operator+(const ValueType& x){
      Complex<ValueType> z;
      z.real = real + x;
      z.imag = imag;
      return z;
    }

    Complex<ValueType> operator-(const Complex<ValueType>& x){
      Complex<ValueType> z;
      z.real = real - x.real;
      z.imag = imag - x.imag;
      return z;
    }

    Complex<ValueType> operator-(const ValueType& x){
      Complex<ValueType> z;
      z.real = real - x;
      z.imag = imag;
      return z;
    }

    ValueType real;
    ValueType imag;
};

template <class ValueType> Complex<ValueType> operator*(const ValueType& x, const Complex<ValueType>& y){
  Complex<ValueType> z;
  z.real = y.real * x;
  z.imag = y.imag * x;
  return z;
}

template <class ValueType> Complex<ValueType> operator+(const ValueType& x, const Complex<ValueType>& y){
  Complex<ValueType> z;
  z.real = y.real + x;
  z.imag = y.imag;
  return z;
}

template <class ValueType> Complex<ValueType> operator-(const ValueType& x, const Complex<ValueType>& y){
  Complex<ValueType> z;
  z.real = y.real - x;
  z.imag = y.imag;
  return z;
}



enum class FFT_Type {R2C, C2C, C2C_INV, C2R};

template <class ValueType> class FFT {

  typedef Complex<ValueType> ComplexType;

  struct FFTPlan {
    std::vector<Matrix<ValueType>> M;
    FFT_Type fft_type;
    Long howmany;
  };

 public:

  void Setup(FFT_Type fft_type, Long howmany, const Vector<Long>& dim_vec) {
    Long rank = dim_vec.Dim();
    plan.fft_type = fft_type;
    plan.howmany = howmany;
    plan.M.resize(0);

    if (fft_type == FFT_Type::R2C) {
      plan.M.push_back(fft_r2c(dim_vec[rank - 1]));
      for (Long i = rank - 2; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]));
    } else if (fft_type == FFT_Type::C2C) {
      for (Long i = rank - 1; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]));
    } else if (fft_type == FFT_Type::C2C_INV) {
      for (Long i = rank - 1; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]).Transpose());
    } else if (fft_type == FFT_Type::C2R) {
      for (Long i = rank - 2; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]).Transpose());
      plan.M.push_back(fft_c2r(dim_vec[rank - 1]));
    }

    Long N0 = howmany * 2;
    Long N1 = howmany * 2;
    for (const auto M : plan.M) {
      N0 = N0 * M.Dim(0) / 2;
      N1 = N1 * M.Dim(1) / 2;
    }
  }

  void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {

    Long howmany = plan.howmany;
    Long N0 = howmany * 2;
    Long N1 = howmany * 2;
    for (const auto M : plan.M) {
      N0 = N0 * M.Dim(0) / 2;
      N1 = N1 * M.Dim(1) / 2;
    }
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);

    Vector<ValueType> buff0(N0 + N1);
    Vector<ValueType> buff1(N0 + N1);
    Long rank = plan.M.size();
    if (rank <= 0) return;
    Long N = N0;

    if (plan.fft_type == FFT_Type::C2R) {
      const Matrix<ValueType>& M = plan.M[rank - 1];
      transpose<ComplexType>(buff0.begin(), in.begin(), N / M.Dim(0), M.Dim(0) / 2);

      for (Long i = 0; i < rank - 1; i++) {
        const Matrix<ValueType>& M = plan.M[i];
        Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff0.begin(), false);
        Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), buff1.begin(), false);
        Matrix<ValueType>::GEMM(vo, vi, M);
        N = N * M.Dim(1) / M.Dim(0);
        transpose<ComplexType>(buff0.begin(), buff1.begin(), N / M.Dim(1), M.Dim(1) / 2);
      }
      transpose<ComplexType>(buff1.begin(), buff0.begin(), N / howmany / 2, howmany);

      Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff1.begin(), false);
      Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), out.begin(), false);
      Matrix<ValueType>::GEMM(vo, vi, M);
    } else {
      memcopy(buff0.begin(), in.begin(), in.Dim());
      for (Long i = 0; i < rank; i++) {
        const Matrix<ValueType>& M = plan.M[i];
        Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff0.begin(), false);
        Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), buff1.begin(), false);
        Matrix<ValueType>::GEMM(vo, vi, M);
        N = N * M.Dim(1) / M.Dim(0);
        transpose<ComplexType>(buff0.begin(), buff1.begin(), N / M.Dim(1), M.Dim(1) / 2);
      }
      transpose<ComplexType>(out.begin(), buff0.begin(), N / howmany / 2, howmany);
    }
  }

  static void test() {
    Vector<Long> fft_dim;
    fft_dim.PushBack(2);
    fft_dim.PushBack(5);
    fft_dim.PushBack(3);

    if (1){ // R2C, C2R
      Vector<ValueType> v0, v1, v2;
      FFT<ValueType> myfft0, myfft1;
      myfft0.Setup(FFT_Type::R2C, 1, fft_dim, v0, v1);
      myfft1.Setup(FFT_Type::C2R, 1, fft_dim, v1, v2);
      for (int i = 0; i < v0.Dim(); i++) v0[i] = 1 + i;
      myfft0.Execute(v0, v1);
      myfft1.Execute(v1, v2);
      { // Print error
        ValueType err = 0;
        SCTL_ASSERT(v0.Dim() == v2.Dim());
        for (Long i=0;i<v0.Dim();i++) err = std::max(err, fabs(v0[i] - v2[i]));
        std::cout<<"Error : "<<err<<'\n';
      }
    }
    std::cout<<'\n';
    { // C2C, C2C_INV
      Vector<ValueType> v0, v1, v2;
      FFT<ValueType> myfft0, myfft1;
      myfft0.Setup(FFT_Type::C2C, 1, fft_dim, v0, v1);
      myfft1.Setup(FFT_Type::C2C_INV, 1, fft_dim, v1, v2);
      for (int i = 0; i < v0.Dim(); i++) v0[i] = 1 + i;
      myfft0.Execute(v0, v1);
      myfft1.Execute(v1, v2);
      { // Print error
        ValueType err = 0;
        SCTL_ASSERT(v0.Dim() == v2.Dim());
        for (Long i=0;i<v0.Dim();i++) err = std::max(err, fabs(v0[i] - v2[i]));
        std::cout<<"Error : "<<err<<'\n';
      }
    }
  }

 private:

  static Matrix<ValueType> fft_r2c(Long N0) {
    ValueType s = 1.0 / sqrt<ValueType>(N0);
    Long N1 = (N0 / 2 + 1);
    Matrix<ValueType> M(N0, 2 * N1);
    for (Long j = 0; j < N0; j++)
      for (Long i = 0; i < N1; i++) {
        M[j][2 * i + 0] = cos<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
        M[j][2 * i + 1] = sin<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
      }
    return M;
  }

  static Matrix<ValueType> fft_c2c(Long N0) {
    ValueType s = 1.0 / sqrt<ValueType>(N0);
    Matrix<ValueType> M(2 * N0, 2 * N0);
    for (Long i = 0; i < N0; i++)
      for (Long j = 0; j < N0; j++) {
        M[2 * i + 0][2 * j + 0] = cos<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
        M[2 * i + 1][2 * j + 0] = sin<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
        M[2 * i + 0][2 * j + 1] = -sin<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
        M[2 * i + 1][2 * j + 1] = cos<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
      }
    return M;
  }

  static Matrix<ValueType> fft_c2r(Long N0) {
    ValueType s = 1.0 / sqrt<ValueType>(N0);
    Long N1 = (N0 / 2 + 1);
    Matrix<ValueType> M(2 * N1, N0);
    for (Long i = 0; i < N1; i++) {
      for (Long j = 0; j < N0; j++) {
        M[2 * i + 0][j] = 2 * cos<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
        M[2 * i + 1][j] = 2 * sin<ValueType>(j * i * (1.0 / N0) * 2.0 * const_pi<ValueType>())*s;
      }
    }
    if (N1 > 0) {
      for (Long j = 0; j < N0; j++) {
        M[0][j] = M[0][j] * 0.5;
        M[1][j] = M[1][j] * 0.5;
      }
    }
    if (N0 % 2 == 0) {
      for (Long j = 0; j < N0; j++) {
        M[2 * N1 - 2][j] = M[2 * N1 - 2][j] * 0.5;
        M[2 * N1 - 1][j] = M[2 * N1 - 1][j] * 0.5;
      }
    }
    return M;
  }

  template <class T> static void transpose(Iterator<ValueType> out, ConstIterator<ValueType> in, Long N0, Long N1) {
    Matrix<T> M0(N0, N1, (Iterator<T>)in, false);
    Matrix<T> M1(N1, N0, (Iterator<T>)out, false);
    M1 = M0.Transpose();
  }

  FFTPlan plan;
};

}  // end namespace

#endif  //_SCTL_FFT_WRAPPER_
