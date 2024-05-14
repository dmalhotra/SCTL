#ifndef _SCTL_FFT_WRAPPER_
#define _SCTL_FFT_WRAPPER_

#if defined(SCTL_HAVE_FFTW) || defined(SCTL_HAVE_FFTWF)
#include <fftw3.h>
#ifdef SCTL_FFTW3_MKL
#include <fftw3_mkl.h>
#endif
#endif

#include <iostream>

#include <sctl/common.hpp>
#include SCTL_INCLUDE(complex.hpp)
#include SCTL_INCLUDE(math_utils.hpp)

namespace SCTL_NAMESPACE {

  template <class ValueType> struct FFTPlan;

  /**
   * Enum class representing different types of FFT transformations.
   */
  enum class FFT_Type {R2C, C2C, C2C_INV, C2R};

  /**
   *Wrapper class for FFTW.
   *
   * @tparam ValueType The value type of the FFT data.
   */
  template <class ValueType> class FFT {
    typedef Complex<ValueType> ComplexType;

    public:

    FFT();

    ~FFT() = default;

    // Delete copy constructor and assignment operator to prevent copying FFT objects
    FFT (const FFT&) = delete;
    FFT& operator= (const FFT&) = delete;

    /**
     * Dimensions of the input and output array are given by Dim(0) and Dim(1) respectively.
     *
     * @return The dimension of the FFT operator.
     */
    Long Dim(Integer i) const;

    /**
     * Setup the FFT operator.
     *
     * @param[in] fft_type The type of transform.
     *
     * @param[in] howmany Number of transforms to compute.
     *
     * @param[in] dim_vec Dimensions of the input data.
     *
     * @param[in] Nthreads Number of threads (default is 1).
     */
    void Setup(FFT_Type fft_type, Long howmany, const Vector<Long>& dim_vec, Integer Nthreads = 1);

    /**
     * Execute the FFT transform.
     *
     * @param[in] in the input data vector.
     *
     * @param[out] out the output data vector.
     */
    void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const;

    /**
     * Test the FFT implementation.
     */
    static void test() {
      const auto inf_norm = [](const Vector<ValueType>& v) {
        ValueType max_val = 0;
        for (const auto& x : v) max_val = std::max<ValueType>(max_val, fabs(x));
        return max_val;
      };

      Vector<Long> fft_dim;
      fft_dim.PushBack(2);
      fft_dim.PushBack(5);
      fft_dim.PushBack(3);
      Long howmany = 3;

      { // R2C, C2R
        FFT myfft0, myfft1;
        myfft0.Setup(FFT_Type::R2C, howmany, fft_dim);
        myfft1.Setup(FFT_Type::C2R, howmany, fft_dim);
        Vector<ValueType> v0(myfft0.Dim(0)), v1, v2;
        for (int i = 0; i < v0.Dim(); i++) v0[i] = (1 + i) / (ValueType)v0.Dim();
        myfft0.Execute(v0, v1);
        myfft1.Execute(v1, v2);

        const auto err = inf_norm(v2-v0);
        std::cout<<"Error : "<<err<<'\n';
        SCTL_ASSERT(err < machine_eps<ValueType>() * 64);
      }

      { // C2C, C2C_INV
        FFT myfft0, myfft1;
        myfft0.Setup(FFT_Type::C2C, howmany, fft_dim);
        myfft1.Setup(FFT_Type::C2C_INV, howmany, fft_dim);
        Vector<ValueType> v0(myfft0.Dim(0)), v1, v2;
        for (int i = 0; i < v0.Dim(); i++) v0[i] = (1 + i) / (ValueType)v0.Dim();
        myfft0.Execute(v0, v1);
        myfft1.Execute(v1, v2);

        const auto err = inf_norm(v2-v0);
        std::cout<<"Error : "<<inf_norm(v2-v0)<<'\n';
        SCTL_ASSERT(err < machine_eps<ValueType>() * 64);
      }
    }

    private:

    //static void check_align(const Vector<ValueType>& in, const Vector<ValueType>& out);

    FFTPlan<ValueType> plan;
    bool copy_input;

    StaticArray<Long,2> dim; // operator dimensions
    FFT_Type fft_type; // type of FFT transform
    Long howmany_; // number of transforms
  };

}  // end namespace

#include SCTL_INCLUDE(fft_wrapper.txx)

#endif  //_SCTL_FFT_WRAPPER_
