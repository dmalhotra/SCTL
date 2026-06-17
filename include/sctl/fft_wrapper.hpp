#ifndef _SCTL_FFT_WRAPPER_HPP_
#define _SCTL_FFT_WRAPPER_HPP_

#include <complex>                // for complex

#include "sctl/common.hpp"        // for Long, Integer, sctl
#include "sctl/static-array.hpp"  // for StaticArray

#if defined(SCTL_HAVE_FFTW) || defined(SCTL_HAVE_FFTWF) || defined(SCTL_HAVE_FFTWL)
#include <fftw3.h>
#endif

namespace sctl {

  template <class ValueType> class Vector;

  template <class ValueType> struct FFTPlan;

  /**
   * Enum class representing different types of FFT transformations.
   *
   * Sign convention and normalization (N = product of `dim_vec` entries):
   *
   * - Forward transforms (`R2C`, `C2C`):
   *   \f[ X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i \, k n / N} \f]
   *
   * - Inverse transforms (`C2C_INV`, `C2R`):
   *   \f[ x_n = \sum_{k=0}^{N-1} X_k \, e^{+2\pi i \, k n / N} \f]
   *
   * UNNORMALIZED (raw FFTW): no scaling either way, so forward followed by
   * inverse returns the input times N. The caller applies any scaling.
   */
  enum class FFT_Type {R2C, C2C, C2C_INV, C2R};

  /**
   * Wrapper class for the FFTW library.  It uses FFTW for double precision calculation when linked
   * with `libfftw3` and the macro `SCTL_HAVE_FFTW` is defined. Similarly, for single precision and
   * long double precision computations, the macros `SCTL_HAVE_FFTWF` and `SCTL_HAVE_FFTWL` must be
   * defined and the code must be linked with `libfftw3f` and `libfftw3l`. If setup in this way, it
   * computes Fourier transform directly and will have lower performance.
   *
   * @tparam ValueType The value type of the FFT data.
   */
  template <class ValueType> class FFT {
    typedef std::complex<ValueType> ComplexType;

    public:

    FFT();

    ~FFT();

    /// Deleted: an FFT owns a non-copyable transform plan.
    FFT (const FFT&) = delete;
    FFT& operator= (const FFT&) = delete;

    /// Move constructor. Takes over `other`'s transform plan, leaving it empty.
    FFT (FFT&& other) noexcept;

    /// Move assignment. Takes over `other`'s transform plan and releases the
    /// plan currently held.
    FFT& operator= (FFT&& other) noexcept;

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
     *
     * @param[in] preserve_input If `false` (default), `Execute` may overwrite
     * its input (only multi-D C2R does); set `true` to keep it intact.
     */
    void Setup(FFT_Type fft_type, Long howmany, const Vector<Long>& dim_vec, Integer Nthreads = 1, bool preserve_input = false);

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
    static void test();

    private:

    /// Swap all state with `other`; backs the move operations.
    void Swap(FFT& other) noexcept;

    //static void check_align(const Vector<ValueType>& in, const Vector<ValueType>& out);

    FFTPlan<ValueType> plan;
    bool copy_input;

    StaticArray<Long,2> dim; // operator dimensions
    FFT_Type fft_type; // type of FFT transform
    Long howmany_; // number of transforms
  };

}  // end namespace

#endif // _SCTL_FFT_WRAPPER_HPP_
