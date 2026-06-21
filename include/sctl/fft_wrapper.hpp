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
     */
    void Setup(FFT_Type fft_type, Long howmany, const Vector<Long>& dim_vec, Integer Nthreads = 1);

    /**
     * Execute the transform, preserving `in` (multi-D C2R may cost a copy).
     *
     * @param[in] in the input data vector.
     *
     * @param[out] out the output data vector.
     *
     * @note FFTW build: `in`/`out` must match the plan's alignment
     * (`SCTL_MEM_ALIGN`): a default `Vector` qualifies, a sub-view only if its
     * offset preserves it.  Misaligned buffers trip an assert (else crash);
     * the Cooley-Tukey fallback has none.
     */
    void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const;

    /**
     * Execute the transform; `in` MAY be overwritten (fast default). Pass
     * `preserve_input = true` to keep it. Alignment rules as above.
     *
     * @param[in] in the input data vector.
     *
     * @param[out] out the output data vector.
     */
    void Execute(Vector<ValueType>& in, Vector<ValueType>& out, bool preserve_input = false) const;

    /**
     * Test the FFT implementation.
     */
    static void test();

    private:

    /// Swap all state with `other`; backs the move operations.
    void Swap(FFT& other) noexcept;

    /// Shared body; preserve_input keeps `in` (via plan_preserve or a scratch copy).
    /// Not const-correct: writes `in` only when preserve_input==false, which only the
    /// non-const Execute overload passes -- so the const_cast never mutates a const object.
    void ExecuteImpl(const Vector<ValueType>& in, Vector<ValueType>& out, bool preserve_input) const;

    FFTPlan<ValueType> plan;          // destroy-input plan (fast)
    FFTPlan<ValueType> plan_preserve; // preserve-input plan; null for multi-D C2R
    Integer align_in, align_out; // plan alignment; Execute asserts in/out match

    StaticArray<Long,2> dim; // operator dimensions
    FFT_Type fft_type; // type of FFT transform
    Long howmany_; // number of transforms
  };

}  // end namespace

#endif // _SCTL_FFT_WRAPPER_HPP_
