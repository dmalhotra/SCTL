#ifndef _SCTL_FFT_WRAPPER_TXX_
#define _SCTL_FFT_WRAPPER_TXX_

#include <algorithm>              // for max
#include <utility>                // for std::swap
#include <iostream>               // for basic_ostream, operator<<, cout
#include <vector>                 // for vector
#ifdef _OPENMP
#include <omp.h>
#endif

#include "sctl/common.hpp"        // for Long, Integer, SCTL_ASSERT, SCTL_AS...
#include "sctl/fft_wrapper.hpp"   // for FFT, FFT_Type
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator
#include "sctl/iterator.txx"      // for Iterator::Iterator<ValueType>, Iter...
#include "sctl/math_utils.hpp"    // for const_pi, cos, sin, sqrt, fabs
#include "sctl/math_utils.txx"    // for machine_eps
#include "sctl/matrix.hpp"        // for Matrix
#include "sctl/scratch_pool.hpp"  // for ScratchBuf
#include "sctl/scratch_pool.txx"  // for ScratchBuf::begin/end (ctor, dtor)
#include "sctl/static-array.hpp"  // for StaticArray
#include "sctl/static-array.txx"  // for StaticArray::operator[], StaticArra...
#include "sctl/vector.hpp"        // for Vector
#include "sctl/vector.txx"        // for Vector::operator[], Vector::PushBack

namespace sctl {

  template <class ValueType> void FFT<ValueType>::test() {
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

  //template <class ValueType> void FFT<ValueType>::check_align(const Vector<ValueType>& in, const Vector<ValueType>& out) {
  //  //SCTL_ASSERT_MSG((((uintptr_t)& in[0]) & ((uintptr_t)(SCTL_MEM_ALIGN - 1))) == 0, "sctl::FFT: Input vector not aligned to " <<SCTL_MEM_ALIGN<<" bytes!");
  //  //SCTL_ASSERT_MSG((((uintptr_t)&out[0]) & ((uintptr_t)(SCTL_MEM_ALIGN - 1))) == 0, "sctl::FFT: Output vector not aligned to "<<SCTL_MEM_ALIGN<<" bytes!");
  //  // TODO: copy to auxiliary array if unaligned
  //}

  // ----------------------------------------------------------------------
  // Mixed-radix Cooley-Tukey fallback. Used for any ValueType without an
  // FFTW specialization (notably QuadReal / __float128) and for builds
  // without FFTW. Public API matches the FFTW-backed path exactly.
  // ----------------------------------------------------------------------

  namespace fft_fallback_internal {

    inline void FactorizeRadix(Long N, Vector<Long>& radices) {
      radices.ReInit(0);
      if (N <= 1) return;
      Long n = N;
      // Greedy extraction of small radices. Order matters: 8 before 4 before 2,
      // 4 before 2 — pulling out the largest power-of-two/composite first
      // gives fewer stages.
      const Long try_radix[] = {8, 7, 5, 4, 3, 2};
      for (Long r : try_radix) {
        while (n > 1 && n % r == 0) {
          radices.PushBack(r);
          n /= r;
        }
      }
      if (n > 1) {
        // Remaining prime factor (or a prime power not in the small set);
        // handled by the generic O(p^2) per-row DFT.
        radices.PushBack(n);
        if (n > 32) {
          std::cerr << "sctl::FFT: warning: 1-D length " << N
                    << " contains a prime factor " << n
                    << " > 32; fallback FFT will run an O(p^2) inner DFT.\n";
        }
      }
    }

    inline Long ProductOf(const Vector<Long>& v) {
      Long p = 1;
      for (Long i = 0; i < v.Dim(); i++) p *= v[i];
      return p;
    }

    template <class T> void BuildTwiddles(const Vector<Long>& radices, int sign, Integer nthreads,
                                          Vector<T>& twiddle, Vector<Long>& stage_off) {
      const Long s = radices.Dim();
      stage_off.ReInit(s + 1);
      Long total = 0;
      Long m = 1;
      for (Long k = 0; k < s; k++) {
        stage_off[k] = total;
        total += 2 * radices[k] * m;
        m *= radices[k];
      }
      stage_off[s] = total;
      twiddle.ReInit(total);

      const T two_pi = (T)2 * const_pi<T>();
      const T sign_t = (T)sign;
      m = 1;
      for (Long k = 0; k < s; k++) {
        const Long r = radices[k];
        const Long rm = r * m;
        T* tw = &twiddle[stage_off[k]];
        const T scale = sign_t * two_pi / (T)rm;
        #pragma omp parallel for num_threads(nthreads) if(rm > 1024) schedule(static)
        for (Long pj = 0; pj < rm; pj++) {
          const Long p = pj / m;
          const Long j = pj - p * m;
          const T angle = scale * (T)(p * j);
          tw[2 * pj]     = cos<T>(angle);
          tw[2 * pj + 1] = sin<T>(angle);
        }
        m *= r;
      }
    }

    template <class T> void BuildSmallDFT(const Vector<Long>& radices, int sign, Integer nthreads,
                                          Vector<T>& dft_flat, Vector<Long>& stage_off) {
      const Long s = radices.Dim();
      stage_off.ReInit(s + 1);
      Long total = 0;
      for (Long k = 0; k < s; k++) {
        stage_off[k] = total;
        total += 2 * radices[k] * radices[k];
      }
      stage_off[s] = total;
      dft_flat.ReInit(total);

      const T two_pi = (T)2 * const_pi<T>();
      const T sign_t = (T)sign;
      for (Long k = 0; k < s; k++) {
        const Long r = radices[k];
        T* mat = &dft_flat[stage_off[k]];
        const T scale = sign_t * two_pi / (T)r;
        #pragma omp parallel for num_threads(nthreads) if(r * r > 256) schedule(static)
        for (Long qp = 0; qp < r * r; qp++) {
          const Long q = qp / r;
          const Long p = qp - q * r;
          const T angle = scale * (T)(q * p);
          mat[2 * qp]     = cos<T>(angle);
          mat[2 * qp + 1] = sin<T>(angle);
        }
      }
    }

    inline void BuildDigitReversal(const Vector<Long>& radices, Integer nthreads,
                                   Vector<Long>& perm) {
      const Long s = radices.Dim();
      Long N = 1;
      for (Long k = 0; k < s; k++) N *= radices[k];
      perm.ReInit(N);
      if (N == 0) return;

      Vector<Long> cumprod(s + 1);
      cumprod[0] = 1;
      for (Long k = 0; k < s; k++) cumprod[k + 1] = cumprod[k] * radices[k];

      const Long* rad = (s ? &radices[0] : nullptr);
      const Long* cp  = &cumprod[0];
      #pragma omp parallel for num_threads(nthreads) if(N > 1024) schedule(static)
      for (Long i = 0; i < N; i++) {
        Long n = i;
        Long rev = 0;
        for (Long k = 0; k < s; k++) {
          const Long r = rad[k];
          const Long digit = n % r;
          n /= r;
          rev += digit * (N / cp[k + 1]);
        }
        perm[i] = rev;
      }
    }

    // Apply digit-reverse permutation to one row of `N` interleaved complex.
    template <class T> inline void DigitReverseRow(const T* src, T* dst, Long N, const Long* perm) {
      for (Long i = 0; i < N; i++) {
        const Long p = perm[i];
        dst[2 * i]     = src[2 * p];
        dst[2 * i + 1] = src[2 * p + 1];
      }
    }

    // ---------------- hand-coded butterflies ----------------
    //
    // Twiddle convention: tw[2*(p*m + j) + 0/1] = re/im of W_{r*m}^{p*j}.
    // For p=0, tw is (1, 0) (cosmetic; we skip multiplying by it). The
    // butterflies avoid materializing a small r×r DFT matrix; the r-point
    // DFT entries that are ±1 or ±i are folded into adds and sign-flips.
    //
    // `sign` is the FFT sign convention (-1 forward, +1 inverse). It only
    // shows up in r-point DFT entries that contain a non-trivial imaginary
    // part — radix-2 is sign-independent, radix-4 picks up ±i, radix-8
    // picks up ±i and ±(c, ±c) for c = cos(π/4).

    template <class T>
    inline void Radix3StageRow(T* data, Long N, Long m, const T* tw, int sign) {
      // r=3.  W_3 = (c, sign*s_abs) with c = -1/2, s_abs = sqrt(3)/2.
      // DFT[3] outputs (after twiddle of b[1], b[2]):
      //   t1 = b[1]+b[2], t2 = b[1]-b[2]
      //   X[0] = b[0] + t1
      //   X[1] = (b[0] + c*t1)  +  i·(sign·s_abs)·t2
      //   X[2] = (b[0] + c*t1)  -  i·(sign·s_abs)·t2
      static const T s_abs = sqrt<T>((T)3) / (T)2;
      const T c = (T)-0.5;
      const T S = (T)sign * s_abs;
      const Long rm = 3 * m;
      for (Long outer = 0; outer < N; outer += rm) {
        T* base = data + 2 * outer;
        for (Long j = 0; j < m; j++) {
          const Long i0 = 2 * j;
          const Long i1 = 2 * (m + j);
          const Long i2 = 2 * (2 * m + j);
          const T w1r = tw[2 * (m + j)],     w1i = tw[2 * (m + j) + 1];
          const T w2r = tw[2 * (2 * m + j)], w2i = tw[2 * (2 * m + j) + 1];

          const T a0r = base[i0], a0i = base[i0 + 1];
          const T a1r = base[i1], a1i = base[i1 + 1];
          const T a2r = base[i2], a2i = base[i2 + 1];
          const T b1r = a1r * w1r - a1i * w1i, b1i = a1r * w1i + a1i * w1r;
          const T b2r = a2r * w2r - a2i * w2i, b2i = a2r * w2i + a2i * w2r;

          const T t1r = b1r + b2r, t1i = b1i + b2i;
          const T t2r = b1r - b2r, t2i = b1i - b2i;
          const T mr  = a0r + c * t1r, mi  = a0i + c * t1i;   // mid = b[0] + c*t1
          const T jr  = -S * t2i, ji  = S * t2r;              // i·S·t2 (post-rotate)

          base[i0]     = a0r + t1r;
          base[i0 + 1] = a0i + t1i;
          base[i1]     = mr + jr;
          base[i1 + 1] = mi + ji;
          base[i2]     = mr - jr;
          base[i2 + 1] = mi - ji;
        }
      }
    }

    template <class T>
    inline void Radix7StageRow(T* data, Long N, Long m, const T* tw, int sign) {
      // r=7.  Use conjugate symmetry W_7^{7-k} = W_7^{-k} = conj(W_7^k).
      // Pair (b[1],b[6]), (b[2],b[5]), (b[3],b[4]) → u_p (sum), v_p (diff).
      // Cosine matrix per X[k] (k=1..3), p=1..3 (k=4..6 mirror via −i·beta):
      //   k=1: C1 C2 C3
      //   k=2: C2 C3 C1
      //   k=3: C3 C1 C2
      // Sine matrix (entries get the sign factor applied):
      //   k=1: +S1 +S2 +S3
      //   k=2: +S2 −S3 −S1
      //   k=3: +S3 −S1 +S2
      // X[k]   = alpha_k + i·beta_k
      // X[7-k] = alpha_k − i·beta_k
      static const T C1  = cos<T>((T)2 * const_pi<T>() / (T)7);
      static const T C2  = cos<T>((T)4 * const_pi<T>() / (T)7);
      static const T C3  = cos<T>((T)6 * const_pi<T>() / (T)7);
      static const T S1A = sin<T>((T)2 * const_pi<T>() / (T)7);
      static const T S2A = sin<T>((T)4 * const_pi<T>() / (T)7);
      static const T S3A = sin<T>((T)6 * const_pi<T>() / (T)7);
      const T S1 = (T)sign * S1A;
      const T S2 = (T)sign * S2A;
      const T S3 = (T)sign * S3A;
      const Long rm = 7 * m;
      for (Long outer = 0; outer < N; outer += rm) {
        T* base = data + 2 * outer;
        for (Long j = 0; j < m; j++) {
          const Long i0 = 2 * j;
          const Long i1 = 2 * (m + j);
          const Long i2 = 2 * (2 * m + j);
          const Long i3 = 2 * (3 * m + j);
          const Long i4 = 2 * (4 * m + j);
          const Long i5 = 2 * (5 * m + j);
          const Long i6 = 2 * (6 * m + j);

          const T w1r = tw[2 * (m + j)],         w1i = tw[2 * (m + j) + 1];
          const T w2r = tw[2 * (2 * m + j)],     w2i = tw[2 * (2 * m + j) + 1];
          const T w3r = tw[2 * (3 * m + j)],     w3i = tw[2 * (3 * m + j) + 1];
          const T w4r = tw[2 * (4 * m + j)],     w4i = tw[2 * (4 * m + j) + 1];
          const T w5r = tw[2 * (5 * m + j)],     w5i = tw[2 * (5 * m + j) + 1];
          const T w6r = tw[2 * (6 * m + j)],     w6i = tw[2 * (6 * m + j) + 1];

          const T a0r = base[i0], a0i = base[i0 + 1];
          const T a1r = base[i1], a1i = base[i1 + 1];
          const T a2r = base[i2], a2i = base[i2 + 1];
          const T a3r = base[i3], a3i = base[i3 + 1];
          const T a4r = base[i4], a4i = base[i4 + 1];
          const T a5r = base[i5], a5i = base[i5 + 1];
          const T a6r = base[i6], a6i = base[i6 + 1];

          const T b1r = a1r * w1r - a1i * w1i, b1i = a1r * w1i + a1i * w1r;
          const T b2r = a2r * w2r - a2i * w2i, b2i = a2r * w2i + a2i * w2r;
          const T b3r = a3r * w3r - a3i * w3i, b3i = a3r * w3i + a3i * w3r;
          const T b4r = a4r * w4r - a4i * w4i, b4i = a4r * w4i + a4i * w4r;
          const T b5r = a5r * w5r - a5i * w5i, b5i = a5r * w5i + a5i * w5r;
          const T b6r = a6r * w6r - a6i * w6i, b6i = a6r * w6i + a6i * w6r;

          const T u1r = b1r + b6r, u1i = b1i + b6i;
          const T v1r = b1r - b6r, v1i = b1i - b6i;
          const T u2r = b2r + b5r, u2i = b2i + b5i;
          const T v2r = b2r - b5r, v2i = b2i - b5i;
          const T u3r = b3r + b4r, u3i = b3i + b4i;
          const T v3r = b3r - b4r, v3i = b3i - b4i;

          base[i0]     = a0r + u1r + u2r + u3r;
          base[i0 + 1] = a0i + u1i + u2i + u3i;

          // alpha_k = b[0] + C_k·(u1,u2,u3)
          const T A1r = a0r + C1 * u1r + C2 * u2r + C3 * u3r;
          const T A1i = a0i + C1 * u1i + C2 * u2i + C3 * u3i;
          const T A2r = a0r + C2 * u1r + C3 * u2r + C1 * u3r;
          const T A2i = a0i + C2 * u1i + C3 * u2i + C1 * u3i;
          const T A3r = a0r + C3 * u1r + C1 * u2r + C2 * u3r;
          const T A3i = a0i + C3 * u1i + C1 * u2i + C2 * u3i;

          // beta_k = S_k·(v1,v2,v3) (signed pattern shown above)
          const T B1r = S1 * v1r + S2 * v2r + S3 * v3r;
          const T B1i = S1 * v1i + S2 * v2i + S3 * v3i;
          const T B2r = S2 * v1r - S3 * v2r - S1 * v3r;
          const T B2i = S2 * v1i - S3 * v2i - S1 * v3i;
          const T B3r = S3 * v1r - S1 * v2r + S2 * v3r;
          const T B3i = S3 * v1i - S1 * v2i + S2 * v3i;

          // X[k] = alpha_k + i·beta_k = (A_r − B_i, A_i + B_r)
          // X[7-k] = alpha_k − i·beta_k = (A_r + B_i, A_i − B_r)
          base[i1]     = A1r - B1i;
          base[i1 + 1] = A1i + B1r;
          base[i6]     = A1r + B1i;
          base[i6 + 1] = A1i - B1r;
          base[i2]     = A2r - B2i;
          base[i2 + 1] = A2i + B2r;
          base[i5]     = A2r + B2i;
          base[i5 + 1] = A2i - B2r;
          base[i3]     = A3r - B3i;
          base[i3 + 1] = A3i + B3r;
          base[i4]     = A3r + B3i;
          base[i4 + 1] = A3i - B3r;
        }
      }
    }

    template <class T>
    inline void Radix5StageRow(T* data, Long N, Long m, const T* tw, int sign) {
      // r=5.  Standard symmetric reduction:
      //   u1 = b[1]+b[4], v1 = b[1]-b[4]
      //   u2 = b[2]+b[3], v2 = b[2]-b[3]
      //   alpha = b[0] + c1*u1 + c2*u2
      //   beta  = b[0] + c2*u1 + c1*u2
      //   X[0] = b[0] + u1 + u2
      //   X[1] = alpha + i·(S1·v1 + S2·v2)
      //   X[4] = alpha − i·(S1·v1 + S2·v2)
      //   X[2] = beta  + i·(S2·v1 − S1·v2)
      //   X[3] = beta  − i·(S2·v1 − S1·v2)
      // with c1=cos(2π/5), c2=cos(4π/5), S1=sign·sin(2π/5), S2=sign·sin(4π/5).
      static const T C1 = cos<T>((T)2 * const_pi<T>() / (T)5);
      static const T C2 = cos<T>((T)4 * const_pi<T>() / (T)5);
      static const T S1A = sin<T>((T)2 * const_pi<T>() / (T)5);
      static const T S2A = sin<T>((T)4 * const_pi<T>() / (T)5);
      const T S1 = (T)sign * S1A;
      const T S2 = (T)sign * S2A;
      const Long rm = 5 * m;
      for (Long outer = 0; outer < N; outer += rm) {
        T* base = data + 2 * outer;
        for (Long j = 0; j < m; j++) {
          const Long i0 = 2 * j;
          const Long i1 = 2 * (m + j);
          const Long i2 = 2 * (2 * m + j);
          const Long i3 = 2 * (3 * m + j);
          const Long i4 = 2 * (4 * m + j);
          const T w1r = tw[2 * (m + j)],         w1i = tw[2 * (m + j) + 1];
          const T w2r = tw[2 * (2 * m + j)],     w2i = tw[2 * (2 * m + j) + 1];
          const T w3r = tw[2 * (3 * m + j)],     w3i = tw[2 * (3 * m + j) + 1];
          const T w4r = tw[2 * (4 * m + j)],     w4i = tw[2 * (4 * m + j) + 1];

          const T a0r = base[i0], a0i = base[i0 + 1];
          const T a1r = base[i1], a1i = base[i1 + 1];
          const T a2r = base[i2], a2i = base[i2 + 1];
          const T a3r = base[i3], a3i = base[i3 + 1];
          const T a4r = base[i4], a4i = base[i4 + 1];

          const T b1r = a1r * w1r - a1i * w1i, b1i = a1r * w1i + a1i * w1r;
          const T b2r = a2r * w2r - a2i * w2i, b2i = a2r * w2i + a2i * w2r;
          const T b3r = a3r * w3r - a3i * w3i, b3i = a3r * w3i + a3i * w3r;
          const T b4r = a4r * w4r - a4i * w4i, b4i = a4r * w4i + a4i * w4r;

          const T u1r = b1r + b4r, u1i = b1i + b4i;
          const T v1r = b1r - b4r, v1i = b1i - b4i;
          const T u2r = b2r + b3r, u2i = b2i + b3i;
          const T v2r = b2r - b3r, v2i = b2i - b3i;

          base[i0]     = a0r + u1r + u2r;
          base[i0 + 1] = a0i + u1i + u2i;

          const T alpr = a0r + C1 * u1r + C2 * u2r;
          const T alpi = a0i + C1 * u1i + C2 * u2i;
          const T betr = a0r + C2 * u1r + C1 * u2r;
          const T beti = a0i + C2 * u1i + C1 * u2i;

          const T gir  = S1 * v1r + S2 * v2r;
          const T gii  = S1 * v1i + S2 * v2i;
          const T dir  = S2 * v1r - S1 * v2r;
          const T dii  = S2 * v1i - S1 * v2i;
          // i·z = (-z_i, z_r)
          const T gr   = -gii, gi = gir;
          const T dr   = -dii, di = dir;

          base[i1]     = alpr + gr;
          base[i1 + 1] = alpi + gi;
          base[i4]     = alpr - gr;
          base[i4 + 1] = alpi - gi;
          base[i2]     = betr + dr;
          base[i2 + 1] = beti + di;
          base[i3]     = betr - dr;
          base[i3 + 1] = beti - di;
        }
      }
    }

    template <class T>
    inline void Radix2StageRow(T* data, Long N, Long m, const T* tw) {
      // r=2.  2-point DFT is [[1,1],[1,-1]] — no sign dependence.
      const Long rm = 2 * m;
      for (Long outer = 0; outer < N; outer += rm) {
        T* base = data + 2 * outer;
        for (Long j = 0; j < m; j++) {
          const Long i0 = 2 * j;
          const Long i1 = 2 * (m + j);
          const T wr = tw[2 * (m + j)];
          const T wi = tw[2 * (m + j) + 1];
          const T a0r = base[i0], a0i = base[i0 + 1];
          const T b1r = base[i1], b1i = base[i1 + 1];
          // t = W * b
          const T tr = b1r * wr - b1i * wi;
          const T ti = b1r * wi + b1i * wr;
          base[i0]     = a0r + tr;
          base[i0 + 1] = a0i + ti;
          base[i1]     = a0r - tr;
          base[i1 + 1] = a0i - ti;
        }
      }
    }

    template <class T>
    inline void Radix4StageRow(T* data, Long N, Long m, const T* tw, int sign) {
      // r=4.  4-point DFT via two radix-2 butterflies.  Sign appears in the
      // ±i factor: forward (sign=-1) → −i on the diagonal, inverse → +i.
      // Output formulas (after twiddle b1, b2, b3):
      //   X[0] = a0 + b2 + b1 + b3
      //   X[2] = a0 + b2 − b1 − b3
      //   X[1] = (a0 − b2) + sign·i·(b3 − b1)         [forward: q − i·s]
      //   X[3] = (a0 − b2) − sign·i·(b3 − b1)         [forward: q + i·s]
      const T S = (T)sign;
      const Long rm = 4 * m;
      for (Long outer = 0; outer < N; outer += rm) {
        T* base = data + 2 * outer;
        for (Long j = 0; j < m; j++) {
          const Long i0 = 2 * j;
          const Long i1 = 2 * (m + j);
          const Long i2 = 2 * (2 * m + j);
          const Long i3 = 2 * (3 * m + j);
          const T w1r = tw[2 * (m + j)],         w1i = tw[2 * (m + j) + 1];
          const T w2r = tw[2 * (2 * m + j)],     w2i = tw[2 * (2 * m + j) + 1];
          const T w3r = tw[2 * (3 * m + j)],     w3i = tw[2 * (3 * m + j) + 1];

          const T a0r = base[i0], a0i = base[i0 + 1];
          const T a1r = base[i1], a1i = base[i1 + 1];
          const T a2r = base[i2], a2i = base[i2 + 1];
          const T a3r = base[i3], a3i = base[i3 + 1];

          const T b1r = a1r * w1r - a1i * w1i, b1i = a1r * w1i + a1i * w1r;
          const T b2r = a2r * w2r - a2i * w2i, b2i = a2r * w2i + a2i * w2r;
          const T b3r = a3r * w3r - a3i * w3i, b3i = a3r * w3i + a3i * w3r;

          const T pr = a0r + b2r, pi = a0i + b2i;     // p = a0 + b2
          const T qr = a0r - b2r, qi = a0i - b2i;     // q = a0 − b2
          const T rr = b1r + b3r, ri = b1i + b3i;     // r = b1 + b3
          const T sr = b1r - b3r, si = b1i - b3i;     // s = b1 − b3

          base[i0]     = pr + rr;
          base[i0 + 1] = pi + ri;
          base[i2]     = pr - rr;
          base[i2 + 1] = pi - ri;
          // forward (S=-1): X[1] = q - i·s  →  (qr + si, qi - sr)
          //                 X[3] = q + i·s  →  (qr - si, qi + sr)
          // inverse (S=+1) flips:           X[1] = (qr - si, qi + sr), X[3] = (qr + si, qi - sr)
          base[i1]     = qr - S * si;
          base[i1 + 1] = qi + S * sr;
          base[i3]     = qr + S * si;
          base[i3 + 1] = qi - S * sr;
        }
      }
    }

    template <class T>
    inline void Radix8StageRow(T* data, Long N, Long m, const T* tw, int sign) {
      // r=8.  Decompose 8-point DFT as two 4-point DFTs (on even/odd inputs)
      // followed by 4 size-2 combining butterflies twiddled by W_8^k for
      // k=0..3 (the non-trivial twiddles are W_8^1 and W_8^3, both ±(c, ±c)
      // with c = √2/2).
      const T S = (T)sign;
      const T c  = sqrt<T>((T)0.5);  // cos(π/4) = sin(π/4)
      const Long rm = 8 * m;
      for (Long outer = 0; outer < N; outer += rm) {
        T* base = data + 2 * outer;
        for (Long j = 0; j < m; j++) {
          // 8 input indices
          Long idx[8];
          for (Long p = 0; p < 8; p++) idx[p] = 2 * (p * m + j);

          // Load + apply twiddles (p=0 trivial)
          T br[8], bi[8];
          br[0] = base[idx[0]]; bi[0] = base[idx[0] + 1];
          for (Long p = 1; p < 8; p++) {
            const T ar = base[idx[p]], ai = base[idx[p] + 1];
            const T wr = tw[2 * (p * m + j)];
            const T wi = tw[2 * (p * m + j) + 1];
            br[p] = ar * wr - ai * wi;
            bi[p] = ar * wi + ai * wr;
          }

          // 4-point DFT on evens (b[0], b[2], b[4], b[6]) → E[0..3]
          // 4-point DFT on odds  (b[1], b[3], b[5], b[7]) → O[0..3]
          auto dft4 = [&](const T& z0r, const T& z0i, const T& z1r, const T& z1i,
                          const T& z2r, const T& z2i, const T& z3r, const T& z3i,
                          T& E0r, T& E0i, T& E1r, T& E1i,
                          T& E2r, T& E2i, T& E3r, T& E3i) {
            const T pr = z0r + z2r, pi = z0i + z2i;
            const T qr = z0r - z2r, qi = z0i - z2i;
            const T rr = z1r + z3r, ri = z1i + z3i;
            const T sr = z1r - z3r, si = z1i - z3i;
            E0r = pr + rr; E0i = pi + ri;
            E2r = pr - rr; E2i = pi - ri;
            E1r = qr - S * si; E1i = qi + S * sr;   // q − S·i·s  (S = sign)
            E3r = qr + S * si; E3i = qi - S * sr;
          };
          T Er[4], Ei[4], Or[4], Oi[4];
          dft4(br[0], bi[0], br[2], bi[2], br[4], bi[4], br[6], bi[6],
               Er[0], Ei[0], Er[1], Ei[1], Er[2], Ei[2], Er[3], Ei[3]);
          dft4(br[1], bi[1], br[3], bi[3], br[5], bi[5], br[7], bi[7],
               Or[0], Oi[0], Or[1], Oi[1], Or[2], Oi[2], Or[3], Oi[3]);

          // Combine: X[k]   = E[k] + W_8^k · O[k]   for k=0..3
          //          X[k+4] = E[k] − W_8^k · O[k]
          // W_8^0 = 1
          // W_8^1 = (c, S·c)     where S·c reads: forward (S=-1) → −c, inverse → +c.
          //   (Actually W_8^k = exp(sign · 2πi · k/8) = exp(sign · iπk/4).)
          //   For k=1: cos(sign·π/4) = c, sin(sign·π/4) = sign·c → (c, S·c).  ✓
          // W_8^2 = (0, S)
          // W_8^3 = (-c, S·c)
          auto cmul = [](const T& xr, const T& xi, const T& yr, const T& yi, T& zr, T& zi) {
            zr = xr * yr - xi * yi;
            zi = xr * yi + xi * yr;
          };
          T t0r = Or[0],          t0i = Oi[0];                          // W^0 * O[0]
          T t1r, t1i; cmul(Or[1], Oi[1],  c,  S * c, t1r, t1i);          // W^1 * O[1]
          T t2r = -S * Oi[2],     t2i =  S * Or[2];                      // W^2 * O[2] = (0, S) * O[2]
          T t3r, t3i; cmul(Or[3], Oi[3], -c,  S * c, t3r, t3i);          // W^3 * O[3]

          base[idx[0]]     = Er[0] + t0r; base[idx[0] + 1] = Ei[0] + t0i;
          base[idx[1]]     = Er[1] + t1r; base[idx[1] + 1] = Ei[1] + t1i;
          base[idx[2]]     = Er[2] + t2r; base[idx[2] + 1] = Ei[2] + t2i;
          base[idx[3]]     = Er[3] + t3r; base[idx[3] + 1] = Ei[3] + t3i;
          base[idx[4]]     = Er[0] - t0r; base[idx[4] + 1] = Ei[0] - t0i;
          base[idx[5]]     = Er[1] - t1r; base[idx[5] + 1] = Ei[1] - t1i;
          base[idx[6]]     = Er[2] - t2r; base[idx[6] + 1] = Ei[2] - t2i;
          base[idx[7]]     = Er[3] - t3r; base[idx[7] + 1] = Ei[3] - t3i;
        }
      }
    }

    // Generic (any radix) DIT stage applied to one row, in place.
    //   See header comments above for arg semantics.
    template <class T>
    inline void RadixStageRow(T* data, Long N, Long m, Long r,
                              const T* tw, const T* dft, int sign,
                              T* tmp_re, T* tmp_im) {
#ifndef SCTL_FFT_NO_HANDROLLED_RADIX
      if (r == 2) { Radix2StageRow(data, N, m, tw);             return; }
      if (r == 3) { Radix3StageRow(data, N, m, tw, sign);       return; }
      if (r == 4) { Radix4StageRow(data, N, m, tw, sign);       return; }
      if (r == 5) { Radix5StageRow(data, N, m, tw, sign);       return; }
      if (r == 7) { Radix7StageRow(data, N, m, tw, sign);       return; }
      if (r == 8) { Radix8StageRow(data, N, m, tw, sign);       return; }
#endif

      const Long rm = r * m;
      // Iterate over outer blocks (each of size r*m complex) and j within
      for (Long outer = 0; outer < N; outer += rm) {
        T* base = data + 2 * outer;
        for (Long j = 0; j < m; j++) {
          // Load + twiddle the r samples spaced m apart
          for (Long p = 0; p < r; p++) {
            const T re = base[2 * (p * m + j)];
            const T im = base[2 * (p * m + j) + 1];
            const T wr = tw[2 * (p * m + j)];
            const T wi = tw[2 * (p * m + j) + 1];
            tmp_re[p] = re * wr - im * wi;
            tmp_im[p] = re * wi + im * wr;
          }
          // Apply r-point DFT: out[q] = sum_p W_r^{q*p} * tmp[p]
          for (Long q = 0; q < r; q++) {
            T sr = 0, si = 0;
            const T* row = dft + 2 * q * r;
            for (Long p = 0; p < r; p++) {
              const T cr = row[2 * p];
              const T ci = row[2 * p + 1];
              sr += tmp_re[p] * cr - tmp_im[p] * ci;
              si += tmp_re[p] * ci + tmp_im[p] * cr;
            }
            base[2 * (q * m + j)]     = sr;
            base[2 * (q * m + j) + 1] = si;
          }
        }
      }
    }

    // --- Public-ish Cooley-Tukey plan + Setup/Execute -----------------
    // These are exposed (under `fft_fallback_internal`) so test code can
    // instantiate the Cooley-Tukey path for *any* ValueType, bypassing the
    // FFTW specialization. `FFT<T>` (generic) is a thin wrapper around them.
    template <class T> struct CTPlan {
      struct Dim1DPlan {
        Long N;                              // 1-D length (complex)
        int sign;                            // -1 forward, +1 inverse
        Vector<Long> radices;                // factorization, stage order
        Vector<Long> stage_twiddle_off;      // size s+1
        Vector<T> twiddle;                   // 2 * sum(r_k * m_k) reals
        Vector<Long> stage_dft_off;          // size s+1
        Vector<T> dft_flat;                  // 2 * sum(r_k * r_k) reals
        Vector<Long> digit_rev;              // size N
      };
      Vector<Long> dim_vec;
      std::vector<Dim1DPlan> dim_plan;
      Integer nthreads = 1;
      T norm_scale = 1;
    };

    template <class T>
    void CTSetup(CTPlan<T>& plan, FFT_Type fft_type, Long howmany,
                 const Vector<Long>& dim_vec, Integer Nthreads,
                 Long& N0_out, Long& N1_out) {
      plan.dim_vec = dim_vec;
      plan.dim_plan.clear();
      plan.nthreads = std::max<Integer>(1, Nthreads);

      const Long rank = dim_vec.Dim();
      Long N0 = 0, N1 = 0;
      if (rank > 0) {
        Long N = howmany;
        for (Long k = 0; k < rank; k++) N *= dim_vec[k];
        if (fft_type == FFT_Type::R2C) {
          N0 = N;
          N1 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
        } else if (fft_type == FFT_Type::C2C || fft_type == FFT_Type::C2C_INV) {
          N0 = N * 2; N1 = N * 2;
        } else if (fft_type == FFT_Type::C2R) {
          N0 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
          N1 = N;
        }
      }
      N0_out = N0;
      N1_out = N1;

      const bool inverse = (fft_type == FFT_Type::C2C_INV) || (fft_type == FFT_Type::C2R);
      plan.dim_plan.resize(rank);
      Long total_N = 1;
      for (Long k = 0; k < rank; k++) {
        auto& dp = plan.dim_plan[k];
        dp.N = dim_vec[k];
        dp.sign = inverse ? +1 : -1;
        total_N *= dp.N;
        FactorizeRadix(dp.N, dp.radices);
        BuildTwiddles<T>(dp.radices, dp.sign, plan.nthreads, dp.twiddle, dp.stage_twiddle_off);
        BuildSmallDFT<T>(dp.radices, dp.sign, plan.nthreads, dp.dft_flat, dp.stage_dft_off);
        BuildDigitReversal(dp.radices, plan.nthreads, dp.digit_rev);
      }
      plan.norm_scale = (total_N > 0) ? (T)1 / sqrt<T>((T)total_N) : (T)1;
    }

    template <class T>
    void CTExecute(const CTPlan<T>& plan, FFT_Type fft_type, Long howmany,
                   Long N0, Long N1, const Vector<T>& in, Vector<T>& out) {
      SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
      if (out.Dim() != N1) out.ReInit(N1);
      if (!N0 || !N1) return;

      const Long rank = plan.dim_vec.Dim();
      if (rank == 0) return;

      Integer nthreads = plan.nthreads;
#ifdef _OPENMP
      if (SCTL_IN_PARALLEL()) nthreads = 1;
#endif

      const Long buf_reals = std::max<Long>(N0, N1) + std::max<Long>(N0, N1);
      ScratchBuf<T> buf_a(buf_reals);
      ScratchBuf<T> buf_b(buf_reals);
      Iterator<T> cur = buf_a.begin();
      Iterator<T> nxt = buf_b.begin();
      const auto swap_bufs = [&](){ Iterator<T> t = cur; cur = nxt; nxt = t; };

      // Stage 1: pack input as a full (howmany, dim_vec...) complex tensor.
      const Long total_complex = howmany * ProductOf(plan.dim_vec);
      {
        const Long Nd = plan.dim_vec[rank - 1];
        const Long Nh = Nd / 2 + 1;
        if (fft_type == FFT_Type::R2C) {
          const Long batch = total_complex / Nd;
          #pragma omp parallel for num_threads(nthreads) if(batch * Nd > 4096) schedule(static)
          for (Long b = 0; b < batch; b++) {
            for (Long i = 0; i < Nd; i++) {
              cur[2 * (b * Nd + i)]     = in[b * Nd + i];
              cur[2 * (b * Nd + i) + 1] = (T)0;
            }
          }
        } else if (fft_type == FFT_Type::C2R) {
          const Long batch = howmany;
          const Long prod_other = total_complex / (batch * Nd);
          const Long Nin_per_batch  = prod_other * Nh;
          const Long Nout_per_batch = prod_other * Nd;
          #pragma omp parallel for num_threads(nthreads) if(batch * Nout_per_batch >= 1024) schedule(static)
          for (Long b = 0; b < batch; b++) {
            const T* in_b = &in[2 * b * Nin_per_batch];
            T* out_b = &cur[2 * b * Nout_per_batch];
            for (Long out_idx = 0; out_idx < Nout_per_batch; out_idx++) {
              Long ks[16] = {};
              SCTL_ASSERT_MSG(rank <= 16, "fallback FFT: rank > 16 not supported");
              Long x = out_idx;
              for (Long axis = rank - 1; axis >= 0; axis--) {
                const Long d = plan.dim_vec[axis];
                ks[axis] = x % d;
                x /= d;
              }
              Long src_idx;
              if (ks[rank - 1] < Nh) {
                src_idx = 0;
                for (Long axis = 0; axis < rank - 1; axis++) {
                  src_idx = src_idx * plan.dim_vec[axis] + ks[axis];
                }
                src_idx = src_idx * Nh + ks[rank - 1];
                out_b[2 * out_idx]     = in_b[2 * src_idx];
                out_b[2 * out_idx + 1] = in_b[2 * src_idx + 1];
              } else {
                src_idx = 0;
                for (Long axis = 0; axis < rank - 1; axis++) {
                  const Long d = plan.dim_vec[axis];
                  const Long ka_refl = (d - ks[axis]) % d;
                  src_idx = src_idx * d + ka_refl;
                }
                src_idx = src_idx * Nh + (Nd - ks[rank - 1]);
                out_b[2 * out_idx]     =  in_b[2 * src_idx];
                out_b[2 * out_idx + 1] = -in_b[2 * src_idx + 1];
              }
            }
          }
        } else {
          const Long n = 2 * total_complex;
          #pragma omp parallel for num_threads(nthreads) if(n > 4096) schedule(static)
          for (Long i = 0; i < n; i++) cur[i] = in[i];
        }
      }

      // Stage 2: rank passes, each axis = {digit-reverse, butterflies, transpose}.
      for (Long axis_idx = 0; axis_idx < rank; axis_idx++) {
        const Long k = (rank - 1 - axis_idx);
        const auto& dp = plan.dim_plan[k];
        const Long N = dp.N;
        const Long rest = total_complex / N;
        const Long* perm = (N > 0) ? &dp.digit_rev[0] : nullptr;

        #pragma omp parallel for num_threads(nthreads) if(rest >= 2 * nthreads) schedule(static)
        for (Long r = 0; r < rest; r++) {
          DigitReverseRow<T>(&cur[2 * r * N], &nxt[2 * r * N], N, perm);
        }
        swap_bufs();

        const Long s = dp.radices.Dim();
        Long m = 1;
        for (Long stg = 0; stg < s; stg++) {
          const Long rad = dp.radices[stg];
          const T* tw  = &dp.twiddle[dp.stage_twiddle_off[stg]];
          const T* dft = &dp.dft_flat[dp.stage_dft_off[stg]];
          #pragma omp parallel num_threads(nthreads) if(rest >= 2 * nthreads)
          {
            ScratchBuf<T> tmp(2 * rad);  // per-thread scratch for the generic O(p^2) radix path
            #pragma omp for schedule(static)
            for (Long r = 0; r < rest; r++) {
              RadixStageRow<T>(&cur[2 * r * N], N, m, rad, tw, dft, dp.sign, &tmp.begin()[0], &tmp.begin()[rad]);
            }
          }
          m *= rad;
        }

        #pragma omp parallel for num_threads(nthreads) if(rest * N >= 1024) collapse(2) schedule(static)
        for (Long r = 0; r < rest; r++) {
          for (Long c = 0; c < N; c++) {
            nxt[2 * (c * rest + r)]     = cur[2 * (r * N + c)];
            nxt[2 * (c * rest + r) + 1] = cur[2 * (r * N + c) + 1];
          }
        }
        swap_bufs();
      }

      // Stage 3: final transpose to bring howmany to the front.
      {
        const Long h = howmany;
        const Long rest = (h > 0) ? (total_complex / h) : 0;
        #pragma omp parallel for num_threads(nthreads) if(rest * h >= 1024) collapse(2) schedule(static)
        for (Long r = 0; r < rest; r++) {
          for (Long c = 0; c < h; c++) {
            nxt[2 * (c * rest + r)]     = cur[2 * (r * h + c)];
            nxt[2 * (c * rest + r) + 1] = cur[2 * (r * h + c) + 1];
          }
        }
        swap_bufs();
      }

      // Stage 4: write output, applying normalization + Hermitian/real extract.
      const T s = plan.norm_scale;
      if (fft_type == FFT_Type::R2C) {
        const Long Nd = plan.dim_vec[rank - 1];
        const Long Nh = Nd / 2 + 1;
        const Long batch = total_complex / Nd;
        #pragma omp parallel for num_threads(nthreads) if(batch * Nh >= 1024) schedule(static)
        for (Long b = 0; b < batch; b++) {
          for (Long i = 0; i < Nh; i++) {
            out[2 * (b * Nh + i)]     = cur[2 * (b * Nd + i)]     * s;
            out[2 * (b * Nh + i) + 1] = cur[2 * (b * Nd + i) + 1] * s;
          }
        }
      } else if (fft_type == FFT_Type::C2R) {
        const Long Nd = plan.dim_vec[rank - 1];
        const Long batch = total_complex / Nd;
        #pragma omp parallel for num_threads(nthreads) if(batch * Nd >= 4096) schedule(static)
        for (Long b = 0; b < batch; b++) {
          for (Long i = 0; i < Nd; i++) {
            out[b * Nd + i] = cur[2 * (b * Nd + i)] * s;
          }
        }
      } else {
        const Long n = 2 * total_complex;
        #pragma omp parallel for num_threads(nthreads) if(n >= 4096) schedule(static)
        for (Long i = 0; i < n; i++) out[i] = cur[i] * s;
      }
    }

  }  // namespace fft_fallback_internal

  template <class ValueType> struct FFTPlan : public fft_fallback_internal::CTPlan<ValueType> {};

  template <class ValueType> FFT<ValueType>::~FFT() {}

  template <class ValueType> FFT<ValueType>::FFT() : copy_input(false), dim{0,0}, fft_type(FFT_Type::R2C), howmany_(0) {}

  // Move via swap: the moved-from object retains the target's previous plan, so
  // its (possibly type-specialized) destructor releases it. Works for both the
  // FFTW-backed and Cooley-Tukey FFTPlan specializations.
  template <class ValueType> FFT<ValueType>::FFT(FFT<ValueType>&& other) noexcept : FFT() { this->Swap(other); }

  template <class ValueType> FFT<ValueType>& FFT<ValueType>::operator=(FFT<ValueType>&& other) noexcept {
    this->Swap(other);
    return *this;
  }

  template <class ValueType> void FFT<ValueType>::Swap(FFT<ValueType>& other) noexcept {
    std::swap(plan, other.plan);
    std::swap(copy_input, other.copy_input);
    std::swap(dim[0], other.dim[0]);
    std::swap(dim[1], other.dim[1]);
    std::swap(fft_type, other.fft_type);
    std::swap(howmany_, other.howmany_);
  }

  template <class ValueType> Long FFT<ValueType>::Dim(Integer i) const { return dim[i]; }

  template <class ValueType> void FFT<ValueType>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    this->fft_type = fft_type_;
    this->howmany_ = howmany_;
    this->copy_input = false;
    fft_fallback_internal::CTSetup<ValueType>(plan, fft_type_, howmany_, dim_vec, Nthreads, this->dim[0], this->dim[1]);
  }

  template <class ValueType> void FFT<ValueType>::Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {
    fft_fallback_internal::CTExecute<ValueType>(plan, fft_type, howmany_, dim[0], dim[1], in, out);
  }



  static inline void FFTWInitThreads(Integer Nthreads) {
#ifdef SCTL_FFTW_THREADS
    static bool once = [](){
      fftw_init_threads();
      return true;
    }();
    SCTL_UNUSED(once);
    fftw_plan_with_nthreads(Nthreads);
#endif
  }

#ifdef SCTL_HAVE_FFTW
  template <> struct FFTPlan<double> {
    FFTPlan() : fftwplan(nullptr) {}
    fftw_plan fftwplan;
  };

  template <> inline FFT<double>::~FFT() {
    if (plan.fftwplan) fftw_destroy_plan(plan.fftwplan);
    plan.fftwplan = nullptr;
  }

  template <> inline void FFT<double>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    fft_type = fft_type_;
    this->howmany_ = howmany_;
    copy_input = false;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) {
      dim_vec_[i] = dim_vec[i];
    }

    Long N0 = 0, N1 = 0;
    if (rank) { // Set N0, N1
      Long N = this->howmany_;
      for (auto ni : dim_vec) N *= ni;
      if (fft_type == FFT_Type::R2C) {
        N0 = N;
        N1 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
      } else if (fft_type == FFT_Type::C2C) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2C_INV) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2R) {
        N0 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
        N1 = N;
      } else {
        N0 = 0;
        N1 = 0;
      }
    }
    dim[0] = N0;
    dim[1] = N1;
    if (!N0 || !N1) return;
    Vector<double> in(N0), out(N1);

    #pragma omp critical(SCTL_FFTW_PLAN)
    {
    FFTWInitThreads(Nthreads);
    if (plan.fftwplan) fftw_destroy_plan(plan.fftwplan);
    plan.fftwplan = nullptr;
    if (fft_type == FFT_Type::R2C) {
      plan.fftwplan = fftw_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], nullptr, 1, N0 / this->howmany_, (fftw_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan.fftwplan = fftw_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, &out[0], nullptr, 1, N1 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan.fftwplan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan.fftwplan = fftw_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], nullptr, 1, N0 / this->howmany_, (fftw_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan.fftwplan = fftw_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, &out[0], nullptr, 1, N1 / this->howmany_, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    }
    SCTL_ASSERT(plan.fftwplan);
  }

  template <> inline void FFT<double>::Execute(const Vector<double>& in, Vector<double>& out) const {
    using ValueType = double;
    Long N0 = Dim(0);
    Long N1 = Dim(1);
    if (!N0 || !N1) return;
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    //check_align(in, out);

    ValueType s = 0;
    Vector<ValueType> tmp;
    auto in_ptr = in.begin();
    if (copy_input) { // Save input
      tmp.ReInit(N0);
      in_ptr = tmp.begin();
      tmp = in;
    }
    if (fft_type == FFT_Type::R2C) {
      s = 1 / sqrt<ValueType>(N0 / this->howmany_);
      fftw_execute_dft_r2c(plan.fftwplan, (double*)&in_ptr[0], (fftw_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C) {
      s = 1 / sqrt<ValueType>(N0 / this->howmany_ * (ValueType)0.5);
      fftw_execute_dft(plan.fftwplan, (fftw_complex*)&in_ptr[0], (fftw_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C_INV) {
      s = 1 / sqrt<ValueType>(N1 / this->howmany_ * (ValueType)0.5);
      fftw_execute_dft(plan.fftwplan, (fftw_complex*)&in_ptr[0], (fftw_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2R) {
      s = 1 / sqrt<ValueType>(N1 / this->howmany_);
      fftw_execute_dft_c2r(plan.fftwplan, (fftw_complex*)&in_ptr[0], (double*)&out[0]);
    }
    for (auto& x : out) x *= s;
  }
#endif

#ifdef SCTL_HAVE_FFTWF
  template <> struct FFTPlan<float> {
    FFTPlan() : fftwplan(nullptr) {}
    fftwf_plan fftwplan;
  };

  template <> inline FFT<float>::~FFT() {
    if (plan.fftwplan) fftwf_destroy_plan(plan.fftwplan);
    plan.fftwplan = nullptr;
  }

  template <> inline void FFT<float>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    fft_type = fft_type_;
    this->howmany_ = howmany_;
    copy_input = false;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) {
      dim_vec_[i] = dim_vec[i];
    }

    Long N0 = 0, N1 = 0;
    if (rank) { // Set N0, N1
      Long N = this->howmany_;
      for (auto ni : dim_vec) N *= ni;
      if (fft_type == FFT_Type::R2C) {
        N0 = N;
        N1 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
      } else if (fft_type == FFT_Type::C2C) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2C_INV) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2R) {
        N0 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
        N1 = N;
      } else {
        N0 = 0;
        N1 = 0;
      }
    }
    dim[0] = N0;
    dim[1] = N1;
    if (!N0 || !N1) return;
    Vector<float> in (N0), out(N1);

    #pragma omp critical(SCTL_FFTW_PLAN)
    {
    FFTWInitThreads(Nthreads);
    if (plan.fftwplan) fftwf_destroy_plan(plan.fftwplan);
    plan.fftwplan = nullptr;
    if (fft_type == FFT_Type::R2C) {
      plan.fftwplan = fftwf_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], nullptr, 1, N0 / this->howmany_, (fftwf_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan.fftwplan = fftwf_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, &out[0], nullptr, 1, N1 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan.fftwplan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan.fftwplan = fftwf_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], nullptr, 1, N0 / this->howmany_, (fftwf_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan.fftwplan = fftwf_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, &out[0], nullptr, 1, N1 / this->howmany_, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    }
    SCTL_ASSERT(plan.fftwplan);
  }

  template <> inline void FFT<float>::Execute(const Vector<float>& in, Vector<float>& out) const {
    using ValueType = float;
    Long N0 = Dim(0);
    Long N1 = Dim(1);
    if (!N0 || !N1) return;
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    //check_align(in, out);

    ValueType s = 0;
    Vector<ValueType> tmp;
    auto in_ptr = in.begin();
    if (copy_input) { // Save input
      tmp.ReInit(N0);
      in_ptr = tmp.begin();
      tmp = in;
    }
    if (fft_type == FFT_Type::R2C) {
      s = 1 / sqrt<ValueType>(N0 / this->howmany_);
      fftwf_execute_dft_r2c(plan.fftwplan, (float*)&in_ptr[0], (fftwf_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C) {
      s = 1 / sqrt<ValueType>(N0 / this->howmany_ * (ValueType)0.5);
      fftwf_execute_dft(plan.fftwplan, (fftwf_complex*)&in_ptr[0], (fftwf_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C_INV) {
      s = 1 / sqrt<ValueType>(N1 / this->howmany_ * (ValueType)0.5);
      fftwf_execute_dft(plan.fftwplan, (fftwf_complex*)&in_ptr[0], (fftwf_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2R) {
      s = 1 / sqrt<ValueType>(N1 / this->howmany_);
      fftwf_execute_dft_c2r(plan.fftwplan, (fftwf_complex*)&in_ptr[0], (float*)&out[0]);
    }
    for (auto& x : out) x *= s;
  }

#endif

#ifdef SCTL_HAVE_FFTWL
  template <> struct FFTPlan<long double> {
    FFTPlan() : fftwplan(nullptr) {}
    fftwl_plan fftwplan;
  };

  template <> inline FFT<long double>::~FFT() {
    if (plan.fftwplan) fftwl_destroy_plan(plan.fftwplan);
    plan.fftwplan = nullptr;
  }

  template <> inline void FFT<long double>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    fft_type = fft_type_;
    this->howmany_ = howmany_;
    copy_input = false;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) dim_vec_[i] = dim_vec[i];

    Long N0 = 0, N1 = 0;
    if (rank) { // Set N0, N1
      Long N = this->howmany_;
      for (auto ni : dim_vec) N *= ni;
      if (fft_type == FFT_Type::R2C) {
        N0 = N;
        N1 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
      } else if (fft_type == FFT_Type::C2C) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2C_INV) {
        N0 = N * 2;
        N1 = N * 2;
      } else if (fft_type == FFT_Type::C2R) {
        N0 = (N / dim_vec[rank - 1]) * (dim_vec[rank - 1] / 2 + 1) * 2;
        N1 = N;
      } else {
        N0 = 0;
        N1 = 0;
      }
    }
    dim[0] = N0;
    dim[1] = N1;
    if (!N0 || !N1) return;
    Vector<long double> in (N0), out(N1);

    #pragma omp critical(SCTL_FFTW_PLAN)
    {
    FFTWInitThreads(Nthreads);
    if (plan.fftwplan) fftwl_destroy_plan(plan.fftwplan);
    plan.fftwplan = nullptr;
    if (fft_type == FFT_Type::R2C) {
      plan.fftwplan = fftwl_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], nullptr, 1, N0 / this->howmany_, (fftwl_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan.fftwplan = fftwl_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, &out[0], nullptr, 1, N1 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan.fftwplan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan.fftwplan = fftwl_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], nullptr, 1, N0 / this->howmany_, (fftwl_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], nullptr, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan.fftwplan = fftwl_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], nullptr, 1, N0 / 2 / this->howmany_, &out[0], nullptr, 1, N1 / this->howmany_, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    }
    SCTL_ASSERT(plan.fftwplan);
  }

  template <> inline void FFT<long double>::Execute(const Vector<long double>& in, Vector<long double>& out) const {
    using ValueType = long double;
    Long N0 = Dim(0);
    Long N1 = Dim(1);
    if (!N0 || !N1) return;
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    //check_align(in, out);

    ValueType s = 0;
    Vector<ValueType> tmp;
    auto in_ptr = in.begin();
    if (copy_input) { // Save input
      tmp.ReInit(N0);
      in_ptr = tmp.begin();
      tmp = in;
    }
    if (fft_type == FFT_Type::R2C) {
      s = 1 / sqrt<ValueType>(N0 / this->howmany_);
      fftwl_execute_dft_r2c(plan.fftwplan, (long double*)&in_ptr[0], (fftwl_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C) {
      s = 1 / sqrt<ValueType>(N0 / this->howmany_ * (ValueType)0.5);
      fftwl_execute_dft(plan.fftwplan, (fftwl_complex*)&in_ptr[0], (fftwl_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2C_INV) {
      s = 1 / sqrt<ValueType>(N1 / this->howmany_ * (ValueType)0.5);
      fftwl_execute_dft(plan.fftwplan, (fftwl_complex*)&in_ptr[0], (fftwl_complex*)&out[0]);
    } else if (fft_type == FFT_Type::C2R) {
      s = 1 / sqrt<ValueType>(N1 / this->howmany_);
      fftwl_execute_dft_c2r(plan.fftwplan, (fftwl_complex*)&in_ptr[0], (long double*)&out[0]);
    }
    for (auto& x : out) x *= s;
  }
#endif

}  // end namespace

#endif // _SCTL_FFT_WRAPPER_TXX_
