// Cross-check and benchmark the new mixed-radix Cooley-Tukey fallback in
// sctl::FFT<ValueType> against the original O(N^2) matrix-DFT fallback
// (embedded here verbatim under the name OldMatrixFFT<T>).
//
//   - Correctness: forward+inverse round-trip for the new FFT.
//   - Cross-check: new FFT output vs OldMatrixFFT output, same input.
//   - Performance: time sweep across sizes / ranks / howmany / nthreads.
//
// Run for double (FFTW path is not linked by default, so FFT<double> exercises
// the new fallback) and for sctl::QuadReal (only ever uses the fallback).

#include "sctl.hpp"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using sctl::Long;
using sctl::Integer;
using sctl::Vector;
using sctl::Matrix;
using sctl::FFT_Type;
using sctl::Iterator;
using sctl::ConstIterator;

// ---------------------------------------------------------------------------
// FFT_CT<T>: a thin wrapper that *always* uses the new Cooley-Tukey path,
// regardless of whether FFTW is linked for type T. Lets us benchmark new vs
// FFTW for double/float when SCTL_HAVE_FFTW(F) is defined (in which case
// `sctl::FFT<double/float>` is FFTW-backed).
// ---------------------------------------------------------------------------
template <class T> class FFT_CT {
  sctl::fft_fallback_internal::CTPlan<T> plan_;
  FFT_Type type_{FFT_Type::R2C};
  Long howmany_{0};
  Long dim_[2]{0, 0};
 public:
  Long Dim(Integer i) const { return dim_[i]; }
  void Setup(FFT_Type t, Long howmany, const Vector<Long>& dv, Integer nt = 1) {
    type_ = t;
    howmany_ = howmany;
    sctl::fft_fallback_internal::CTSetup<T>(plan_, t, howmany, dv, nt, dim_[0], dim_[1]);
  }
  void Execute(const Vector<T>& in, Vector<T>& out) const {
    sctl::fft_fallback_internal::CTExecute<T>(plan_, type_, howmany_, dim_[0], dim_[1], in, out);
  }
};

// ---------------------------------------------------------------------------
// OldMatrixFFT<T>: verbatim port of the pre-Cooley-Tukey generic fallback.
// Builds dense DFT matrices per axis, executes via Matrix::GEMM + transposes.
// ---------------------------------------------------------------------------
template <class ValueType> class OldMatrixFFT {
  typedef std::complex<ValueType> ComplexType;
 public:
  OldMatrixFFT() : dim_{0, 0}, fft_type_(FFT_Type::R2C), howmany_(0) {}

  Long Dim(Integer i) const { return dim_[i]; }

  void Setup(FFT_Type fft_type, Long howmany, const Vector<Long>& dim_vec) {
    // Wrapper is unnormalized now: drop the 1/sqrt(N) the reference used.
    const auto fft_r2c = [](Long N0) {
      Long N1 = (N0 / 2 + 1);
      Matrix<ValueType> M(N0, 2 * N1);
      for (Long j = 0; j < N0; j++)
        for (Long i = 0; i < N1; i++) {
          M[j][2 * i + 0] =  sctl::cos<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
          M[j][2 * i + 1] = -sctl::sin<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
        }
      return M;
    };
    const auto fft_c2c = [](Long N0) {
      Matrix<ValueType> M(2 * N0, 2 * N0);
      for (Long i = 0; i < N0; i++)
        for (Long j = 0; j < N0; j++) {
          M[2 * i + 0][2 * j + 0] =  sctl::cos<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
          M[2 * i + 1][2 * j + 0] =  sctl::sin<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
          M[2 * i + 0][2 * j + 1] = -sctl::sin<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
          M[2 * i + 1][2 * j + 1] =  sctl::cos<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
        }
      return M;
    };
    const auto fft_c2r = [](Long N0) {
      Long N1 = (N0 / 2 + 1);
      Matrix<ValueType> M(2 * N1, N0);
      for (Long i = 0; i < N1; i++) {
        for (Long j = 0; j < N0; j++) {
          M[2 * i + 0][j] =  2 * sctl::cos<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
          M[2 * i + 1][j] = -2 * sctl::sin<ValueType>(2 * sctl::const_pi<ValueType>() * j * i / N0);
        }
      }
      if (N1 > 0) {
        for (Long j = 0; j < N0; j++) {
          M[0][j] = M[0][j] * (ValueType)0.5;
          M[1][j] = M[1][j] * (ValueType)0.5;
        }
      }
      if (N0 % 2 == 0) {
        for (Long j = 0; j < N0; j++) {
          M[2 * N1 - 2][j] = M[2 * N1 - 2][j] * (ValueType)0.5;
          M[2 * N1 - 1][j] = M[2 * N1 - 1][j] * (ValueType)0.5;
        }
      }
      return M;
    };

    fft_type_ = fft_type;
    howmany_ = howmany;
    M_.clear();
    const Long rank = dim_vec.Dim();
    Long N0 = 0, N1 = 0;
    if (rank) {
      if (fft_type == FFT_Type::R2C) {
        M_.push_back(fft_r2c(dim_vec[rank - 1]));
        for (Long i = rank - 2; i >= 0; i--) M_.push_back(fft_c2c(dim_vec[i]));
      } else if (fft_type == FFT_Type::C2C) {
        for (Long i = rank - 1; i >= 0; i--) M_.push_back(fft_c2c(dim_vec[i]));
      } else if (fft_type == FFT_Type::C2C_INV) {
        for (Long i = rank - 1; i >= 0; i--) M_.push_back(fft_c2c(dim_vec[i]).Transpose());
      } else if (fft_type == FFT_Type::C2R) {
        for (Long i = rank - 2; i >= 0; i--) M_.push_back(fft_c2c(dim_vec[i]).Transpose());
        M_.push_back(fft_c2r(dim_vec[rank - 1]));
      }
      N0 = howmany_ * 2;
      N1 = howmany_ * 2;
      for (const auto& M : M_) {
        N0 = N0 * M.Dim(0) / 2;
        N1 = N1 * M.Dim(1) / 2;
      }
    }
    dim_[0] = N0;
    dim_[1] = N1;
  }

  void Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {
    const auto transpose = [](Iterator<ValueType> out, ConstIterator<ValueType> in, Long N0, Long N1) {
      const Matrix<ComplexType> M0(N0, N1, (Iterator<ComplexType>)in, false);
      Matrix<ComplexType> M1(N1, N0, (Iterator<ComplexType>)out, false);
      M1 = M0.Transpose();
    };

    Long N0 = dim_[0], N1 = dim_[1];
    if (out.Dim() != N1) out.ReInit(N1);

    Vector<ValueType> buff0(N0 + N1);
    Vector<ValueType> buff1(N0 + N1);
    Long rank = (Long)M_.size();
    if (rank <= 0) return;
    Long N = N0;

    if (fft_type_ == FFT_Type::C2R) {
      const Matrix<ValueType>& M = M_[rank - 1];
      transpose(buff0.begin(), in.begin(), N / M.Dim(0), M.Dim(0) / 2);
      for (Long i = 0; i < rank - 1; i++) {
        const Matrix<ValueType>& Mi = M_[i];
        Matrix<ValueType> vi(N / Mi.Dim(0), Mi.Dim(0), buff0.begin(), false);
        Matrix<ValueType> vo(N / Mi.Dim(0), Mi.Dim(1), buff1.begin(), false);
        Matrix<ValueType>::GEMM(vo, vi, Mi);
        N = N * Mi.Dim(1) / Mi.Dim(0);
        transpose(buff0.begin(), buff1.begin(), N / Mi.Dim(1), Mi.Dim(1) / 2);
      }
      transpose(buff1.begin(), buff0.begin(), N / howmany_ / 2, howmany_);
      Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff1.begin(), false);
      Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), out.begin(), false);
      Matrix<ValueType>::GEMM(vo, vi, M);
    } else {
      sctl::omp_par::memcpy(buff0.begin(), in.begin(), in.Dim());
      for (Long i = 0; i < rank; i++) {
        const Matrix<ValueType>& M = M_[i];
        Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff0.begin(), false);
        Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), buff1.begin(), false);
        Matrix<ValueType>::GEMM(vo, vi, M);
        N = N * M.Dim(1) / M.Dim(0);
        transpose(buff0.begin(), buff1.begin(), N / M.Dim(1), M.Dim(1) / 2);
      }
      transpose(out.begin(), buff0.begin(), N / howmany_ / 2, howmany_);
    }
  }

 private:
  sctl::StaticArray<Long, 2> dim_;
  FFT_Type fft_type_;
  Long howmany_;
  std::vector<Matrix<ValueType>> M_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
template <class T> T inf_norm(const Vector<T>& v) {
  T m = 0;
  for (const auto& x : v) m = std::max<T>(m, sctl::fabs(x));
  return m;
}
template <class T> T inf_norm_diff(const Vector<T>& a, const Vector<T>& b) {
  SCTL_ASSERT(a.Dim() == b.Dim());
  T m = 0;
  for (Long i = 0; i < a.Dim(); i++) m = std::max<T>(m, sctl::fabs(a[i] - b[i]));
  return m;
}
template <class T> void fill_random(Vector<T>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (Long i = 0; i < v.Dim(); i++) v[i] = (T)dist(rng);
}

double wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)std::clock() / CLOCKS_PER_SEC;
#endif
}

// ---------------------------------------------------------------------------
// Correctness checks
// ---------------------------------------------------------------------------
template <class T>
void check_round_trip(FFT_Type fwd, FFT_Type inv,
                      const Vector<Long>& dim_vec, Long howmany, const char* tag) {
  sctl::FFT<T> f, fi;
  f .Setup(fwd, howmany, dim_vec);
  fi.Setup(inv, howmany, dim_vec);
  Vector<T> v0(f.Dim(0)), v1, v2;
  fill_random(v0, 12345u);
  f .Execute(v0, v1);
  fi.Execute(v1, v2);
  // Unnormalized: fwd then inv scales by N; undo before comparing.
  Long N = 1;
  for (Long i = 0; i < dim_vec.Dim(); i++) N *= dim_vec[i];
  v2 *= (T)1 / (T)N;
  T err = inf_norm_diff(v0, v2);
  T tol = sctl::machine_eps<T>() * 64 * std::max<Long>(1, f.Dim(0));
  std::cout << "  [" << tag << "] round-trip err = " << err << " (tol " << tol << ")\n";
  SCTL_ASSERT_MSG(err < tol, "round-trip error too large");
}

template <class T>
void check_vs_old(FFT_Type type, const Vector<Long>& dim_vec, Long howmany, const char* tag) {
  sctl::FFT<T> f_new;
  OldMatrixFFT<T> f_old;
  f_new.Setup(type, howmany, dim_vec);
  f_old.Setup(type, howmany, dim_vec);
  SCTL_ASSERT(f_new.Dim(0) == f_old.Dim(0));
  SCTL_ASSERT(f_new.Dim(1) == f_old.Dim(1));

  Vector<T> v0(f_new.Dim(0)), vn, vo;
  if (type == FFT_Type::C2R) {
    // C2R is only well-defined on a Hermitian-symmetric spectrum; on arbitrary
    // input FFTW and the matrix reference apply different (implementation-
    // defined) conventions for the multi-dim case. Feed a valid spectrum by
    // forward-transforming random real data via R2C.
    sctl::FFT<T> f_r2c;
    f_r2c.Setup(FFT_Type::R2C, howmany, dim_vec);
    Vector<T> real_in(f_r2c.Dim(0));
    fill_random(real_in, 13579u);
    f_r2c.Execute(real_in, v0);
  } else {
    fill_random(v0, 67890u);
  }
  // C2R (preserve_input=false) may destroy v0: reference on a copy, new FFT second.
  Vector<T> v0_copy = v0;
  f_old.Execute(v0_copy, vo);
  f_new.Execute(v0, vn);
  T err = inf_norm_diff(vn, vo);
  T tol = sctl::machine_eps<T>() * 64 * std::max<Long>(1, f_new.Dim(0));
  std::cout << "  [" << tag << "] vs-old err = " << err << " (tol " << tol << ")\n";
  SCTL_ASSERT_MSG(err < tol, "new vs old mismatch");
}

// ---------------------------------------------------------------------------
// Performance harness
// ---------------------------------------------------------------------------
template <class F, class T>
double bench_execute(F& f, const Vector<T>& v0, double min_time) {
  Vector<T> vo;
  f.Execute(v0, vo);  // warm
  double t0 = wtime();
  Long iters = 0;
  while (iters < 1 || (wtime() - t0) < min_time) {
    f.Execute(v0, vo);
    iters++;
  }
  return (wtime() - t0) / iters;
}

template <class T>
void perf_one(FFT_Type type, const Vector<Long>& dim_vec, Long howmany, Integer nthreads,
              bool run_old, bool run_fftw, double min_time = 0.05) {
  FFT_CT<T> f_ct;
  f_ct.Setup(type, howmany, dim_vec, nthreads);
  Vector<T> v0(f_ct.Dim(0));
  fill_random(v0, 0xdeadbeefu);

  const double t_ct = bench_execute(f_ct, v0, min_time);

  double t_old = 0;
  if (run_old) {
    OldMatrixFFT<T> f_old;
    f_old.Setup(type, howmany, dim_vec);
    t_old = bench_execute(f_old, v0, std::min<double>(min_time, 0.1));
  }

  double t_fftw = 0;
  if (run_fftw) {
    sctl::FFT<T> f_fftw;       // FFTW-backed when SCTL_HAVE_FFTW(F/L) is defined for T
    f_fftw.Setup(type, howmany, dim_vec, nthreads);
    t_fftw = bench_execute(f_fftw, v0, min_time);
  }

  std::string dims;
  for (Long i = 0; i < dim_vec.Dim(); i++) {
    if (i) dims += "x";
    dims += std::to_string(dim_vec[i]);
  }
  std::cout << "  " << std::setw(14) << dims
            << " rank=" << dim_vec.Dim()
            << " hm=" << std::setw(2) << howmany
            << " nt=" << std::setw(2) << nthreads
            << "  t_ct=" << std::scientific << std::setprecision(3) << t_ct << "s";
  if (run_old) {
    std::cout << "  t_old=" << t_old << "s  ct/old=" << std::fixed
              << std::setprecision(2) << (t_old / t_ct) << "x";
  }
  if (run_fftw) {
    std::cout << std::scientific << std::setprecision(3)
              << "  t_fftw=" << t_fftw << "s  fftw/ct="
              << std::fixed << std::setprecision(2) << (t_ct / t_fftw) << "x";
  }
  std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------
template <class T>
void run_correctness(const char* tname) {
  std::cout << "\n=== Correctness: " << tname << " ===\n";
  auto V = [](std::initializer_list<Long> l) {
    Vector<Long> v;
    for (Long x : l) v.PushBack(x);
    return v;
  };

  // round-trip across types
  for (Long hm : {1, 3}) {
    check_round_trip<T>(FFT_Type::R2C,     FFT_Type::C2R,     V({8}),         hm, "1D pow2 R2C");
    check_round_trip<T>(FFT_Type::C2C,     FFT_Type::C2C_INV, V({8}),         hm, "1D pow2 C2C");
    check_round_trip<T>(FFT_Type::R2C,     FFT_Type::C2R,     V({30}),        hm, "1D smooth R2C");
    check_round_trip<T>(FFT_Type::C2C,     FFT_Type::C2C_INV, V({30}),        hm, "1D smooth C2C");
    check_round_trip<T>(FFT_Type::R2C,     FFT_Type::C2R,     V({13}),        hm, "1D prime R2C");
    check_round_trip<T>(FFT_Type::C2C,     FFT_Type::C2C_INV, V({13}),        hm, "1D prime C2C");
    check_round_trip<T>(FFT_Type::R2C,     FFT_Type::C2R,     V({2, 5, 3}),   hm, "3D mixed R2C");
    check_round_trip<T>(FFT_Type::C2C,     FFT_Type::C2C_INV, V({2, 5, 3}),   hm, "3D mixed C2C");
    check_round_trip<T>(FFT_Type::R2C,     FFT_Type::C2R,     V({6, 10}),     hm, "2D smooth R2C");
    check_round_trip<T>(FFT_Type::C2C,     FFT_Type::C2C_INV, V({6, 10}),     hm, "2D smooth C2C");
  }

  // vs old matrix DFT (small sizes — old is O(N^2))
  check_vs_old<T>(FFT_Type::R2C,     V({16}),       1, "vs-old 1D R2C");
  check_vs_old<T>(FFT_Type::C2C,     V({16}),       1, "vs-old 1D C2C");
  check_vs_old<T>(FFT_Type::C2C_INV, V({16}),       1, "vs-old 1D C2C_INV");
  check_vs_old<T>(FFT_Type::C2R,     V({16}),       1, "vs-old 1D C2R");
  check_vs_old<T>(FFT_Type::C2C,     V({30}),       1, "vs-old 1D mixed C2C");
  check_vs_old<T>(FFT_Type::R2C,     V({13}),       1, "vs-old 1D prime R2C");
  check_vs_old<T>(FFT_Type::R2C,     V({2, 5, 3}),  3, "vs-old 3D R2C");
  check_vs_old<T>(FFT_Type::C2C,     V({2, 5, 3}),  3, "vs-old 3D C2C");
  check_vs_old<T>(FFT_Type::C2R,     V({2, 5, 3}),  3, "vs-old 3D C2R");
  check_vs_old<T>(FFT_Type::C2C,     V({6, 10}),    2, "vs-old 2D");
}

template <class T>
void run_perf(const char* tname, bool include_old, bool include_fftw, Long size_cap = 1<<30) {
  std::cout << "\n=== Performance: " << tname
            << (include_fftw ? " (vs FFTW)" : "") << " ===\n";
  auto V = [](std::initializer_list<Long> l) {
    Vector<Long> v;
    for (Long x : l) v.PushBack(x);
    return v;
  };

  const Integer max_threads =
#ifdef _OPENMP
      (Integer)omp_get_max_threads();
#else
      1;
#endif
  const Integer nt_hi = std::min<Integer>(max_threads, 8);

  // Threshold beyond which we skip the old O(N^2) matrix-DFT comparison
  // (it dwarfs total bench time at large N).
  const Long old_cap = 4096;
  auto inc_old = [&](Long sz) { return include_old && sz <= old_cap; };

  // Pow-2 sweep
  for (Long N : {64L, 256L, 1024L, 4096L, 16384L, 65536L}) {
    if (N > size_cap) continue;
    perf_one<T>(FFT_Type::C2C, V({N}), 1, 1, inc_old(N), include_fftw);
    if (nt_hi > 1) perf_one<T>(FFT_Type::C2C, V({N}), 1, nt_hi, /*old=*/false, include_fftw);
  }
  // Smooth-composite sweep
  for (Long N : {60L, 120L, 360L, 720L, 5040L}) {
    if (N > size_cap) continue;
    perf_one<T>(FFT_Type::C2C, V({N}), 1, 1, inc_old(N), include_fftw);
  }
  // Prime sweep
  for (Long N : {13L, 17L, 31L}) {
    perf_one<T>(FFT_Type::C2C, V({N}), 1, 1, inc_old(N), include_fftw);
  }
  // 2D / 3D with batches
  for (Long N : {16L, 32L, 64L, 128L}) {
    if (N * N > size_cap) continue;
    perf_one<T>(FFT_Type::C2C, V({N, N}), 4, 1, inc_old(N * N), include_fftw);
    if (nt_hi > 1) perf_one<T>(FFT_Type::C2C, V({N, N}), 4, nt_hi, /*old=*/false, include_fftw);
  }
  for (Long N : {8L, 16L, 32L}) {
    if (N * N * N > size_cap) continue;
    perf_one<T>(FFT_Type::R2C, V({N, N, N}), 1, 1, inc_old(N * N * N), include_fftw);
  }
}

}  // namespace

int main() {
  std::cout << std::scientific;
  run_correctness<float>("float");
  run_correctness<double>("double");
  run_correctness<long double>("long double");
#ifdef SCTL_QUAD_T
  run_correctness<sctl::QuadReal>("QuadReal");
#endif

  const bool have_fftw_double =
#ifdef SCTL_HAVE_FFTW
      true;
#else
      false;
#endif
  const bool have_fftw_float =
#ifdef SCTL_HAVE_FFTWF
      true;
#else
      false;
#endif

  run_perf<double>("double", /*include_old=*/true, /*include_fftw=*/have_fftw_double, /*size_cap=*/65536);
  run_perf<float> ("float",  /*include_old=*/false, /*include_fftw=*/have_fftw_float,  /*size_cap=*/65536);
#ifdef SCTL_QUAD_T
  run_perf<sctl::QuadReal>("QuadReal", /*include_old=*/true, /*include_fftw=*/false, /*size_cap=*/256);
#endif

  std::cout << "\nAll tests passed.\n";
  return 0;
}
