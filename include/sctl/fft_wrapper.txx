#include <vector>

#include <sctl/common.hpp>
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(math_utils.hpp)

namespace SCTL_NAMESPACE {

  //template <class ValueType> void FFT<ValueType>::check_align(const Vector<ValueType>& in, const Vector<ValueType>& out) {
  //  //SCTL_ASSERT_MSG((((uintptr_t)& in[0]) & ((uintptr_t)(SCTL_MEM_ALIGN - 1))) == 0, "sctl::FFT: Input vector not aligned to " <<SCTL_MEM_ALIGN<<" bytes!");
  //  //SCTL_ASSERT_MSG((((uintptr_t)&out[0]) & ((uintptr_t)(SCTL_MEM_ALIGN - 1))) == 0, "sctl::FFT: Output vector not aligned to "<<SCTL_MEM_ALIGN<<" bytes!");
  //  // TODO: copy to auxiliary array if unaligned
  //}

  template <class ValueType> struct FFTPlan { std::vector<Matrix<ValueType>> M; };

  template <class ValueType> FFT<ValueType>::FFT() : dim{0,0}, fft_type(FFT_Type::R2C), howmany_(0) {}

  template <class ValueType> Long FFT<ValueType>::Dim(Integer i) const { return dim[i]; }

  template <class ValueType> void FFT<ValueType>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    const auto fft_r2c = [](Long N0) {
      ValueType s = 1 / sqrt<ValueType>(N0);
      Long N1 = (N0 / 2 + 1);
      Matrix<ValueType> M(N0, 2 * N1);
      for (Long j = 0; j < N0; j++)
        for (Long i = 0; i < N1; i++) {
          M[j][2 * i + 0] =  cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
          M[j][2 * i + 1] = -sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
        }
      return M;
    };
    const auto fft_c2c = [](Long N0) {
      ValueType s = 1 / sqrt<ValueType>(N0);
      Matrix<ValueType> M(2 * N0, 2 * N0);
      for (Long i = 0; i < N0; i++)
        for (Long j = 0; j < N0; j++) {
          M[2 * i + 0][2 * j + 0] =  cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
          M[2 * i + 1][2 * j + 0] =  sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
          M[2 * i + 0][2 * j + 1] = -sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
          M[2 * i + 1][2 * j + 1] =  cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
        }
      return M;
    };
    const auto fft_c2r = [](Long N0) {
      ValueType s = 1 / sqrt<ValueType>(N0);
      Long N1 = (N0 / 2 + 1);
      Matrix<ValueType> M(2 * N1, N0);
      for (Long i = 0; i < N1; i++) {
        for (Long j = 0; j < N0; j++) {
          M[2 * i + 0][j] =  2 * cos<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
          M[2 * i + 1][j] = -2 * sin<ValueType>(2 * const_pi<ValueType>() * j * i / N0)*s;
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

    Long rank = dim_vec.Dim();
    this->fft_type = fft_type_;
    this->howmany_ = howmany_;
    plan.M.resize(0);

    if (this->fft_type == FFT_Type::R2C) {
      plan.M.push_back(fft_r2c(dim_vec[rank - 1]));
      for (Long i = rank - 2; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]));
    } else if (this->fft_type == FFT_Type::C2C) {
      for (Long i = rank - 1; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]));
    } else if (this->fft_type == FFT_Type::C2C_INV) {
      for (Long i = rank - 1; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]).Transpose());
    } else if (this->fft_type == FFT_Type::C2R) {
      for (Long i = rank - 2; i >= 0; i--) plan.M.push_back(fft_c2c(dim_vec[i]).Transpose());
      plan.M.push_back(fft_c2r(dim_vec[rank - 1]));
    }

    Long N0 = this->howmany_ * 2;
    Long N1 = this->howmany_ * 2;
    for (const auto& M : plan.M) {
      N0 = N0 * M.Dim(0) / 2;
      N1 = N1 * M.Dim(1) / 2;
    }
    this->dim[0] = N0;
    this->dim[1] = N1;
  }

  template <class ValueType> void FFT<ValueType>::Execute(const Vector<ValueType>& in, Vector<ValueType>& out) const {
    const auto transpose = [](Iterator<ValueType> out, ConstIterator<ValueType> in, Long N0, Long N1) {
      const Matrix<ComplexType> M0(N0, N1, (Iterator<ComplexType>)in, false);
      Matrix<ComplexType> M1(N1, N0, (Iterator<ComplexType>)out, false);
      M1 = M0.Transpose();
    };

    Long N0 = this->Dim(0);
    Long N1 = this->Dim(1);
    SCTL_ASSERT_MSG(in.Dim() == N0, "FFT: Wrong input size.");
    if (out.Dim() != N1) out.ReInit(N1);
    //this->check_align(in, out);

    Vector<ValueType> buff0(N0 + N1);
    Vector<ValueType> buff1(N0 + N1);
    Long rank = plan.M.size();
    if (rank <= 0) return;
    Long N = N0;

    if (this->fft_type == FFT_Type::C2R) {
      const Matrix<ValueType>& M = plan.M[rank - 1];
      transpose(buff0.begin(), in.begin(), N / M.Dim(0), M.Dim(0) / 2);

      for (Long i = 0; i < rank - 1; i++) {
        const Matrix<ValueType>& M = plan.M[i];
        Matrix<ValueType> vi(N / M.Dim(0), M.Dim(0), buff0.begin(), false);
        Matrix<ValueType> vo(N / M.Dim(0), M.Dim(1), buff1.begin(), false);
        Matrix<ValueType>::GEMM(vo, vi, M);
        N = N * M.Dim(1) / M.Dim(0);
        transpose(buff0.begin(), buff1.begin(), N / M.Dim(1), M.Dim(1) / 2);
      }
      transpose(buff1.begin(), buff0.begin(), N / this->howmany_ / 2, this->howmany_);

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
        transpose(buff0.begin(), buff1.begin(), N / M.Dim(1), M.Dim(1) / 2);
      }
      transpose(out.begin(), buff0.begin(), N / this->howmany_ / 2, this->howmany_);
    }
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
  template <> struct FFTPlan<double> { fftw_plan fftwplan; };

  template <> FFT<double>::~FFT() { if (Dim(0) && Dim(1)) fftw_destroy_plan(plan.fftwplan); }

  template <> void FFT<double>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    FFTWInitThreads(Nthreads);
    if (Dim(0) && Dim(1)) fftw_destroy_plan(plan.fftwplan);
    fft_type = fft_type_;
    this->howmany_ = howmany_;
    copy_input = false;
    plan.fftwplan = NULL;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) {
      dim_vec_[i] = dim_vec[i];
    }

    Long N0 = 0, N1 = 0;
    { // Set N0, N1
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
      dim[0] = N0;
      dim[1] = N1;
    }
    if (!N0 || !N1) return;
    Vector<double> in(N0), out(N1);

    if (fft_type == FFT_Type::R2C) {
      plan.fftwplan = fftw_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], NULL, 1, N0 / this->howmany_, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan.fftwplan = fftw_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, &out[0], NULL, 1, N1 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan.fftwplan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan.fftwplan = fftw_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], NULL, 1, N0 / this->howmany_, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan.fftwplan = fftw_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftw_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan.fftwplan = fftw_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftw_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, &out[0], NULL, 1, N1 / this->howmany_, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    SCTL_ASSERT(plan.fftwplan);
  }

  template <> void FFT<double>::Execute(const Vector<double>& in, Vector<double>& out) const {
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
  template <> struct FFTPlan<float> { fftwf_plan fftwplan; };

  template <> FFT<float>::~FFT() { if (Dim(0) && Dim(1)) fftwf_destroy_plan(plan.fftwplan); }

  template <> void FFT<float>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    FFTWInitThreads(Nthreads);
    if (Dim(0) && Dim(1)) fftwf_destroy_plan(plan.fftwplan);
    fft_type = fft_type_;
    this->howmany_ = howmany_;
    copy_input = false;
    plan.fftwplan = NULL;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) {
      dim_vec_[i] = dim_vec[i];
    }

    Long N0, N1;
    { // Set N0, N1
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
      dim[0] = N0;
      dim[1] = N1;
    }
    if (!N0 || !N1) return;
    Vector<float> in (N0), out(N1);

    if (fft_type == FFT_Type::R2C) {
      plan.fftwplan = fftwf_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], NULL, 1, N0 / this->howmany_, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan.fftwplan = fftwf_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, &out[0], NULL, 1, N1 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan.fftwplan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan.fftwplan = fftwf_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], NULL, 1, N0 / this->howmany_, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan.fftwplan = fftwf_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwf_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan.fftwplan = fftwf_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwf_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, &out[0], NULL, 1, N1 / this->howmany_, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    SCTL_ASSERT(plan.fftwplan);
  }

  template <> void FFT<float>::Execute(const Vector<float>& in, Vector<float>& out) const {
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
  template <> struct FFTPlan<long double> { fftwl_plan fftwplan; };

  template <> FFT<long double>::~FFT() { if (Dim(0) && Dim(1)) fftwl_destroy_plan(plan.fftwplan); }

  template <> void FFT<long double>::Setup(FFT_Type fft_type_, Long howmany_, const Vector<Long>& dim_vec, Integer Nthreads) {
    FFTWInitThreads(Nthreads);
    if (Dim(0) && Dim(1)) fftwl_destroy_plan(plan.fftwplan);
    fft_type = fft_type_;
    this->howmany_ = howmany_;
    copy_input = false;
    plan.fftwplan = NULL;

    Long rank = dim_vec.Dim();
    Vector<int> dim_vec_(rank);
    for (Integer i = 0; i < rank; i++) dim_vec_[i] = dim_vec[i];

    Long N0, N1;
    { // Set N0, N1
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
      dim[0] = N0;
      dim[1] = N1;
    }
    if (!N0 || !N1) return;
    Vector<long double> in (N0), out(N1);

    if (fft_type == FFT_Type::R2C) {
      plan.fftwplan = fftwl_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], NULL, 1, N0 / this->howmany_, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C) {
      plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2C_INV) {
      plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    } else if (fft_type == FFT_Type::C2R) {
      plan.fftwplan = fftwl_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, &out[0], NULL, 1, N1 / this->howmany_, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    }
    if (!plan.fftwplan) { // Build plan without FFTW_PRESERVE_INPUT
      if (fft_type == FFT_Type::R2C) {
        plan.fftwplan = fftwl_plan_many_dft_r2c(rank, &dim_vec_[0], this->howmany_, &in[0], NULL, 1, N0 / this->howmany_, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C) {
        plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_FORWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2C_INV) {
        plan.fftwplan = fftwl_plan_many_dft(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, (fftwl_complex*)&out[0], NULL, 1, N1 / 2 / this->howmany_, FFTW_BACKWARD, FFTW_ESTIMATE);
      } else if (fft_type == FFT_Type::C2R) {
        plan.fftwplan = fftwl_plan_many_dft_c2r(rank, &dim_vec_[0], this->howmany_, (fftwl_complex*)&in[0], NULL, 1, N0 / 2 / this->howmany_, &out[0], NULL, 1, N1 / this->howmany_, FFTW_ESTIMATE);
      }
      copy_input = true;
    }
    SCTL_ASSERT(plan.fftwplan);
  }

  template <> void FFT<long double>::Execute(const Vector<long double>& in, Vector<long double>& out) const {
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

