#ifndef _SCTL_KERNEL_FUNCTIONS_HPP_
#define _SCTL_KERNEL_FUNCTIONS_HPP_

#include SCTL_INCLUDE(vec.hpp)
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(common.hpp)

namespace SCTL_NAMESPACE {

template <class uKernel, Integer KDIM0, Integer KDIM1, Integer DIM, Integer N_DIM> struct uKerHelper {
  template <Integer digits, class VecType> static void MatEval(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const VecType (&n)[N_DIM], const void* ctx_ptr) {
    uKernel::template uKerMatrix<digits>(u, r, n, ctx_ptr);
  }
};
template <class uKernel, Integer KDIM0, Integer KDIM1, Integer DIM> struct uKerHelper<uKernel,KDIM0,KDIM1,DIM,0> {
  template <Integer digits, class VecType, class NormalType> static void MatEval(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const NormalType& n, const void* ctx_ptr) {
    uKernel::template uKerMatrix<digits>(u, r, ctx_ptr);
  }
};

template <class uKernel> class GenericKernel : public uKernel {

    template <class VecType, Integer K0, Integer K1, Integer D, class ...T> static constexpr Integer get_DIM  (void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], T... args)) { return D;  }
    template <class VecType, Integer K0, Integer K1, Integer D, class ...T> static constexpr Integer get_KDIM0(void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], T... args)) { return K0; }
    template <class VecType, Integer K0, Integer K1, Integer D, class ...T> static constexpr Integer get_KDIM1(void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], T... args)) { return K1; }

    static constexpr Integer DIM   = get_DIM  (uKernel::template uKerMatrix<0,Vec<double,1>>);
    static constexpr Integer KDIM0 = get_KDIM0(uKernel::template uKerMatrix<0,Vec<double,1>>);
    static constexpr Integer KDIM1 = get_KDIM1(uKernel::template uKerMatrix<0,Vec<double,1>>);


    template <Integer cnt> static constexpr Integer argsize_helper() { return 0; }
    template <Integer cnt, class T, class ...T1> static constexpr Integer argsize_helper() { return (cnt == 0 ? sizeof(T) : 0) + argsize_helper<cnt-1, T1...>(); }
    template <Integer idx, class ...T1> static constexpr Integer argsize(void (uKer)(T1... args)) { return argsize_helper<idx, T1...>(); }

    template <Integer cnt> static constexpr Integer argcount_helper() { return cnt; }
    template <Integer cnt, class T, class ...T1> static constexpr Integer argcount_helper() { return argcount_helper<cnt+1, T1...>(); }
    template <class ...T1> static constexpr Integer argcount(void (uKer)(T1... args)) { return argcount_helper<0, T1...>(); }

    static constexpr Integer ARGCNT = argcount(uKernel::template uKerMatrix<0,Vec<double,1>>);
    static constexpr Integer N_DIM = (ARGCNT > 3 ? argsize<2>(uKernel::template uKerMatrix<0,Vec<double,1>>)/sizeof(Vec<double,1>) : 0);
    static constexpr Integer N_DIM_ = (N_DIM?N_DIM:1); // non-zero

  public:

    GenericKernel() : ctx_ptr(nullptr) {}

    static constexpr Integer CoordDim() {
      return DIM;
    }
    static constexpr Integer NormalDim() {
      return N_DIM;
    }
    static constexpr Integer SrcDim() {
      return KDIM0;
    }
    static constexpr Integer TrgDim() {
      return KDIM1;
    }

    const void* GetCtxPtr() const {
      return ctx_ptr;
    }

    template <class Real, bool enable_openmp> static void Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self) {
      if (digits < 8) {
        if (digits < 4) {
          if (digits == -1) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,-1>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  0) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 0>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  1) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 1>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  2) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 2>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  3) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 3>(v_trg, r_trg, r_src, n_src, v_src);
        } else {
          if (digits ==  7) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 7>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  6) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 6>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  5) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 5>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  4) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 4>(v_trg, r_trg, r_src, n_src, v_src);
        }
      } else {
        if (digits < 12) {
          if (digits ==  8) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 8>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits ==  9) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp, 9>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits == 10) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,10>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits == 11) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,11>(v_trg, r_trg, r_src, n_src, v_src);
        } else {
          if (digits == 12) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,12>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits == 13) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,13>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits == 14) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,14>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits == 15) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,15>(v_trg, r_trg, r_src, n_src, v_src);
          if (digits >= 16) ((ConstIterator<GenericKernel<uKernel>>)self)->template Eval<Real, enable_openmp,-1>(v_trg, r_trg, r_src, n_src, v_src);
        }
      }
    }

    template <class Real, bool enable_openmp=false, Integer digits=-1> void Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src) const {
      static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<Real>::SigBits*0.3010299957) : digits);
      static constexpr Integer VecLen = DefaultVecLen<Real>();
      using RealVec = Vec<Real, VecLen>;

      auto uKerEval = [this](RealVec (&vt)[KDIM1], const RealVec (&xt)[DIM], const RealVec (&xs)[DIM], const RealVec (&ns)[N_DIM_], const RealVec (&vs)[KDIM0]) {
        RealVec dX[DIM], U[KDIM0][KDIM1];
        for (Integer i = 0; i < DIM; i++) dX[i] = xt[i] - xs[i];
        uKerMatrix<digits_>(U, dX, ns, ctx_ptr);
        for (Integer k0 = 0; k0 < KDIM0; k0++) {
          for (Integer k1 = 0; k1 < KDIM1; k1++) {
            vt[k1] = FMA(U[k0][k1], vs[k0], vt[k1]);
          }
        }
      };

      const Long Ns = r_src.Dim() / DIM;
      const Long Nt = r_trg.Dim() / DIM;
      SCTL_ASSERT(r_trg.Dim() == Nt*DIM);
      SCTL_ASSERT(r_src.Dim() == Ns*DIM);
      SCTL_ASSERT(v_src.Dim() == Ns*KDIM0);
      SCTL_ASSERT(n_src.Dim() == Ns*N_DIM || !N_DIM);
      if (v_trg.Dim() != Nt*KDIM1) {
        v_trg.ReInit(Nt*KDIM1);
        v_trg.SetZero();
      }

      const Long NNt = ((Nt + VecLen - 1) / VecLen) * VecLen;
      if (NNt == VecLen) {
        RealVec xt[DIM], vt[KDIM1], xs[DIM], ns[N_DIM_], vs[KDIM0];
        for (Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
        for (Integer k = 0; k < DIM; k++) {
          alignas(sizeof(RealVec)) StaticArray<Real,VecLen> Xt;
          RealVec::Zero().StoreAligned(&Xt[0]);
          for (Integer i = 0; i < Nt; i++) Xt[i] = r_trg[i*DIM+k];
          xt[k] = RealVec::LoadAligned(&Xt[0]);
        }
        for (Long s = 0; s < Ns; s++) {
          for (Integer k = 0; k < DIM; k++) xs[k] = RealVec::Load1(&r_src[s*DIM+k]);
          for (Integer k = 0; k < N_DIM; k++) ns[k] = RealVec::Load1(&n_src[s*N_DIM+k]);
          for (Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&v_src[s*KDIM0+k]);
          uKerEval(vt, xt, xs, ns, vs);
        }
        for (Integer k = 0; k < KDIM1; k++) {
          alignas(sizeof(RealVec)) StaticArray<Real,VecLen> out;
          vt[k].StoreAligned(&out[0]);
          for (Long t = 0; t < Nt; t++) {
            v_trg[t*KDIM1+k] += out[t] * uKernel::template uKerScaleFactor<Real>();
          }
        }
      } else {
        const Matrix<Real> Xs_(Ns, DIM, (Iterator<Real>)r_src.begin(), false);
        const Matrix<Real> Ns_(Ns, N_DIM, (Iterator<Real>)n_src.begin(), false);
        const Matrix<Real> Vs_(Ns, KDIM0, (Iterator<Real>)v_src.begin(), false);

        Matrix<Real> Xt_(DIM, NNt), Vt_(KDIM1, NNt);
        for (Long k = 0; k < DIM; k++) { // Set Xt_
          for (Long i = 0; i < Nt; i++) {
            Xt_[k][i] = r_trg[i*DIM+k];
          }
          for (Long i = Nt; i < NNt; i++) {
            Xt_[k][i] = 0;
          }
        }
        if (enable_openmp) { // Compute Vt_
          #pragma omp parallel for schedule(static)
          for (Long t = 0; t < NNt; t += VecLen) {
            RealVec xt[DIM], vt[KDIM1], xs[DIM], ns[N_DIM_], vs[KDIM0];
            for (Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
            for (Integer k = 0; k < DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
            for (Long s = 0; s < Ns; s++) {
              for (Integer k = 0; k < DIM; k++) xs[k] = RealVec::Load1(&Xs_[s][k]);
              for (Integer k = 0; k < N_DIM; k++) ns[k] = RealVec::Load1(&Ns_[s][k]);
              for (Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[s][k]);
              uKerEval(vt, xt, xs, ns, vs);
            }
            for (Integer k = 0; k < KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
          }
        } else {
          for (Long t = 0; t < NNt; t += VecLen) {
            RealVec xt[DIM], vt[KDIM1], xs[DIM], ns[N_DIM_], vs[KDIM0];
            for (Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
            for (Integer k = 0; k < DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
            for (Long s = 0; s < Ns; s++) {
              for (Integer k = 0; k < DIM; k++) xs[k] = RealVec::Load1(&Xs_[s][k]);
              for (Integer k = 0; k < N_DIM; k++) ns[k] = RealVec::Load1(&Ns_[s][k]);
              for (Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[s][k]);
              uKerEval(vt, xt, xs, ns, vs);
            }
            for (Integer k = 0; k < KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
          }
        }

        for (Long k = 0; k < KDIM1; k++) { // v_trg += Vt_
          for (Long i = 0; i < Nt; i++) {
            v_trg[i*KDIM1+k] += Vt_[k][i] * uKernel::template uKerScaleFactor<Real>();
          }
        }
      }
    }

    template <class Real, bool enable_openmp=false, Integer digits=-1> void KernelMatrix(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn) const {
      static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<Real>::SigBits*0.3010299957) : digits);
      static constexpr Integer VecLen = DefaultVecLen<Real>();
      using VecType = Vec<Real, VecLen>;

      const Long Ns = Xs.Dim()/DIM;
      const Long Nt = Xt.Dim()/DIM;
      if (M.Dim(0) != Ns*KDIM0 || M.Dim(1) != Nt*KDIM1) {
        M.ReInit(Ns*KDIM0, Nt*KDIM1);
        M.SetZero();
      }

      if (Xt.Dim() == DIM) {
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> Xs_[DIM];
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> Xn_[N_DIM_];
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> M_[KDIM0*KDIM1];
        for (Integer k = 0; k < DIM; k++) VecType::Zero().StoreAligned(&Xs_[k][0]);
        for (Integer k = 0; k < N_DIM; k++) VecType::Zero().StoreAligned(&Xn_[k][0]);

        VecType vec_Xt[DIM], vec_dX[DIM], vec_Xn[N_DIM_], vec_M[KDIM0][KDIM1];
        for (Integer k = 0; k < DIM; k++) { // Set vec_Xt
          vec_Xt[k] = VecType::Load1(&Xt[k]);
        }
        for (Long i0 = 0; i0 < Ns; i0+=VecLen) {
          const Long Ns_ = std::min<Long>(VecLen, Ns-i0);

          for (Long i1 = 0; i1 < Ns_; i1++) { // Set Xs_
            for (Long k = 0; k < DIM; k++) {
              Xs_[k][i1] = Xs[(i0+i1)*DIM+k];
            }
          }
          for (Long k = 0; k < DIM; k++) { // Set vec_dX
            vec_dX[k] = vec_Xt[k] - VecType::LoadAligned(&Xs_[k][0]);
          }
          if (N_DIM) { // Set vec_Xn
            for (Long i1 = 0; i1 < Ns_; i1++) { // Set Xn_
              for (Long k = 0; k < N_DIM; k++) {
                Xn_[k][i1] = Xn[(i0+i1)*N_DIM+k];
              }
            }
            for (Long k = 0; k < N_DIM; k++) { // Set vec_Xn
              vec_Xn[k] = VecType::LoadAligned(&Xn_[k][0]);
            }
          }

          uKerMatrix<digits_>(vec_M, vec_dX, vec_Xn, ctx_ptr);
          for (Integer k0 = 0; k0 < KDIM0; k0++) { // Set M_
            for (Integer k1 = 0; k1 < KDIM1; k1++) {
              vec_M[k0][k1].StoreAligned(&M_[k0*KDIM1+k1][0]);
            }
          }
          for (Long i1 = 0; i1 < Ns_; i1++) { // Set M
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                M[(i0+i1)*KDIM0+k0][k1] = M_[k0*KDIM1+k1][i1] * uKernel::template uKerScaleFactor<Real>();
              }
            }
          }
        }
      } else if (Xs.Dim() == DIM) {
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> Xt_[DIM];
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> M_[KDIM0*KDIM1];
        for (Integer k = 0; k < DIM; k++) VecType::Zero().StoreAligned(&Xt_[k][0]);

        VecType vec_Xs[DIM], vec_dX[DIM], vec_Xn[N_DIM_], vec_M[KDIM0][KDIM1];
        for (Integer k = 0; k < DIM; k++) { // Set vec_Xs
          vec_Xs[k] = VecType::Load1(&Xs[k]);
        }
        for (Long k = 0; k < N_DIM; k++) { // Set vec_Xn
          vec_Xn[k] = VecType::Load1(&Xn[k]);
        }
        for (Long i0 = 0; i0 < Nt; i0+=VecLen) {
          const Long Nt_ = std::min<Long>(VecLen, Nt-i0);

          for (Long i1 = 0; i1 < Nt_; i1++) { // Set Xt_
            for (Long k = 0; k < DIM; k++) {
              Xt_[k][i1] = Xt[(i0+i1)*DIM+k];
            }
          }
          for (Long k = 0; k < DIM; k++) { // Set vec_dX
            vec_dX[k] = VecType::LoadAligned(&Xt_[k][0]) - vec_Xs[k];
          }

          uKerMatrix<digits_>(vec_M, vec_dX, vec_Xn, ctx_ptr);
          for (Integer k0 = 0; k0 < KDIM0; k0++) { // Set M_
            for (Integer k1 = 0; k1 < KDIM1; k1++) {
              vec_M[k0][k1].StoreAligned(&M_[k0*KDIM1+k1][0]);
            }
          }
          for (Long i1 = 0; i1 < Nt_; i1++) { // Set M
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                M[k0][(i0+i1)*KDIM1+k1] = M_[k0*KDIM1+k1][i1] * uKernel::template uKerScaleFactor<Real>();
              }
            }
          }
        }
      } else {
        if (enable_openmp) {
          #pragma omp parallel for schedule(static)
          for (Long i = 0; i < Ns; i++) {
            Matrix<Real> M_(KDIM0, Nt*KDIM1, M.begin() + i*KDIM0*Nt*KDIM1, false);
            const Vector<Real> Xs_(DIM, (Iterator<Real>)Xs.begin() + i*DIM, false);
            const Vector<Real> Xn_(N_DIM, (Iterator<Real>)Xn.begin() + i*N_DIM, false);
            KernelMatrix<Real,enable_openmp,digits>(M_, Xt, Xs_, Xn_);
          }
        } else {
          for (Long i = 0; i < Ns; i++) {
            Matrix<Real> M_(KDIM0, Nt*KDIM1, M.begin() + i*KDIM0*Nt*KDIM1, false);
            const Vector<Real> Xs_(DIM, (Iterator<Real>)Xs.begin() + i*DIM, false);
            const Vector<Real> Xn_(N_DIM, (Iterator<Real>)Xn.begin() + i*N_DIM, false);
            KernelMatrix<Real,enable_openmp,digits>(M_, Xt, Xs_, Xn_);
          }
        }
      }
    }

    template <Integer digits, class VecType, class NormalType> static void uKerMatrix(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const NormalType& n, const void* ctx_ptr) {
      uKerHelper<uKernel,KDIM0,KDIM1,DIM,N_DIM>::template MatEval<digits>(u, r, n, ctx_ptr);
    };

  private:
    void* ctx_ptr;
};

namespace kernel_impl {
struct Laplace3D_FxU {
  static const std::string& Name() {
    static const std::string name = "Laplace3D-FxU";
    return name;
  }
  template <class Real> static constexpr Real uKerScaleFactor() {
    return 1 / (4 * const_pi<Real>());
  }
  template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    u[0][0] = rinv;
  }
};
struct Laplace3D_DxU {
  static const std::string& Name() {
    static const std::string name = "Laplace3D-DxU";
    return name;
  }
  template <class Real> static constexpr Real uKerScaleFactor() {
    return 1 / (4 * const_pi<Real>());
  }
  template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
    VecType rinv3 = rinv * rinv * rinv;
    u[0][0] = rdotn * rinv3;
  }
};
struct Laplace3D_FxdU {
  static const std::string& Name() {
    static const std::string name = "Laplace3D-FxdU";
    return name;
  }
  template <class Real> static constexpr Real uKerScaleFactor() {
    return -1 / (4 * const_pi<Real>());
  }
  template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[1][3], const VecType (&r)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rinv3 = rinv * rinv * rinv;
    u[0][0] = r[0] * rinv3;
    u[0][1] = r[1] * rinv3;
    u[0][2] = r[2] * rinv3;
  }
};
struct Stokes3D_FxU {
  static const std::string& Name() {
    static const std::string name = "Stokes3D-FxU";
    return name;
  }
  template <class Real> static constexpr Real uKerScaleFactor() {
    return 1 / (8 * const_pi<Real>());
  }
  template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rinv3 = rinv*rinv*rinv;
    for (Integer i = 0; i < 3; i++) {
      for (Integer j = 0; j < 3; j++) {
        u[i][j] = (i==j ? rinv : VecType((typename VecType::ScalarType)0)) + r[i]*r[j]*rinv3;
      }
    }
  }
};
struct Stokes3D_DxU {
  static const std::string& Name() {
    static const std::string name = "Stokes3D-DxU";
    return name;
  }
  template <class Real> static constexpr Real uKerScaleFactor() {
    return 3 / (4 * const_pi<Real>());
  }
  template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rinv2 = rinv*rinv;
    VecType rinv5 = rinv2*rinv2*rinv;
    VecType rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
    for (Integer i = 0; i < 3; i++) {
      for (Integer j = 0; j < 3; j++) {
        u[i][j] = r[i]*r[j]*rdotn*rinv5;
      }
    }
  }
};
struct Stokes3D_FxT {
  static const std::string& Name() {
    static const std::string name = "Stokes3D-FxT";
    return name;
  }
  template <class Real> static constexpr Real uKerScaleFactor() {
    return -3 / (4 * const_pi<Real>());
  }
  template <Integer digits, class VecType> static void uKerMatrix(VecType (&u)[3][9], const VecType (&r)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rinv2 = rinv*rinv;
    VecType rinv5 = rinv2*rinv2*rinv;
    for (Integer i = 0; i < 3; i++) {
      for (Integer j = 0; j < 3; j++) {
        for (Integer k = 0; k < 3; k++) {
          u[i][j*3+k] = r[i]*r[j]*r[k]*rinv5;
        }
      }
    }
  }
};
}  // namespace kernel_impl

struct Laplace3D_FxU : public GenericKernel<kernel_impl::Laplace3D_FxU> {};
struct Laplace3D_DxU : public GenericKernel<kernel_impl::Laplace3D_DxU> {};
struct Laplace3D_FxdU : public GenericKernel<kernel_impl::Laplace3D_FxdU>{};
struct Stokes3D_FxU : public GenericKernel<kernel_impl::Stokes3D_FxU> {};
struct Stokes3D_DxU : public GenericKernel<kernel_impl::Stokes3D_DxU> {};
struct Stokes3D_FxT : public GenericKernel<kernel_impl::Stokes3D_FxT> {};

template <class Real, Integer DIM> class ParticleFMM {

  public:

    ParticleFMM(const ParticleFMM&) = delete;
    ParticleFMM& operator= (const ParticleFMM&) = delete;

    ParticleFMM(const Comm& comm = Comm::Self()) : comm_(comm), digits_(10) {
      fmm_ker.ker_m2m = NullIterator<char>();
      fmm_ker.ker_m2l = NullIterator<char>();
      fmm_ker.ker_l2l = NullIterator<char>();
    }

    ~ParticleFMM() {
      Vector<std::string> src_lst, trg_lst;
      Vector<std::pair<std::string,std::string>> s2t_lst;
      for (auto& it : src_map) src_lst.PushBack(it.first);
      for (auto& it : trg_map) trg_lst.PushBack(it.first);
      for (auto& it : s2t_map) s2t_lst.PushBack(it.first);

      for (const auto& name : src_lst) DeleteSrc(name);
      for (const auto& name : trg_lst) DeleteTrg(name);
      for (const auto& name : s2t_lst) DeleteS2T(name.first, name.second);

      if (fmm_ker.ker_m2m != NullIterator<char>()) fmm_ker.delete_ker_m2m(fmm_ker.ker_m2m);
      if (fmm_ker.ker_m2l != NullIterator<char>()) fmm_ker.delete_ker_m2l(fmm_ker.ker_m2l);
      if (fmm_ker.ker_l2l != NullIterator<char>()) fmm_ker.delete_ker_l2l(fmm_ker.ker_l2l);
    }

    void SetComm(const Comm& comm) {
      comm_ = comm;
    }

    void SetAccuracy(Integer digits) {
      digits_ = digits;
    }

    template <class KerM2M, class KerM2L, class KerL2L> void SetKernels(const KerM2M& ker_m2m, const KerM2L& ker_m2l, const KerL2L& ker_l2l) {
      if (fmm_ker.ker_m2m != NullIterator<char>()) fmm_ker.delete_ker_m2m(fmm_ker.ker_m2m);
      if (fmm_ker.ker_m2l != NullIterator<char>()) fmm_ker.delete_ker_m2l(fmm_ker.ker_m2l);
      if (fmm_ker.ker_l2l != NullIterator<char>()) fmm_ker.delete_ker_l2l(fmm_ker.ker_l2l);

      fmm_ker.ker_m2m = (Iterator<char>)aligned_new<KerM2M>(1);
      fmm_ker.ker_m2l = (Iterator<char>)aligned_new<KerM2L>(1);
      fmm_ker.ker_l2l = (Iterator<char>)aligned_new<KerL2L>(1);
      (*(Iterator<KerM2M>)fmm_ker.ker_m2m) = ker_m2m;
      (*(Iterator<KerM2L>)fmm_ker.ker_m2l) = ker_m2l;
      (*(Iterator<KerL2L>)fmm_ker.ker_l2l) = ker_l2l;

      fmm_ker.dim_mul_eq = ker_m2m.SrcDim();
      fmm_ker.dim_mul_ch = ker_m2m.TrgDim();
      fmm_ker.dim_loc_eq = ker_l2l.SrcDim();
      fmm_ker.dim_loc_ch = ker_l2l.TrgDim();
      SCTL_ASSERT(ker_m2m.CoordDim() == DIM);
      SCTL_ASSERT(ker_m2l.CoordDim() == DIM);
      SCTL_ASSERT(ker_l2l.CoordDim() == DIM);
      SCTL_ASSERT(ker_m2l.SrcDim() == fmm_ker.dim_mul_eq);
      SCTL_ASSERT(ker_m2l.TrgDim() == fmm_ker.dim_loc_ch);

      fmm_ker.ker_m2m_eval = KerM2M::template Eval<Real,false>;
      fmm_ker.ker_m2l_eval = KerM2L::template Eval<Real,false>;
      fmm_ker.ker_l2l_eval = KerL2L::template Eval<Real,false>;

      fmm_ker.delete_ker_m2m = DeleteKer<KerM2M>;
      fmm_ker.delete_ker_m2l = DeleteKer<KerM2L>;
      fmm_ker.delete_ker_l2l = DeleteKer<KerL2L>;
    }
    template <class KerS2M, class KerS2L> void AddSrc(const std::string& name, const KerS2M& ker_s2m, const KerS2L& ker_s2l) {
      SCTL_ASSERT_MSG(src_map.find(name) == src_map.end(), "Source name already exists.");
      src_map[name] = SrcData();
      auto& data = src_map[name];

      data.ker_s2m = (Iterator<char>)aligned_new<KerS2M>(1);
      data.ker_s2l = (Iterator<char>)aligned_new<KerS2L>(1);

      (*(Iterator<KerS2M>)data.ker_s2m) = ker_s2m;
      (*(Iterator<KerS2L>)data.ker_s2l) = ker_s2l;

      data.dim_src = ker_s2m.SrcDim();
      data.dim_mul_ch = ker_s2m.TrgDim();
      data.dim_loc_ch = ker_s2l.TrgDim();
      data.dim_normal = ker_s2m.NormalDim();
      SCTL_ASSERT(ker_s2m.CoordDim() == DIM);
      SCTL_ASSERT(ker_s2l.CoordDim() == DIM);
      SCTL_ASSERT(ker_s2l.SrcDim() == data.dim_src);
      SCTL_ASSERT(ker_s2l.NormalDim() == data.dim_normal);

      data.ker_s2m_eval = KerS2M::template Eval<Real,false>;
      data.ker_s2l_eval = KerS2L::template Eval<Real,false>;

      data.delete_ker_s2m = DeleteKer<KerS2M>;
      data.delete_ker_s2l = DeleteKer<KerS2L>;
    }
    template <class KerM2T, class KerL2T> void AddTrg(const std::string& name, const KerM2T& ker_m2t, const KerL2T& ker_l2t) {
      SCTL_ASSERT_MSG(trg_map.find(name) == trg_map.end(), "Target name already exists.");
      trg_map[name] = TrgData();
      auto& data = trg_map[name];

      data.ker_m2t = (Iterator<char>)aligned_new<KerM2T>(1);
      data.ker_l2t = (Iterator<char>)aligned_new<KerL2T>(1);

      (*(Iterator<KerM2T>)data.ker_m2t) = ker_m2t;
      (*(Iterator<KerL2T>)data.ker_l2t) = ker_l2t;

      data.dim_trg = ker_l2t.TrgDim();
      data.dim_mul_eq = ker_m2t.SrcDim();
      data.dim_loc_eq = ker_l2t.SrcDim();
      SCTL_ASSERT(ker_m2t.CoordDim() == DIM);
      SCTL_ASSERT(ker_l2t.CoordDim() == DIM);
      SCTL_ASSERT(ker_m2t.TrgDim() == data.dim_trg);

      data.ker_m2t_eval = KerM2T::template Eval<Real,false>;
      data.ker_l2t_eval = KerL2T::template Eval<Real,false>;

      data.delete_ker_m2t = DeleteKer<KerM2T>;
      data.delete_ker_l2t = DeleteKer<KerL2T>;
    }
    template <class KerS2T> void SetKernelS2T(const std::string& src_name, const std::string& trg_name, const KerS2T& ker_s2t) {
      const auto name = std::make_pair(src_name, trg_name);
      SCTL_ASSERT_MSG(s2t_map.find(name) == s2t_map.end(), "S2T name already exists.");
      s2t_map[name] = S2TData();
      auto& data = s2t_map[name];

      data.ker_s2t = (Iterator<char>)aligned_new<KerS2T>(1);
      (*(Iterator<KerS2T>)data.ker_s2t) = ker_s2t;

      data.dim_src = ker_s2t.SrcDim();
      data.dim_trg = ker_s2t.TrgDim();
      data.dim_normal = ker_s2t.NormalDim();
      SCTL_ASSERT(ker_s2t.CoordDim() == DIM);

      data.ker_s2t_eval = KerS2T::template Eval<Real,false>;
      data.ker_s2t_eval_omp = KerS2T::template Eval<Real,true>;
      data.delete_ker_s2t = DeleteKer<KerS2T>;
    }

    void DeleteSrc(const std::string& name) {
      SCTL_ASSERT_MSG(src_map.find(name) != src_map.end(), "Source name does not exist.");
      auto& data = src_map[name];

      data.delete_ker_s2m(data.ker_s2m);
      data.delete_ker_s2l(data.ker_s2l);
      src_map.erase(name);
    }
    void DeleteTrg(const std::string& name) {
      SCTL_ASSERT_MSG(trg_map.find(name) != trg_map.end(), "Target name does not exist.");
      auto& data = trg_map[name];

      data.delete_ker_m2t(data.ker_m2t);
      data.delete_ker_l2t(data.ker_l2t);
      trg_map.erase(name);
    }
    void DeleteS2T(const std::string& src_name, const std::string& trg_name) {
      const auto name = std::make_pair(src_name, trg_name);
      SCTL_ASSERT_MSG(s2t_map.find(name) != s2t_map.end(), "S2T name does not exist.");
      auto& data = s2t_map[name];

      data.delete_ker_s2t(data.ker_s2t);
      s2t_map.erase(name);
    }

    void SetSrcCoord(const std::string& name, const Vector<Real>& src_coord, const Vector<Real>& src_normal = Vector<Real>()) {
      SCTL_ASSERT_MSG(src_map.find(name) != src_map.end(), "Target name does not exist.");
      auto& data = src_map[name];
      data.X = src_coord;
      data.Xn = src_normal;
    }
    void SetSrcDensity(const std::string& name, const Vector<Real>& src_density) {
      SCTL_ASSERT_MSG(src_map.find(name) != src_map.end(), "Target name does not exist.");
      auto& data = src_map[name];
      data.F = src_density;
    }
    void SetTrgCoord(const std::string& name, const Vector<Real>& trg_coord) {
      SCTL_ASSERT_MSG(trg_map.find(name) != trg_map.end(), "Target name does not exist.");
      auto& data = trg_map[name];
      data.X = trg_coord;
    }

    void Eval(Vector<Real>& U, const std::string& trg_name) const {
      CheckKernelDims();

      #ifdef SCTL_HAVE_PVFMM
      EvalPVFMM(U, trg_name);
      #else
      EvalDirect(U, trg_name);
      #endif
    }
    void EvalDirect(Vector<Real>& U, const std::string& trg_name) const {
      const Integer rank = comm_.Rank();
      const Integer np = comm_.Size();

      SCTL_ASSERT_MSG(trg_map.find(trg_name) != trg_map.end(), "Source name does not exist.");
      const auto& trg_data = trg_map.at(trg_name);
      const Integer TrgDim = trg_data.dim_trg;
      const auto& Xt = trg_data.X;

      const Long Nt = Xt.Dim() / DIM;
      SCTL_ASSERT(Xt.Dim() == Nt * DIM);
      if (U.Dim() != Nt * TrgDim) {
        U.ReInit(Nt * TrgDim);
        U.SetZero();
      }

      for (auto& it : s2t_map) {
        if (it.first.second != trg_name) continue;
        const std::string src_name = it.first.first;

        SCTL_ASSERT_MSG(src_map.find(src_name) != src_map.end(), "Source name does not exist.");
        const auto& src_data = src_map.at(src_name);
        const Integer SrcDim = src_data.dim_src;
        const Integer NorDim = src_data.dim_normal;
        const auto& Xs = src_data.X;
        const auto& F = src_data.F;

        const Vector<Real> Xn_dummy;
        const auto& Xn = (NorDim ? src_data.Xn : Xn_dummy);

        const Long Ns = Xs.Dim() / DIM;
        SCTL_ASSERT(Xs.Dim() == Ns * DIM);
        SCTL_ASSERT(F.Dim() == Ns * SrcDim);
        SCTL_ASSERT(Xn.Dim() == Ns * NorDim);

        Vector<Real> Xs_, Xn_, F_;
        for (Long i = 0; i < np; i++) {
          auto send_recv_vec = [this,rank,np](Vector<Real>& X_, const Vector<Real>& X, Integer offset){
            Integer send_partner = (rank + offset) % np;
            Integer recv_partner = (rank + np - offset) % np;

            Long send_cnt = X.Dim(), recv_cnt = 0;
            void* recv_req = comm_.Irecv(     Ptr2Itr<Long>(&recv_cnt,1), 1, recv_partner, offset);
            void* send_req = comm_.Isend(Ptr2ConstItr<Long>(&send_cnt,1), 1, send_partner, offset);
            comm_.Wait(recv_req);
            comm_.Wait(send_req);

            X_.ReInit(recv_cnt);
            recv_req = comm_.Irecv(X_.begin(), recv_cnt, recv_partner, offset);
            send_req = comm_.Isend(X .begin(), send_cnt, send_partner, offset);
            comm_.Wait(recv_req);
            comm_.Wait(send_req);
          };
          send_recv_vec(Xs_, Xs, i);
          send_recv_vec(Xn_, Xn, i);
          send_recv_vec(F_ , F , i);
          it.second.ker_s2t_eval_omp(U, Xt, Xs_, Xn_, F_, digits_, it.second.ker_s2t);
        }
      }
    }

    static void test(const Comm& comm) {
      Laplace3D_FxU kernel_sl;
      Laplace3D_DxU kernel_dl;

      // Create target and source vectors.
      Vector<Real> trg_coord(5000*DIM);
      Vector<Real>  sl_coord(5000*DIM);
      Vector<Real>  dl_coord(5000*DIM);
      Vector<Real>  dl_norml(5000*DIM);
      for (auto& a : trg_coord) a = drand48();
      for (auto& a :  sl_coord) a = drand48();
      for (auto& a :  dl_coord) a = drand48();
      for (auto& a :  dl_norml) a = drand48();
      Long n_sl  =  sl_coord.Dim()/DIM;
      Long n_dl  =  dl_coord.Dim()/DIM;

      // Set source charges.
      Vector<Real> sl_den(n_sl*kernel_sl.SrcDim());
      Vector<Real> dl_den(n_dl*kernel_dl.SrcDim());
      for (auto& a : sl_den) a = drand48() - 0.5;
      for (auto& a : dl_den) a = drand48() - 0.5;

      ParticleFMM fmm(comm);
      fmm.SetKernels(kernel_sl, kernel_sl, kernel_sl);
      fmm.AddTrg("LaplacePoten", kernel_sl, kernel_sl);
      fmm.SetTrgCoord("LaplacePoten", trg_coord);

      fmm.AddSrc("LaplaceSL", kernel_sl, kernel_sl);
      fmm.SetKernelS2T("LaplaceSL", "LaplacePoten",kernel_sl);
      fmm.SetSrcCoord("LaplaceSL", sl_coord);

      fmm.AddSrc("LaplaceDL", kernel_dl, kernel_dl);
      fmm.SetKernelS2T("LaplaceDL", "LaplacePoten",kernel_dl);
      fmm.SetSrcCoord("LaplaceDL", dl_coord, dl_norml);

      Vector<Real> Ufmm, Uref;
      fmm.SetSrcDensity("LaplaceSL", sl_den);
      fmm.SetSrcDensity("LaplaceDL", dl_den);
      fmm.Eval(Ufmm, "LaplacePoten");
      fmm.EvalDirect(Uref, "LaplacePoten");

      Vector<Real> Uerr = Uref - Ufmm;
      { // Print error
        StaticArray<Real,2> loc_err{0,0}, glb_err{0,0};
        for (const auto& a : Uerr) loc_err[0] = std::max<Real>(loc_err[0], fabs(a));
        for (const auto& a : Uref) loc_err[1] = std::max<Real>(loc_err[1], fabs(a));
        comm.Allreduce<Real>(loc_err, glb_err, 2, Comm::CommOp::MAX);
        if (!comm.Rank()) std::cout<<loc_err[0]/loc_err[1]<<'\n';
      }
    }

  private:

    template <class Ker> static void DeleteKer(Iterator<char> ker) {
      aligned_delete((Iterator<Ker>)ker);
    }

    void CheckKernelDims() const {
      SCTL_ASSERT(fmm_ker.ker_m2m != NullIterator<char>());
      SCTL_ASSERT(fmm_ker.ker_m2l != NullIterator<char>());
      SCTL_ASSERT(fmm_ker.ker_l2l != NullIterator<char>());
      const Integer DimMulEq = fmm_ker.dim_mul_eq;
      const Integer DimMulCh = fmm_ker.dim_mul_ch;
      const Integer DimLocEq = fmm_ker.dim_loc_eq;
      const Integer DimLocCh = fmm_ker.dim_loc_ch;

      for (auto& it : s2t_map) {
        const auto& src_name = it.first.first;
        const auto& trg_name = it.first.second;
        const Integer SrcDim = it.second.dim_src;
        const Integer TrgDim = it.second.dim_trg;
        const Integer NormalDim = it.second.dim_normal;

        SCTL_ASSERT_MSG(src_map.find(src_name) != src_map.end(), "Source name does not exist.");
        SCTL_ASSERT_MSG(trg_map.find(trg_name) != trg_map.end(), "Source name does not exist.");
        const auto& src_data = src_map.at(src_name);
        const auto& trg_data = trg_map.at(trg_name);

        SCTL_ASSERT(trg_data.dim_trg == TrgDim);
        SCTL_ASSERT(src_data.dim_src == SrcDim);
        SCTL_ASSERT(src_data.dim_normal == NormalDim);

        SCTL_ASSERT(src_data.dim_mul_ch == DimMulCh);
        SCTL_ASSERT(src_data.dim_loc_ch == DimLocCh);
        SCTL_ASSERT(trg_data.dim_mul_eq == DimMulEq);
        SCTL_ASSERT(trg_data.dim_loc_eq == DimLocEq);
      }
    }

    #ifdef SCTL_HAVE_PVFMM
    void EvalPVFMM(Vector<Real>& U, const std::string& trg_name) const {
      const Integer rank = comm_.Rank();
      const Integer np = comm_.Size();

      SCTL_ASSERT_MSG(trg_map.find(trg_name) != trg_map.end(), "Source name does not exist.");
      const auto& trg_data = trg_map.at(trg_name);
      const Integer TrgDim = trg_data.dim_trg;
      const auto& Xt = trg_data.X;

      const Long Nt = Xt.Dim() / DIM;
      SCTL_ASSERT(Xt.Dim() == Nt * DIM);
      if (U.Dim() != Nt * TrgDim) {
        U.ReInit(Nt * TrgDim);
        U.SetZero();
      }

    }
    #endif

    struct FMMKernels {
      Iterator<char> ker_m2m, ker_m2l, ker_l2l;
      Integer dim_mul_ch, dim_mul_eq;
      Integer dim_loc_ch, dim_loc_eq;

      void (*ker_m2m_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_m2l_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_l2l_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_m2m)(Iterator<char> ker);
      void (*delete_ker_m2l)(Iterator<char> ker);
      void (*delete_ker_l2l)(Iterator<char> ker);
    };
    struct SrcData {
      Vector<Real> X, Xn, F;
      Iterator<char> ker_s2m, ker_s2l;
      Integer dim_src, dim_mul_ch, dim_loc_ch, dim_normal;

      void (*ker_s2m_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_s2l_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_s2m)(Iterator<char> ker);
      void (*delete_ker_s2l)(Iterator<char> ker);
    };
    struct TrgData {
      Vector<Real> X, U;
      Iterator<char> ker_m2t, ker_l2t;
      Integer dim_mul_eq, dim_loc_eq, dim_trg;

      void (*ker_m2t_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_l2t_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_m2t)(Iterator<char> ker);
      void (*delete_ker_l2t)(Iterator<char> ker);
    };
    struct S2TData {
      Iterator<char> ker_s2t;
      Integer dim_src, dim_trg, dim_normal;

      void (*ker_s2t_eval)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);
      void (*ker_s2t_eval_omp)(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

      void (*delete_ker_s2t)(Iterator<char> ker);
    };

    FMMKernels fmm_ker;
    std::map<std::string, SrcData> src_map;
    std::map<std::string, TrgData> trg_map;
    std::map<std::pair<std::string,std::string>, S2TData> s2t_map;

    Comm comm_;
    Integer digits_;
};

}  // end namespace

#endif  //_SCTL_KERNEL_FUNCTIONS_HPP_
