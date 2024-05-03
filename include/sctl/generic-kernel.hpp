#ifndef _SCTL_GENERIC_KERNEL_HPP_
#define _SCTL_GENERIC_KERNEL_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(vec.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)

namespace SCTL_NAMESPACE {

template <class ValueType> class Matrix;
template <class ValueType> class Vector;

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

        Matrix<Real> Xt_, Vt_;
        constexpr Integer Nbuff = 16*1024;
        StaticArray<Real,Nbuff> buff;
        if (DIM*NNt < Nbuff) {
          Xt_.ReInit(DIM, NNt, buff, false);
        } else {
          Xt_.ReInit(DIM, NNt);
        }
        if ((DIM+KDIM1)*NNt < Nbuff) {
          Vt_.ReInit(KDIM1, NNt, buff+DIM*NNt, false);
        } else {
          Vt_.ReInit(KDIM1, NNt);
        }

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
      Profile::IncrementCounter(ProfileCounter::FLOP, Ns*Nt*uKernel::FLOPS());
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
      Profile::IncrementCounter(ProfileCounter::FLOP, Ns*Nt*uKernel::FLOPS());

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

}  // end namespace

#endif  //_SCTL_GENERIC_KERNEL_HPP_
