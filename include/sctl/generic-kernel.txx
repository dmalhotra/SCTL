#ifndef _SCTL_GENERIC_KERNEL_TXX_
#define _SCTL_GENERIC_KERNEL_TXX_

#include <algorithm>                // for min
#include <type_traits>              // for false_type, bool_constant, integral_constant, void_t

#include "sctl/common.hpp"          // for Integer, Long, SCTL_ASSERT, SCTL_...
#include "sctl/generic-kernel.hpp"  // for GenericKernel
#include "sctl/intrin-wrapper.hpp"  // for TypeTraits
#include "sctl/iterator.hpp"        // for ConstIterator, Iterator
#include "sctl/matrix.hpp"          // for Matrix
#include "sctl/profile.hpp"         // for Profile, ProfileCounter
#include "sctl/profile.txx"         // for Profile::IncrementCounter
#include "sctl/scratch_pool.hpp"    // for ScratchBuf
#include "sctl/scratch_pool.txx"    // for ScratchBuf
#include "sctl/static-array.hpp"    // for StaticArray
#include "sctl/vec.hpp"             // for Vec
#include "sctl/vec.txx"             // for DefaultVecLen, FMA
#include "sctl/vector.hpp"          // for Vector

namespace sctl {

namespace detail {

// Dispatches uKerMatrix on whether the micro-kernel takes a source normal.
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

// Detects the optional fused apply described in kernel_functions.hpp: a kernel
// opts in by declaring FUSED_APPLY = true and providing uKerApply(). Eval uses
// it when present; KernelMatrix always goes through uKerMatrix, since it has to
// produce the matrix itself.
template <class uKernel, class = void> struct uKerFusedApply : std::false_type {};
template <class uKernel> struct uKerFusedApply<uKernel, std::void_t<decltype(uKernel::FUSED_APPLY)>> : std::bool_constant<uKernel::FUSED_APPLY> {};

}  // namespace detail

  template <class uKernel> GenericKernel<uKernel>::GenericKernel() : ctx_ptr(nullptr) {}

  template <class uKernel> constexpr Integer GenericKernel<uKernel>::CoordDim() {
    return DIM;
  }

  template <class uKernel> constexpr Integer GenericKernel<uKernel>::NormalDim() {
    return N_DIM;
  }

  template <class uKernel> constexpr Integer GenericKernel<uKernel>::SrcDim() {
    return KDIM0;
  }

  template <class uKernel> constexpr Integer GenericKernel<uKernel>::TrgDim() {
    return KDIM1;
  }

  template <class uKernel> void GenericKernel<uKernel>::SetCtxPtr(void* ctx) {
    ctx_ptr = ctx;
  }

  template <class uKernel> const void* GenericKernel<uKernel>::GetCtxPtr() const {
    return ctx_ptr;
  }

  template <class uKernel> template <class F> void GenericKernel<uKernel>::DigitsDispatch(const Integer digits, const F& f) {
    switch (digits) {
      case  0: return f(std::integral_constant<Integer, 0>());
      case  1: return f(std::integral_constant<Integer, 1>());
      case  2: return f(std::integral_constant<Integer, 2>());
      case  3: return f(std::integral_constant<Integer, 3>());
      case  4: return f(std::integral_constant<Integer, 4>());
      case  5: return f(std::integral_constant<Integer, 5>());
      case  6: return f(std::integral_constant<Integer, 6>());
      case  7: return f(std::integral_constant<Integer, 7>());
      case  8: return f(std::integral_constant<Integer, 8>());
      case  9: return f(std::integral_constant<Integer, 9>());
      case 10: return f(std::integral_constant<Integer,10>());
      case 11: return f(std::integral_constant<Integer,11>());
      case 12: return f(std::integral_constant<Integer,12>());
      case 13: return f(std::integral_constant<Integer,13>());
      case 14: return f(std::integral_constant<Integer,14>());
      case 15: return f(std::integral_constant<Integer,15>());
      default: SCTL_ASSERT(digits == -1 || digits >= 16);
               return f(std::integral_constant<Integer,-1>());
    }
  }

  template <class uKernel> template <class Real, bool enable_openmp> void GenericKernel<uKernel>::Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self) {
    const GenericKernel<uKernel>& ker = *(ConstIterator<GenericKernel<uKernel>>)self;
    DigitsDispatch(digits, [&ker, &v_trg, &r_trg, &r_src, &n_src, &v_src](const auto d) {
      ker.template Eval<Real, enable_openmp, decltype(d)::value>(v_trg, r_trg, r_src, n_src, v_src);
    });
  }

  template <class uKernel> template <class Real, bool enable_openmp, Integer digits> void GenericKernel<uKernel>::Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src) const {
    static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<Real>::SigBits*0.3010299957) : digits);
    static constexpr Integer VecLen = DefaultVecLen<Real>();
    using RealVec = Vec<Real, VecLen>;

    auto uKerEval = [this](RealVec (&vt)[KDIM1], const RealVec (&xt)[DIM], const RealVec (&xs)[DIM], const RealVec (&ns)[N_DIM_], const RealVec (&vs)[KDIM0]) {
      RealVec dX[DIM];
      for (Integer i = 0; i < DIM; i++) dX[i] = xt[i] - xs[i];
      if constexpr (detail::uKerFusedApply<uKernel>::value) { // skip the KDIM0 x KDIM1 matrix
        uKernel::template uKerApply<digits_,1>(vt, dX, ns, vs, ctx_ptr);
      } else {
        RealVec U[KDIM0][KDIM1];
        uKerMatrix<digits_>(U, dX, ns, ctx_ptr);
        for (Integer k0 = 0; k0 < KDIM0; k0++) {
          for (Integer k1 = 0; k1 < KDIM1; k1++) {
            vt[k1] = FMA(U[k0][k1], vs[k0], vt[k1]);
          }
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

      ScratchBuf<Real> buff_storage((DIM + KDIM1) * NNt);
      Iterator<Real> buff = buff_storage.begin();
      Matrix<Real> Xt_(DIM, NNt, buff + 0, false);
      Matrix<Real> Vt_(KDIM1, NNt, buff + DIM*NNt, false);

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

  template <class uKernel> template <class Real, bool enable_openmp> void GenericKernel<uKernel>::KernelMatrix(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn, Integer digits, ConstIterator<char> self) {
    const GenericKernel<uKernel>& ker = *(ConstIterator<GenericKernel<uKernel>>)self;
    DigitsDispatch(digits, [&ker, &M, &Xt, &Xs, &Xn](const auto d) {
      ker.template KernelMatrix<Real, enable_openmp, decltype(d)::value>(M, Xt, Xs, Xn);
    });
  }

  template <class uKernel> template <class Real, bool enable_openmp, Integer digits> void GenericKernel<uKernel>::KernelMatrix(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn) const {
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

  template <class uKernel> template <Integer digits, class VecType, class NormalType> void GenericKernel<uKernel>::uKerMatrix(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const NormalType& n, const void* ctx_ptr) {
    detail::uKerHelper<uKernel,KDIM0,KDIM1,DIM,N_DIM>::template MatEval<digits>(u, r, n, ctx_ptr);
  };

}  // end namespace

#endif // _SCTL_GENERIC_KERNEL_TXX_
