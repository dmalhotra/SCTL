#ifndef _SCTL_KERNEL_FUNCTIONS_HPP_
#define _SCTL_KERNEL_FUNCTIONS_HPP_

#include SCTL_INCLUDE(vec.hpp)
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(common.hpp)

namespace SCTL_NAMESPACE {

template <class uKernel> class GenericKernel {

    template <class VecType, Integer D, Integer ND, Integer K0, Integer K1> static constexpr Integer get_DIM  (void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], const VecType (&n)[ND], const void* ctx_ptr)) { return D; }
    template <class VecType, Integer D, Integer ND, Integer K0, Integer K1> static constexpr Integer get_KDIM0(void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], const VecType (&n)[ND], const void* ctx_ptr)) { return K0; }
    template <class VecType, Integer D, Integer ND, Integer K0, Integer K1> static constexpr Integer get_KDIM1(void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], const VecType (&n)[ND], const void* ctx_ptr)) { return K1; }
    template <class VecType, Integer D, Integer ND, Integer K0, Integer K1> static constexpr Integer get_N_DIM(void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], const VecType (&n)[ND], const void* ctx_ptr)) { return ND; }

    static constexpr Integer DIM   = get_DIM  (uKernel::template uKerMatrix<Vec<double,1>,0>);
    static constexpr Integer KDIM0 = get_KDIM0(uKernel::template uKerMatrix<Vec<double,1>,0>);
    static constexpr Integer KDIM1 = get_KDIM1(uKernel::template uKerMatrix<Vec<double,1>,0>);
    static constexpr Integer N_DIM = get_N_DIM(uKernel::template uKerMatrix<Vec<double,1>,0>);

  public:

    GenericKernel() : ctx_ptr(nullptr) {}

    static constexpr Integer CoordDim() {
      return DIM;
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

    template <class Real, bool enable_openmp=false, Integer digits=-1> void Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src) const {
      static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<Real>::SigBits*0.3010299957) : digits);
      static constexpr Integer VecLen = DefaultVecLen<Real>();
      using RealVec = Vec<Real, VecLen>;

      auto uKerEval = [this](RealVec (&vt)[KDIM1], const RealVec (&xt)[DIM], const RealVec (&xs)[DIM], const RealVec (&ns)[DIM], const RealVec (&vs)[KDIM0]) {
        RealVec dX[DIM], U[KDIM0][KDIM1];
        for (Integer i = 0; i < DIM; i++) dX[i] = xt[i] - xs[i];
        uKernel::template uKerMatrix<RealVec,digits_>(U, dX, ns, ctx_ptr);
        for (Integer k0 = 0; k0 < KDIM0; k0++) {
          for (Integer k1 = 0; k1 < KDIM1; k1++) {
            vt[k1] = FMA(U[k0][k1], vs[k0], vt[k1]);
          }
        }
      };

      const Long Ns = r_src.Dim() / DIM;
      const Long Nt = r_trg.Dim() / DIM;
      assert(r_trg.Dim() == Nt*DIM);
      assert(r_src.Dim() == Ns*DIM);
      assert(n_src.Dim() == Ns*DIM);
      assert(v_src.Dim() == Ns*KDIM0);
      if (v_trg.Dim() != Nt*KDIM1) {
        v_trg.ReInit(Nt*KDIM1);
        v_trg.SetZero();
      }

      const Long NNt = ((Nt + VecLen - 1) / VecLen) * VecLen;
      if (NNt == VecLen) {
        RealVec xt[DIM], vt[KDIM1], xs[DIM], ns[DIM], vs[KDIM0];
        for (Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
        for (Integer k = 0; k < DIM; k++) {
          alignas(sizeof(RealVec)) StaticArray<Real,VecLen> Xt;
          RealVec::Zero().StoreAligned(&Xt[0]);
          for (Integer i = 0; i < Nt; i++) Xt[i] = r_trg[i*DIM+k];
          xt[k] = RealVec::LoadAligned(&Xt[0]);
        }
        for (Long s = 0; s < Ns; s++) {
          for (Integer k = 0; k < DIM; k++) xs[k] = RealVec::Load1(&r_src[s*DIM+k]);
          for (Integer k = 0; k < DIM; k++) ns[k] = RealVec::Load1(&n_src[s*DIM+k]);
          for (Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&v_src[s*KDIM0+k]);
          uKerEval(vt, xt, xs, ns, vs);
        }
        for (Integer k = 0; k < KDIM1; k++) {
          alignas(sizeof(RealVec)) StaticArray<Real,VecLen> out;
          vt[k].StoreAligned(&out[0]);
          for (Long t = 0; t < Nt; t++) {
            v_trg[t*KDIM1+k] += out[t];
          }
        }
      } else {
        const Matrix<Real> Xs_(Ns, DIM, (Iterator<Real>)r_src.begin(), false);
        const Matrix<Real> Ns_(Ns, DIM, (Iterator<Real>)n_src.begin(), false);
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
            RealVec xt[DIM], vt[KDIM1], xs[DIM], ns[DIM], vs[KDIM0];
            for (Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
            for (Integer k = 0; k < DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
            for (Long s = 0; s < Ns; s++) {
              for (Integer k = 0; k < DIM; k++) xs[k] = RealVec::Load1(&Xs_[s][k]);
              for (Integer k = 0; k < DIM; k++) ns[k] = RealVec::Load1(&Ns_[s][k]);
              for (Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[s][k]);
              uKerEval(vt, xt, xs, ns, vs);
            }
            for (Integer k = 0; k < KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
          }
        } else {
          for (Long t = 0; t < NNt; t += VecLen) {
            RealVec xt[DIM], vt[KDIM1], xs[DIM], ns[DIM], vs[KDIM0];
            for (Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
            for (Integer k = 0; k < DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
            for (Long s = 0; s < Ns; s++) {
              for (Integer k = 0; k < DIM; k++) xs[k] = RealVec::Load1(&Xs_[s][k]);
              for (Integer k = 0; k < DIM; k++) ns[k] = RealVec::Load1(&Ns_[s][k]);
              for (Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[s][k]);
              uKerEval(vt, xt, xs, ns, vs);
            }
            for (Integer k = 0; k < KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
          }
        }

        for (Long k = 0; k < KDIM1; k++) { // v_trg += Vt_
          for (Long i = 0; i < Nt; i++) {
            v_trg[i*KDIM1+k] += Vt_[k][i];
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
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> Xn_[N_DIM];
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> M_[KDIM0*KDIM1];
        for (Integer k = 0; k < DIM; k++) VecType::Zero().StoreAligned(&Xs_[k][0]);
        for (Integer k = 0; k < N_DIM; k++) VecType::Zero().StoreAligned(&Xn_[k][0]);

        VecType vec_Xt[DIM], vec_dX[DIM], vec_Xn[N_DIM], vec_M[KDIM0][KDIM1];
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
            for (Long k = 0; k < DIM; k++) { // Set vec_Xn
              vec_Xn[k] = VecType::LoadAligned(&Xn_[k][0]);
            }
          }

          uKernel::template uKerMatrix<VecType,digits_>(vec_M, vec_dX, vec_Xn, ctx_ptr);
          for (Integer k0 = 0; k0 < KDIM0; k0++) { // Set M_
            for (Integer k1 = 0; k1 < KDIM1; k1++) {
              vec_M[k0][k1].StoreAligned(&M_[k0*KDIM1+k1][0]);
            }
          }
          for (Long i1 = 0; i1 < Ns_; i1++) { // Set M
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                M[(i0+i1)*KDIM0+k0][k1] = M_[k0*KDIM1+k1][i1];
              }
            }
          }
        }
      } else if (Xs.Dim() == DIM) {
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> Xt_[DIM];
        alignas(sizeof(VecType)) StaticArray<Real,VecLen> M_[KDIM0*KDIM1];
        for (Integer k = 0; k < DIM; k++) VecType::Zero().StoreAligned(&Xt_[k][0]);

        VecType vec_Xs[DIM], vec_dX[DIM], vec_Xn[N_DIM], vec_M[KDIM0][KDIM1];
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

          uKernel::template uKerMatrix<VecType,digits_>(vec_M, vec_dX, vec_Xn, ctx_ptr);
          for (Integer k0 = 0; k0 < KDIM0; k0++) { // Set M_
            for (Integer k1 = 0; k1 < KDIM1; k1++) {
              vec_M[k0][k1].StoreAligned(&M_[k0*KDIM1+k1][0]);
            }
          }
          for (Long i1 = 0; i1 < Nt_; i1++) { // Set M
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                M[k0][(i0+i1)*KDIM1+k1] = M_[k0*KDIM1+k1][i1];
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

  private:
    void* ctx_ptr;
};

namespace kernel_impl {
struct Laplace3D_FxU {
  static const std::string& QuadRuleName() {
    static const std::string name = "Laplace3D-FxU";
    return name;
  }
  template <class Real> static constexpr Real ScaleFactor() {
    return 1 / (4 * const_pi<Real>());
  }
  template <class VecType, Integer digits> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    u[0][0] = rinv;
  }
};
struct Laplace3D_DxU {
  static const std::string& QuadRuleName() {
    static const std::string name = "Laplace3D-DxU";
    return name;
  }
  template <class Real> static constexpr Real ScaleFactor() {
    return 1 / (4 * const_pi<Real>());
  }
  template <class VecType, Integer digits> static void uKerMatrix(VecType (&u)[1][1], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
    VecType rinv3 = rinv * rinv * rinv;
    u[0][0] = rdotn * rinv3;
  }
};
struct Laplace3D_FxdU {
  static const std::string& QuadRuleName() {
    static const std::string name = "Laplace3D-FxdU";
    return name;
  }
  template <class Real> static constexpr Real ScaleFactor() {
    return -1 / (4 * const_pi<Real>());
  }
  template <class VecType, Integer digits> static void uKerMatrix(VecType (&u)[1][3], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
    VecType r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    VecType rinv = approx_rsqrt<digits>(r2, r2 > VecType::Zero());
    VecType rinv3 = rinv * rinv * rinv;
    u[0][0] = r[0] * rinv3;
    u[0][1] = r[1] * rinv3;
    u[0][2] = r[2] * rinv3;
  }
};
struct Stokes3D_FxU {
  static const std::string& QuadRuleName() {
    static const std::string name = "Stokes3D-FxU";
    return name;
  }
  template <class Real> static constexpr Real ScaleFactor() {
    return 1 / (8 * const_pi<Real>());
  }
  template <class VecType, Integer digits> static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
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
  static const std::string& QuadRuleName() {
    static const std::string name = "Stokes3D-DxU";
    return name;
  }
  template <class Real> static constexpr Real ScaleFactor() {
    return 3 / (4 * const_pi<Real>());
  }
  template <class VecType, Integer digits> static void uKerMatrix(VecType (&u)[3][3], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
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
  static const std::string& QuadRuleName() {
    static const std::string name = "Stokes3D-FxT";
    return name;
  }
  template <class Real> static constexpr Real ScaleFactor() {
    return -3 / (4 * const_pi<Real>());
  }
  template <class VecType, Integer digits> static void uKerMatrix(VecType (&u)[3][9], const VecType (&r)[3], const VecType (&n)[3], const void* ctx_ptr) {
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

    template <class Kernel> static void Eval(Vector<Real>& U, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn, const Vector<Real>& F, const Kernel& kernel, const Comm& comm) {
      #ifdef SCTL_HAVE_PVFMM
      SCTL_ASSERT(false);
      #else
      constexpr Integer KDIM0 = Kernel::SrcDim();
      constexpr Integer KDIM1 = Kernel::TrgDim();
      const Integer rank = comm.Rank();
      const Integer np = comm.Size();
      const Long Ns = Xs.Dim() / DIM;
      const Long Nt = Xt.Dim() / DIM;
      SCTL_ASSERT(Xt.Dim() == Nt * DIM);

      const Long dof = (Ns ? F.Dim() / (Ns * KDIM0) : 1);
      SCTL_ASSERT(F.Dim() == Ns * dof * KDIM0);
      SCTL_ASSERT(dof == 1); // TODO

      Vector<Real> Xs_, Xn_, F_, U_(Nt * dof * KDIM1);
      U_.SetZero();
      for (Long i = 0; i < np; i++) {
        auto send_recv_vec = [comm,rank,np](Vector<Real>& X_, const Vector<Real>& X, Integer offset){
          Integer send_partner = (rank + offset) % np;
          Integer recv_partner = (rank + np - offset) % np;

          Long send_cnt = X.Dim(), recv_cnt;
          void* recv_req = comm.Irecv(     Ptr2Itr<Long>(&recv_cnt,1), 1, recv_partner, offset);
          void* send_req = comm.Isend(Ptr2ConstItr<Long>(&send_cnt,1), 1, send_partner, offset);
          comm.Wait(recv_req);
          comm.Wait(send_req);

          X_.ReInit(recv_cnt);
          recv_req = comm.Irecv(X_.begin(), recv_cnt, recv_partner, offset);
          send_req = comm.Isend(X .begin(), send_cnt, send_partner, offset);
          comm.Wait(recv_req);
          comm.Wait(send_req);
        };
        send_recv_vec(Xs_, Xs, i);
        send_recv_vec(Xn_, Xn, i);
        send_recv_vec(F_ , F , i);
        kernel.template Eval<Real,true>(U_, Xt, Xs_, Xn_, F_);
      }

      if (U.Dim() != Nt * dof * KDIM1) {
        U.ReInit(Nt * dof * KDIM1);
        U.SetZero();
      }
      U += U_ * kernel.template ScaleFactor<Real>();
      #endif
    }

  private:
};

}  // end namespace

#endif  //_SCTL_KERNEL_FUNCTIONS_HPP_
