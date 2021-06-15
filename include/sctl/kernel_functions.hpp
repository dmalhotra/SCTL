#ifndef _SCTL_KERNEL_FUNCTIONS_HPP_
#define _SCTL_KERNEL_FUNCTIONS_HPP_

#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(math_utils.hpp)
#include SCTL_INCLUDE(common.hpp)

namespace SCTL_NAMESPACE {

template <class uKernel> class GenericKernel {

    template <class Real, Integer D, Integer K0, Integer K1> static constexpr Integer get_DIM  (void (*uKer)(Real (&u)[K0][K1], const Real (&r)[D], const Real (&n)[D], void* ctx_ptr)) { return D; }
    template <class Real, Integer D, Integer K0, Integer K1> static constexpr Integer get_KDIM0(void (*uKer)(Real (&u)[K0][K1], const Real (&r)[D], const Real (&n)[D], void* ctx_ptr)) { return K0; }
    template <class Real, Integer D, Integer K0, Integer K1> static constexpr Integer get_KDIM1(void (*uKer)(Real (&u)[K0][K1], const Real (&r)[D], const Real (&n)[D], void* ctx_ptr)) { return K1; }

    static constexpr Integer DIM   = get_DIM  (uKernel::template uKerMatrix<double>);
    static constexpr Integer KDIM0 = get_KDIM0(uKernel::template uKerMatrix<double>);
    static constexpr Integer KDIM1 = get_KDIM1(uKernel::template uKerMatrix<double>);

    template <class Real, Integer DOF> void uKernelEval(Iterator<Real> u, ConstIterator<Real> r, ConstIterator<Real> n, ConstIterator<Real> f) const {
      Real M[KDIM0][KDIM1];
      uKernel::uKerMatrix(M, *(Real (*)[DIM])&r[0], *(Real (*)[DIM])&n[0], ctx_ptr);
      for (Integer i = 0; i < DOF; i++) {
        for (Integer k0 = 0; k0 < KDIM0; k0++) {
          for (Integer k1 = 0; k1 < KDIM1; k1++) {
            u[i*KDIM1+k1] += M[k0][k1] * f[i*KDIM0+k0];
          }
        }
      }
    }

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

    template <class Real, Integer DOF_ = 0> void Eval(Vector<Real>& U, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn, const Vector<Real>& F) const {
      Long Ns = Xs.Dim() / DIM;
      Long Nt = Xt.Dim() / DIM;
      Long DOF = (DOF_ ? DOF_ : F.Dim() / Ns / KDIM0);
      SCTL_ASSERT(Xs.Dim() == Ns * DIM);
      SCTL_ASSERT(Xt.Dim() == Nt * DIM);
      SCTL_ASSERT(F.Dim() == Ns * DOF * KDIM0);

      if (!DOF_) {
        if (DOF == 1) Eval<Real,1>(U,Xt,Xs,Xn,F);
        if (DOF == 2) Eval<Real,2>(U,Xt,Xs,Xn,F);
        if (DOF == 3) Eval<Real,3>(U,Xt,Xs,Xn,F);
        if (DOF == 4) Eval<Real,4>(U,Xt,Xs,Xn,F);
        if (DOF == 5) Eval<Real,5>(U,Xt,Xs,Xn,F);
        if (DOF == 6) Eval<Real,6>(U,Xt,Xs,Xn,F);
        if (DOF <= 6) return;
      }

      if (U.Dim() != Nt * DOF * KDIM1) {
        U.ReInit(Nt * DOF * KDIM1);
        U.SetZero();
      }
      #pragma omp parallel for schedule(static)
      for (Long t = 0; t < Nt; t++) {
        Iterator<Real> u;
        Vector<Real> u_buff0;
        StaticArray<Real,DOF_*KDIM1> u_buff1;
        if (!DOF_) { // Set u
          u_buff0.ReInit(DOF*KDIM1);
          u = u_buff0.begin();
        } else {
          u = (Iterator<Real>)u_buff1;
        }
        for (Integer k = 0; k < DOF*KDIM1; k++) {
          u[k] = 0;
        }

        StaticArray<Real,DIM> r;
        for (Long s = 0; s < Ns; s++) {
          for (Integer k = 0; k < DIM; k++) {
            r[k] = Xs[s*DIM+k] - Xt[t*DIM+k];
          }
          ConstIterator<Real> n = Xn.begin() + s*DIM;
          if (DOF_) {
            ConstIterator<Real> f = F.begin() + s*DOF_*KDIM0;
            this->template uKernelEval<Real,DOF_>(u,r,n,f);
          } else {
            ConstIterator<Real> f = F.begin() + s*DOF*KDIM0;
            for (Integer k = 0; k < DOF; k++) {
              this->template uKernelEval<Real,1>(u + k*KDIM1,r,n,f + k*KDIM0);
            }
          }
        }
        for (Integer k = 0; k < DOF*KDIM1; k++) {
          U[t*DOF*KDIM1 + k] += u[k];
        }
      }
    }

    template <class Real> void KernelMatrix(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn) const {
      Long Ns = Xs.Dim() / DIM;
      Long Nt = Xt.Dim() / DIM;
      SCTL_ASSERT(Xs.Dim() == Ns * DIM);
      SCTL_ASSERT(Xt.Dim() == Nt * DIM);

      if (M.Dim(0) != Ns * KDIM0 || M.Dim(1) != Nt * KDIM1) {
        M.ReInit(Ns * KDIM0, Nt * KDIM1);
      }
      #pragma omp parallel for schedule(static)
      for (Long t = 0; t < Nt; t++) {
        StaticArray<Real,DIM> r;
        StaticArray<Real,KDIM0> f;
        for (Integer k = 0; k < KDIM0; k++) {
          f[k] = 0;
        }
        for (Long s = 0; s < Ns; s++) {
          for (Integer k = 0; k < DIM; k++) {
            r[k] = Xs[s*DIM+k] - Xt[t*DIM+k];
          }
          ConstIterator<Real> n = Xn.begin() + s*DIM;

          for (Integer k = 0; k < KDIM0; k++) {
            f[k] = 1;
            Iterator<Real> u = M[s*KDIM0+k] + t*KDIM1;
            for (Integer i = 0; i < KDIM1; i++) u[i] = 0;
            this->template uKernelEval<Real,1>(u,r,n,f);
            f[k] = 0;
          }
        }
      }
    }

  private:
    void* ctx_ptr;
};

struct Laplace3D_FxU : public GenericKernel<Laplace3D_FxU> {
  template <class Real> static constexpr Real ScaleFactor() {
    return 1 / (4 * const_pi<Real>());
  }
  template <class Real> static void uKerMatrix(Real (&u)[1][1], const Real (&r)[3], const Real (&n)[3], void* ctx_ptr) {
    Real r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    Real rinv = (r2>0 ? 1/sqrt<Real>(r2) : 0);
    u[0][0] = rinv;
  }
};
struct Laplace3D_DxU : public GenericKernel<Laplace3D_DxU> {
  template <class Real> static constexpr Real ScaleFactor() {
    return 1 / (4 * const_pi<Real>());
  }
  template <class Real> static void uKerMatrix(Real (&u)[1][1], const Real (&r)[3], const Real (&n)[3], void* ctx_ptr) {
    Real r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    Real rinv = (r2>0 ? 1/sqrt<Real>(r2) : 0);
    Real rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
    Real rinv3 = rinv * rinv * rinv;
    u[0][0] = -rdotn * rinv3;
  }
};
struct Laplace3D_FxdU : public GenericKernel<Laplace3D_FxdU> {
  template <class Real> static constexpr Real ScaleFactor() {
    return 1 / (4 * const_pi<Real>());
  }
  template <class Real> static void uKerMatrix(Real (&u)[1][3], const Real (&r)[3], const Real (&n)[3], void* ctx_ptr) {
    Real r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    Real rinv = (r2>0 ? 1/sqrt<Real>(r2) : 0);
    Real rinv3 = rinv * rinv * rinv;
    u[0][0] = -r[0] * rinv3;
    u[0][1] = -r[1] * rinv3;
    u[0][2] = -r[2] * rinv3;
  }
};

struct Stokes3D_FxU : public GenericKernel<Stokes3D_FxU> {
  template <class Real> static constexpr Real ScaleFactor() {
    return 1 / (8 * const_pi<Real>());
  }
  template <class Real> static void uKerMatrix(Real (&u)[3][3], const Real (&r)[3], const Real (&n)[3], void* ctx_ptr) {
    Real r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    Real rinv = (r2>0 ? 1/sqrt<Real>(r2) : 0);
    Real rinv3 = rinv*rinv*rinv;
    for (Integer i = 0; i < 3; i++) {
      for (Integer j = 0; j < 3; j++) {
        u[i][j] = (i==j ? rinv : 0) + r[i]*r[j]*rinv3;
      }
    }
  }
};
struct Stokes3D_DxU : public GenericKernel<Stokes3D_DxU> {
  template <class Real> static constexpr Real ScaleFactor() {
    return -3 / (4 * const_pi<Real>());
  }
  template <class Real> static void uKerMatrix(Real (&u)[3][3], const Real (&r)[3], const Real (&n)[3], void* ctx_ptr) {
    Real r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    Real rinv = (r2>0 ? 1/sqrt<Real>(r2) : 0);
    Real rinv2 = rinv*rinv;
    Real rinv5 = rinv2*rinv2*rinv;
    Real rdotn = r[0]*n[0] + r[1]*n[1] + r[2]*n[2];
    for (Integer i = 0; i < 3; i++) {
      for (Integer j = 0; j < 3; j++) {
        u[i][j] = r[i]*r[j]*rdotn*rinv5;
      }
    }
  }
};
struct Stokes3D_FxT : public GenericKernel<Stokes3D_FxT> {
  template <class Real> static constexpr Real ScaleFactor() {
    return -3 / (4 * const_pi<Real>());
  }
  template <class Real> static void uKerMatrix(Real (&u)[3][9], const Real (&r)[3], const Real (&n)[3], void* ctx_ptr) {
    Real r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    Real rinv = (r2>0 ? 1/sqrt<Real>(r2) : 0);
    Real rinv2 = rinv*rinv;
    Real rinv5 = rinv2*rinv2*rinv;
    for (Integer i = 0; i < 3; i++) {
      for (Integer j = 0; j < 3; j++) {
        for (Integer k = 0; k < 3; k++) {
          u[i][j*3+k] = r[i]*r[j]*r[k]*rinv5;
        }
      }
    }
  }
};


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

      const Long dof = F.Dim() / (Ns * KDIM0);
      SCTL_ASSERT(Ns && F.Dim() == Ns * dof * KDIM0);

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
        kernel.Eval(U_, Xt, Xs_, Xn_, F_);
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
