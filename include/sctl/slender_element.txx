#include SCTL_INCLUDE(complex.hpp)
#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(tensor.hpp)
#include SCTL_INCLUDE(quadrule.hpp)
#include SCTL_INCLUDE(ompUtils.hpp)
#include SCTL_INCLUDE(profile.hpp)
#include SCTL_INCLUDE(fft_wrapper.hpp)
#include SCTL_INCLUDE(vtudata.hpp)
#include SCTL_INCLUDE(lagrange-interp.hpp)
#include SCTL_INCLUDE(ode-solver.hpp)
#include SCTL_INCLUDE(boundary_integral.hpp)

#include <fstream>
#include <functional>

namespace SCTL_NAMESPACE {


  template <class Real, Integer Nm = 12, Integer Nr = 20, Integer Nt = 16> class ToroidalGreensFn { // deprecated
      static constexpr Integer COORD_DIM = 3;
      static constexpr Real min_dist = 0.0;
      static constexpr Real max_dist = 0.2;

    public:

      /**
       * Constructor
       */
      ToroidalGreensFn() {}

      /**
       * Precompute tables for modal Green's funcation
       */
      template <class Kernel> void Setup(const Kernel& ker, Real R0);

      /**
       * Build modal Green's function operator for a given target point
       * (x0,x1,x2).
       */
      template <class Kernel> void BuildOperatorModal(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const;

    private:

      /**
       * Basis functions in which to represent the potential.
       */
      template <class ValueType> class BasisFn { // p(x) log(x) + q(x) + 1/x
        public:
          static ValueType Eval(const Vector<ValueType>& coeff, ValueType x);
          static void EvalBasis(Vector<ValueType>& f, ValueType x);
          static const Vector<ValueType>& nds(Integer ORDER);
      };

      /**
       * Precompute tables for modal Green's funcation
       */
      template <class ValueType, class Kernel> void PrecompToroidalGreensFn(const Kernel& ker, ValueType R0);

      /**
       * Compute reference potential using adaptive integration.
       */
      template <class ValueType, class Kernel> static void ComputePotential(Vector<ValueType>& U, const Vector<ValueType>& Xtrg, ValueType R0, const Vector<ValueType>& F_, const Kernel& ker, ValueType tol = 1e-18);

      /**
       * Compute modal Green's function operator using trapezoidal quadrature
       * rule (for distant target points).
       */
      template <Integer Nnds, class Kernel> void BuildOperatorModalDirect(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const;

      Real R0_;
      FFT<Real> fft_Nm_R2C, fft_Nm_C2R;
      Matrix<Real> Mnds2coeff0, Mnds2coeff1;
      Vector<Real> U; // KDIM0*Nmm*KDIM1*Nr*Ntt
      Vector<Real> Ut; // Nr*Ntt*KDIM0*Nmm*KDIM1
  };

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::Setup(const Kernel& ker, Real R0) {
    #ifdef SCTL_QUAD_T
    using ValueType = QuadReal;
    #else
    using ValueType = long double;
    #endif
    PrecompToroidalGreensFn<ValueType>(ker, R0);
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::BuildOperatorModal(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const {
    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim();
    constexpr Integer Nmm = (Nm/2+1)*2;
    constexpr Integer Ntt = (Nt/2+1)*2;

    StaticArray<Real,2*Nr> buff0;
    StaticArray<Real,Ntt> buff1;
    Vector<Real> r_basis(Nr,buff0,false);
    Vector<Real> interp_r(Nr,buff0+Nr,false);
    Vector<Real> interp_Ntt(Ntt,buff1,false);
    if (M.Dim(0) != KDIM0*Nmm || M.Dim(1) != KDIM1) M.ReInit(KDIM0*Nmm,KDIM1);
    { // Set M
      const Real r = sqrt<Real>(x0*x0 + x1*x1);
      const Real rho = sqrt<Real>((r-R0_)*(r-R0_) + x2*x2);
      if (rho < max_dist*R0_) {
        const Real r_inv = 1/r;
        const Real rho_inv = 1/rho;
        const Real cos_theta = x0*r_inv;
        const Real sin_theta = x1*r_inv;
        const Real cos_phi = x2*rho_inv;
        const Real sin_phi = (r-R0_)*rho_inv;

        { // Set interp_r
          interp_r = 0;
          const Real rho0 = (rho/R0_-min_dist)/(max_dist-min_dist);
          BasisFn<Real>::EvalBasis(r_basis, rho0);
          for (Long i = 0; i < Nr; i++) {
            Real fn_val = 0;
            for (Long j = 0; j < Nr; j++) {
              fn_val += Mnds2coeff1[0][i*Nr+j] * r_basis[j];
            }
            for (Long j = 0; j < Nr; j++) {
              interp_r[j] += Mnds2coeff0[0][i*Nr+j] * fn_val;
            }
          }
        }
        { // Set interp_Ntt
          interp_Ntt[0] = 0.5;
          interp_Ntt[1] = 0.0;
          Complex<Real> exp_t(cos_phi, sin_phi);
          Complex<Real> exp_jt(cos_phi, sin_phi);
          for (Long j = 1; j < Ntt/2; j++) {
            interp_Ntt[j*2+0] = exp_jt.real;
            interp_Ntt[j*2+1] =-exp_jt.imag;
            exp_jt *= exp_t;
          }
        }

        M = 0;
        for (Long j = 0; j < Nr; j++) {
          for (Long k = 0; k < Ntt; k++) {
            Real interp_wt = interp_r[j] * interp_Ntt[k];
            ConstIterator<Real> Ut_ = Ut.begin() + (j*Ntt+k)*KDIM0*Nmm*KDIM1;
            for (Long i = 0; i < KDIM0*Nmm*KDIM1; i++) { // Set M
              M[0][i] += Ut_[i] * interp_wt;
            }
          }
        }
        { // Rotate by theta
          Complex<Real> exp_iktheta(1,0), exp_itheta(cos_theta, -sin_theta);
          for (Long k = 0; k < Nmm/2; k++) {
            for (Long i = 0; i < KDIM0; i++) {
              for (Long j = 0; j < KDIM1; j++) {
                Complex<Real> c(M[i*Nmm+2*k+0][j],M[i*Nmm+2*k+1][j]);
                c *= exp_iktheta;
                M[i*Nmm+2*k+0][j] = c.real;
                M[i*Nmm+2*k+1][j] = c.imag;
              }
            }
            exp_iktheta *= exp_itheta;
          }
        }
      } else if (rho < max_dist*R0_*1.25) {
        BuildOperatorModalDirect<110>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*1.67) {
        BuildOperatorModalDirect<88>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*2.5) {
        BuildOperatorModalDirect<76>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*5) {
        BuildOperatorModalDirect<50>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*10) {
        BuildOperatorModalDirect<25>(M, x0, x1, x2, ker);
      } else if (rho < max_dist*R0_*20) {
        BuildOperatorModalDirect<14>(M, x0, x1, x2, ker);
      } else {
        BuildOperatorModalDirect<Nm>(M, x0, x1, x2, ker);
      }
    }
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType> ValueType ToroidalGreensFn<Real,Nm,Nr,Nt>::BasisFn<ValueType>::Eval(const Vector<ValueType>& coeff, ValueType x) {
    if (1) {
      ValueType sum = 0;
      ValueType log_x = log(x);
      Long Nsplit = std::max<Long>(0,(coeff.Dim()-1)/2);
      ValueType x_i = 1;
      for (Long i = 0; i < Nsplit; i++) {
        sum += coeff[i] * x_i;
        x_i *= x;
      }
      x_i = 1;
      for (Long i = coeff.Dim()-2; i >= Nsplit; i--) {
        sum += coeff[i] * log_x * x_i;
        x_i *= x;
      }
      if (coeff.Dim()-1 >= 0) sum += coeff[coeff.Dim()-1] / x;
      return sum;
    }
    if (0) {
      ValueType sum = 0;
      Long Nsplit = coeff.Dim()/2;
      for (Long i = 0; i < Nsplit; i++) {
        sum += coeff[i] * sctl::pow<ValueType,Long>(x,i);
      }
      for (Long i = Nsplit; i < coeff.Dim(); i++) {
        sum += coeff[i] * log(x) * sctl::pow<ValueType,Long>(x,coeff.Dim()-1-i);
      }
      return sum;
    }
  }
  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType> void ToroidalGreensFn<Real,Nm,Nr,Nt>::BasisFn<ValueType>::EvalBasis(Vector<ValueType>& f, ValueType x) {
    const Long N = f.Dim();
    const Long Nsplit = std::max<Long>(0,(N-1)/2);

    ValueType xi = 1;
    for (Long i = 0; i < Nsplit; i++) {
      f[i] = xi;
      xi *= x;
    }

    ValueType xi_logx = log(x);
    for (Long i = N-2; i >= Nsplit; i--) {
      f[i] = xi_logx;
      xi_logx *= x;
    }

    if (N-1 >= 0) f[N-1] = 1/x;
  }
  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType> const Vector<ValueType>& ToroidalGreensFn<Real,Nm,Nr,Nt>::BasisFn<ValueType>::nds(Integer ORDER) {
    ValueType fn_start = 1e-7, fn_end = 1.0;
    auto compute_nds = [&ORDER,&fn_start,&fn_end]() {
      Vector<ValueType> nds, wts;
      auto integrands = [&ORDER,&fn_start,&fn_end](const Vector<ValueType>& nds) {
        const Integer K = ORDER;
        const Long N = nds.Dim();
        Matrix<ValueType> M(N,K);
        for (Long j = 0; j < N; j++) {
          Vector<ValueType> f(K,M[j],false);
          EvalBasis(f, nds[j]*(fn_end-fn_start)+fn_start);
        }
        return M;
      };
      InterpQuadRule<ValueType>::Build(nds, wts, integrands, 0, 1, sqrt(machine_eps<ValueType>()), ORDER);
      return nds*(fn_end-fn_start)+fn_start;
    };
    static Vector<ValueType> nds = compute_nds();
    return nds;
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType, class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::PrecompToroidalGreensFn(const Kernel& ker, ValueType R0) {
    SCTL_ASSERT(ker.CoordDim() == COORD_DIM);
    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim();
    constexpr Long Nmm = (Nm/2+1)*2;
    constexpr Long Ntt = (Nt/2+1)*2;
    R0_ = (Real)R0;

    const auto& nds = BasisFn<ValueType>::nds(Nr);
    { // Set Mnds2coeff0, Mnds2coeff1
      Matrix<ValueType> M(Nr,Nr);
      Vector<ValueType> coeff(Nr); coeff = 0;
      for (Long i = 0; i < Nr; i++) {
        coeff[i] = 1;
        for (Long j = 0; j < Nr; j++) {
          M[i][j] = BasisFn<ValueType>::Eval(coeff, nds[j]);
        }
        coeff[i] = 0;
      }

      Matrix<ValueType> U, S, Vt;
      M.SVD(U, S, Vt);
      for (Long i = 0; i < S.Dim(0); i++) {
        S[i][i] = 1/S[i][i];
      }
      auto Mnds2coeff0_ = S * Vt;
      auto Mnds2coeff1_ = U.Transpose();
      Mnds2coeff0.ReInit(Mnds2coeff0_.Dim(0), Mnds2coeff0_.Dim(1));
      Mnds2coeff1.ReInit(Mnds2coeff1_.Dim(0), Mnds2coeff1_.Dim(1));
      for (Long i = 0; i < Mnds2coeff0.Dim(0)*Mnds2coeff0.Dim(1); i++) Mnds2coeff0[0][i] = (Real)Mnds2coeff0_[0][i];
      for (Long i = 0; i < Mnds2coeff1.Dim(0)*Mnds2coeff1.Dim(1); i++) Mnds2coeff1[0][i] = (Real)Mnds2coeff1_[0][i];
    }
    { // Setup fft_Nm_R2C
      Vector<Long> dim_vec(1);
      dim_vec[0] = Nm;
      fft_Nm_R2C.Setup(FFT_Type::R2C, KDIM0, dim_vec);
      fft_Nm_C2R.Setup(FFT_Type::C2R, KDIM0*KDIM1, dim_vec);
    }

    Vector<ValueType> Xtrg(Nr*Nt*COORD_DIM);
    for (Long i = 0; i < Nr; i++) {
      for (Long j = 0; j < Nt; j++) {
        Xtrg[(i*Nt+j)*COORD_DIM+0] = R0 * (1.0 + (min_dist+(max_dist-min_dist)*nds[i]) * sin<ValueType>(j*2*const_pi<ValueType>()/Nt));
        Xtrg[(i*Nt+j)*COORD_DIM+1] = R0 * (0.0);
        Xtrg[(i*Nt+j)*COORD_DIM+2] = R0 * (0.0 + (min_dist+(max_dist-min_dist)*nds[i]) * cos<ValueType>(j*2*const_pi<ValueType>()/Nt));
      }
    }

    Vector<ValueType> U0(KDIM0*Nmm*Nr*KDIM1*Nt);
    { // Set U0
      FFT<ValueType> fft_Nm_C2R;
      { // Setup fft_Nm_C2R
        Vector<Long> dim_vec(1);
        dim_vec[0] = Nm;
        fft_Nm_C2R.Setup(FFT_Type::C2R, KDIM0, dim_vec);
      }
      Vector<ValueType> Fcoeff(KDIM0*Nmm), F, U_;
      for (Long i = 0; i < KDIM0*Nmm; i++) {
        Fcoeff = 0; Fcoeff[i] = 1;
        { // Set F
          fft_Nm_C2R.Execute(Fcoeff, F);
          Matrix<ValueType> FF(KDIM0,Nm,F.begin(), false);
          FF = FF.Transpose();
        }
        ComputePotential<ValueType>(U_, Xtrg, R0, F, ker);
        SCTL_ASSERT(U_.Dim() == Nr*Nt*KDIM1);

        for (Long j = 0; j < Nr; j++) {
          for (Long l = 0; l < Nt; l++) {
            for (Long k = 0; k < KDIM1; k++) {
              U0[((i*Nr+j)*KDIM1+k)*Nt+l] = U_[(j*Nt+l)*KDIM1+k];
            }
          }
        }
      }
    }

    Vector<ValueType> U1(KDIM0*Nmm*Nr*KDIM1*Ntt);
    { // U1 <-- fft_Nt(U0)
      FFT<ValueType> fft_Nt;
      Vector<Long> dim_vec(1); dim_vec = Nt;
      fft_Nt.Setup(FFT_Type::R2C, KDIM0*Nmm*Nr*KDIM1, dim_vec);
      fft_Nt.Execute(U0, U1);
      if (Nt%2==0 && Nt) {
        for (Long i = Ntt-2; i < U1.Dim(); i += Ntt) {
          U1[i] *= 0.5;
        }
      }
      U1 *= 1.0/sqrt<ValueType>(Nt);
    }

    U.ReInit(KDIM0*Nmm*KDIM1*Nr*Ntt);
    { // U <-- rearrange(U1)
      for (Long i0 = 0; i0 < KDIM0*Nmm; i0++) {
        for (Long i1 = 0; i1 < Nr; i1++) {
          for (Long i2 = 0; i2 < KDIM1; i2++) {
            for (Long i3 = 0; i3 < Ntt; i3++) {
              U[((i0*Nr+i1)*KDIM1+i2)*Ntt+i3] = (Real)U1[((i0*KDIM1+i2)*Nr+i1)*Ntt+i3];
            }
          }
        }
      }
    }

    Ut.ReInit(Nr*Ntt*KDIM0*Nmm*KDIM1);
    { // Set Ut
      Matrix<Real> Ut_(Nr*Ntt,KDIM0*Nmm*KDIM1, Ut.begin(), false);
      Matrix<Real> U_(KDIM0*Nmm*KDIM1,Nr*Ntt, U.begin(), false);
      Ut_ = U_.Transpose()*2.0;
    }
  }

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <class ValueType, class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::ComputePotential(Vector<ValueType>& U, const Vector<ValueType>& Xtrg, ValueType R0, const Vector<ValueType>& F_, const Kernel& ker, ValueType tol) {
   constexpr Integer KDIM0 = Kernel::SrcDim();
    Vector<ValueType> F_fourier_coeff;
    const Long Nt_ = F_.Dim() / KDIM0; // number of Fourier modes
    SCTL_ASSERT(F_.Dim() == Nt_ * KDIM0);

    { // Transpose F_
      Matrix<ValueType> FF(Nt_,KDIM0,(Iterator<ValueType>)F_.begin(), false);
      FF = FF.Transpose();
    }
    { // Set F_fourier_coeff
      FFT<ValueType> fft_plan;
      Vector<Long> dim_vec(1); dim_vec[0] = Nt_;
      fft_plan.Setup(FFT_Type::R2C, KDIM0, dim_vec);
      fft_plan.Execute(F_, F_fourier_coeff);
      if (Nt_%2==0 && F_fourier_coeff.Dim()) {
        F_fourier_coeff[F_fourier_coeff.Dim()-2] *= 0.5;
      }
    }
    auto EvalFourierExp = [&Nt_](Vector<ValueType>& F, const Vector<ValueType>& F_fourier_coeff, Integer dof, const Vector<ValueType>& theta) {
      const Long N = F_fourier_coeff.Dim() / dof / 2;
      SCTL_ASSERT(F_fourier_coeff.Dim() == dof * N * 2);
      const Long Ntheta = theta.Dim();
      if (F.Dim() != Ntheta*dof) F.ReInit(Ntheta*dof);
      for (Integer k = 0; k < dof; k++) {
        for (Long j = 0; j < Ntheta; j++) {
          Complex<ValueType> F_(0,0);
          for (Long i = 0; i < N; i++) {
            Complex<ValueType> c(F_fourier_coeff[(k*N+i)*2+0],F_fourier_coeff[(k*N+i)*2+1]);
            Complex<ValueType> exp_t(cos<ValueType>(theta[j]*i), sin<ValueType>(theta[j]*i));
            F_ += exp_t * c * (i==0?1:2);
          }
          F[j*dof+k] = F_.real/sqrt<ValueType>(Nt_);
        }
      }
    };

    constexpr Integer QuadOrder = 18;
    std::function<Vector<ValueType>(ValueType,ValueType,ValueType)>  compute_potential = [&](ValueType a, ValueType b, ValueType tol) -> Vector<ValueType> {
      auto GetGeomCircle = [&R0] (Vector<ValueType>& Xsrc, Vector<ValueType>& Nsrc, const Vector<ValueType>& nds) {
        Long N = nds.Dim();
        if (Xsrc.Dim() != N * COORD_DIM) Xsrc.ReInit(N*COORD_DIM);
        if (Nsrc.Dim() != N * COORD_DIM) Nsrc.ReInit(N*COORD_DIM);
        for (Long i = 0; i < N; i++) {
          Xsrc[i*COORD_DIM+0] = R0 * cos<ValueType>(nds[i]);
          Xsrc[i*COORD_DIM+1] = R0 * sin<ValueType>(nds[i]);
          Xsrc[i*COORD_DIM+2] = R0 * 0;
          Nsrc[i*COORD_DIM+0] = cos<ValueType>(nds[i]);
          Nsrc[i*COORD_DIM+1] = sin<ValueType>(nds[i]);
          Nsrc[i*COORD_DIM+2] = 0;
        }
      };

      Vector<ValueType> nds0, nds1, wts0, wts1;
      ChebQuadRule<ValueType>::ComputeNdsWts(&nds0, &wts0, QuadOrder+1);
      ChebQuadRule<ValueType>::ComputeNdsWts(&nds1, &wts1, QuadOrder+0);

      Vector<ValueType> U0;
      Vector<ValueType> Xsrc, Nsrc, Fsrc;
      GetGeomCircle(Xsrc, Nsrc, a+(b-a)*nds0);
      EvalFourierExp(Fsrc, F_fourier_coeff, KDIM0, a+(b-a)*nds0);
      for (Long i = 0; i < nds0.Dim(); i++) {
        for (Long j = 0; j < KDIM0; j++) {
          Fsrc[i*KDIM0+j] *= ((b-a) * wts0[i]);
        }
      }
      ker.Eval(U0, Xtrg, Xsrc, Nsrc, Fsrc);

      Vector<ValueType> U1;
      GetGeomCircle(Xsrc, Nsrc, a+(b-a)*nds1);
      EvalFourierExp(Fsrc, F_fourier_coeff, KDIM0, a+(b-a)*nds1);
      for (Long i = 0; i < nds1.Dim(); i++) {
        for (Long j = 0; j < KDIM0; j++) {
          Fsrc[i*KDIM0+j] *= ((b-a) * wts1[i]);
        }
      }
      ker.Eval(U1, Xtrg, Xsrc, Nsrc, Fsrc);

      ValueType err = 0, max_val = 0;
      for (Long i = 0; i < U1.Dim(); i++) {
        err = std::max<ValueType>(err, fabs(U0[i]-U1[i]));
        max_val = std::max<ValueType>(max_val, fabs(U0[i]));
      }
      if (err < tol || (b-a)<tol) {
      //if ((a != 0 && b != 2*const_pi<ValueType>()) || (b-a)<tol) {
        std::cout<<a<<' '<<b-a<<' '<<err<<' '<<tol<<'\n';
        return U1;
      } else {
        U0 = compute_potential(a, (a+b)*0.5, tol);
        U1 = compute_potential((a+b)*0.5, b, tol);
        return U0 + U1;
      }
    };
    U = compute_potential(0, 2*const_pi<ValueType>(), tol);
  };

  template <class Real, Integer Nm, Integer Nr, Integer Nt> template <Integer Nnds, class Kernel> void ToroidalGreensFn<Real,Nm,Nr,Nt>::BuildOperatorModalDirect(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const {
    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim();
    constexpr Integer Nmm = (Nm/2+1)*2;

    auto get_sin_theta = [](Long N){
      Vector<Real> sin_theta(N);
      for (Long i = 0; i < N; i++) {
        sin_theta[i] = sin<Real>(2*const_pi<Real>()*i/N);
      }
      return sin_theta;
    };
    auto get_cos_theta = [](Long N){
      Vector<Real> cos_theta(N);
      for (Long i = 0; i < N; i++) {
        cos_theta[i] = cos<Real>(2*const_pi<Real>()*i/N);
      }
      return cos_theta;
    };
    auto get_circle_coord = [](Long N, Real R0){
      Vector<Real> X(N*COORD_DIM);
      for (Long i = 0; i < N; i++) {
        X[i*COORD_DIM+0] = R0*cos<Real>(2*const_pi<Real>()*i/N);
        X[i*COORD_DIM+1] = R0*sin<Real>(2*const_pi<Real>()*i/N);
        X[i*COORD_DIM+2] = 0;
      }
      return X;
    };
    constexpr Real scal = 2/sqrt<Real>(Nm);

    static const Vector<Real> sin_nds = get_sin_theta(Nnds);
    static const Vector<Real> cos_nds = get_cos_theta(Nnds);
    static const Vector<Real> Xn = get_circle_coord(Nnds,1);

    StaticArray<Real,Nnds*COORD_DIM> buff0;
    Vector<Real> Xs(Nnds*COORD_DIM,buff0,false);
    Xs = Xn * R0_;

    StaticArray<Real,COORD_DIM> Xt = {x0,x1,x2};
    StaticArray<Real,KDIM0*KDIM1*Nnds> mem_buff2;
    Matrix<Real> Mker(KDIM0*Nnds, KDIM1, mem_buff2, false);
    ker.KernelMatrix(Mker, Vector<Real>(COORD_DIM,(Iterator<Real>)Xt,false), Xs, Xn);

    StaticArray<Real,4*Nnds> mem_buff3;
    Vector<Complex<Real>> exp_itheta(Nnds, (Iterator<Complex<Real>>)(mem_buff3+0*Nnds), false);
    Vector<Complex<Real>> exp_iktheta_da(Nnds, (Iterator<Complex<Real>>)(mem_buff3+2*Nnds), false);
    for (Integer j = 0; j < Nnds; j++) {
      exp_itheta[j].real = cos_nds[j];
      exp_itheta[j].imag =-sin_nds[j];
      exp_iktheta_da[j].real = 2*const_pi<Real>()/Nnds*scal;
      exp_iktheta_da[j].imag = 0;
    }
    for (Integer k = 0; k < Nmm/2; k++) { // apply Mker to complex exponentials
      // TODO: FFT might be faster since points are uniform
      Tensor<Real,true,KDIM0,KDIM1> Mk0, Mk1;
      for (Integer i0 = 0; i0 < KDIM0; i0++) {
        for (Integer i1 = 0; i1 < KDIM1; i1++) {
          Mk0(i0,i1) = 0;
          Mk1(i0,i1) = 0;
        }
      }
      for (Integer j = 0; j < Nnds; j++) {
        Tensor<Real,false,KDIM0,KDIM1> Mker_(Mker[j*KDIM0]);
        Mk0 = Mk0 + Mker_ * exp_iktheta_da[j].real;
        Mk1 = Mk1 + Mker_ * exp_iktheta_da[j].imag;
      }
      for (Integer i0 = 0; i0 < KDIM0; i0++) {
        for (Integer i1 = 0; i1 < KDIM1; i1++) {
          M[i0*Nmm+(k*2+0)][i1] = Mk0(i0,i1);
          M[i0*Nmm+(k*2+1)][i1] = Mk1(i0,i1);
        }
      }
      exp_iktheta_da *= exp_itheta;
    }
    for (Integer i0 = 0; i0 < KDIM0; i0++) {
      for (Integer i1 = 0; i1 < KDIM1; i1++) {
        M[i0*Nmm+0][i1] *= 0.5;
        M[i0*Nmm+1][i1] *= 0.5;
        if (Nm%2 == 0) {
          M[(i0+1)*Nmm-2][i1] *= 0.5;
          M[(i0+1)*Nmm-1][i1] *= 0.5;
        }
      }
    }
  }






  template <class ValueType> static void ReadFile(Vector<Vector<ValueType>>& data, const std::string fname) {
    FILE* f = fopen(fname.c_str(), "r");
    if (f == nullptr) {
      std::cout << "Unable to open file for reading:" << fname << '\n';
    } else {
      uint64_t data_len;
      Long readlen = fread(&data_len, sizeof(uint64_t), 1, f);
      SCTL_ASSERT(readlen == 1);
      if (data_len) {
        data.ReInit(data_len);
        for (Long i = 0; i < data.Dim(); i++) {
          readlen = fread(&data_len, sizeof(uint64_t), 1, f);
          SCTL_ASSERT(readlen == 1);
          data[i].ReInit(data_len);
          if (data_len) {
            readlen = fread(&data[i][0], sizeof(ValueType), data_len, f);
            SCTL_ASSERT(readlen == (Long)data_len);
          }
        }
      }
      fclose(f);
    }
  }
  template <class ValueType> static void WriteFile(const Vector<Vector<ValueType>>& data, const std::string fname) {
    FILE* f = fopen(fname.c_str(), "wb+");
    if (f == nullptr) {
      std::cout << "Unable to open file for writing:" << fname << '\n';
      exit(0);
    }
    uint64_t data_len = data.Dim();
    fwrite(&data_len, sizeof(uint64_t), 1, f);

    for (Integer i = 0; i < data.Dim(); i++) {
      data_len = data[i].Dim();
      fwrite(&data_len, sizeof(uint64_t), 1, f);
      if (data_len) fwrite(&data[i][0], sizeof(ValueType), data_len, f);
    }
    fclose(f);
  }

  template <class ValueType> static ValueType dot_prod(const Tensor<ValueType,true,3,1>& u, const Tensor<ValueType,true,3,1>& v) {
    ValueType u_dot_v = 0;
    u_dot_v += u(0,0) * v(0,0);
    u_dot_v += u(1,0) * v(1,0);
    u_dot_v += u(2,0) * v(2,0);
    return u_dot_v;
  }
  template <class ValueType> static Tensor<ValueType,true,3,1> cross_prod(const Tensor<ValueType,true,3,1>& u, const Tensor<ValueType,true,3,1>& v) {
    Tensor<ValueType,true,3,1> uxv;
    uxv(0,0) = u(1,0) * v(2,0) - u(2,0) * v(1,0);
    uxv(1,0) = u(2,0) * v(0,0) - u(0,0) * v(2,0);
    uxv(2,0) = u(0,0) * v(1,0) - u(1,0) * v(0,0);
    return uxv;
  }

  template <class Real> static const Vector<Real>& sin_theta(const Integer ORDER) {
    constexpr Integer MaxOrder = 256;
    auto compute_sin_theta = [MaxOrder](){
      Vector<Vector<Real>> sin_theta_lst(MaxOrder);
      for (Long k = 0; k < MaxOrder; k++) {
        sin_theta_lst[k].ReInit(k);
        for (Long i = 0; i < k; i++) {
          sin_theta_lst[k][i] = sin<Real>(2*const_pi<Real>()*i/k);
        }
      }
      return sin_theta_lst;
    };
    static const auto sin_theta_lst = compute_sin_theta();

    SCTL_ASSERT(ORDER < MaxOrder);
    return sin_theta_lst[ORDER];
  }
  template <class Real> static const Vector<Real>& cos_theta(const Integer ORDER) {
    constexpr Integer MaxOrder = 256;
    auto compute_cos_theta = [MaxOrder](){
      Vector<Vector<Real>> cos_theta_lst(MaxOrder);
      for (Long k = 0; k < MaxOrder; k++) {
        cos_theta_lst[k].ReInit(k);
        for (Long i = 0; i < k; i++) {
          cos_theta_lst[k][i] = cos<Real>(2*const_pi<Real>()*i/k);
        }
      }
      return cos_theta_lst;
    };
    static const auto cos_theta_lst = compute_cos_theta();

    SCTL_ASSERT(ORDER < MaxOrder);
    return cos_theta_lst[ORDER];
  }
  template <class Real> static const Matrix<Real>& fourier_matrix(Integer Nmodes, Integer Nnodes) {
    constexpr Integer MaxOrder = 128;
    auto compute_fourier_matrix = [](Integer Nmodes, Integer Nnodes) {
      if (Nnodes == 0 || Nmodes == 0) return Matrix<Real>();
      Matrix<Real> M_fourier(2*Nmodes,Nnodes);
      for (Long i = 0; i < Nnodes; i++) {
        Real theta = 2*const_pi<Real>()*i/Nnodes;
        for (Long k = 0; k < Nmodes; k++) {
          M_fourier[k*2+0][i] = cos<Real>(k*theta);
          M_fourier[k*2+1][i] = sin<Real>(k*theta);
        }
      }
      return M_fourier;
    };
    auto compute_all = [&compute_fourier_matrix, MaxOrder]() {
      Matrix<Matrix<Real>> Mall(MaxOrder, MaxOrder);
      for (Long i = 0; i < MaxOrder; i++) {
        for (Long j = 0; j < MaxOrder; j++) {
          Mall[i][j] = compute_fourier_matrix(i,j);
        }
      }
      return Mall;
    };
    static const Matrix<Matrix<Real>> Mall = compute_all();

    SCTL_ASSERT(Nmodes < MaxOrder && Nnodes < MaxOrder);
    return Mall[Nmodes][Nnodes];
  }
  template <class Real> static const Matrix<Real>& fourier_matrix_inv(Integer Nnodes, Integer Nmodes) {
    constexpr Integer MaxOrder = 128;
    auto compute_fourier_matrix_inv = [](Integer Nnodes, Integer Nmodes) {
      if (Nmodes > Nnodes/2+1 || Nnodes == 0 || Nmodes == 0) return Matrix<Real>();
      const Real scal = 2/(Real)Nnodes;

      Matrix<Real> M_fourier_inv(Nnodes,2*Nmodes);
      for (Long i = 0; i < Nnodes; i++) {
        Real theta = 2*const_pi<Real>()*i/Nnodes;
        for (Long k = 0; k < Nmodes; k++) {
          M_fourier_inv[i][k*2+0] = cos<Real>(k*theta)*scal;
          M_fourier_inv[i][k*2+1] = sin<Real>(k*theta)*scal;
        }
      }
      for (Long i = 0; i < Nnodes; i++) {
        M_fourier_inv[i][0] *= 0.5;
      }
      if (Nnodes == (Nmodes-1)*2) {
        for (Long i = 0; i < Nnodes; i++) {
          M_fourier_inv[i][Nnodes] *= 0.5;
        }
      }
      return M_fourier_inv;
    };
    auto compute_all = [&compute_fourier_matrix_inv, MaxOrder]() {
      Matrix<Matrix<Real>> Mall(MaxOrder, MaxOrder);
      for (Long i = 0; i < MaxOrder; i++) {
        for (Long j = 0; j < MaxOrder; j++) {
          Mall[i][j] = compute_fourier_matrix_inv(i,j);
        }
      }
      return Mall;
    };
    static const Matrix<Matrix<Real>> Mall = compute_all();

    SCTL_ASSERT(Nnodes < MaxOrder && Nmodes < MaxOrder);
    return Mall[Nnodes][Nmodes];
  }
  template <class Real> static const Matrix<Real>& fourier_matrix_inv_transpose(Integer Nnodes, Integer Nmodes) {
    constexpr Integer MaxOrder = 128;
    auto compute_all = [MaxOrder]() {
      Matrix<Matrix<Real>> Mall(MaxOrder, MaxOrder);
      for (Long i = 0; i < MaxOrder; i++) {
        for (Long j = 0; j < MaxOrder; j++) {
          Mall[i][j] = fourier_matrix_inv<Real>(i,j).Transpose();
        }
      }
      return Mall;
    };
    static const Matrix<Matrix<Real>> Mall = compute_all();

    SCTL_ASSERT(Nnodes < MaxOrder && Nmodes < MaxOrder);
    return Mall[Nnodes][Nmodes];
  }

  template <class ValueType> static const std::pair<Vector<ValueType>,Vector<ValueType>>& LegendreQuadRule(Integer ORDER) {
    constexpr Integer max_order = 50;
    auto compute_nds_wts = [max_order]() {
      Vector<std::pair<Vector<ValueType>,Vector<ValueType>>> nds_wts(max_order);
      for (Integer order = 1; order < max_order; order++) {
        auto& x_ = nds_wts[order].first;
        auto& w_ = nds_wts[order].second;
        LegQuadRule<ValueType>::ComputeNdsWts(&x_, &w_, order);
      }
      return nds_wts;
    };
    static const auto nds_wts = compute_nds_wts();

    SCTL_ASSERT(ORDER < max_order);
    return nds_wts[ORDER];
  }
  template <class ValueType> static const std::pair<Vector<ValueType>,Vector<ValueType>>& LogSingularityQuadRule(Integer ORDER) {
    constexpr Integer MaxOrder = 50;
    auto compute_nds_wts_lst = [MaxOrder]() {
      #ifdef SCTL_QUAD_T
      using RealType = QuadReal;
      #else
      using RealType = long double;
      #endif
      Vector<Vector<RealType>> data;
      ReadFile<RealType>(data, std::string(SCTL_QUOTEME(SCTL_DATA_PATH)) + "/log_quad");
      if (data.Dim() < MaxOrder*2) {
        data.ReInit(MaxOrder*2);
        #pragma omp parallel for
        for (Integer order = 1; order < MaxOrder; order++) {
          auto integrands = [order](const Vector<RealType>& nds) {
            const Integer K = order;
            const Long N = nds.Dim();
            Matrix<RealType> M(N,K);
            for (Long j = 0; j < N; j++) {
              for (Long i = 0; i < (K+1)/2; i++) {
                M[j][i] = pow<RealType,Long>(nds[j],i);
              }
              for (Long i = (K+1)/2; i < K; i++) {
                M[j][i] = pow<RealType,Long>(nds[j],i-(K+1)/2) * log<RealType>(nds[j]);
              }
            }
            return M;
          };
          InterpQuadRule<RealType>::Build(data[order*2+0], data[order*2+1], integrands, 0, 1, machine_eps<RealType>(), order, 2e-4, 1.0, false);
        }
        WriteFile<RealType>(data, std::string(SCTL_QUOTEME(SCTL_DATA_PATH)) + "/log_quad");
      }

      Vector<std::pair<Vector<ValueType>,Vector<ValueType>>> nds_wts_lst(MaxOrder);
      #pragma omp parallel for
      for (Integer order = 1; order < MaxOrder; order++) {
        const auto& nds = data[order*2+0];
        const auto& wts = data[order*2+1];
        auto& nds_ = nds_wts_lst[order].first;
        auto& wts_ = nds_wts_lst[order].second;
        nds_.ReInit(nds.Dim());
        wts_.ReInit(wts.Dim());
        for (Long i = 0; i < nds.Dim(); i++) {
          nds_[i] = (ValueType)nds[i];
          wts_[i] = (ValueType)wts[i];
        }
      }
      return nds_wts_lst;
    };
    static const auto nds_wts_lst = compute_nds_wts_lst();

    SCTL_ASSERT(ORDER < MaxOrder);
    return nds_wts_lst[ORDER];
  }

  template <class RealType, class Kernel, Integer adap> static Vector<Vector<RealType>> BuildToroidalSpecialQuadRules(Integer Nmodes, Integer VecLen) {
    const std::string fname = std::string(SCTL_QUOTEME(SCTL_DATA_PATH)) + std::string("/toroidal_quad_rule_m") + std::to_string(Nmodes) + "_" + Kernel::Name();
    constexpr Integer COORD_DIM = 3;
    constexpr Integer max_adap_depth = 30; // build quadrature rules for points up to 2*pi*0.5^max_adap_depth from source loop
    constexpr Integer crossover_adap_depth = 2;
    constexpr Integer max_digits = 20;

    #ifdef SCTL_QUAD_T
    using ValueType = QuadReal;
    #else
    using ValueType = long double;
    #endif

    auto DyadicPanelQuadRule = [](Vector<ValueType>& nds, Vector<ValueType>& wts, const Integer depth, const Integer LegOrder, const Integer PanelRepeat) {
      Vector<ValueType> panel_nds, panel_wts;
      { // Set panel_nds, panel_wts
        auto leg_quad = LegendreQuadRule<ValueType>(LegOrder);
        const auto& leg_nds = leg_quad.first;
        const auto& leg_wts = leg_quad.second;

        const Long rep = PanelRepeat;
        const ValueType scal = 1/(ValueType)rep;
        for (Long i = 0; i < rep; i++) {
          for (Long j = 0; j < leg_nds.Dim(); j++) {
            panel_nds.PushBack(leg_nds[j]*scal + i*scal);
            panel_wts.PushBack(leg_wts[j]*scal);
          }
        }
      }

      SCTL_ASSERT(depth);
      Long N = 2*depth;
      ValueType l = 0.5;
      nds.ReInit(N*panel_nds.Dim());
      wts.ReInit(N*panel_nds.Dim());
      for (Integer idx = 0; idx < depth; idx++) {
        l *= (idx<depth-1 ? 0.5 : 1.0);
        Vector<ValueType> nds0(panel_nds.Dim(), nds.begin()+(  idx  )*panel_nds.Dim(), false);
        Vector<ValueType> nds1(panel_nds.Dim(), nds.begin()+(N-idx-1)*panel_nds.Dim(), false);
        Vector<ValueType> wts0(panel_wts.Dim(), wts.begin()+(  idx  )*panel_wts.Dim(), false);
        Vector<ValueType> wts1(panel_wts.Dim(), wts.begin()+(N-idx-1)*panel_wts.Dim(), false);
        for (Long i = 0; i < panel_nds.Dim(); i++) {
          ValueType s = panel_nds[i]*l + (idx<depth-1 ? l : 0);
          nds0[panel_nds.Dim()-1-i] =-s;
          nds1[                  i] = s;
          wts0[panel_nds.Dim()-1-i] = panel_wts[i]*l;
          wts1[                  i] = panel_wts[i]*l;
        }
      }
    };

    Vector<Vector<ValueType>> data;
    if (!adap) { // read from file
      ReadFile(data, fname);
    } else { // use dyadically refined panel quadrature rules
      data.ReInit(max_adap_depth * max_digits);
      for (Integer idx = 0; idx < max_adap_depth; idx++) {
        const ValueType dist = 4*const_pi<ValueType>()*pow<ValueType,Long>(0.5,idx);
        const Integer DyadicRefDepth = std::max<Integer>(1,(Integer)(log(dist/2/const_pi<ValueType>())/log(0.5)+0.5));

        Vector<ValueType> quad_nds, quad_wts;
        for (Integer digits = 0; digits < max_digits; digits++) {
          const Integer LegOrder = (Integer)(digits*1.5);
          DyadicPanelQuadRule(quad_nds, quad_wts, DyadicRefDepth, LegOrder, adap);

          const Long N = quad_nds.Dim();
          data[idx*max_digits+digits].ReInit(3*N);
          for (Long i = 0; i < N; i++) {
            data[idx*max_digits+digits][i*3+0] = cos<ValueType>(2*const_pi<ValueType>()*quad_nds[i]);
            data[idx*max_digits+digits][i*3+1] = sin<ValueType>(2*const_pi<ValueType>()*quad_nds[i]);
            data[idx*max_digits+digits][i*3+2] = (2*const_pi<ValueType>()*quad_wts[i]);
          }
        }
      }
    }
    if (!adap && data.Dim() != max_adap_depth*max_digits) { // If file is not-found then compute quadrature rule and write to file
      data.ReInit(max_adap_depth * max_digits);
      for (Integer idx = 0; idx < max_adap_depth; idx++) {
        Vector<Vector<ValueType>> quad_nds,  quad_wts;
        { // generate special quadrature rule
          Vector<ValueType> nds, wts;
          Matrix<ValueType> Mintegrands;
          auto discretize_basis_functions = [Nmodes,&DyadicPanelQuadRule](Matrix<ValueType>& Mintegrands, Vector<ValueType>& nds, Vector<ValueType>& wts, const ValueType dist, const Integer LegOrder) {
            auto trg_coord = [](ValueType dist, Long M) {
              Vector<ValueType> Xtrg; //(M*M*COORD_DIM);
              for (Long i = 0; i < M; i++) {
                for (Long j = 0; j < M; j++) {
                  ValueType theta = i*2*const_pi<ValueType>()/(M);
                  ValueType r = (0.5 + i*0.5/(M)) * dist;
                  ValueType x0 = r*cos<ValueType>(theta);
                  ValueType x1 = 0;
                  ValueType x2 = r*sin<ValueType>(theta);
                  if (x0 > -1) {
                    Xtrg.PushBack(x0);
                    Xtrg.PushBack(x1);
                    Xtrg.PushBack(x2);
                  }
                }
              }
              return Xtrg;
            };
            const Vector<ValueType> Xtrg = trg_coord(dist, 25); // TODO: determine optimal sample count
            const Long Ntrg = Xtrg.Dim()/COORD_DIM;

            const Integer DyadicRefDepth = std::max<Integer>(1,(Integer)(log(dist/2/const_pi<ValueType>())/log(0.5)+0.5));
            DyadicPanelQuadRule(nds, wts, DyadicRefDepth, LegOrder, 1);

            const Long Nnds = nds.Dim();
            Vector<Complex<ValueType>> exp_itheta(Nnds), exp_iktheta(Nnds);
            Vector<ValueType> Xsrc(Nnds*COORD_DIM), Xn(Nnds*COORD_DIM);
            for (Long i = 0; i < Nnds; i++) {
              const ValueType cos_t = cos<ValueType>(2*const_pi<ValueType>()*nds[i]);
              const ValueType sin_t = sin<ValueType>(2*const_pi<ValueType>()*nds[i]);
              exp_iktheta[i].real = 1;
              exp_iktheta[i].imag = 0;
              exp_itheta[i].real = cos_t;
              exp_itheta[i].imag = sin_t;
              Xsrc[i*COORD_DIM+0] = -2*sin<ValueType>(const_pi<ValueType>()*nds[i])*sin<ValueType>(const_pi<ValueType>()*nds[i]); // == cos_t - 1
              Xsrc[i*COORD_DIM+1] = sin_t;
              Xsrc[i*COORD_DIM+2] = 0;
              Xn[i*COORD_DIM+0] = cos_t;
              Xn[i*COORD_DIM+1] = sin_t;
              Xn[i*COORD_DIM+2] = 0;
            }

            Kernel ker;
            Matrix<ValueType> Mker;
            ker.template KernelMatrix<ValueType,true>(Mker, Xtrg, Xsrc, Xn);
            SCTL_ASSERT(Mker.Dim(0) == Nnds * Kernel::SrcDim());
            SCTL_ASSERT(Mker.Dim(1) == Ntrg * Kernel::TrgDim());

            Mintegrands.ReInit(Nnds, (Nmodes*2)*Kernel::SrcDim() * Ntrg*Kernel::TrgDim());
            for (Long k = 0; k < Nmodes; k++) {
              for (Long i = 0; i < Nnds; i++) {
                for (Long j = 0; j < Ntrg; j++) {
                  for (Long k0 = 0; k0 < Kernel::SrcDim(); k0++) {
                    for (Long k1 = 0; k1 < Kernel::TrgDim(); k1++) {
                      Mintegrands[i][(((k*2+0)*Kernel::SrcDim()+k0) *Ntrg+j)*Kernel::TrgDim()+k1] = Mker[i*Kernel::SrcDim()+k0][j*Kernel::TrgDim()+k1] * exp_iktheta[i].real;
                      Mintegrands[i][(((k*2+1)*Kernel::SrcDim()+k0) *Ntrg+j)*Kernel::TrgDim()+k1] = Mker[i*Kernel::SrcDim()+k0][j*Kernel::TrgDim()+k1] * exp_iktheta[i].imag;
                    }
                  }
                }
              }
              for (Long i = 0; i < Nnds; i++) {
                exp_iktheta[i] *= exp_itheta[i];
              }
            }
          };
          const ValueType dist = 4*const_pi<ValueType>()*pow<ValueType,Long>(0.5,idx); // distance of target points from the source loop (which is a unit circle)
          discretize_basis_functions(Mintegrands, nds, wts, dist, 35); // TODO: adaptively select Legendre order

          Vector<ValueType> eps_vec;
          for (Long k = 0; k < max_digits; k++) eps_vec.PushBack(pow<ValueType,Long>(0.1,k));
          std::cout<<"Level = "<<idx<<" of "<<max_adap_depth<<'\n';
          if (1) { // make symmetric quadrature rules
            const Long Nnds = Mintegrands.Dim(0), Nint = Mintegrands.Dim(1);
            #pragma omp parallel for schedule(static)
            for (Long i = 0; i < Nnds/2; i++) { // make integrands symmetric
              for (Long j = 0; j < Nint; j++) {
                Mintegrands[i][j] += Mintegrands[Nnds-1-i][j];
              }
            }
            const Matrix<ValueType> Mintegrands_(Nnds/2, Nint, Mintegrands.begin(), false);
            const Vector<ValueType> nds_(Nnds/2, nds.begin(), false);
            const Vector<ValueType> wts_(Nnds/2, wts.begin(), false);
            Vector<Vector<ValueType>> quad_nds_, quad_wts_;
            auto cond_num_vec = InterpQuadRule<ValueType>::Build(quad_nds_, quad_wts_, Mintegrands_, nds_, wts_, eps_vec);
            quad_nds.ReInit(quad_nds_.Dim());
            quad_wts.ReInit(quad_wts_.Dim());
            for (Long i = 0; i < quad_nds.Dim(); i++) {
              const Long N = quad_nds_[i].Dim();
              quad_nds[i].ReInit(2*N);
              quad_wts[i].ReInit(2*N);
              for (Long j = 0; j < N; j++) {
                quad_nds[i][      j] = quad_nds_[i][j];
                quad_nds[i][2*N-1-j] =-quad_nds_[i][j];
                quad_wts[i][      j] = quad_wts_[i][j];
                quad_wts[i][2*N-1-j] = quad_wts_[i][j];
              }
            }
          } else {
            auto cond_num_vec = InterpQuadRule<ValueType>::Build(quad_nds, quad_wts, Mintegrands, nds, wts, eps_vec);
          }
        }
        for (Integer digits = 0; digits < max_digits; digits++) {
          Long N = quad_nds[digits].Dim();
          data[idx*max_digits+digits].ReInit(3*N);
          for (Long i = 0; i < N; i++) {
            data[idx*max_digits+digits][i*3+0] = cos<ValueType>(2*const_pi<ValueType>()*quad_nds[digits][i]);
            data[idx*max_digits+digits][i*3+1] = sin<ValueType>(2*const_pi<ValueType>()*quad_nds[digits][i]);
            data[idx*max_digits+digits][i*3+2] = (2*const_pi<ValueType>()*quad_wts[digits][i]);
          }
        }
      }
      WriteFile(data, fname);
    }
    for (Integer idx = 0; idx < crossover_adap_depth; idx++) { // Use trapezoidal rule up to crossover_adap_depth
      for (Integer digits = 0; digits < max_digits; digits++) {
        Long N = std::max<Long>(digits*pow<Long,Long>(2,idx), Nmodes); // TODO: determine optimal order by testing error or adaptively
        data[idx*max_digits+digits].ReInit(3*N);
        for (Long i = 0; i < N; i++) {
          ValueType quad_nds = i/(ValueType)N;
          ValueType quad_wts = 1/(ValueType)N;
          data[idx*max_digits+digits][i*3+0] = cos<ValueType>(2*const_pi<ValueType>()*quad_nds);
          data[idx*max_digits+digits][i*3+1] = sin<ValueType>(2*const_pi<ValueType>()*quad_nds);
          data[idx*max_digits+digits][i*3+2] = (2*const_pi<ValueType>()*quad_wts);
        }
      }
    }

    Vector<Vector<RealType>> quad_rule_lst;
    quad_rule_lst.ReInit(data.Dim()*4);
    for (Integer i = 0; i < data.Dim(); i++) {
      const Long Nnds_ = data[i].Dim()/3;
      const Integer Nnds = ((Nnds_+VecLen-1)/VecLen)*VecLen;
      quad_rule_lst[i*4+0].ReInit(Nnds); quad_rule_lst[i*4+0].SetZero();
      quad_rule_lst[i*4+1].ReInit(Nnds); quad_rule_lst[i*4+1].SetZero();
      quad_rule_lst[i*4+2].ReInit(Nnds); quad_rule_lst[i*4+2].SetZero();
      quad_rule_lst[i*4+3].ReInit(Nmodes*2*Nnds); quad_rule_lst[i*4+3].SetZero();
      for (Long j = 0; j < Nnds_; j++) {
        Complex<ValueType> exp_itheta(data[i][j*3+0], data[i][j*3+1]);
        quad_rule_lst[i*4+0][j] = (RealType)(exp_itheta.real-1);
        quad_rule_lst[i*4+1][j] = (RealType)(exp_itheta.imag);
        quad_rule_lst[i*4+2][j] = (RealType)data[i][j*3+2];

        Complex<ValueType> exp_iktheta(1,0);
        for (Long k = 0; k < Nmodes; k++) {
          quad_rule_lst[i*4+3][(k*2+0)*Nnds+j] = (RealType)exp_iktheta.real;
          quad_rule_lst[i*4+3][(k*2+1)*Nnds+j] = (RealType)exp_iktheta.imag;
          exp_iktheta *= exp_itheta;
        }
      }
    }
    return quad_rule_lst;
  }
  template <class RealType, Integer VecLen, Integer ModalUpsample, class Kernel, Integer adap> static bool ToroidalSpecialQuadRule(Matrix<RealType>& Mfourier, Vector<RealType>& nds_cos_theta, Vector<RealType>& nds_sin_theta, Vector<RealType>& wts, const Integer Nmodes, RealType r_R0, Integer digits) {
    static constexpr Integer max_adap_depth = 30; // build quadrature rules for points up to 2*pi*0.5^max_adap_depth from source loop
    static constexpr Integer crossover_adap_depth = 2;
    static constexpr Integer max_digits = 20;
    if (digits >= max_digits) digits = max_digits-1;
    //SCTL_ASSERT(digits<max_digits);

    Long adap_depth = 0;
    for (RealType s = r_R0; s<2*const_pi<RealType>(); s*=2) adap_depth++;
    if (adap_depth >= max_adap_depth) {
      SCTL_WARN("Toroidal quadrature evaluation is outside of the range of precomputed quadratures; accuracy may be severely degraded.");
      adap_depth = max_adap_depth-1;
    }

    SCTL_ASSERT(Nmodes < 100);
    static Vector<Vector<Matrix<RealType>>> all_fourier_basis(100);
    static Vector<Vector<Vector<RealType>>> all_quad_nds_cos_theta(100);
    static Vector<Vector<Vector<RealType>>> all_quad_nds_sin_theta(100);
    static Vector<Vector<Vector<RealType>>> all_quad_wts(100);
    #pragma omp critical(SCTL_ToroidalSpecialQuadRule)
    if (all_quad_wts[Nmodes].Dim() == 0) {
      auto quad_rules = BuildToroidalSpecialQuadRules<RealType,Kernel,adap>(Nmodes, VecLen);
      const Long Nrules = quad_rules.Dim()/4;

      Vector<Matrix<RealType>> fourier_basis(Nrules);
      Vector<Vector<RealType>> quad_nds_cos_theta(Nrules);
      Vector<Vector<RealType>> quad_nds_sin_theta(Nrules);
      Vector<Vector<RealType>> quad_wts(Nrules);
      for (Long i = 0; i < Nrules; i++) { // Set quad_nds_cos_theta, quad_nds_sin_theta, quad_wts, fourier_basis
        const Integer Nnds = quad_rules[i*4+0].Dim();
        SCTL_ASSERT(Nnds%VecLen == 0);
        quad_wts[i] = quad_rules[i*4+2];
        quad_nds_cos_theta[i] = quad_rules[i*4+0];
        quad_nds_sin_theta[i] = quad_rules[i*4+1];
        fourier_basis[i] = Matrix<RealType>((Nmodes-ModalUpsample)*2, Nnds, quad_rules[i*4+3].begin()).Transpose();
      }
      all_fourier_basis[Nmodes].Swap(fourier_basis);
      all_quad_nds_cos_theta[Nmodes].Swap(quad_nds_cos_theta);
      all_quad_nds_sin_theta[Nmodes].Swap(quad_nds_sin_theta);
      all_quad_wts[Nmodes].Swap(quad_wts);
    }

    { // Set Mfourier, nds_cos_theta, nds_sin_theta, wts
      const Long quad_idx = adap_depth*max_digits+digits;
      const auto& Mfourier0 = all_fourier_basis[Nmodes][quad_idx];
      const auto& nds0_cos_theta = all_quad_nds_cos_theta[Nmodes][quad_idx];
      const auto& nds0_sin_theta = all_quad_nds_sin_theta[Nmodes][quad_idx];
      const auto& wts0 = all_quad_wts[Nmodes][quad_idx];
      const Long N = wts0.Dim();

      Mfourier.ReInit(Mfourier0.Dim(0), Mfourier0.Dim(1), (Iterator<RealType>)Mfourier0.begin(), false);
      nds_cos_theta.ReInit(N, (Iterator<RealType>)nds0_cos_theta.begin(), false);
      nds_sin_theta.ReInit(N, (Iterator<RealType>)nds0_sin_theta.begin(), false);
      wts.ReInit(N, (Iterator<RealType>)wts0.begin(), false);
    }

    // return whether an adaptive quadrature rule has been used
    return (adap_depth >= crossover_adap_depth);
  }
  template <Integer digits, Integer ModalUpsample, bool trg_dot_prod, class RealType, class Kernel, Integer adap=(sizeof(RealType)>sizeof(double))> static void toroidal_greens_fn_batched(Matrix<RealType>& M, const Tensor<RealType,true,3,1>& x_trg, const Tensor<RealType,true,3,1>& e_trg, const RealType r_trg, const Tensor<RealType,true,3,1>& n_trg, const Matrix<RealType>& x_src, const Matrix<RealType>& dx_src, const Matrix<RealType>& d2x_src, const Matrix<RealType>& r_src, const Matrix<RealType>& dr_src, const Matrix<RealType>& e1_src, const Kernel& ker, const Integer FourierModes) {
    static constexpr Integer VecLen = DefaultVecLen<RealType>();
    using VecType = Vec<RealType, VecLen>;

    constexpr Integer COORD_DIM = 3;
    using Vec3 = Tensor<RealType,true,COORD_DIM,1>;
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1 = Kernel::TrgDim()/(trg_dot_prod?COORD_DIM:1);
    static constexpr Integer Nbuff = 10000; // TODO

    const Long BatchSize = M.Dim(0);
    SCTL_ASSERT(M.Dim(1) == KDIM0*KDIM1*FourierModes*2);
    SCTL_ASSERT(  x_src.Dim(1) == BatchSize &&   x_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT( dx_src.Dim(1) == BatchSize &&  dx_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT(d2x_src.Dim(1) == BatchSize && d2x_src.Dim(0) == COORD_DIM);
    SCTL_ASSERT(  r_src.Dim(1) == BatchSize &&   r_src.Dim(0) ==         1);
    SCTL_ASSERT( dr_src.Dim(1) == BatchSize &&  dr_src.Dim(0) ==         1);
    SCTL_ASSERT( e1_src.Dim(1) == BatchSize &&  e1_src.Dim(0) == COORD_DIM);
    const VecType n_trg_[COORD_DIM] = {n_trg(0,0),n_trg(1,0),n_trg(2,0)};
    const Vec3 y_trg = x_trg + e_trg*r_trg;
    for (Long ii = 0; ii < BatchSize; ii++) {
      RealType r = r_src[0][ii], dr = dr_src[0][ii];
      if (r < 0) r = 0;
      Vec3 x, dx, d2x, e1;
      for (Integer k = 0; k < COORD_DIM; k++) { // Set x, dx, d2x, e1
        x  (k,0) =   x_src[k][ii];
        dx (k,0) =  dx_src[k][ii];
        d2x(k,0) = d2x_src[k][ii];
        e1 (k,0) =  e1_src[k][ii];
      }

      auto toroidal_greens_fn = [&ker,&n_trg_](Matrix<RealType>& M, const Vec3& Xt, const Vec3& x, const Vec3& dx, const Vec3& d2x, const Vec3& e1_, const RealType r, const RealType dr, const Integer FourierModes) {
        SCTL_ASSERT(M.Dim(0) ==    KDIM0*KDIM1);
        SCTL_ASSERT(M.Dim(1) == FourierModes*2);
        const auto Xt_X0 = Xt-x;

        RealType dist;
        Vec3 e1, e2, e3;
        { // Set dist, e1, e2, e3
          e3 = dx*(-1/sqrt<RealType>(dot_prod(dx,dx)));
          e1 = Xt_X0 - e3 * dot_prod(Xt_X0,e3);
          if (dot_prod(e1,e1) == 0) e1 = e1_;
          e1 = e1 * (1/sqrt<RealType>(dot_prod(e1,e1)));
          e2 = cross_prod(e3, e1);
          e2 = e2 * (1/sqrt<RealType>(dot_prod(e2,e2)));

          RealType dist0 = dot_prod(Xt_X0, e1) - r;
          RealType dist1 = dot_prod(Xt_X0, e3);
          dist = sqrt<RealType>(dist0*dist0 + dist1*dist1);
        }
        const auto exp_theta = Complex<RealType>(dot_prod(e1,e1_), -dot_prod(e2,e1_));

        Matrix<RealType> Mexp_iktheta;
        Vector<RealType> nds_cos_theta, nds_sin_theta, wts;
        ToroidalSpecialQuadRule<RealType,VecLen,ModalUpsample,Kernel,adap>(Mexp_iktheta, nds_cos_theta, nds_sin_theta, wts, FourierModes+ModalUpsample, dist/r, digits);
        const Long Nnds = wts.Dim();
        SCTL_ASSERT(Nnds < Nbuff);

        { // Set M
          const RealType d2x_dot_e1 = e1(0,0)*d2x(0,0) + e1(1,0)*d2x(1,0) + e1(2,0)*d2x(2,0);
          const RealType d2x_dot_e2 = e2(0,0)*d2x(0,0) + e2(1,0)*d2x(1,0) + e2(2,0)*d2x(2,0);
          const RealType norm_dx_ = sqrt<RealType>(dot_prod(dx,dx));
          const RealType inv_norm_dx = 1/norm_dx_;
          const VecType norm_dx(norm_dx_);

          const VecType vec_dx[3] = {dx(0,0), dx(1,0), dx(2,0)};
          const VecType vec_dy0[3] = {Xt(0,0)-x(0,0), Xt(1,0)-x(1,0), Xt(2,0)-x(2,0)};

          alignas(sizeof(VecType)) StaticArray<char,KDIM0*KDIM1*Nbuff*sizeof(RealType)> mem_buff;
          Matrix<RealType> Mker_da(KDIM0*KDIM1, Nnds, (Iterator<RealType>)(Iterator<char>)mem_buff, false);
          for (Integer j = 0; j < Nnds; j+=VecLen) { // Set Mker_da
            VecType dy[3], n[3], da;
            { // Set dy, n, da
              VecType cost = VecType::LoadAligned(&nds_cos_theta[j])+(RealType)1;
              VecType sint = VecType::LoadAligned(&nds_sin_theta[j]);

              dy[0] = vec_dy0[0] - cost*(r*e1(0,0)) - sint*(r*e2(0,0));
              dy[1] = vec_dy0[1] - cost*(r*e1(1,0)) - sint*(r*e2(1,0));
              dy[2] = vec_dy0[2] - cost*(r*e1(2,0)) - sint*(r*e2(2,0));

              VecType norm_dy = norm_dx - (cost*d2x_dot_e1 + sint*d2x_dot_e2) * (r*inv_norm_dx);
              n[0] = cost*e1(0,0)*norm_dy + sint*e2(0,0)*norm_dy - vec_dx[0]*(dr*inv_norm_dx);
              n[1] = cost*e1(1,0)*norm_dy + sint*e2(1,0)*norm_dy - vec_dx[1]*(dr*inv_norm_dx);
              n[2] = cost*e1(2,0)*norm_dy + sint*e2(2,0)*norm_dy - vec_dx[2]*(dr*inv_norm_dx);

              VecType da2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];
              VecType inv_da = approx_rsqrt<digits>(da2);
              da = da2 * inv_da * r;

              n[0] = n[0] * inv_da;
              n[1] = n[1] * inv_da;
              n[2] = n[2] * inv_da;

              //da = norm_dx*r - (n[0]*vec_d2x[0]+n[1]*vec_d2x[1]+n[2]*vec_d2x[2])*(r*r*inv_norm_dx); // dr == 0
            }

            VecType Mker[KDIM0][Kernel::TrgDim()];
            ker.template uKerMatrix<digits, VecType>(Mker, dy, n, ker.GetCtxPtr());
            VecType da_wts = VecType::LoadAligned(&wts[j]) * da;
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                if (trg_dot_prod) {
                  VecType Mker_dot_n = FMA(Mker[k0][k1*COORD_DIM+0],n_trg_[0],
                                       FMA(Mker[k0][k1*COORD_DIM+1],n_trg_[1],
                                           Mker[k0][k1*COORD_DIM+2]*n_trg_[2]));
                  (Mker_dot_n*da_wts).StoreAligned(&Mker_da[k0*KDIM1+k1][j]);
                } else {
                  (Mker[k0][k1]*da_wts).StoreAligned(&Mker_da[k0*KDIM1+k1][j]);
                }
              }
            }
          }
          Matrix<RealType>::GEMM(M, Mker_da, Mexp_iktheta);

          Complex<RealType> exp_iktheta(1,0);
          for (Integer j = 0; j < FourierModes; j++) {
            for (Integer k = 0; k < KDIM0*KDIM1; k++) {
              Complex<RealType> Mjk(M[k][j*2+0],M[k][j*2+1]);
              Mjk *= exp_iktheta;
              M[k][j*2+0] = Mjk.real;
              M[k][j*2+1] = Mjk.imag;
            }
            exp_iktheta *= exp_theta;
          }
        }
      };
      Matrix<RealType> M_toroidal_greens_fn(KDIM0*KDIM1, FourierModes*2, M[ii], false);
      toroidal_greens_fn(M_toroidal_greens_fn, y_trg, x, dx, d2x, e1, r, dr, FourierModes);
    }

    return;
    if (adap==0) { // Print toroidal quadrature error
      using ValueType = RealType;
      auto copy_matrix = [](Matrix<ValueType>& M_, const Matrix<RealType>& M) {
        M_.ReInit(M.Dim(0), M.Dim(1));
        for (Long i = 0; i < M.Dim(0)*M.Dim(1); i++) {
          M_[0][i] = (ValueType)M[0][i];
        }
      };
      Matrix<ValueType> M_;
      Tensor<ValueType,true,3,1> x_trg_;
      Tensor<ValueType,true,3,1> e_trg_;
      Tensor<ValueType,true,3,1> n_trg_;
      for (Long i = 0; i < 3; i++) {
        x_trg_(i,0) = (ValueType)x_trg(i,0);
        e_trg_(i,0) = (ValueType)e_trg(i,0);
        n_trg_(i,0) = (ValueType)n_trg(i,0);
      }
      Matrix<ValueType> x_src_  ;
      Matrix<ValueType> dx_src_ ;
      Matrix<ValueType> d2x_src_;
      Matrix<ValueType> r_src_  ;
      Matrix<ValueType> dr_src_ ;
      Matrix<ValueType> e1_src_ ;
      copy_matrix(M_  , M);
      copy_matrix(x_src_  , x_src);
      copy_matrix(dx_src_ , dx_src);
      copy_matrix(d2x_src_, d2x_src);
      copy_matrix(r_src_  , r_src);
      copy_matrix(dr_src_ , dr_src);
      copy_matrix(e1_src_ , e1_src);

      toroidal_greens_fn_batched<(digits<32?digits+1:digits), ModalUpsample, trg_dot_prod, ValueType, Kernel, 2>(M_, x_trg_, e_trg_/sqrt<ValueType>(dot_prod(e_trg_,e_trg_)), (ValueType)r_trg, n_trg_, x_src_, dx_src_, d2x_src_, r_src_, dr_src_, e1_src_, ker, FourierModes);

      static RealType max_rel_err = 0;
      RealType max_err = 0, max_val = 0;
      for (Long i = 0; i < BatchSize*KDIM0*KDIM1; i++) {
        RealType err = fabs(M[0][i*FourierModes*2] - (RealType)M_[0][i*FourierModes*2]);
        RealType val = fabs(M[0][i*FourierModes*2]);
        if (err > max_err) max_err = err;
        if (val > max_val) max_val = val;
      }
      if (max_val>0 && max_err/max_val > max_rel_err) {
        max_rel_err = max_err/max_val;
        std::cout<<max_rel_err<<' '<<max_err<<'\n';
      }

      for (Long i = 0; i < M.Dim(0)*M.Dim(1); i++) {
        M[0][i] = (RealType)M_[0][i];
      }
    }
  }

  template <class ValueType> static void DyadicQuad_s(Vector<ValueType>& nds, Vector<ValueType>& wts, const Integer LegQuadOrder, const Integer LogQuadOrder, const ValueType s, const Integer levels, bool sort) {
    const auto& log_quad_nds = LogSingularityQuadRule<ValueType>(LogQuadOrder).first;
    const auto& log_quad_wts = LogSingularityQuadRule<ValueType>(LogQuadOrder).second;
    const auto& leg_nds = LegendreQuadRule<ValueType>(LegQuadOrder).first;
    const auto& leg_wts = LegendreQuadRule<ValueType>(LegQuadOrder).second;

    ValueType len0 = std::min(pow<ValueType>(0.5,levels), std::min(s, (1-s)));
    ValueType len1 = std::min<ValueType>(s, 1-s);
    ValueType len2 = std::max<ValueType>(s, 1-s);

    for (Long i = 0; i < log_quad_nds.Dim(); i++) {
      nds.PushBack( len0*log_quad_nds[i]);
      nds.PushBack(-len0*log_quad_nds[i]);
      wts.PushBack(len0*log_quad_wts[i]);
      wts.PushBack(len0*log_quad_wts[i]);
    }

    for (ValueType start = len0; start < len1; start*=2) {
      ValueType step_ = std::min(start, len1-start);
      for (Long i = 0; i < leg_nds.Dim(); i++) {
        nds.PushBack( start + step_*leg_nds[i]);
        nds.PushBack(-start - step_*leg_nds[i]);
        wts.PushBack(step_*leg_wts[i]);
        wts.PushBack(step_*leg_wts[i]);
      }
    }

    for (ValueType start = len1; start < len2; start*=2) {
      ValueType step_ = std::min(start, len2-start);
      for (Long i = 0; i < leg_nds.Dim(); i++) {
        if (s + start + step_*leg_nds[i] <= 1.0) {
          nds.PushBack( start + step_*leg_nds[i]);
          wts.PushBack(step_*leg_wts[i]);
        }
        if (s - start - step_*leg_nds[i] >= 0.0) {
          nds.PushBack(-start - step_*leg_nds[i]);
          wts.PushBack(step_*leg_wts[i]);
        }
      }
    }

    if (!sort) return;
    Vector<ValueType> nds_(nds.Dim());
    Vector<ValueType> wts_(wts.Dim());
    Vector<std::pair<ValueType,Long>> sort_pair;
    for (Long i = 0; i < nds.Dim(); i++) {
      sort_pair.PushBack(std::pair<ValueType,Long>{nds[i], i});
    }
    std::sort(sort_pair.begin(), sort_pair.end());
    for (Long i = 0; i < nds.Dim(); i++) {
      const Long idx = sort_pair[i].second;
      nds_[i] = nds[idx];
      wts_[i] = wts[idx];
    }
    nds = nds_;
    wts = wts_;
  };
  template <Integer ModalUpsample, class ValueType, class Kernel, bool trg_dot_prod> static void SpecialQuadBuildBasisMatrix(Matrix<ValueType>& M, Vector<ValueType>& quad_nds, Vector<ValueType>& quad_wts, const Integer Ncheb, const Integer FourierModes, const ValueType s_trg, const Integer max_digits, const ValueType elem_length, const Integer RefLevels, const Kernel& ker) {
    // TODO: cleanup
    constexpr Integer COORD_DIM = 3;
    using Vec3 = Tensor<ValueType,true,COORD_DIM,1>;

    const Long LegQuadOrder = 2*max_digits;
    constexpr Long LogQuadOrder = 16; // this has non-negative weights

    constexpr Integer KDIM0 = Kernel::SrcDim();
    constexpr Integer KDIM1 = Kernel::TrgDim() / (trg_dot_prod ? COORD_DIM : 1);

    // Adaptive quadrature rule
    DyadicQuad_s(quad_nds, quad_wts, LegQuadOrder, LogQuadOrder, s_trg, RefLevels, true);
    quad_nds += s_trg; // TODO: remove this

    Matrix<ValueType> Minterp_quad_nds;
    { // Set Minterp_quad_nds
      Minterp_quad_nds.ReInit(Ncheb, quad_nds.Dim());
      Vector<ValueType> Vinterp_quad_nds(Ncheb*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
      LagrangeInterp<ValueType>::Interpolate(Vinterp_quad_nds, SlenderElemList<ValueType>::CenterlineNodes(Ncheb), quad_nds);
    }

    Vec3 x_trg, e_trg, n_trg;
    x_trg(0,0) = 0;
    x_trg(1,0) = 0;
    x_trg(2,0) = 0;
    e_trg(0,0) = 1;
    e_trg(1,0) = 0;
    e_trg(2,0) = 0;
    n_trg(0,0) = 1;
    n_trg(1,0) = 0;
    n_trg(2,0) = 0;

    Vector<ValueType> radius(          Ncheb);
    Vector<ValueType> coord (COORD_DIM*Ncheb);
    Vector<ValueType> dr    (          Ncheb);
    Vector<ValueType> dx    (COORD_DIM*Ncheb);
    Vector<ValueType> d2x   (COORD_DIM*Ncheb);
    Vector<ValueType> e1    (COORD_DIM*Ncheb);
    for (Long i = 0; i < Ncheb; i++) {
      radius[i] = 1;
      dr[i] = 0;

      coord[0*Ncheb+i] = 0;
      coord[1*Ncheb+i] = 0;
      coord[2*Ncheb+i] = SlenderElemList<ValueType>::CenterlineNodes(Ncheb)[i] * elem_length - s_trg * elem_length;

      dx[0*Ncheb+i] = 0;
      dx[1*Ncheb+i] = 0;
      dx[2*Ncheb+i] = elem_length;

      d2x[0*Ncheb+i] = 0;
      d2x[1*Ncheb+i] = 0;
      d2x[2*Ncheb+i] = 0;

      e1[0*Ncheb+i] = 1;
      e1[1*Ncheb+i] = 0;
      e1[2*Ncheb+i] = 0;
    }

    Matrix<ValueType> r_src, dr_src, x_src, dx_src, d2x_src, e1_src;
    r_src  .ReInit(        1,quad_nds.Dim());
    dr_src .ReInit(        1,quad_nds.Dim());
    x_src  .ReInit(COORD_DIM,quad_nds.Dim());
    dx_src .ReInit(COORD_DIM,quad_nds.Dim());
    d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
    e1_src .ReInit(COORD_DIM,quad_nds.Dim());
    Matrix<ValueType>::GEMM(  x_src, Matrix<ValueType>(COORD_DIM,Ncheb, coord.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM( dx_src, Matrix<ValueType>(COORD_DIM,Ncheb,    dx.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM(d2x_src, Matrix<ValueType>(COORD_DIM,Ncheb,   d2x.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM(  r_src, Matrix<ValueType>(        1,Ncheb,radius.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM( dr_src, Matrix<ValueType>(        1,Ncheb,    dr.begin(),false), Minterp_quad_nds);
    Matrix<ValueType>::GEMM( e1_src, Matrix<ValueType>(COORD_DIM,Ncheb,    e1.begin(),false), Minterp_quad_nds);
    for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
      Vec3 e1, dx;
      for (Integer k = 0; k < COORD_DIM; k++) {
        e1(k,0) = e1_src[k][j];
        dx(k,0) = dx_src[k][j];
      }
      e1 = e1 - dx * dot_prod(e1, dx) * (1/dot_prod(dx,dx));
      e1 = e1 * (1/sqrt<ValueType>(dot_prod(e1,e1)));

      for (Integer k = 0; k < COORD_DIM; k++) {
        e1_src[k][j] = e1(k,0);
      }
    }

    Matrix<ValueType> M_tor(quad_nds.Dim(), KDIM0*KDIM1*FourierModes*2);
    constexpr Integer TorGreensFnDigits = (Integer)(TypeTraits<ValueType>::SigBits*0.3010299957);
    constexpr Integer adap_tor_greens_fn = 1;
    toroidal_greens_fn_batched<TorGreensFnDigits,ModalUpsample,trg_dot_prod,ValueType,Kernel,adap_tor_greens_fn>(M_tor, x_trg, e_trg, (ValueType)1, n_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, ker, FourierModes);

    M.ReInit(quad_nds.Dim(), Ncheb*FourierModes*2*KDIM0*KDIM1);
    for (Long i = 0; i < quad_nds.Dim(); i++) {
      for (Long j = 0; j < Ncheb; j++) {
        for (Long k = 0; k < KDIM0*KDIM1*FourierModes*2; k++) {
          M[i][j*KDIM0*KDIM1*FourierModes*2+k] = Minterp_quad_nds[j][i] * M_tor[i][k];
        }
      }
    }
  }
  template <Integer ModalUpsample, class ValueType, class Kernel, bool trg_dot_prod, bool symmetric=true/*must be set true for hypersingular kernels*/> static Vector<Vector<ValueType>> BuildSpecialQuadRules(const Integer Ncheb, const Integer FourierModes, const Integer trg_node_idx, const ValueType elem_length) {
    constexpr Integer Nlen = 20; // number of length samples in [elem_length/sqrt(2), elem_length*sqrt(2)]
    constexpr Integer max_digits = 19;
    const ValueType s_trg = SlenderElemList<ValueType>::CenterlineNodes(Ncheb)[trg_node_idx];
    const Integer adap_depth = (Integer)(log<ValueType>(elem_length)/log<ValueType>(2)+4);
    const ValueType eps_buffer = std::min<ValueType>(3e-2/elem_length, 3e-4); // distance of closest node points to s_trg
    const ValueType eps = 8*machine_eps<ValueType>();

    Kernel ker;
    Vector<ValueType> nds, wts;
    Matrix<ValueType> Mintegrands;
    { // Set nds, wts, Mintegrands
      Vector<Matrix<ValueType>> Mker(Nlen);
      Vector<Vector<ValueType>> nds_(Nlen), wts_(Nlen);
      #pragma omp parallel for schedule(static)
      for (Long k = 0; k < Nlen; k++) {
        ValueType length = elem_length/sqrt<ValueType>(2.0)*k/(Nlen-1) + elem_length*sqrt<ValueType>(2.0)*(Nlen-k-1)/(Nlen-1);
        SpecialQuadBuildBasisMatrix<ModalUpsample,ValueType,Kernel,trg_dot_prod>(Mker[k], nds_[k], wts_[k], Ncheb, FourierModes, s_trg, max_digits, length, adap_depth, ker);
      }
      const Long N0 = nds_[0].Dim();

      Vector<Long> cnt(Nlen), dsp(Nlen); dsp[0] = 0;
      for (Long k = 0; k < Nlen; k++) {
        cnt[k] = Mker[k].Dim(1);
      }
      omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());

      const Long Nsplit = (symmetric ? std::lower_bound(nds_[0].begin(), nds_[0].end(), s_trg) - nds_[0].begin() : N0);
      const Long N = std::max<Long>(N0 - Nsplit, Nsplit);

      nds.ReInit(N);
      wts.ReInit(N);
      Mintegrands.ReInit(N, dsp[Nlen-1] + cnt[Nlen-1]);
      if (N == Nsplit) {
        #pragma omp parallel for schedule(static)
        for (Long k = 0; k < Nlen; k++) {
          for (Long i = 0; i < Nsplit; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[i][dsp[k]+j] = Mker[k][i][j];
            }
          }

          for (Long i = Nsplit; i < N0; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[2*Nsplit-i-1][dsp[k]+j] += Mker[k][i][j];
            }
          }
        }

        for (Long i = 0; i < Nsplit; i++) {
          nds[i] = nds_[0][i];
          wts[i] = wts_[0][i];
        }
        for (Long i = Nsplit; i < N0; i++) {
          SCTL_ASSERT(fabs(nds[2*Nsplit-i-1] + nds_[0][i] - 2*s_trg) < eps);
          SCTL_ASSERT(fabs(wts[2*Nsplit-i-1] - wts_[0][i]) < eps);
        }
      } else {
        #pragma omp parallel for schedule(static)
        for (Long k = 0; k < Nlen; k++) {
          for (Long i = Nsplit; i < N0; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[i-Nsplit][dsp[k]+j] = Mker[k][i][j];
            }
          }

          for (Long i = 0; i < Nsplit; i++) {
            for (Long j = 0; j < cnt[k]; j++) {
              Mintegrands[Nsplit-i-1][dsp[k]+j] += Mker[k][i][j];
            }
          }
        }

        for (Long i = Nsplit; i < N0; i++) {
          nds[i-Nsplit] = nds_[0][i];
          wts[i-Nsplit] = wts_[0][i];
        }
        for (Long i = 0; i < Nsplit; i++) {
          SCTL_ASSERT(fabs(nds[Nsplit-i-1] + nds_[0][i] - 2*s_trg) < eps);
          SCTL_ASSERT(fabs(wts[Nsplit-i-1] - wts_[0][i]) < eps);
        }
      }
    }

    Vector<Vector<ValueType>> nds_wts(max_digits*2);
    { // Set nds_wts
      Vector<ValueType> eps_vec;
      Vector<Vector<ValueType>> quad_nds, quad_wts;
      for (Long k = 0; k < max_digits; k++) eps_vec.PushBack(pow<ValueType,Long>(0.1,k));
      ValueType range0 = s_trg>=0.5 ? 0 : s_trg+eps_buffer;
      ValueType range1 = s_trg>=0.5 ? s_trg-eps_buffer : 1;
      InterpQuadRule<ValueType>::Build(quad_nds, quad_wts,  Mintegrands, nds, wts, eps_vec, Vector<Long>(), range0, range1);
      SCTL_ASSERT(quad_nds.Dim() == max_digits);
      SCTL_ASSERT(quad_wts.Dim() == max_digits);
      for (Long k = 0; k < max_digits; k++) {
        for (Long i = 0; i < quad_nds[k].Dim(); i++) {
          const ValueType qx0 = quad_nds[k][i];
          const ValueType qx1 = 2*s_trg - qx0;
          const ValueType qw = quad_wts[k][i];

          nds_wts[k*2+0].PushBack(qx0);
          nds_wts[k*2+1].PushBack(qw);

          if (symmetric && 0 <= qx1 && qx1 <= (ValueType)1) {
            nds_wts[k*2+0].PushBack(qx1);
            nds_wts[k*2+1].PushBack(qw);
          }
        }
      }
    }
    return nds_wts;
  }
  template <Integer ModalUpsample, class Real, class Kernel, bool trg_dot_prod, bool adap_quad=false> static void SpecialQuadRule(Vector<Real>& nds, Vector<Real>& wts, const Integer ChebOrder, const Integer trg_node_idx, const Real elem_radius, const Real elem_length, const Integer digits) {
    static constexpr Integer max_adap_depth = 30+7; // TODO
    constexpr Integer MaxFourierModes = 8; // TODO
    constexpr Integer MaxChebOrder = 100;
    constexpr Integer max_digits = 19;

    auto LogSingularQuadOrder = [](Integer digits) { return 2*digits; }; // TODO: determine optimal order
    auto LegQuadOrder = [](Integer digits) { return digits; }; // TODO: determine optimal order

    #ifdef SCTL_QUAD_T
    using ValueType = QuadReal;
    #else
    using ValueType = long double;
    #endif
    if (0) { // Compute quadratures on-the-fly
      static ValueType aspect_ratio = 0;
      static Vector<Vector<Vector<ValueType>>> nds_wts;
      #pragma omp critical(mytest)
      if (elem_length/elem_radius < aspect_ratio/1.42 || aspect_ratio*1.42 < elem_length/elem_radius) {
        nds_wts.ReInit(ChebOrder);
        for (Long i = 0; i < ChebOrder; i++) {
          nds_wts[i] = BuildSpecialQuadRules<ModalUpsample,ValueType,Kernel,trg_dot_prod>(ChebOrder, MaxFourierModes, i, elem_length/elem_radius);
        }
        aspect_ratio = elem_length/elem_radius;
      }
      const Long Nnds = nds_wts[trg_node_idx][digits*2+0].Dim();
      nds.ReInit(Nnds);
      wts.ReInit(Nnds);
      for (Long i = 0; i < Nnds; i++) {
        static const auto cheb_nds_ = SlenderElemList<ValueType>::CenterlineNodes(ChebOrder);
        nds[i] = (Real)(nds_wts[trg_node_idx][digits*2+0][i] - cheb_nds_[trg_node_idx]);
        wts[i] = (Real)nds_wts[trg_node_idx][digits*2+1][i];
      }
      return;
    }

    if (!adap_quad) {
      auto load_special_quad_rule = [](Vector<Vector<Real>>& nds_lst, Vector<Vector<Real>>& wts_lst, const Integer ChebOrder){
        const std::string fname = std::string(SCTL_QUOTEME(SCTL_DATA_PATH)) + std::string("/special_quad_q") + std::to_string(ChebOrder) + "_" + Kernel::Name() + (trg_dot_prod ? "_dotXn" : "");
        const auto cheb_nds_ = SlenderElemList<ValueType>::CenterlineNodes(ChebOrder);

        Vector<Vector<ValueType>> data;
        ReadFile(data, fname);
        if (data.Dim() != max_adap_depth*ChebOrder*max_digits*2) { // build quadrature rules
          data.ReInit(max_adap_depth*ChebOrder*max_digits*2);
          ValueType length = pow<max_adap_depth-7,ValueType>((ValueType)2);
          for (Integer i = 0; i < max_adap_depth; i++) {
            std::cout<<"length = "<<length<<'\n';
            for (Integer trg_node_idx = 0; trg_node_idx < ChebOrder; trg_node_idx++) {
              auto nds_wts = BuildSpecialQuadRules<ModalUpsample,ValueType,Kernel,trg_dot_prod>(ChebOrder, MaxFourierModes, trg_node_idx, length);
              for (Long j = 0; j < max_digits; j++) {
                data[((i*ChebOrder+trg_node_idx) * max_digits+j)*2+0] = nds_wts[j*2+0];
                data[((i*ChebOrder+trg_node_idx) * max_digits+j)*2+1] = nds_wts[j*2+1];
              }
            }
            length *= (ValueType)0.5;
          }
          WriteFile(data, fname);
        }

        nds_lst.ReInit(max_adap_depth*ChebOrder*max_digits);
        wts_lst.ReInit(max_adap_depth*ChebOrder*max_digits);
        for (Long i = 0; i < max_adap_depth*ChebOrder*max_digits; i++) { // Set nds_wts_lst
          const Long trg_node_idx = (i/max_digits)%ChebOrder;
          const auto& nds_ = data[i*2+0];
          const auto& wts_ = data[i*2+1];
          const Long Nnds = wts_.Dim();

          nds_lst[i].ReInit(Nnds);
          wts_lst[i].ReInit(Nnds);
          for (Long j = 0; j < Nnds; j++) {
            nds_lst[i][j] = (Real)(nds_[j] - cheb_nds_[trg_node_idx]);
            wts_lst[i][j] = (Real)wts_[j];
          }
        }
      };
      static Vector<Vector<Vector<Real>>> nds_lst(MaxChebOrder);
      static Vector<Vector<Vector<Real>>> wts_lst(MaxChebOrder);
      SCTL_ASSERT(ChebOrder < MaxChebOrder);
      #pragma omp critical(SCTL_SpecialQuadRule)
      if (!wts_lst[ChebOrder].Dim()) {
        load_special_quad_rule(nds_lst[ChebOrder], wts_lst[ChebOrder], ChebOrder);
      }

      Long quad_idx = (Long)((max_adap_depth-7) - log2((double)(elem_length/elem_radius*sqrt<Real>(0.5))));
      if (quad_idx < 0 || quad_idx > max_adap_depth-1) {
        SCTL_WARN("Slender element aspect-ratio is outside of the range of precomputed quadratures; accuracy may be severely degraded.");
      }
      quad_idx = std::max<Integer>(0, std::min<Integer>(max_adap_depth-1, quad_idx));

      const auto& wts0 = wts_lst[ChebOrder][(quad_idx*ChebOrder+trg_node_idx) * max_digits+digits];
      const auto& nds0 = nds_lst[ChebOrder][(quad_idx*ChebOrder+trg_node_idx) * max_digits+digits];
      wts.ReInit(wts0.Dim(), (Iterator<Real>)wts0.begin(), false);
      nds.ReInit(nds0.Dim(), (Iterator<Real>)nds0.begin(), false);
    } else {
      const Integer RefLevels = (Integer)(log<Real>(elem_length/elem_radius)/log<Real>(2)-1);
      const auto& cheb_nds = SlenderElemList<Real>::CenterlineNodes(ChebOrder);
      const Real s_trg = cheb_nds[trg_node_idx];
      DyadicQuad_s(nds, wts, LegQuadOrder(digits), LogSingularQuadOrder(digits), s_trg, RefLevels, false);
    }
  }




  template <class Real> template <class ValueType> SlenderElemList<Real>::SlenderElemList(const Vector<Long>& cheb_order0, const Vector<Long>& fourier_order0, const Vector<ValueType>& coord0, const Vector<ValueType>& radius0, const Vector<ValueType>& orientation0) {
    Init(cheb_order0, fourier_order0, coord0, radius0, orientation0);
  }
  template <class Real> template <class ValueType> void SlenderElemList<Real>::Init(const Vector<Long>& cheb_order0, const Vector<Long>& fourier_order0, const Vector<ValueType>& coord0, const Vector<ValueType>& radius0, const Vector<ValueType>& orientation0) {
    const Long Nelem = cheb_order0.Dim();
    SCTL_ASSERT(fourier_order0.Dim() == Nelem);

    cheb_order = cheb_order0;
    fourier_order = fourier_order0;
    elem_dsp.ReInit(Nelem);
    if (Nelem) elem_dsp[0] = 0;
    omp_par::scan(cheb_order.begin(), elem_dsp.begin(), Nelem);

    const Long Nnodes = (Nelem ? cheb_order[Nelem-1]+elem_dsp[Nelem-1] : 0);
    SCTL_ASSERT_MSG(coord0.Dim() == Nnodes * COORD_DIM, "Length of the coordinate vector does not match the number of nodes.");
    SCTL_ASSERT_MSG(radius0.Dim() == Nnodes, "Length of the radius vector does not match the number of nodes.");

    radius.ReInit(          Nnodes);
    coord .ReInit(COORD_DIM*Nnodes);
    e1    .ReInit(COORD_DIM*Nnodes);
    dr    .ReInit(          Nnodes);
    dx    .ReInit(COORD_DIM*Nnodes);
    d2x   .ReInit(COORD_DIM*Nnodes);
    for (Long i = 0; i < Nelem; i++) { // Set coord, radius, dr, ds, d2s
      const Long Ncheb = cheb_order[i];
      const Vector<ValueType> radius0_(          Ncheb, (Iterator<ValueType>)radius0.begin()+elem_dsp[i]          , false);
      const Vector<ValueType> coord0_ (COORD_DIM*Ncheb, (Iterator<ValueType>) coord0.begin()+elem_dsp[i]*COORD_DIM, false);

      const auto&      radius__ = radius0_;
      Vector<ValueType> coord__(COORD_DIM*Ncheb);
      Vector<ValueType>    e1__(COORD_DIM*Ncheb);
      Vector<ValueType>    dr__(          Ncheb);
      Vector<ValueType>    dx__(COORD_DIM*Ncheb);
      Vector<ValueType>   d2x__(COORD_DIM*Ncheb);
      for (Long j = 0; j < Ncheb; j++) { // Set coord__
        for (Long k = 0; k < COORD_DIM; k++) {
          coord__[k*Ncheb+j] = coord0_[j*COORD_DIM+k];
        }
      }
      LagrangeInterp<ValueType>::Derivative( dr__, radius__, SlenderElemList<ValueType>::CenterlineNodes(Ncheb));
      LagrangeInterp<ValueType>::Derivative( dx__,  coord__, SlenderElemList<ValueType>::CenterlineNodes(Ncheb));
      LagrangeInterp<ValueType>::Derivative(d2x__,     dx__, SlenderElemList<ValueType>::CenterlineNodes(Ncheb));

      if (orientation0.Dim()) { // Set e1__
        SCTL_ASSERT(orientation0.Dim() == Nnodes*COORD_DIM);
        const Vector<ValueType> orientation0_(COORD_DIM*Ncheb, (Iterator<ValueType>)orientation0.begin()+elem_dsp[i]*COORD_DIM, false);
        for (Long j = 0; j < Ncheb; j++) {
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1__[k*Ncheb+j] = orientation0_[j*COORD_DIM+k];
          }
        }
      } else {
        using Vec3 = Tensor<ValueType,true,COORD_DIM,1>;
        const auto orthonormalize = [&Ncheb,&dx__](Vec3& e1_vec, const Integer j) { // orthonormalize
          Vec3 dx_vec;
          for (Integer k = 0; k < COORD_DIM; k++) dx_vec(k,0) = dx__[k*Ncheb+j];
          e1_vec = e1_vec - dx_vec*(dot_prod(dx_vec,e1_vec)/dot_prod(dx_vec,dx_vec));
          e1_vec = e1_vec * (1.0/sqrt<ValueType>(dot_prod(e1_vec,e1_vec)));
        };

        Vec3 e1_vec((ValueType)0);
        { // Set e1_vec
          Integer orient_dir = 0;
          for (Integer k = 0; k < COORD_DIM; k++) {
            if (fabs(dx__[k*Ncheb+0]) < fabs(dx__[orient_dir*Ncheb+0])) orient_dir = k;
          }
          e1_vec(orient_dir,0) = 1;
        }

        if (0) for (Long j = 0; j < Ncheb; j++) { // first-order method using orthonormal projections
          orthonormalize(e1_vec, j);
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1__[k*Ncheb+j] = e1_vec(k,0);
          }
        }

        if (0) { // Solve de = e x (d2x x dx) / (dx.dx), using spectral deferred correction
          const auto& nodes = SlenderElemList<ValueType>::CenterlineNodes(Ncheb);
          const SDC<ValueType> ode_solve(10, Comm::Self());
          const ValueType tol = std::max((ValueType)1e-10, sqrt<ValueType>(machine_eps<ValueType>()));
          constexpr bool continue_with_errors = true;

          Vector<ValueType> e1_0(4), e1_j(4); e1_0 = 0;
          const auto fn_de1 = [this,&Ncheb,&d2x__,&dx__,&nodes](Vector<ValueType>* de_, const Vector<ValueType>& e_) {
            Vec3 e, dx, d2x, de;
            Vector<ValueType> interp_wts;
            LagrangeInterp<ValueType>::Interpolate(interp_wts, nodes, Vector<ValueType>(1, (Iterator<ValueType>)e_.begin()+COORD_DIM, false));
            for (Long k = 0; k < COORD_DIM; k++) {
              dx(k,0) = 0;
              d2x(k,0) = 0;
              e(k,0) = e_[k];
              for (Long i = 0; i < Ncheb; i++) {
                d2x(k,0) += interp_wts[i] * d2x__[k*Ncheb+i];
                dx(k,0) += interp_wts[i] * dx__[k*Ncheb+i];
              }
            }

            de = cross_prod(e, cross_prod(d2x, dx)) / dot_prod(dx,dx);
            for (Long k = 0; k < COORD_DIM; k++) (*de_)[k] = de(k,0);
            (*de_)[COORD_DIM] = 1;
          };
          for (Long j = 0; j < Ncheb; j++) { // ODE solve
            orthonormalize(e1_vec, j);
            for (Integer k = 0; k < COORD_DIM; k++) { // e1_0, e1__ <-- e1_vec
              e1__[k*Ncheb+j] = e1_vec(k,0);
              e1_0[k] = e1_vec(k,0);
            }
            e1_0[COORD_DIM] = nodes[j];

            ValueType error = 0;
            const ValueType T = (j+1<Ncheb ? nodes[j+1] : (ValueType)1) - nodes[j];
            ode_solve.AdaptiveSolve(&e1_j, T/10, T, e1_0, fn_de1, tol, nullptr, continue_with_errors, &error);
            for (Integer k = 0; k < COORD_DIM; k++) e1_vec(k,0) = e1_j[k];
          }
        }

        if (1) { // Solve de = e x (d2x x dx) / (dx.dx), by building Chebyshev spectral operators
          const auto& nodes = SlenderElemList<ValueType>::CenterlineNodes(Ncheb);
          orthonormalize(e1_vec, 0);

          Matrix<ValueType> M1(Ncheb*COORD_DIM, (Ncheb+1)*COORD_DIM); M1 = 0;
          for (Long i = 0; i < Ncheb; i++) { // M <-- operator( . x (d2x x dx) / (dx.dx) )
            Vec3 dx, d2x;
            for (Long k = 0; k < COORD_DIM; k++) {
              dx(k,0) = dx__[k*Ncheb+i];
              d2x(k,0) = d2x__[k*Ncheb+i];
            }
            const Vec3 v = cross_prod(d2x, dx) / dot_prod(dx,dx);
            M1[i*COORD_DIM+0][i*COORD_DIM+1] =-v(0,2);
            M1[i*COORD_DIM+0][i*COORD_DIM+2] = v(0,1);
            M1[i*COORD_DIM+1][i*COORD_DIM+0] = v(0,2);
            M1[i*COORD_DIM+1][i*COORD_DIM+2] =-v(0,0);
            M1[i*COORD_DIM+2][i*COORD_DIM+0] =-v(0,1);
            M1[i*COORD_DIM+2][i*COORD_DIM+1] = v(0,0);
          }

          Matrix<ValueType> M2(Ncheb*COORD_DIM, (Ncheb+1)*COORD_DIM); M2 = 0;
          for (Long i = 0; i < Ncheb; i++) { // differentiation matrix
            Vector<ValueType> dx(Ncheb), x(Ncheb); x = 0; x[i] = 1;
            LagrangeInterp<ValueType>::Derivative(dx, x, nodes);
            for (Long j = 0; j < Ncheb; j++) {
              for (Long k = 0; k < COORD_DIM; k++) {
                M2[i*COORD_DIM+k][j*COORD_DIM+k] = dx[j];
              }
            }
          }

          Matrix<ValueType> A = M2 - M1;
          A[0][Ncheb*COORD_DIM+0] = 1; // Add boundary conditions
          A[1][Ncheb*COORD_DIM+1] = 1;
          A[2][Ncheb*COORD_DIM+2] = 1;

          Matrix<ValueType> b(1, (Ncheb+1)*COORD_DIM); b = 0;
          b[0][Ncheb*COORD_DIM+0] = e1_vec(0,0);
          b[0][Ncheb*COORD_DIM+1] = e1_vec(0,1);
          b[0][Ncheb*COORD_DIM+2] = e1_vec(0,2);
          Matrix<ValueType> e1 = b * A.pinv();

          for (Long i = 0; i < Ncheb; i++) {
            Vec3 e1_;
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1_(0,k) = e1[0][i*COORD_DIM+k];
            }
            orthonormalize(e1_, i);
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1__[k*Ncheb+i] = e1_(0,k);
            }
          }
        }
      }

      Vector<Real> radius_(          Ncheb, radius.begin()+          elem_dsp[i], false);
      Vector<Real>  coord_(COORD_DIM*Ncheb,  coord.begin()+COORD_DIM*elem_dsp[i], false);
      Vector<Real>     e1_(COORD_DIM*Ncheb,     e1.begin()+COORD_DIM*elem_dsp[i], false);
      Vector<Real>     dr_(          Ncheb,     dr.begin()+          elem_dsp[i], false);
      Vector<Real>     dx_(COORD_DIM*Ncheb,     dx.begin()+COORD_DIM*elem_dsp[i], false);
      Vector<Real>    d2x_(COORD_DIM*Ncheb,    d2x.begin()+COORD_DIM*elem_dsp[i], false);
      for (Long j = 0; j < COORD_DIM*Ncheb; j++) {
        coord_[j] = (Real)coord__[j];
        e1_   [j] = (Real)e1__   [j];
        dx_   [j] = (Real)dx__   [j];
        d2x_  [j] = (Real)d2x__  [j];
      }
      for (Long j = 0; j < Ncheb; j++) {
        radius_[j] = (Real)radius__[j];
        dr_    [j] = (Real)dr__    [j];
      }
    }
  }

  template <class Real> Long SlenderElemList<Real>::Size() const {
    return cheb_order.Dim();
  }

  template <class Real> void SlenderElemList<Real>::GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const {
    const Long Nelem = cheb_order.Dim();
    Vector<Long> node_cnt(Nelem), node_dsp(Nelem);
    { // Set node_cnt, node_dsp
      for (Long i = 0; i < Nelem; i++) {
        node_cnt[i] = cheb_order[i] * fourier_order[i];
      }
      if (Nelem) node_dsp[0] = 0;
      omp_par::scan(node_cnt.begin(), node_dsp.begin(), Nelem);
    }

    const Long Nnodes = (Nelem ? node_dsp[Nelem-1]+node_cnt[Nelem-1] : 0);
    if (element_wise_node_cnt) (*element_wise_node_cnt) = node_cnt;
    if (X  != nullptr && X ->Dim() != Nnodes*COORD_DIM) X ->ReInit(Nnodes*COORD_DIM);
    if (Xn != nullptr && Xn->Dim() != Nnodes*COORD_DIM) Xn->ReInit(Nnodes*COORD_DIM);
    for (Long i = 0; i < Nelem; i++) {
      Vector<Real> X_, Xn_;
      if (X  != nullptr) X_ .ReInit(node_cnt[i]*COORD_DIM, X ->begin()+node_dsp[i]*COORD_DIM, false);
      if (Xn != nullptr) Xn_.ReInit(node_cnt[i]*COORD_DIM, Xn->begin()+node_dsp[i]*COORD_DIM, false);
      GetGeom((X==nullptr?nullptr:&X_), (Xn==nullptr?nullptr:&Xn_), nullptr,nullptr,nullptr, CenterlineNodes(cheb_order[i]), sin_theta<Real>(fourier_order[i]), cos_theta<Real>(fourier_order[i]), i);
    }
  }
  template <class Real> void SlenderElemList<Real>::GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const {
    const Long Nelem = cheb_order.Dim();
    Vector<Long> node_cnt(Nelem), node_dsp(Nelem);
    { // Set node_cnt, node_dsp
      for (Long i = 0; i < Nelem; i++) {
        node_cnt[i] = cheb_order[i]*FARFIELD_UPSAMPLE * fourier_order[i]*FARFIELD_UPSAMPLE;
      }
      if (Nelem) node_dsp[0] = 0;
      omp_par::scan(node_cnt.begin(), node_dsp.begin(), Nelem);
    }

    StaticArray<Real,6000> static_buff;
    Vector<Real> buff(6000, static_buff, false);

    element_wise_node_cnt = node_cnt;
    const Long Nnodes = (Nelem ? node_dsp[Nelem-1]+node_cnt[Nelem-1] : 0);
    if (X       .Dim() != Nnodes*COORD_DIM) X       .ReInit(Nnodes*COORD_DIM);
    if (Xn      .Dim() != Nnodes*COORD_DIM) Xn      .ReInit(Nnodes*COORD_DIM);
    if (wts     .Dim() != Nnodes          ) wts     .ReInit(Nnodes          );
    if (dist_far.Dim() != Nnodes          ) dist_far.ReInit(Nnodes          );
    for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
      Vector<Real>        X_(node_cnt[elem_idx]*COORD_DIM,        X.begin()+node_dsp[elem_idx]*COORD_DIM, false);
      Vector<Real>       Xn_(node_cnt[elem_idx]*COORD_DIM,       Xn.begin()+node_dsp[elem_idx]*COORD_DIM, false);
      Vector<Real>      wts_(node_cnt[elem_idx]          ,      wts.begin()+node_dsp[elem_idx]          , false);
      Vector<Real> dist_far_(node_cnt[elem_idx]          , dist_far.begin()+node_dsp[elem_idx]          , false);

      const Long ChebOrder = cheb_order[elem_idx];
      const Long FourierOrder = fourier_order[elem_idx];
      SCTL_ASSERT(node_cnt[elem_idx] == ChebOrder*FARFIELD_UPSAMPLE * FourierOrder*FARFIELD_UPSAMPLE);

      Vector<Real> dX_ds, dX_dt;
      const Long Nnds = ChebOrder*FARFIELD_UPSAMPLE*FourierOrder*FARFIELD_UPSAMPLE;
      if (buff.Dim() < 2*Nnds*COORD_DIM) {
        buff.ReInit(2*Nnds*COORD_DIM);
      } else {
        dX_ds.ReInit(Nnds*COORD_DIM, buff.begin(), false);
        dX_dt.ReInit(Nnds*COORD_DIM, buff.begin()+Nnds*COORD_DIM, false);
      }

      const auto& leg_nds = LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).first;
      const auto& leg_wts = LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).second;
      GetGeom(&X_, &Xn_, &wts_, &dX_ds, &dX_dt, leg_nds, sin_theta<Real>(FourierOrder*FARFIELD_UPSAMPLE), cos_theta<Real>(FourierOrder*FARFIELD_UPSAMPLE), elem_idx);

      Vector<Real> dist_far_gauss(ChebOrder*FARFIELD_UPSAMPLE);
      for (Long i = 0; i < ChebOrder*FARFIELD_UPSAMPLE; i++) { // Set dist_far_gauss
        const Real rho=pow<Real>((64/(15*tol)), (1/(Real)(2*ChebOrder*FARFIELD_UPSAMPLE)));
        const Real a = (rho-1/rho)/4;
        const Real b = (rho+1/rho)/4;
        const Real c = 0.5;

        dist_far_gauss[i] = b - fabs(leg_nds[i]-0.5);
        const Real cos_t = b * (leg_nds[i]-0.5) / (c*c);
        if (fabs(cos_t) <= 1) dist_far_gauss[i] = a * sqrt<Real>(1 + ((a*a)/(b*b)-1) * cos_t*cos_t);
      }
      const Real dist_far_trapezoidal = (pow<Real>(4*const_pi<Real>()/tol+1, 1/(Real)(FourierOrder*FARFIELD_UPSAMPLE)) - 1); // from theorem 2.2 in doi:10.1137/130932132

      const Real theta_quad_wt = 2*const_pi<Real>()/(FourierOrder*FARFIELD_UPSAMPLE);
      for (Long i = 0; i < ChebOrder*FARFIELD_UPSAMPLE; i++) { // Set wts *= leg_wts * theta_quad_wt
        Real quad_wt = leg_wts[i] * theta_quad_wt;
        for (Long j = 0; j < FourierOrder*FARFIELD_UPSAMPLE; j++) {
          wts_[i*FourierOrder*FARFIELD_UPSAMPLE+j] *= quad_wt;
        }
      }
      for (Long i = 0; i < ChebOrder*FARFIELD_UPSAMPLE; i++) { // Set dist_far
        for (Long j = 0; j < FourierOrder*FARFIELD_UPSAMPLE; j++) {
          const Long node_idx = i*FourierOrder*FARFIELD_UPSAMPLE + j;
          Real dxdt = sqrt<Real>(dX_dt[node_idx*COORD_DIM+0]*dX_dt[node_idx*COORD_DIM+0] + dX_dt[node_idx*COORD_DIM+1]*dX_dt[node_idx*COORD_DIM+1] + dX_dt[node_idx*COORD_DIM+2]*dX_dt[node_idx*COORD_DIM+2]);
          Real dxds = sqrt<Real>(dX_ds[node_idx*COORD_DIM+0]*dX_ds[node_idx*COORD_DIM+0] + dX_ds[node_idx*COORD_DIM+1]*dX_ds[node_idx*COORD_DIM+1] + dX_ds[node_idx*COORD_DIM+2]*dX_ds[node_idx*COORD_DIM+2]);
          dist_far_[node_idx] = std::max(dist_far_gauss[i] * dxds, dist_far_trapezoidal * dxdt);
        }
      }
    }
  }
  template <class Real> void SlenderElemList<Real>::GetFarFieldDensity(Vector<Real>& Fout, const Vector<Real>& Fin) const {
    constexpr Integer MaxOrderFourier = 128/FARFIELD_UPSAMPLE;
    constexpr Integer MaxOrderCheb = 50/FARFIELD_UPSAMPLE;
    auto compute_Mfourier_upsample_transpose = [MaxOrderFourier]() {
      Vector<Matrix<Real>> M_lst(MaxOrderFourier);
      for (Long k = 1; k < MaxOrderFourier; k++) {
        const Integer FourierOrder = k;
        const Integer FourierModes = FourierOrder/2+1;
        const Matrix<Real>& Mfourier_inv = fourier_matrix_inv<Real>(FourierOrder,FourierModes);
        const Matrix<Real>& Mfourier = fourier_matrix<Real>(FourierModes,FourierOrder*FARFIELD_UPSAMPLE);
        M_lst[k] = (FARFIELD_UPSAMPLE != 1 ? (Mfourier_inv * Mfourier).Transpose() : Matrix<Real>(0,0));
      }
      return M_lst;
    };
    auto compute_Mcheb_upsample_transpose = [MaxOrderCheb]() {
      Vector<Matrix<Real>> M_lst(MaxOrderCheb);
      for (Long k = 0; k < MaxOrderCheb; k++) {
        const Integer ChebOrder = k;
        Matrix<Real> Minterp(ChebOrder, ChebOrder*FARFIELD_UPSAMPLE);
        Vector<Real> Vinterp(ChebOrder*ChebOrder*FARFIELD_UPSAMPLE, Minterp.begin(), false);
        LagrangeInterp<Real>::Interpolate(Vinterp, CenterlineNodes(ChebOrder), LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).first);
        M_lst[k] = Minterp.Transpose();
      }
      return M_lst;
    };
    static const Vector<Matrix<Real>> Mfourier_transpose = compute_Mfourier_upsample_transpose();
    static const Vector<Matrix<Real>> Mcheb_transpose = compute_Mcheb_upsample_transpose();

    const Long Nelem = cheb_order.Dim();
    Vector<Long> node_cnt(Nelem), node_dsp(Nelem);
    { // Set node_cnt, node_dsp
      for (Long i = 0; i < Nelem; i++) {
        node_cnt[i] = cheb_order[i] * fourier_order[i];
      }
      if (Nelem) node_dsp[0] = 0;
      omp_par::scan(node_cnt.begin(), node_dsp.begin(), Nelem);
    }

    const Long Nnodes = (Nelem ? node_dsp[Nelem-1]+node_cnt[Nelem-1] : 0);
    const Long density_dof = (Nnodes ? Fin.Dim() / Nnodes : 0);
    SCTL_ASSERT(Fin.Dim() == Nnodes * density_dof);

    if (Fout.Dim() != Nnodes*(FARFIELD_UPSAMPLE*FARFIELD_UPSAMPLE) * density_dof) {
      Fout.ReInit(Nnodes*(FARFIELD_UPSAMPLE*FARFIELD_UPSAMPLE) * density_dof);
    }
    Matrix<Real> F0; // pre-allocate
    for (Long i = 0; i < Nelem; i++) { // TODO: parallelize
      const Integer ChebOrder = cheb_order[i];
      const Integer FourierOrder = fourier_order[i];

      const auto& Mfourier_ = Mfourier_transpose[FourierOrder];
      const Matrix<Real> Fin_(ChebOrder, FourierOrder*density_dof, (Iterator<Real>)Fin.begin()+node_dsp[i]*density_dof, false);
      if (Mfourier_.Dim(0) && Mfourier_.Dim(1)) {
        F0.ReInit(ChebOrder, FourierOrder*FARFIELD_UPSAMPLE*density_dof);
        for (Long l = 0; l < ChebOrder; l++) { // Set F0
          for (Long j0 = 0; j0 < FourierOrder*FARFIELD_UPSAMPLE; j0++) {
            for (Long k = 0; k < density_dof; k++) {
              Real f = 0;
              for (Long j1 = 0; j1 < FourierOrder; j1++) {
                f += Fin_[l][j1*density_dof+k] * Mfourier_[j0][j1];
              }
              F0[l][j0*density_dof+k] = f;
            }
          }
        }
      }
      const auto& F0_ = (Mfourier_.Dim(0) && Mfourier_.Dim(1) ? F0 : Fin_);

      Matrix<Real> Fout_(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*FARFIELD_UPSAMPLE*density_dof, Fout.begin()+node_dsp[i]*FARFIELD_UPSAMPLE*FARFIELD_UPSAMPLE*density_dof, false);
      Matrix<Real>::GEMM(Fout_, Mcheb_transpose[ChebOrder], F0_);
    }
  }
  template <class Real> void SlenderElemList<Real>::FarFieldDensityOperatorTranspose(Matrix<Real>& Mout, const Matrix<Real>& Min, const Long elem_idx) const {
    constexpr Integer MaxOrderFourier = 128/FARFIELD_UPSAMPLE;
    constexpr Integer MaxOrderCheb = 50/FARFIELD_UPSAMPLE;
    auto compute_Mfourier_upsample = [MaxOrderFourier]() {
      Vector<Matrix<Real>> M_lst(MaxOrderFourier);
      for (Long k = 1; k < MaxOrderFourier; k++) {
        const Integer FourierOrder = k;
        const Integer FourierModes = FourierOrder/2+1;
        const Matrix<Real>& Mfourier_inv = fourier_matrix_inv<Real>(FourierOrder,FourierModes);
        const Matrix<Real>& Mfourier = fourier_matrix<Real>(FourierModes,FourierOrder*FARFIELD_UPSAMPLE);
        M_lst[k] = Mfourier_inv * Mfourier;
      }
      return M_lst;
    };
    auto compute_Mcheb_upsample = [MaxOrderCheb]() {
      Vector<Matrix<Real>> M_lst(MaxOrderCheb);
      for (Long k = 0; k < MaxOrderCheb; k++) {
        const Integer ChebOrder = k;
        Matrix<Real> Minterp(ChebOrder, ChebOrder*FARFIELD_UPSAMPLE);
        Vector<Real> Vinterp(ChebOrder*ChebOrder*FARFIELD_UPSAMPLE, Minterp.begin(), false);
        LagrangeInterp<Real>::Interpolate(Vinterp, CenterlineNodes(ChebOrder), LegendreQuadRule<Real>(ChebOrder*FARFIELD_UPSAMPLE).first);
        M_lst[k] = Minterp;
      }
      return M_lst;
    };
    static const Vector<Matrix<Real>> Mfourier = compute_Mfourier_upsample();
    static const Vector<Matrix<Real>> Mcheb = compute_Mcheb_upsample();

    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];

    const Long N = Min.Dim(1);
    const Long density_dof = Min.Dim(0) / (ChebOrder*FARFIELD_UPSAMPLE*FourierOrder*FARFIELD_UPSAMPLE);
    SCTL_ASSERT(Min.Dim(0) == ChebOrder*FARFIELD_UPSAMPLE*FourierOrder*FARFIELD_UPSAMPLE*density_dof);
    if (Mout.Dim(0) != ChebOrder*FourierOrder*density_dof || Mout.Dim(1) != N) {
      Mout.ReInit(ChebOrder*FourierOrder*density_dof,N);
      Mout.SetZero();
    }

    Matrix<Real> Mtmp(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*density_dof*N);
    const Matrix<Real> Min_(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*FARFIELD_UPSAMPLE*density_dof*N, (Iterator<Real>)Min.begin(), false);
    if (FARFIELD_UPSAMPLE != 1) { // Appyl Mfourier // TODO: optimize
      const auto& Mfourier_ = Mfourier[FourierOrder];
      for (Long l = 0; l < ChebOrder*FARFIELD_UPSAMPLE; l++) {
        for (Long j0 = 0; j0 < FourierOrder; j0++) {
          for (Long k = 0; k < density_dof*N; k++) {
            Real f_tmp = 0;
            for (Long j1 = 0; j1 < FourierOrder*FARFIELD_UPSAMPLE; j1++) {
              f_tmp += Min_[l][j1*density_dof*N+k] * Mfourier_[j0][j1];
            }
            Mtmp[l][j0*density_dof*N+k] = f_tmp;
          }
        }
      }
    }else{
      Mtmp.ReInit(ChebOrder*FARFIELD_UPSAMPLE, FourierOrder*density_dof*N, (Iterator<Real>)Min.begin(), false);
    }

    Matrix<Real> Mout_(ChebOrder, FourierOrder*density_dof*N, Mout.begin(), false);
    Matrix<Real>::GEMM(Mout_, Mcheb[ChebOrder], Mtmp);
  }

  template <class Real> template <class Kernel> void SlenderElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    const auto& elem_lst = *dynamic_cast<const SlenderElemList*>(self);
    const Long Nelem = elem_lst.cheb_order.Dim();

    for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) { // Initialize quadrature tables
      const Integer ChebOrder = elem_lst.cheb_order[elem_idx];
      const Integer FourierOrder = elem_lst.fourier_order[elem_idx];
      const Integer FourierModes = FourierOrder/2+1;

      LogSingularityQuadRule<Real>(0);

      Matrix<Real> Mfourier;
      Vector<Real> nds_cos, nds_sin, wts;
      ToroidalSpecialQuadRule<Real,DefaultVecLen<Real>(),ModalUpsample,Kernel,false>(Mfourier, nds_cos, nds_sin, wts, FourierModes+ModalUpsample, (Real)1, 1); // TODO: replace by toroidal_greens_fn_batched

      Vector<Real> quad_nds, quad_wts;
      Matrix<Real> Minterp_quad_nds;
      if (trg_dot_prod) {
        SpecialQuadRule<ModalUpsample,Real,Kernel,true>(quad_nds, quad_wts, ChebOrder, 0, (Real)1, (Real)1, 1);
      } else {
        SpecialQuadRule<ModalUpsample,Real,Kernel,false>(quad_nds, quad_wts, ChebOrder, 0, (Real)1, (Real)1, 1);
      }
    }

    if (M_lst.Dim() != Nelem) M_lst.ReInit(Nelem);
    if (trg_dot_prod) {
      #pragma omp parallel for //schedule(static)
      for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
        if      (tol <= pow<15,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<15,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<14,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<14,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<13,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<13,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<12,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<12,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<11,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<11,true,Kernel>(ker, elem_idx);
        else if (tol <= pow<10,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<10,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 9,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 9,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 8,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 8,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 7,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 7,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 6,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 6,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 5,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 5,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 4,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 4,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 3,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 3,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 2,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 2,true,Kernel>(ker, elem_idx);
        else if (tol <= pow< 1,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 1,true,Kernel>(ker, elem_idx);
        else                                     M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 0,true,Kernel>(ker, elem_idx);
      }
    } else {
      #pragma omp parallel for //schedule(static)
      for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
        if      (tol <= pow<15,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<15,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<14,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<14,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<13,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<13,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<12,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<12,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<11,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<11,false,Kernel>(ker, elem_idx);
        else if (tol <= pow<10,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper<10,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 9,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 9,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 8,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 8,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 7,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 7,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 6,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 6,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 5,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 5,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 4,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 4,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 3,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 3,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 2,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 2,false,Kernel>(ker, elem_idx);
        else if (tol <= pow< 1,Real>((Real)0.1)) M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 1,false,Kernel>(ker, elem_idx);
        else                                     M_lst[elem_idx] = elem_lst.template SelfInteracHelper< 0,false,Kernel>(ker, elem_idx);
      }
    }
  }
  template <class Real> template <class Kernel> void SlenderElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    const auto& elem_lst = *dynamic_cast<const SlenderElemList*>(self);
    if (normal_trg.Dim()) {
      if      (tol <= pow<15,Real>((Real)0.1)) elem_lst.template NearInteracHelper<15,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<14,Real>((Real)0.1)) elem_lst.template NearInteracHelper<14,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<13,Real>((Real)0.1)) elem_lst.template NearInteracHelper<13,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<12,Real>((Real)0.1)) elem_lst.template NearInteracHelper<12,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<11,Real>((Real)0.1)) elem_lst.template NearInteracHelper<11,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<10,Real>((Real)0.1)) elem_lst.template NearInteracHelper<10,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 9,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 9,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 8,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 8,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 7,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 7,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 6,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 6,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 5,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 5,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 4,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 4,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 3,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 3,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 2,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 2,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 1,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 1,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else                                     elem_lst.template NearInteracHelper< 0,true,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
    } else {
      if      (tol <= pow<15,Real>((Real)0.1)) elem_lst.template NearInteracHelper<15,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<14,Real>((Real)0.1)) elem_lst.template NearInteracHelper<14,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<13,Real>((Real)0.1)) elem_lst.template NearInteracHelper<13,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<12,Real>((Real)0.1)) elem_lst.template NearInteracHelper<12,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<11,Real>((Real)0.1)) elem_lst.template NearInteracHelper<11,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow<10,Real>((Real)0.1)) elem_lst.template NearInteracHelper<10,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 9,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 9,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 8,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 8,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 7,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 7,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 6,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 6,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 5,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 5,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 4,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 4,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 3,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 3,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 2,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 2,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else if (tol <= pow< 1,Real>((Real)0.1)) elem_lst.template NearInteracHelper< 1,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
      else                                     elem_lst.template NearInteracHelper< 0,false,Kernel>(M, Xtrg, normal_trg, ker, elem_idx);
    }
  }
  template <class Real> template <Integer digits, bool trg_dot_prod, class Kernel> void SlenderElemList<Real>::NearInteracHelper(Matrix<Real>& M, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Long elem_idx) const {
    constexpr Integer MAX_THREADS=1000;
    constexpr Integer MAX_BUFF_SIZE=10000000;
    SCTL_ASSERT(omp_get_num_threads() < MAX_THREADS);
    static Vector<Vector<Real>> buff_(MAX_THREADS);
    Vector<Real>& buff = buff_[omp_get_thread_num()];
    if (buff.Dim() == 0) buff.ReInit(MAX_BUFF_SIZE);

    using Vec3 = Tensor<Real,true,COORD_DIM,1>;
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1 = Kernel::TrgDim()/(trg_dot_prod?COORD_DIM:1);

    constexpr double rho = (double)2.5;
    //const Integer digits = (Integer)(log(tol)/log(0.1)+0.5);
    static const Integer LegQuadOrder = (Integer)ceil(-log(((15.0*(rho*rho-1))/64.0)*(double)pow<digits,Real>((Real)0.1))/log(rho)*0.5+1);

    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];
    const Integer FourierModes = FourierOrder/2+1;
    const Matrix<Real>& M_fourier_inv = fourier_matrix_inv_transpose<Real>(FourierOrder,FourierModes);

    const Vector<Real>  coord(COORD_DIM*ChebOrder,(Iterator<Real>)this-> coord.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>     dx(COORD_DIM*ChebOrder,(Iterator<Real>)this->    dx.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>    d2x(COORD_DIM*ChebOrder,(Iterator<Real>)this->   d2x.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real> radius(        1*ChebOrder,(Iterator<Real>)this->radius.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     dr(        1*ChebOrder,(Iterator<Real>)this->    dr.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     e1(COORD_DIM*ChebOrder,(Iterator<Real>)this->    e1.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Real dx_max = [&dx,&ChebOrder](){
      Real dx2_max = 0;
      for (Long i = 0; i < ChebOrder; i++) {
        Real dx2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) {
          const Real dx_ = dx[k*ChebOrder+i];
          dx2 += dx_*dx_;
        }
        if (dx2 > dx2_max) dx2_max = dx2;
      }
      return sqrt<Real>(dx2_max);
    }();

    const Long Ntrg = Xtrg.Dim() / COORD_DIM;
    if (M.Dim(0) != ChebOrder*FourierOrder*KDIM0 || M.Dim(1) != Ntrg*KDIM1) {
      M.ReInit(ChebOrder*FourierOrder*KDIM0, Ntrg*KDIM1);
    }

    //#pragma omp parallel for
    for (Long i = 0; i < Ntrg; i++) {
      Long buff_offset = 0;
      const Vec3 Xt((Iterator<Real>)Xtrg.begin()+i*COORD_DIM);
      const Vec3 n_trg = (trg_dot_prod ? Vec3((Iterator<Real>)normal_trg.begin()+i*COORD_DIM) : Vec3((Real)0));

      Matrix<Real> M_modal;
      if (MAX_BUFF_SIZE-buff_offset >= ChebOrder * KDIM0*KDIM1*FourierModes*2) {
        M_modal.ReInit(ChebOrder, KDIM0*KDIM1*FourierModes*2, buff.begin()+buff_offset, false);
        buff_offset += ChebOrder * KDIM0*KDIM1*FourierModes*2;
      } else {
        M_modal.ReInit(ChebOrder, KDIM0*KDIM1*FourierModes*2);
      }
      { // Set M_modal
        Vector<Real> quad_nds, quad_wts; // Quadrature rule in s
        auto adap_quad_rule = [&ChebOrder,&radius,&dr,&coord,&dx,&d2x,&dx_max,&buff,&buff_offset,&MAX_BUFF_SIZE](Vector<Real>& quad_nds, Vector<Real>& quad_wts, const Vec3& x_trg) {
          const auto& leg_nds = LegendreQuadRule<Real>(LegQuadOrder).first;
          const auto& leg_wts = LegendreQuadRule<Real>(LegQuadOrder).second;
          auto adap_ref = [&leg_nds,&leg_wts](Vector<Real>& nds, Vector<Real>& wts, Real a, Real b, Integer levels) {
            if (nds.Dim() != levels * LegQuadOrder) nds.ReInit(levels*LegQuadOrder);
            if (wts.Dim() != levels * LegQuadOrder) wts.ReInit(levels*LegQuadOrder);
            Vector<Real> nds_(nds.Dim(), nds.begin(), false);
            Vector<Real> wts_(wts.Dim(), wts.begin(), false);

            while (levels) {
              Vector<Real> nds0(LegQuadOrder, nds_.begin(), false);
              Vector<Real> wts0(LegQuadOrder, wts_.begin(), false);
              Vector<Real> nds1((levels-1)*LegQuadOrder, nds_.begin()+LegQuadOrder, false);
              Vector<Real> wts1((levels-1)*LegQuadOrder, wts_.begin()+LegQuadOrder, false);

              const Real end_point = (levels==1 ? b : (a+b)*0.5);
              const Real panel_scal = (end_point-a);
              const Real panel_scal_abs = fabs<Real>(end_point-a);
              for (Long i = 0; i < LegQuadOrder; i++) {
                nds0[i] = leg_nds[i] * panel_scal + a;
                wts0[i] = leg_wts[i] * panel_scal_abs;
              }

              nds_.Swap(nds1);
              wts_.Swap(wts1);
              a = end_point;
              levels--;
            }
          };

          // TODO: develop special quadrature rule instead of adaptive integration
          if (0) { // dyadic refinement on element ends
            const Integer levels = 6;
            quad_nds.ReInit(2*levels*LegQuadOrder);
            quad_wts.ReInit(2*levels*LegQuadOrder);
            Vector<Real> nds0(levels*LegQuadOrder,quad_nds.begin(),false);
            Vector<Real> wts0(levels*LegQuadOrder,quad_wts.begin(),false);
            Vector<Real> nds1(levels*LegQuadOrder,quad_nds.begin()+levels*LegQuadOrder,false);
            Vector<Real> wts1(levels*LegQuadOrder,quad_wts.begin()+levels*LegQuadOrder,false);
            adap_ref(nds0, wts0, 0.5, 0.0, levels);
            adap_ref(nds1, wts1, 0.5, 1.0, levels);
          }
          if (0) { // dyadic refinement near target point
            Real dist_min, s_min, dxds;
            { // Set dist_min, s_min, dxds
              auto get_dist = [&ChebOrder,&radius,&coord,&dx] (const Vec3& x_trg, Real s) -> Real {
                StaticArray<Real,20> buff;
                Vector<Real> interp_wts(ChebOrder, buff, false);
                if (ChebOrder > 20) interp_wts.ReInit(ChebOrder);
                LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(ChebOrder), Vector<Real>(1,Ptr2Itr<Real>(&s,1),false));

                Real r0 = 0;
                Vec3 x0, dx_ds0;
                for (Long i = 0; i < COORD_DIM; i++) {
                  x0(i,0) = 0;
                  dx_ds0(i,0) = 0;
                }
                for (Long i = 0; i < ChebOrder; i++) {
                  r0 += radius[i] * interp_wts[i];
                  x0(0,0) += coord[0*ChebOrder+i] * interp_wts[i];
                  x0(1,0) += coord[1*ChebOrder+i] * interp_wts[i];
                  x0(2,0) += coord[2*ChebOrder+i] * interp_wts[i];
                  dx_ds0(0,0) += dx[0*ChebOrder+i] * interp_wts[i];
                  dx_ds0(1,0) += dx[1*ChebOrder+i] * interp_wts[i];
                  dx_ds0(2,0) += dx[2*ChebOrder+i] * interp_wts[i];
                }
                Vec3 dx = x0 - x_trg;
                Vec3 n0 = dx_ds0 * sqrt<Real>(1/dot_prod(dx_ds0, dx_ds0));
                Real dz = dot_prod(dx, n0);
                Vec3 dr = dx - n0*dz;
                Real dR = sqrt<Real>(dot_prod(dr,dr)) - r0;
                return sqrt<Real>(dR*dR + dz*dz);
              };
              const auto bin_search = [&get_dist](Real& s_min, Real& dist_min, const Vec3& x_trg) {
                StaticArray<Real,2> dist;
                StaticArray<Real,2> s_val{0,1};
                dist[0] = get_dist(x_trg, s_val[0]);
                dist[1] = get_dist(x_trg, s_val[1]);
                for (Long i = 0; i < 90; i++) { // Binary search: set dist, s_val
                  Real ss0 = (s_val[0]*2 + s_val[1])/3;
                  Real ss1 = (s_val[0] + s_val[1]*2)/3;
                  Real dd0 = get_dist(x_trg, ss0);
                  Real dd1 = get_dist(x_trg, ss1);
                  if (dd0 > dd1) {
                    dist[0] = dd0;
                    s_val[0] = ss0;
                  } else {
                    dist[1] = dd1;
                    s_val[1] = ss1;
                  }
                }
                if (dist[0] < dist[1]) { // Set dis_min, s_min
                  dist_min = dist[0];
                  s_min = s_val[0];
                } else {
                  dist_min = dist[1];
                  s_min = s_val[1];
                }
              };

              const auto newton_iter_step = [&ChebOrder,&radius,&dr,&coord,&dx,&d2x](Real& ds, Real& dist2, Real& dyds2, const Real s, const Vec3& x_trg) {
                StaticArray<Real,20> buff;
                Vector<Real> interp_wts(ChebOrder, buff, false);
                if (ChebOrder > 20) interp_wts.ReInit(ChebOrder);
                LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(ChebOrder), Vector<Real>(1,(Iterator<Real>)Ptr2ConstItr<Real>(&s,1),false));

                Vec3 x0, dx0, d2x0;
                Real r0 = 0, dr0 = 0;
                for (Long i = 0; i < COORD_DIM; i++) {
                  x0(i,0) = 0;
                  dx0(i,0) = 0;
                  d2x0(i,0) = 0;
                }
                for (Long i = 0; i < ChebOrder; i++) {
                  x0(0,0) += coord[0*ChebOrder+i] * interp_wts[i];
                  x0(1,0) += coord[1*ChebOrder+i] * interp_wts[i];
                  x0(2,0) += coord[2*ChebOrder+i] * interp_wts[i];
                  dx0(0,0) += dx[0*ChebOrder+i] * interp_wts[i];
                  dx0(1,0) += dx[1*ChebOrder+i] * interp_wts[i];
                  dx0(2,0) += dx[2*ChebOrder+i] * interp_wts[i];
                  d2x0(0,0) += d2x[0*ChebOrder+i] * interp_wts[i];
                  d2x0(1,0) += d2x[1*ChebOrder+i] * interp_wts[i];
                  d2x0(2,0) += d2x[2*ChebOrder+i] * interp_wts[i];
                  r0 += radius[i] * interp_wts[i];
                  dr0 += dr[i] * interp_wts[i];
                }

                Vec3 n0, dy;
                { // Set n0, dy
                  const Vec3 Xt_X0 = x_trg - x0;
                  n0 = -cross_prod(cross_prod(Xt_X0, dx0), dx0);
                  Real scal = (1/sqrt<Real>(dot_prod(n0,n0)));
                  n0 = n0 * scal;
                  Vec3 dn0 = -(cross_prod(cross_prod(Xt_X0, d2x0), dx0) + cross_prod(cross_prod(Xt_X0, dx0), d2x0)) * scal;
                  dn0 = dn0 - n0 * dot_prod(dn0,n0);
                  dy = dx0 + n0 * dr0 + dn0 * r0;
                }

                const Vec3 y_Xt = x0 + n0 * r0 - x_trg;
                dyds2 = dot_prod(dy, dy);
                dist2 = dot_prod(y_Xt, y_Xt);
                ds = dot_prod(y_Xt,dy)/dot_prod(dy,dy);
              };
              const auto newton_iter = [&newton_iter_step,&bin_search](Real& s0, Real& dist0, const Vec3& x_trg) {
                static const Real eps_sqrt = sqrt<Real>(machine_eps<Real>());
                constexpr Integer max_iter = 100;
                const Real tol2 = 1e-4;

                Real dyds2;
                Real d2, d2_;
                Real ds, ds_;
                newton_iter_step(ds_, d2_, dyds2, 0, x_trg);
                newton_iter_step(ds, d2, dyds2, 1, x_trg);
                if (d2_ < d2) {
                  if (ds_ > 0) {
                    s0 = 0;
                    dist0 = sqrt<Real>(d2_);
                    return;
                  }
                } else {
                  if (ds < 0) {
                    s0 = 1;
                    dist0 = sqrt<Real>(d2);
                    return;
                  }
                }

                Real s, s_ = (Real)0.5;
                Real s_min = (Real)0, s_max = (Real)1;
                newton_iter_step(ds_, d2_, dyds2, s_, x_trg);
                s = std::min(s_max, std::max<Real>(s_min, s_ - ds_));
                for (Integer iter = 0; iter < max_iter; iter++) {
                  newton_iter_step(ds, d2, dyds2, s, x_trg);
                  if ((ds*ds)*dyds2 < tol2*d2 || (s==0 && ds>0) || (s==1 && ds<0)) break;
                  if (d2 < d2_ || (d2 < d2_*(1+eps_sqrt) && fabs(ds)<fabs(ds_))) {
                    if (ds > 0) s_max = s;
                    else s_min = s;
                    SCTL_ASSERT(s_min < s_max);

                    Real scal = std::min<Real>((Real)10.0, (ds!=ds_ ? (s-s_)/(ds-ds_) : (Real)1));
                    if (scal <= 0.1) scal = (Real)1;
                    Real s__ = s - ds * scal;

                    ds_ = ds;
                    d2_ = d2;
                    s_ = s;

                    s = std::min((Real)1, std::max<Real>((Real)0, s__));
                    if (s < s_min || s > s_max) s = (s_min + s_max) * (Real)0.5;
                  } else {
                    s = (s+s_)*0.5;
                  }
                  if (iter == max_iter-1) {
                    SCTL_WARN("Newton iterations failed to converge");
                    bin_search(s0, dist0, x_trg);
                  }
                }

                s0 = s;
                dist0 = sqrt<Real>(d2);
              };
              newton_iter(s_min, dist_min, x_trg);
              //bin_search(s_min, dist_min, x_trg);

              { // Set dx_ds;
                StaticArray<Real,20> buff;
                Vector<Real> interp_wts(ChebOrder, buff, false);
                if (ChebOrder > 20) interp_wts.ReInit(ChebOrder);
                LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(ChebOrder), Vector<Real>(1,Ptr2Itr<Real>(&s_min,1),false));

                Vec3 dxds_vec;
                for (Long i = 0; i < COORD_DIM; i++) {
                  dxds_vec(i,0) = 0;
                }
                for (Long i = 0; i < ChebOrder; i++) {
                  dxds_vec(0,0) += dx[0*ChebOrder+i] * interp_wts[i];
                  dxds_vec(1,0) += dx[1*ChebOrder+i] * interp_wts[i];
                  dxds_vec(2,0) += dx[2*ChebOrder+i] * interp_wts[i];
                }
                dxds = sqrt<Real>(dot_prod(dxds_vec,dxds_vec))*const_pi<Real>()/2;
              }
            }

            const Real h0 =   (s_min)*dxds;
            const Real h1 = (1-s_min)*dxds;
            static const double log2_inv = 1/log(2.0);
            const Integer adap_levels0 = (s_min==0 ? 0 : std::max<Integer>(0,(Integer)ceil(log((double)(h0/dist_min))*log2_inv))+1);
            const Integer adap_levels1 = (s_min==1 ? 0 : std::max<Integer>(0,(Integer)ceil(log((double)(h1/dist_min))*log2_inv))+1);

            Long N0 = adap_levels0 * LegQuadOrder;
            Long N1 = adap_levels1 * LegQuadOrder;
            if (MAX_BUFF_SIZE-buff_offset >= 2*(N0+N1)) {
              quad_nds.ReInit(N0+N1, buff.begin()+buff_offset+0*(N0+N1), false);
              quad_wts.ReInit(N0+N1, buff.begin()+buff_offset+1*(N0+N1), false);
              buff_offset += 2*(N0+N1);
            } else {
              quad_nds.ReInit(N0+N1);
              quad_wts.ReInit(N0+N1);
            }
            Vector<Real> nds0(N0, quad_nds.begin(), false);
            Vector<Real> wts0(N0, quad_wts.begin(), false);
            Vector<Real> nds1(N1, quad_nds.begin()+N0, false);
            Vector<Real> wts1(N1, quad_wts.begin()+N0, false);
            adap_ref(nds0, wts0, 0, s_min, adap_levels0);
            adap_ref(nds1, wts1, 1, s_min, adap_levels1);
          }
          if (1) { // adaptive refinement
            Long Npanel = 0;
            Vector<Real> s_vec;
            { // Set s_vec, Npanel (7% - 15% of near interaction time)
              const auto get_geom = [&ChebOrder,&radius,&dr,&coord,&dx,&d2x](Vec3& y, Real& dist_xt, Real& dyds, const Real s, const Vec3& x_trg, const Vec3& yy) {
                StaticArray<Real,20> buff;
                Vector<Real> interp_wts(ChebOrder, buff, false);
                if (ChebOrder > 20) interp_wts.ReInit(ChebOrder);
                LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(ChebOrder), Vector<Real>(1,(Iterator<Real>)Ptr2ConstItr<Real>(&s,1),false));

                Vec3 x0, dx0, d2x0;
                Real r0 = 0, dr0 = 0;
                for (Long i = 0; i < COORD_DIM; i++) {
                  x0(i,0) = -x_trg(i,0);
                  dx0(i,0) = 0;
                  d2x0(i,0) = 0;
                }
                for (Long i = 0; i < ChebOrder; i++) {
                  x0(0,0) += coord[0*ChebOrder+i] * interp_wts[i];
                  x0(1,0) += coord[1*ChebOrder+i] * interp_wts[i];
                  x0(2,0) += coord[2*ChebOrder+i] * interp_wts[i];
                  dx0(0,0) += dx[0*ChebOrder+i] * interp_wts[i];
                  dx0(1,0) += dx[1*ChebOrder+i] * interp_wts[i];
                  dx0(2,0) += dx[2*ChebOrder+i] * interp_wts[i];
                  d2x0(0,0) += d2x[0*ChebOrder+i] * interp_wts[i];
                  d2x0(1,0) += d2x[1*ChebOrder+i] * interp_wts[i];
                  d2x0(2,0) += d2x[2*ChebOrder+i] * interp_wts[i];
                  r0 += radius[i] * interp_wts[i];
                  dr0 += dr[i] * interp_wts[i];
                }

                Vec3 n0, dy;
                { // Set n0, dy
                  n0 = cross_prod(cross_prod(x0, dx0), dx0);
                  Real scal = (1/sqrt<Real>(dot_prod(n0,n0)));
                  n0 = n0 * scal;
                  Vec3 dn0 = (cross_prod(cross_prod(x0, d2x0), dx0) + cross_prod(cross_prod(x0, dx0), d2x0)) * scal;
                  dn0 = dn0 - n0 * dot_prod(dn0,n0);
                  dy = dx0 + n0 * dr0 + dn0 * r0;
                }

                y = x0 + n0 * r0;
                //ds = dot_prod(y,dy)/dot_prod(dy,dy);
                dist_xt = sqrt<Real>(dot_prod(y, y));
                dyds = sqrt<Real>(dot_prod(dy, dy));

                const Vec3 nn = cross_prod(cross_prod(x0-yy, dx0), dx0);
                const Vec3 y_yy = (x0-yy) +  nn * ((1/sqrt<Real>(dot_prod(nn,nn))) * r0);
                return sqrt<Real>(dot_prod(y_yy, y_yy));
              };

              constexpr Long MaxPanels = 1000, max_iter = 2000;
              if (MAX_BUFF_SIZE-buff_offset >= MaxPanels+1) { // Allocate s_vec
                s_vec.ReInit(MaxPanels+1, buff.begin()+buff_offset, false);
                buff_offset += MaxPanels+1;
              } else {
                s_vec.ReInit(MaxPanels+1);
              }

              Vec3 y_;
              s_vec[0] = 0;
              Real dist_, dyds_, s_ = 0;
              get_geom(y_, dist_, dyds_, s_, x_trg, x_trg);
              Real step_size = (2/(rho+1/rho))*(3*dist_/std::max<Real>(dyds_,dx_max));
              for (Long iter = 0; iter < max_iter; iter++) {
                Vec3 y;
                Real dist, dyds;
                const Real s = std::min<Real>(1, s_+step_size);
                const Real panel_len = get_geom(y, dist, dyds, s, x_trg, y_);
                if (dist+dist_ > ((rho+1/rho)/2) * std::max<Real>(std::max<Real>(dx_max,std::max<Real>(dyds,dyds_))*(s-s_), panel_len)) {
                  Npanel++;
                  SCTL_ASSERT(Npanel <= MaxPanels);
                  s_vec[Npanel] = s;

                  s_ = s;
                  y_ = y;
                  dist_ = dist;
                  dyds_ = dyds;
                  step_size = (2/(rho+1/rho))*(3*dist_/std::max<Real>(dyds_,dx_max));
                } else {
                  step_size = (s-s_) * std::max<Real>(0.5, 0.9 * (dist+dist_)*(2/(rho+1/rho))/std::max<Real>(std::max<Real>(dx_max,std::max<Real>(dyds,dyds_))*(s-s_), panel_len));
                }
                if (s_ == 1) {
                  //if (1) { // display iteration count
                  //  static Long max_iter = 0;
                  //  if (iter > max_iter) {
                  //    std::cout<<iter<<' '<<Npanel<<'\n';
                  //    max_iter = iter;
                  //  }
                  //}
                  break;
                }
              }
              if (s_vec[Npanel] < 1) {
                SCTL_WARN("Adaptive refinement failed");
              }
            }

            const Long N = Npanel * LegQuadOrder;
            if (MAX_BUFF_SIZE-buff_offset >= 2*N) { // Allocate quad_nds, quad_wts
              quad_nds.ReInit(N, buff.begin()+buff_offset+0*N, false);
              quad_wts.ReInit(N, buff.begin()+buff_offset+1*N, false);
              buff_offset += 2*N;
            } else {
              quad_nds.ReInit(N);
              quad_wts.ReInit(N);
            }
            for (Long j = 0; j < Npanel; j++) { // Set quad_nds, quad_wts
              for (Long k = 0; k < LegQuadOrder; k++) {
                quad_nds[j*LegQuadOrder+k] = s_vec[j] + (s_vec[j+1]-s_vec[j]) * leg_nds[k];
                quad_wts[j*LegQuadOrder+k] = (s_vec[j+1]-s_vec[j]) * leg_wts[k];
              }
            }
          }
        };
        adap_quad_rule(quad_nds, quad_wts, Xt);

        Matrix<Real> Minterp_quad_nds;
        { // Set Minterp_quad_nds
          if (MAX_BUFF_SIZE-buff_offset >= ChebOrder*quad_nds.Dim()) {
            Minterp_quad_nds.ReInit(ChebOrder, quad_nds.Dim(), buff.begin()+buff_offset, false);
            buff_offset += ChebOrder*quad_nds.Dim();
          } else {
            Minterp_quad_nds.ReInit(ChebOrder, quad_nds.Dim());
          }
          Vector<Real> Vinterp_quad_nds(ChebOrder*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
          LagrangeInterp<Real>::Interpolate(Vinterp_quad_nds, CenterlineNodes(ChebOrder), quad_nds);
        }

        Vec3 x_trg = Xt;
        Matrix<Real> r_src, dr_src, x_src, dx_src, d2x_src, e1_src;
        if (MAX_BUFF_SIZE-buff_offset >= 14*quad_nds.Dim()) {
          r_src  .ReInit(        1,quad_nds.Dim(), buff.begin()+buff_offset+ 0*quad_nds.Dim(), false);
          dr_src .ReInit(        1,quad_nds.Dim(), buff.begin()+buff_offset+ 1*quad_nds.Dim(), false);
          x_src  .ReInit(COORD_DIM,quad_nds.Dim(), buff.begin()+buff_offset+ 2*quad_nds.Dim(), false);
          dx_src .ReInit(COORD_DIM,quad_nds.Dim(), buff.begin()+buff_offset+ 5*quad_nds.Dim(), false);
          d2x_src.ReInit(COORD_DIM,quad_nds.Dim(), buff.begin()+buff_offset+ 8*quad_nds.Dim(), false);
          e1_src .ReInit(COORD_DIM,quad_nds.Dim(), buff.begin()+buff_offset+11*quad_nds.Dim(), false);
          buff_offset += 14*quad_nds.Dim();
        } else {
          r_src  .ReInit(        1,quad_nds.Dim());
          dr_src .ReInit(        1,quad_nds.Dim());
          x_src  .ReInit(COORD_DIM,quad_nds.Dim());
          dx_src .ReInit(COORD_DIM,quad_nds.Dim());
          d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
          e1_src .ReInit(COORD_DIM,quad_nds.Dim());
        }
        { // Set x_src, x_trg (improve numerical stability)
          Matrix<Real> x_nodes;
          StaticArray<Real,30> buff;
          if (COORD_DIM*ChebOrder <= 30) {
            x_nodes.ReInit(COORD_DIM,ChebOrder, buff, false);
          } else {
            x_nodes.ReInit(COORD_DIM,ChebOrder);
          }
          for (Integer k = 0; k < COORD_DIM; k++) {
            for (Long j = 0; j < ChebOrder; j++) {
              x_nodes[k][j] = coord[k*ChebOrder+j] - x_trg(k,0);
            }
          }
          Matrix<Real>::GEMM(  x_src, x_nodes, Minterp_quad_nds);
          for (Integer k = 0; k < COORD_DIM; k++) {
            x_trg(k,0) = 0;
          }
        }
        //Matrix<Real>::GEMM(  x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>) coord.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM( dx_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    dx.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM(d2x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)   d2x.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM(  r_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)radius.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM( dr_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)    dr.begin(),false), Minterp_quad_nds);
        Matrix<Real>::GEMM( e1_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    e1.begin(),false), Minterp_quad_nds);
        for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
          Vec3 e1, dx;
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1(k,0) = e1_src[k][j];
            dx(k,0) = dx_src[k][j];
          }
          e1 = e1 - dx * dot_prod(e1, dx) * (1/dot_prod(dx,dx));
          e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

          for (Integer k = 0; k < COORD_DIM; k++) {
            e1_src[k][j] = e1(k,0);
          }
        }

        const Vec3 y_trg = x_trg;
        Matrix<Real> M_tor;
        if (MAX_BUFF_SIZE-buff_offset >= quad_nds.Dim()*KDIM0*KDIM1*FourierModes*2) {
          M_tor.ReInit(quad_nds.Dim(), KDIM0*KDIM1*FourierModes*2, buff.begin()+buff_offset, false);
          buff_offset += quad_nds.Dim()*KDIM0*KDIM1*FourierModes*2;
        } else {
          M_tor.ReInit(quad_nds.Dim(), KDIM0*KDIM1*FourierModes*2);
        }
        toroidal_greens_fn_batched<digits+1,ModalUpsample,trg_dot_prod>(M_tor, y_trg, Vec3((Real)1), (Real)0, n_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, ker, FourierModes);

        for (Long ii = 0; ii < M_tor.Dim(0); ii++) {
          for (Long jj = 0; jj < M_tor.Dim(1); jj++) {
            M_tor[ii][jj] *= quad_wts[ii];
          }
        }
        Matrix<Real>::GEMM(M_modal, Minterp_quad_nds, M_tor);
      }

      Matrix<Real> M_nodal;
      if (MAX_BUFF_SIZE-buff_offset >= ChebOrder * KDIM0*KDIM1*FourierOrder) {
        M_nodal.ReInit(ChebOrder, KDIM0*KDIM1*FourierOrder, buff.begin()+buff_offset, false);
        buff_offset += ChebOrder * KDIM0*KDIM1*FourierOrder;
      } else {
        M_nodal.ReInit(ChebOrder, KDIM0*KDIM1*FourierOrder);
      }
      { // Set M_nodal
        Matrix<Real> M_nodal_(ChebOrder*KDIM0*KDIM1, FourierOrder, M_nodal.begin(), false);
        const Matrix<Real> M_modal_(ChebOrder*KDIM0*KDIM1, FourierModes*2, M_modal.begin(), false);
        Matrix<Real>::GEMM(M_nodal_, M_modal_, M_fourier_inv);
      }

      { // Set M
        for (Integer i0 = 0; i0 < ChebOrder; i0++) {
          for (Integer i1 = 0; i1 < FourierOrder; i1++) {
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
              for (Integer k1 = 0; k1 < KDIM1; k1++) {
                M[(i0*FourierOrder+i1)*KDIM0+k0][i*KDIM1+k1] = M_nodal[i0][(k0*KDIM1+k1)*FourierOrder+i1] * ker.template uKerScaleFactor<Real>();
              }
            }
          }
        }
      }
    }
  }

  template <class Real> const Vector<Real>& SlenderElemList<Real>::CenterlineNodes(Integer Order) {
    return ChebQuadRule<Real>::nds(Order);
  }

  template <class Real> void SlenderElemList<Real>::Write(const std::string& fname, const Comm& comm) const {
    auto allgather = [&comm](Vector<Real>& v_out, const Vector<Real>& v_in) {
      const Long Nproc = comm.Size();
      StaticArray<Long,1> len{v_in.Dim()};
      Vector<Long> cnt(Nproc), dsp(Nproc);
      comm.Allgather(len+0, 1, cnt.begin(), 1); dsp = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), Nproc);

      v_out.ReInit(dsp[Nproc-1]+cnt[Nproc-1]);
      comm.Allgatherv(v_in.begin(), v_in.Dim(), v_out.begin(), cnt.begin(), dsp.begin());
    };
    auto allgatherl = [&comm](Vector<Long>& v_out, const Vector<Long>& v_in) {
      const Long Nproc = comm.Size();
      StaticArray<Long,1> len{v_in.Dim()};
      Vector<Long> cnt(Nproc), dsp(Nproc);
      comm.Allgather(len+0, 1, cnt.begin(), 1); dsp = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), Nproc);

      v_out.ReInit(dsp[Nproc-1]+cnt[Nproc-1]);
      comm.Allgatherv(v_in.begin(), v_in.Dim(), v_out.begin(), cnt.begin(), dsp.begin());
    };

    Vector<Real> radius_, coord_, e1_;
    allgather(radius_, radius);
    allgather( coord_,  coord);
    allgather(    e1_,     e1);

    Vector<Long> cheb_order_, elem_dsp_, fourier_order_;
    allgatherl(   cheb_order_, cheb_order   );
    allgatherl(fourier_order_, fourier_order);
    elem_dsp_.ReInit(cheb_order_.Dim()); elem_dsp_ = 0;
    omp_par::scan(cheb_order_.begin(), elem_dsp_.begin(), cheb_order_.Dim());

    if (comm.Rank()) return;
    const Integer precision = (Integer)ceil(-log((double)machine_eps<Real>())/log((double)10));
    const Integer width = precision + 8;
    std::ofstream file;
    file.open(fname, std::ofstream::out | std::ofstream::trunc);
    if (!file.good()) {
      std::cout << "Unable to open file for writing:" << fname << '\n';
    }

    // Header
    file<<"#";
    file<<std::setw(width-1)<<"X";
    file<<std::setw(width)<<"Y";
    file<<std::setw(width)<<"Z";
    file<<std::setw(width)<<"r";
    file<<std::setw(width)<<"orient-x";
    file<<std::setw(width)<<"orient-y";
    file<<std::setw(width)<<"orient-z";
    file<<std::setw(width)<<"ChebOrder";
    file<<std::setw(width)<<"FourierOrder";
    file<<'\n';

    file<<std::scientific<<std::setprecision(precision);
    for (Long i = 0; i < cheb_order_.Dim(); i++) {
      for (Long j = 0; j < cheb_order_[i]; j++) {
        for (Integer k = 0; k < COORD_DIM; k++) {
          file<<std::setw(width)<<coord_[elem_dsp_[i]*COORD_DIM + k*cheb_order_[i]+j];
        }
        file<<std::setw(width)<<radius_[elem_dsp_[i] + j];
        for (Integer k = 0; k < COORD_DIM; k++) {
          file<<std::setw(width)<<e1_[elem_dsp_[i]*COORD_DIM + k*cheb_order_[i]+j];
        }
        if (!j) {
          file<<std::setw(width)<<cheb_order_[i];
          file<<std::setw(width)<<fourier_order_[i];
        }
        file<<"\n";
      }
    }
    file.close();
  }
  template <class Real> template <class ValueType> void SlenderElemList<Real>::Read(const std::string& fname, const Comm& comm) {
    std::ifstream file;
    file.open(fname, std::ifstream::in);
    if (!file.good()) {
      std::cout << "Unable to open file for reading:" << fname << '\n';
    }

    std::string line;
    Vector<ValueType> coord_, radius_, e1_;
    Vector<Long> cheb_order_, fourier_order_;
    while (std::getline(file, line)) { // Set coord_, radius_, e1_, cheb_order_, fourier_order_
      size_t first_char_pos = line.find_first_not_of(' ');
      if (first_char_pos == std::string::npos || line[first_char_pos] == '#') continue;

      std::istringstream iss(line);
      for (Integer k = 0; k < COORD_DIM; k++) { // read coord_
        ValueType a;
        iss>>a;
        SCTL_ASSERT(!iss.fail());
        coord_.PushBack(a);
      }
      { // read radius_
        ValueType a;
        iss>>a;
        SCTL_ASSERT(!iss.fail());
        radius_.PushBack(a);
      }
      for (Integer k = 0; k < COORD_DIM; k++) { // read e1_
        ValueType a;
        iss>>a;
        SCTL_ASSERT(!iss.fail());
        e1_.PushBack(a);
      }

      Integer ChebOrder, FourierOrder;
      if (iss>>ChebOrder>>FourierOrder) {
        cheb_order_.PushBack(ChebOrder);
        fourier_order_.PushBack(FourierOrder);
      } else {
        cheb_order_.PushBack(-1);
        fourier_order_.PushBack(-1);
      }
    }
    file.close();

    Long offset = 0;
    Vector<Long> cheb_order__, fourier_order__;
    while (offset < cheb_order_.Dim()) { // Set cheb_order__, fourier_order__
      Integer ChebOrder = cheb_order_[offset];
      Integer FourierOrder = fourier_order_[offset];
      for (Integer j = 1; j < ChebOrder; j++) {
        SCTL_ASSERT(cheb_order_[offset+j] == ChebOrder || cheb_order_[offset+j] == -1);
        SCTL_ASSERT(fourier_order_[offset+j] == FourierOrder || fourier_order_[offset+j] == -1);
      }
      cheb_order__.PushBack(ChebOrder);
      fourier_order__.PushBack(FourierOrder);
      offset += ChebOrder;
    }
    { // Distribute across processes and init SlenderElemList
      const Long Np = comm.Size();
      const Long pid = comm.Rank();
      const Long Nelem = cheb_order__.Dim();

      const Long i0 = Nelem*(pid+0)/Np;
      const Long i1 = Nelem*(pid+1)/Np;

      Vector<Long> cheb_order, fourier_order;
      cheb_order.ReInit(i1-i0, cheb_order__.begin()+i0, false);
      fourier_order.ReInit(i1-i0, fourier_order__.begin()+i0, false);

      Vector<Long> elem_offset(Nelem+1); elem_offset = 0;
      omp_par::scan(cheb_order__.begin(), elem_offset.begin(), Nelem);
      elem_offset[Nelem] = (Nelem ? elem_offset[Nelem-1] + cheb_order__[Nelem-1] : 0);
      const Long j0 = elem_offset[i0];
      const Long j1 = elem_offset[i1];

      Vector<ValueType> radius, coord, e1;
      radius.ReInit((j1-j0), radius_.begin()+j0, false);
      coord.ReInit((j1-j0)*COORD_DIM, coord_.begin()+j0*COORD_DIM, false);
      if (e1_.Dim()) e1.ReInit((j1-j0)*COORD_DIM, e1_.begin()+j0*COORD_DIM, false);

      Init<ValueType>(cheb_order, fourier_order, coord, radius, e1);
    }
  }

  template <class Real> void SlenderElemList<Real>::GetVTUData(VTUData& vtu_data, const Vector<Real>& F, const Long elem_idx) const {
    if (elem_idx == -1) {
      const Long Nelem = cheb_order.Dim();
      Long dof = 0, offset = 0;
      if (F.Dim()) { // Set dof
        Long Nnodes = 0;
        for (Long i = 0; i < Nelem; i++) {
          Nnodes += cheb_order[i] * fourier_order[i];
        }
        dof = F.Dim() / Nnodes;
        SCTL_ASSERT(F.Dim() == Nnodes * dof);
      }
      for (Long i = 0; i < Nelem; i++) {
        const Vector<Real> F_(cheb_order[i]*fourier_order[i]*dof, (Iterator<Real>)F.begin()+offset, false);
        GetVTUData(vtu_data, F_, i);
        offset += F_.Dim();
      }
      return;
    }

    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];
    Vector<Real> X, s_nodes(ChebOrder+2);
    s_nodes[0] = 0;
    s_nodes[ChebOrder+1] = 1;
    Vector<Real>(ChebOrder, s_nodes.begin()+1, false) = CenterlineNodes(ChebOrder);
    GetGeom(&X,nullptr,nullptr,nullptr,nullptr, s_nodes, sin_theta<Real>(FourierOrder), cos_theta<Real>(FourierOrder), elem_idx);

    Vector<Real> F_(F.Dim()/ChebOrder*s_nodes.Dim());
    if (F.Dim()) {
      Matrix<Real> M(ChebOrder, s_nodes.Dim());
      Vector<Real> M_(ChebOrder*s_nodes.Dim(), M.begin(), false);
      LagrangeInterp<Real>::Interpolate(M_, CenterlineNodes(ChebOrder), s_nodes);

      const Matrix<Real> Mf(ChebOrder, F.Dim()/ChebOrder, (Iterator<Real>)F.begin(), false);
      Matrix<Real> Mf_(s_nodes.Dim(), F_.Dim()/s_nodes.Dim(), F_.begin(), false);
      Mf_ = M.Transpose() * Mf;
    }

    Long point_offset = vtu_data.coord.Dim() / COORD_DIM;
    for (const auto& x : X) vtu_data.coord.PushBack((VTUData::VTKReal)x);
    for (const auto& f : F_) vtu_data.value.PushBack((VTUData::VTKReal)f);
    for (Long i = 0; i < s_nodes.Dim()-1; i++) {
      for (Long j = 0; j <= FourierOrder; j++) {
        vtu_data.connect.PushBack(point_offset + (i+0)*FourierOrder+(j%FourierOrder));
        vtu_data.connect.PushBack(point_offset + (i+1)*FourierOrder+(j%FourierOrder));
      }
      vtu_data.offset.PushBack(vtu_data.connect.Dim());
      vtu_data.types.PushBack(6);
    }
  }
  template <class Real> void SlenderElemList<Real>::WriteVTK(const std::string& fname, const Vector<Real>& F, const Comm& comm) const {
    VTUData vtu_data;
    GetVTUData(vtu_data, F);
    vtu_data.WriteVTK(fname, comm);
  }

  template <class Real> template <class Kernel> void SlenderElemList<Real>::test(const Comm& comm, Real tol) {
    sctl::Profile::Enable(false);
    const Long pid = comm.Rank();
    const Long Np = comm.Size();

    SlenderElemList<Real> elem_lst0;
    //elem_lst0.Read("data/geom.data"); // Read geometry from file
    if (1) { // Initialize elem_lst0 in code
      const Long Nelem = 16;
      const Long ChebOrder = 10;
      const Long FourierOrder = 8;

      Vector<Real> coord, radius;
      Vector<Long> cheb_order, fourier_order;
      const Long k0 = (Nelem*(pid+0))/Np;
      const Long k1 = (Nelem*(pid+1))/Np;
      for (Long k = k0; k < k1; k++) {
        cheb_order.PushBack(ChebOrder);
        fourier_order.PushBack(FourierOrder);
        const auto& nds = CenterlineNodes(ChebOrder);
        for (Long i = 0; i < nds.Dim(); i++) {
          Real theta = 2*const_pi<Real>()*(k+nds[i])/Nelem;
          coord.PushBack(cos<Real>(theta));
          coord.PushBack(sin<Real>(theta));
          coord.PushBack(0.1*sin<Real>(2*theta));
          radius.PushBack(0.01*(2+sin<Real>(theta+sqrt<Real>(2))));
        }
      }
      elem_lst0.Init(cheb_order, fourier_order, coord, radius);
    }

    Kernel ker_fn;
    BoundaryIntegralOp<Real,Kernel> BIOp(ker_fn, false, comm);
    BIOp.AddElemList(elem_lst0);
    BIOp.SetAccuracy(tol);

    // Warm-up run
    Vector<Real> F(BIOp.Dim(0)), U; F = 1;
    BIOp.ComputePotential(U,F);
    BIOp.ClearSetup();
    U = 0;

    Profile::Enable(true);
    Profile::Tic("Setup+Eval", &comm, true);
    BIOp.ComputePotential(U,F);
    Profile::Toc();

    Vector<Real> Uerr = U + 0.5;
    elem_lst0.WriteVTK("Uerr_", Uerr, comm); // Write VTK
    { // Print error
      StaticArray<Real,2> max_err{0,0};
      for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
      comm.Allreduce(max_err+0, max_err+1, 1, CommOp::MAX);
      if (!pid) std::cout<<"Error = "<<max_err[1]<<'\n';
    }
    Profile::Enable(false);
    Profile::print(&comm);
  }
  template <class Real> void SlenderElemList<Real>::test_greens_identity(const Comm& comm, Real tol) {
    using KerSL = Laplace3D_FxU;
    using KerDL = Laplace3D_DxU;
    using KerGrad = Laplace3D_FxdU;

    const auto concat_vecs = [](Vector<Real>& v, const Vector<Vector<Real>>& vec_lst) {
      const Long N = vec_lst.Dim();
      Vector<Long> dsp(N+1); dsp[0] = 0;
      for (Long i = 0; i < N; i++) {
        dsp[i+1] = dsp[i] + vec_lst[i].Dim();
      }
      if (v.Dim() != dsp[N]) v.ReInit(dsp[N]);
      for (Long i = 0; i < N; i++) {
        Vector<Real> v_(vec_lst[i].Dim(), v.begin()+dsp[i], false);
        v_ = vec_lst[i];
      }
    };
    auto loop_geom = [](Real& x, Real& y, Real& z, Real& r, const Real theta){
      x = cos<Real>(theta);
      y = sin<Real>(theta);
      z = 0.1*sin<Real>(theta-sqrt<Real>(2));
      r = 0.01*(2+sin<Real>(theta+sqrt<Real>(2)));
    };
    sctl::Profile::Enable(false);
    const Long pid = comm.Rank();
    const Long Np = comm.Size();

    SlenderElemList elem_lst0;
    SlenderElemList elem_lst1;
    { // Set elem_lst0, elem_lst1
      const Long Nelem = 16;
      const Long idx0 = Nelem*(pid+0)/Np;
      const Long idx1 = Nelem*(pid+1)/Np;

      Vector<Real> coord0, radius0;
      Vector<Long> cheb_order0, fourier_order0;
      for (Long k = idx0; k < idx1; k++) { // Init elem_lst0
      const Integer ChebOrder = 8, FourierOrder = 14;
        const auto& nds = CenterlineNodes(ChebOrder);
        for (Long i = 0; i < nds.Dim(); i++) {
          Real x, y, z, r;
          loop_geom(x, y, z, r, const_pi<Real>()*(k+nds[i])/Nelem);
          coord0.PushBack(x);
          coord0.PushBack(y);
          coord0.PushBack(z);
          radius0.PushBack(r);
        }
        cheb_order0.PushBack(ChebOrder);
        fourier_order0.PushBack(FourierOrder);
      }
      elem_lst0.Init(cheb_order0, fourier_order0, coord0, radius0);

      Vector<Real> coord1, radius1;
      Vector<Long> cheb_order1, fourier_order1;
      for (Long k = idx0; k < idx1; k++) { // Init elem_lst1
        const Integer ChebOrder = 10, FourierOrder = 14;
        const auto& nds = CenterlineNodes(ChebOrder);
        for (Long i = 0; i < nds.Dim(); i++) {
          Real x, y, z, r;
          loop_geom(x, y, z, r, const_pi<Real>()*(1+(k+nds[i])/Nelem));
          coord1.PushBack(x);
          coord1.PushBack(y);
          coord1.PushBack(z);
          radius1.PushBack(r);
        }
        cheb_order1.PushBack(ChebOrder);
        fourier_order1.PushBack(FourierOrder);
      }
      elem_lst1.Init(cheb_order1, fourier_order1, coord1, radius1);
    }

    KerSL kernel_sl;
    KerDL kernel_dl;
    KerGrad kernel_grad;
    BoundaryIntegralOp<Real,KerSL> BIOpSL(kernel_sl, false, comm);
    BoundaryIntegralOp<Real,KerDL> BIOpDL(kernel_dl, false, comm);
    BIOpSL.AddElemList(elem_lst0, "elem_lst0");
    BIOpSL.AddElemList(elem_lst1, "elem_lst1");
    BIOpDL.AddElemList(elem_lst0, "elem_lst0");
    BIOpDL.AddElemList(elem_lst1, "elem_lst1");
    BIOpSL.SetAccuracy(tol);
    BIOpDL.SetAccuracy(tol);

    Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
    { // Get X, Xn
      Vector<Vector<Real>> X_(2), Xn_(2);
      elem_lst0.GetNodeCoord(&X_[0], &Xn_[0], nullptr);
      elem_lst1.GetNodeCoord(&X_[1], &Xn_[1], nullptr);
      concat_vecs(X, X_);
      concat_vecs(Xn, Xn_);
    }
    { // Set Fs, Fd, Uref
      Vector<Real> X0{0.3,0.6,0.2}, Xn0{0,0,0}, F0{1}, dU;
      kernel_sl.Eval(Uref, X, X0, Xn0, F0);
      kernel_grad.Eval(dU, X, X0, Xn0, F0);

      Fd = Uref;
      { // Set Fs <-- -dot_prod(dU, Xn)
        Fs.ReInit(X.Dim()/COORD_DIM);
        for (Long i = 0; i < Fs.Dim(); i++) {
          Real dU_dot_Xn = 0;
          for (Long k = 0; k < COORD_DIM; k++) {
            dU_dot_Xn += dU[i*COORD_DIM+k] * Xn[i*COORD_DIM+k];
          }
          Fs[i] = -dU_dot_Xn;
        }
      }
    }

    // Warm-up run
    BIOpSL.ComputePotential(Us,Fs);
    BIOpDL.ComputePotential(Ud,Fd);
    BIOpSL.ClearSetup();
    BIOpDL.ClearSetup();
    Us = 0; Ud = 0;

    sctl::Profile::Enable(true);
    Profile::Tic("Setup+Eval", &comm);
    BIOpSL.ComputePotential(Us,Fs);
    BIOpDL.ComputePotential(Ud,Fd);
    Profile::Toc();

    Vector<Real> Uerr = Fd*0.5 + (Us - Ud) - Uref;
    { // Write VTK
      Vector<Vector<Real>> X_(2);
      elem_lst0.GetNodeCoord(&X_[0], nullptr, nullptr);
      elem_lst1.GetNodeCoord(&X_[1], nullptr, nullptr);
      const Long N0 = X_[0].Dim()/COORD_DIM;
      const Long N1 = X_[1].Dim()/COORD_DIM;
      elem_lst0.WriteVTK("Uerr0", Vector<Real>(N0,Uerr.begin()+ 0,false), comm);
      elem_lst1.WriteVTK("Uerr1", Vector<Real>(N1,Uerr.begin()+N0,false), comm);
    }
    { // Print error
      StaticArray<Real,2> max_err{0,0};
      StaticArray<Real,2> max_val{0,0};
      for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
      for (auto x : Uref) max_val[0] = std::max<Real>(max_val[0], fabs(x));
      comm.Allreduce(max_err+0, max_err+1, 1, CommOp::MAX);
      comm.Allreduce(max_val+0, max_val+1, 1, CommOp::MAX);
      if (!pid) std::cout<<"Error = "<<max_err[1]/max_val[1]<<'\n';
    }

    sctl::Profile::print(&comm);
    sctl::Profile::Enable(false);
  }

  template <class Real> void SlenderElemList<Real>::GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_ds, Vector<Real>* dX_dt, const Vector<Real>& s_param, const Vector<Real>& sin_theta_, const Vector<Real>& cos_theta_, const Long elem_idx) const {
    SCTL_ASSERT_MSG(elem_idx < Size(), "element index is greater than number of elements in the list!");
    using Vec3 = Tensor<Real,true,COORD_DIM,1>;
    const Integer ChebOrder = cheb_order[elem_idx];
    const Long Nt = sin_theta_.Dim();
    const Long Ns = s_param.Dim();
    const Long N = Ns * Nt;

    if (X     && X    ->Dim() != N*COORD_DIM) X    ->ReInit(N*COORD_DIM);
    if (Xn    && Xn   ->Dim() != N*COORD_DIM) Xn   ->ReInit(N*COORD_DIM);
    if (Xa    && Xa   ->Dim() != N          ) Xa   ->ReInit(N);
    if (dX_ds && dX_ds->Dim() != N*COORD_DIM) dX_ds->ReInit(N*COORD_DIM);
    if (dX_dt && dX_dt->Dim() != N*COORD_DIM) dX_dt->ReInit(N*COORD_DIM);

    Matrix<Real> M_lagrange_interp;
    { // Set M_lagrange_interp
      M_lagrange_interp.ReInit(ChebOrder, Ns);
      Vector<Real> V_lagrange_interp(ChebOrder*Ns, M_lagrange_interp.begin(), false);
      LagrangeInterp<Real>::Interpolate(V_lagrange_interp, CenterlineNodes(ChebOrder), s_param);
    }

    Matrix<Real> r_, dr_, x_, dx_, d2x_, e1_;
    r_  .ReInit(        1,Ns);
    x_  .ReInit(COORD_DIM,Ns);
    dx_ .ReInit(COORD_DIM,Ns);
    e1_ .ReInit(COORD_DIM,Ns);
    Matrix<Real>::GEMM(  x_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>) coord.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
    Matrix<Real>::GEMM( dx_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    dx.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
    Matrix<Real>::GEMM(  r_, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)radius.begin()+          elem_dsp[elem_idx],false), M_lagrange_interp);
    Matrix<Real>::GEMM( e1_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    e1.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
    if (Xn || Xa || dX_ds || dX_dt) { // Set dr_, d2x_
      dr_ .ReInit(        1,Ns);
      d2x_.ReInit(COORD_DIM,Ns);
      Matrix<Real>::GEMM(d2x_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)   d2x.begin()+COORD_DIM*elem_dsp[elem_idx],false), M_lagrange_interp);
      Matrix<Real>::GEMM( dr_, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)    dr.begin()+          elem_dsp[elem_idx],false), M_lagrange_interp);
    }
    auto compute_coord = [](Vec3& y, const Vec3& x, const Vec3& e1, const Vec3& e2, const Real r, const Real sint, const Real cost) {
      y = x + e1*(r*cost) + e2*(r*sint);
    };
    auto compute_normal_area_elem_tangents = [](Vec3& n, Real& da, Vec3& dy_ds, Vec3& dy_dt, const Vec3& dx, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const Real r, const Real dr, const Real sint, const Real cost) {
      dy_ds = dx + e1*(dr*cost) + e2*(dr*sint) + de1*(r*cost) + de2*(r*sint);
      dy_dt = e1*(-r*sint) + e2*(r*cost);

      n = cross_prod(dy_ds, dy_dt);
      da = sqrt<Real>(dot_prod(n,n));
      n = n * (1/da);
    };

    for (Long j = 0; j < Ns; j++) {
      Real r, inv_dx2;
      Vec3 x, dx, e1, e2;
      { // Set x, dx, e1, r, inv_dx2
        for (Integer k = 0; k < COORD_DIM; k++) {
          x(k,0)  = x_[k][j];
          dx(k,0) = dx_[k][j];
          e1(k,0) = e1_[k][j];
        }
        inv_dx2 = 1/dot_prod(dx,dx);
        r = r_[0][j];

        e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
        e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

        e2 = cross_prod(e1, dx);
        e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
      }

      if (X) {
        for (Integer i = 0; i < Nt; i++) { // Set X
          Vec3 y;
          compute_coord(y, x, e1, e2, r, sin_theta_[i], cos_theta_[i]);
          for (Integer k = 0; k < COORD_DIM; k++) {
            (*X)[(j*Nt+i)*COORD_DIM+k] = y(k,0);
          }
        }
      }
      if (Xn || Xa || dX_ds || dX_dt) {
        Vec3 d2x, de1, de2;
        for (Integer k = 0; k < COORD_DIM; k++) {
          d2x(k,0) = d2x_[k][j];
        }
        de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
        de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
        Real dr = dr_[0][j];

        for (Integer i = 0; i < Nt; i++) { // Set X, Xn, Xa, dX_ds, dX_dt
          Real da;
          Vec3 n, dx_ds, dx_dt;
          compute_normal_area_elem_tangents(n, da, dx_ds, dx_dt, dx, e1, e2, de1, de2, r, dr, sin_theta_[i], cos_theta_[i]);
          if (Xn) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              (*Xn)[(j*Nt+i)*COORD_DIM+k] = n(k,0);
            }
          }
          if (Xa) {
            (*Xa)[j*Nt+i] = da;
          }
          if (dX_ds) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              (*dX_ds)[(j*Nt+i)*COORD_DIM+k] = dx_ds(k,0);
            }
          }
          if (dX_dt) {
            for (Integer k = 0; k < COORD_DIM; k++) {
              (*dX_dt)[(j*Nt+i)*COORD_DIM+k] = dx_dt(k,0);
            }
          }
        }
      }
    }
  }

  template <class Real> template <Integer digits, bool trg_dot_prod, class Kernel> Matrix<Real> SlenderElemList<Real>::SelfInteracHelper(const Kernel& ker, const Long elem_idx) const {
    constexpr Integer MAX_THREADS=1000;
    constexpr Integer MAX_BUFF_SIZE=10000000;
    SCTL_ASSERT(omp_get_num_threads() < MAX_THREADS);
    static Vector<Vector<Real>> buff_(MAX_THREADS);
    Vector<Real>& buff = buff_[omp_get_thread_num()];
    if (buff.Dim() == 0) buff.ReInit(MAX_BUFF_SIZE);
    Long buff_offset = 0;

    using Vec3 = Tensor<Real,true,COORD_DIM,1>;
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1 = Kernel::TrgDim()/(trg_dot_prod?COORD_DIM:1);
    //const Integer digits = (Integer)(log(tol)/log(0.1)+0.5);

    const Integer ChebOrder = cheb_order[elem_idx];
    const Integer FourierOrder = fourier_order[elem_idx];
    const Integer FourierModes = FourierOrder/2+1;
    const Matrix<Real>& M_fourier_inv = fourier_matrix_inv_transpose<Real>(FourierOrder,FourierModes);
    const auto& cheb_nds = SlenderElemList<Real>::CenterlineNodes(ChebOrder);

    const Vector<Real>  coord(COORD_DIM*ChebOrder,(Iterator<Real>)this-> coord.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>     dx(COORD_DIM*ChebOrder,(Iterator<Real>)this->    dx.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real>    d2x(COORD_DIM*ChebOrder,(Iterator<Real>)this->   d2x.begin()+COORD_DIM*elem_dsp[elem_idx],false);
    const Vector<Real> radius(        1*ChebOrder,(Iterator<Real>)this->radius.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     dr(        1*ChebOrder,(Iterator<Real>)this->    dr.begin()+          elem_dsp[elem_idx],false);
    const Vector<Real>     e1(COORD_DIM*ChebOrder,(Iterator<Real>)this->    e1.begin()+COORD_DIM*elem_dsp[elem_idx],false);

    const Real dtheta = 2*const_pi<Real>()/FourierOrder;
    const Complex<Real> exp_dtheta(cos<Real>(dtheta), sin<Real>(dtheta));

    Matrix<Real> M_modal;
    if (MAX_BUFF_SIZE-buff_offset >= ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1*FourierModes*2) {
      M_modal.ReInit(ChebOrder*FourierOrder, ChebOrder*KDIM0*KDIM1*FourierModes*2, buff.begin()+buff_offset, false);
      buff_offset += ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1*FourierModes*2;
    } else {
      M_modal.ReInit(ChebOrder*FourierOrder, ChebOrder*KDIM0*KDIM1*FourierModes*2);
    }
    //#pragma omp parallel for
    for (Long i = 0; i < ChebOrder; i++) {
      Long buff_offset0 = buff_offset;

      Real r_trg = radius[i];
      Real dr_trg = dr[i];
      Vec3 x_trg, dx_trg, d2x_trg, e1_trg, e2_trg;
      { // Set x_trg, dx_trg, d2x_trg, e1_trg
        for (Integer k = 0; k < COORD_DIM; k++) {
          x_trg (k,0) = coord[k*ChebOrder+i];
          e1_trg(k,0) = e1[k*ChebOrder+i];
          dx_trg(k,0) = dx[k*ChebOrder+i];
          d2x_trg(k,0) = d2x[k*ChebOrder+i];
        }
        Real inv_dx2 = 1/dot_prod(dx_trg,dx_trg);
        e1_trg = e1_trg - dx_trg * dot_prod(e1_trg, dx_trg) * inv_dx2;
        e1_trg = e1_trg * (1/sqrt<Real>(dot_prod(e1_trg,e1_trg)));

        e2_trg = cross_prod(e1_trg, dx_trg);
        e2_trg = e2_trg * (1/sqrt<Real>(dot_prod(e2_trg,e2_trg)));
      }
      const Real norm_dx_trg = sqrt<Real>(dot_prod(dx_trg,dx_trg));
      const Real inv_norm_dx_trg = 1/norm_dx_trg;

      Vector<Real> quad_nds, quad_wts; // Quadrature rule in s
      SpecialQuadRule<ModalUpsample,Real,Kernel,trg_dot_prod>(quad_nds, quad_wts, ChebOrder, i, r_trg, sqrt<Real>(dot_prod(dx_trg, dx_trg)), digits);
      const Long Nq = quad_nds.Dim();

      Matrix<Real> Minterp_quad_nds;
      { // Set Minterp_quad_nds
        if (MAX_BUFF_SIZE-buff_offset0 >= ChebOrder*Nq) {
          Minterp_quad_nds.ReInit(ChebOrder, Nq, buff.begin()+buff_offset0, false);
          buff_offset0 += ChebOrder*Nq;
        } else {
          Minterp_quad_nds.ReInit(ChebOrder, Nq);
        }
        Vector<Real> Vinterp_quad_nds(ChebOrder*Nq, Minterp_quad_nds.begin(), false);

        StaticArray<Real,20> buff0;
        Vector<Real> cheb_nds0(ChebOrder, (ChebOrder>20?NullIterator<Real>():buff0), (ChebOrder>20));
        for (Long j = 0; j < ChebOrder; j++) cheb_nds0[j] = cheb_nds[j] - cheb_nds[i];
        LagrangeInterp<Real>::Interpolate(Vinterp_quad_nds, cheb_nds0, quad_nds);
      }

      Matrix<Real> r_src, dr_src, x_src, dx_src, d2x_src, e1_src;
      if (MAX_BUFF_SIZE-buff_offset0 >= 14*Nq) {
        r_src  .ReInit(        1, Nq, buff.begin()+buff_offset0+ 0*Nq, false);
        dr_src .ReInit(        1, Nq, buff.begin()+buff_offset0+ 1*Nq, false);
        x_src  .ReInit(COORD_DIM, Nq, buff.begin()+buff_offset0+ 2*Nq, false);
        dx_src .ReInit(COORD_DIM, Nq, buff.begin()+buff_offset0+ 5*Nq, false);
        d2x_src.ReInit(COORD_DIM, Nq, buff.begin()+buff_offset0+ 8*Nq, false);
        e1_src .ReInit(COORD_DIM, Nq, buff.begin()+buff_offset0+11*Nq, false);
        buff_offset0 += 14*Nq;
      } else {
        r_src  .ReInit(        1, Nq);
        dr_src .ReInit(        1, Nq);
        x_src  .ReInit(COORD_DIM, Nq);
        dx_src .ReInit(COORD_DIM, Nq);
        d2x_src.ReInit(COORD_DIM, Nq);
        e1_src .ReInit(COORD_DIM, Nq);
      }
      { // Set x_src, x_trg (improve numerical stability)
        Matrix<Real> x_nodes;
        StaticArray<Real,30> buff;
        if (COORD_DIM*ChebOrder <= 30) {
          x_nodes.ReInit(COORD_DIM,ChebOrder, buff, false);
        } else {
          x_nodes.ReInit(COORD_DIM,ChebOrder);
        }
        for (Integer k = 0; k < COORD_DIM; k++) {
          for (Long j = 0; j < ChebOrder; j++) {
            x_nodes[k][j] = coord[k*ChebOrder+j] - x_trg(k,0);
          }
        }
        Matrix<Real>::GEMM(  x_src, x_nodes, Minterp_quad_nds);
        for (Integer k = 0; k < COORD_DIM; k++) {
          x_trg(k,0) = 0;
        }
      }
      //Matrix<Real>::GEMM(  x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>) coord.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM( dx_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    dx.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM(d2x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)   d2x.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM(  r_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)radius.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM( dr_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)    dr.begin(),false), Minterp_quad_nds);
      Matrix<Real>::GEMM( e1_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)    e1.begin(),false), Minterp_quad_nds);
      for (Long j = 0; j < Nq; j++) { // Set e1_src
        Vec3 e1, dx;
        for (Integer k = 0; k < COORD_DIM; k++) {
          e1(k,0) = e1_src[k][j];
          dx(k,0) = dx_src[k][j];
        }
        e1 = e1 - dx * dot_prod(e1, dx) * (1/dot_prod(dx,dx));
        e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

        for (Integer k = 0; k < COORD_DIM; k++) {
          e1_src[k][j] = e1(k,0);
        }
      }


      Complex<Real> exp_theta_trg(1,0);
      for (Long j = 0; j < ChebOrder; j++) { // Minterp_quad_nds *= quad_wts
        for (Long k = 0; k < Nq; k++) {
          Minterp_quad_nds[j][k] *= quad_wts[k];
        }
      }
      for (Long j = 0; j < FourierOrder; j++) {
        Long buff_offset1 = buff_offset0;

        auto compute_Xn_trg = [&exp_theta_trg,&dx_trg,&d2x_trg,&e1_trg,&e2_trg,&r_trg,&dr_trg,&norm_dx_trg,&inv_norm_dx_trg]() { // Set n_trg
          Real cost = exp_theta_trg.real;
          Real sint = exp_theta_trg.imag;

          Real d2x_dot_e1 = dot_prod(d2x_trg, e1_trg);
          Real d2x_dot_e2 = dot_prod(d2x_trg, e2_trg);

          Vec3 n_trg;
          Real norm_dy = norm_dx_trg - (cost*d2x_dot_e1 + sint*d2x_dot_e2) * (r_trg*inv_norm_dx_trg);
          n_trg(0,0) = e1_trg(0,0)*norm_dy*cost + e2_trg(0,0)*norm_dy*sint - dx_trg(0,0)*(dr_trg*inv_norm_dx_trg);
          n_trg(1,0) = e1_trg(1,0)*norm_dy*cost + e2_trg(1,0)*norm_dy*sint - dx_trg(1,0)*(dr_trg*inv_norm_dx_trg);
          n_trg(2,0) = e1_trg(2,0)*norm_dy*cost + e2_trg(2,0)*norm_dy*sint - dx_trg(2,0)*(dr_trg*inv_norm_dx_trg);
          Real scale = 1/sqrt<Real>(dot_prod(n_trg,n_trg));
          return n_trg*scale;
        };
        //const Vec3 y_trg = x_trg + e1_trg*r_trg*exp_theta_trg.real + e2_trg*r_trg*exp_theta_trg.imag;
        const Vec3 e_trg = e1_trg*exp_theta_trg.real + e2_trg*exp_theta_trg.imag;
        const Vec3 n_trg(trg_dot_prod ? compute_Xn_trg() : Vec3((Real)0));

        Matrix<Real> M_tor;
        if (MAX_BUFF_SIZE-buff_offset1 >= Nq * KDIM0*KDIM1*FourierModes*2) {
          M_tor.ReInit(Nq, KDIM0*KDIM1*FourierModes*2, buff.begin()+buff_offset1, false);
          buff_offset1 += Nq * KDIM0*KDIM1*FourierModes*2;
        } else {
          M_tor.ReInit(Nq, KDIM0*KDIM1*FourierModes*2);
        }
        toroidal_greens_fn_batched<digits+1,ModalUpsample,trg_dot_prod>(M_tor, x_trg, e_trg, r_trg, n_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, ker, FourierModes);

        Matrix<Real> M_modal_(ChebOrder, KDIM0*KDIM1*FourierModes*2, M_modal[i*FourierOrder+j], false);
        Matrix<Real>::GEMM(M_modal_, Minterp_quad_nds, M_tor);
        exp_theta_trg *= exp_dtheta;
      }
    }

    Matrix<Real> M_nodal;
    if (MAX_BUFF_SIZE-buff_offset >= ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1*FourierOrder) {
      M_nodal.ReInit(ChebOrder*FourierOrder, ChebOrder*KDIM0*KDIM1*FourierOrder, buff.begin()+buff_offset, false);
      buff_offset += ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1*FourierOrder;
    } else {
      M_nodal.ReInit(ChebOrder*FourierOrder, ChebOrder*KDIM0*KDIM1*FourierOrder);
    }
    { // Set M_nodal
      const Matrix<Real> M_modal_(ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1, FourierModes*2, M_modal.begin(), false);
      Matrix<Real> M_nodal_(ChebOrder*FourierOrder * ChebOrder*KDIM0*KDIM1, FourierOrder, M_nodal.begin(), false);
      Matrix<Real>::GEMM(M_nodal_, M_modal_, M_fourier_inv);
    }

    Matrix<Real> M(ChebOrder*FourierOrder*KDIM0, ChebOrder*FourierOrder*KDIM1); // TODO: pass M by reference
    { // Set M
      const Integer Nnds = ChebOrder*FourierOrder;
      for (Integer i = 0; i < Nnds; i++) {
        for (Integer j0 = 0; j0 < ChebOrder; j0++) {
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1; k1++) {
              for (Integer j1 = 0; j1 < FourierOrder; j1++) {
                M[(j0*FourierOrder+j1)*KDIM0+k0][i*KDIM1+k1] = M_nodal[i][((j0*KDIM0+k0)*KDIM1+k1)*FourierOrder+j1] * ker.template uKerScaleFactor<Real>();
              }
            }
          }
        }
      }
    }
    return M;
  }

  template <class Real> template <class ValueType> void SlenderElemList<Real>::Copy(SlenderElemList<ValueType>& elem_lst) const {
    elem_lst.radius.ReInit(radius.Dim());
    elem_lst. coord.ReInit( coord.Dim());
    elem_lst.    e1.ReInit(    e1.Dim());
    elem_lst.    dr.ReInit(    dr.Dim());
    elem_lst.    dx.ReInit(    dx.Dim());
    elem_lst.   d2x.ReInit(   d2x.Dim());
    for (Long i = 0; i < radius.Dim(); i++) elem_lst.radius[i] = (ValueType)radius[i];
    for (Long i = 0; i <  coord.Dim(); i++) elem_lst. coord[i] = (ValueType) coord[i];
    for (Long i = 0; i <     e1.Dim(); i++) elem_lst.    e1[i] = (ValueType)    e1[i];
    for (Long i = 0; i <     dr.Dim(); i++) elem_lst.    dr[i] = (ValueType)    dr[i];
    for (Long i = 0; i <     dx.Dim(); i++) elem_lst.    dx[i] = (ValueType)    dx[i];
    for (Long i = 0; i <    d2x.Dim(); i++) elem_lst.   d2x[i] = (ValueType)   d2x[i];
    elem_lst.   cheb_order =    cheb_order;
    elem_lst.fourier_order = fourier_order;
    elem_lst.     elem_dsp =      elem_dsp;
  }
}

