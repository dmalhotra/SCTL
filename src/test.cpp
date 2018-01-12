#include "pvfmm.hpp"
#include <pvfmm/fft_wrapper.hpp>

namespace pvfmm {

  template <class Real> class Kernel {
      static constexpr Integer COORD_DIM = 3;
      static constexpr Integer KER_DIM0 = 3;
      static constexpr Integer KER_DIM1 = 3;

    public:

      virtual Integer Dim(Integer i) const { return i == 0 ? KER_DIM0 : KER_DIM1; }

      virtual void operator()(const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, const Vector<Real>& r_trg, Vector<Real>& v_trg) const {
        auto ker = [&](Iterator<Real> v, ConstIterator<Real> x, ConstIterator<Real> n, ConstIterator<Real> f) {
          Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
          Real invr = (r > 0 ? 1.0 / r : 0.0);

          Real fdotr = 0;
          Real ndotr = 0;
          Real invr2 = invr*invr;
          Real invr3 = invr2*invr;
          Real invr5 = invr2*invr3;
          for(Integer k=0;k<COORD_DIM;k++) fdotr += f[k] * x[k];
          for(Integer k=0;k<COORD_DIM;k++) ndotr += n[k] * x[k];
          v[0] += x[0] * fdotr * ndotr * invr5;
          v[1] += x[1] * fdotr * ndotr * invr5;
          v[2] += x[2] * fdotr * ndotr * invr5;
        };

        Long Ns = r_src.Dim() / COORD_DIM;
        Long Nt = r_trg.Dim() / COORD_DIM;
        PVFMM_ASSERT(v_src.Dim() == Dim(0) * Ns);
        PVFMM_ASSERT(n_src.Dim() == COORD_DIM * Ns);
        PVFMM_ASSERT(r_src.Dim() == COORD_DIM * Ns);
        PVFMM_ASSERT(r_trg.Dim() == COORD_DIM * Nt);
        if(v_trg.Dim() != Dim(1) * Nt) {
          v_trg.ReInit(Dim(1) * Nt);
          v_trg.SetZero();
        }

        static const Real scal = 3.0 / (4.0 * const_pi<Real>());
        #pragma omp parallel for schedule(static)
        for (Long t = 0; t < Nt; t++) {
          StaticArray<Real, COORD_DIM> v = {0, 0, 0};
          for (Long s = 0; s < Ns; s++) {
            StaticArray<Real, COORD_DIM> r = {r_trg[0 * Nt + t] - r_src[0 * Ns + s],
                                              r_trg[1 * Nt + t] - r_src[1 * Ns + s],
                                              r_trg[2 * Nt + t] - r_src[2 * Ns + s]};
            StaticArray<Real, COORD_DIM> n = {n_src[0 * Ns + s], n_src[1 * Ns + s], n_src[2 * Ns + s]};
            StaticArray<Real, COORD_DIM> f = {v_src[0 * Ns + s], v_src[1 * Ns + s], v_src[2 * Ns + s]};
            ker(v, r, n, f);
          }
          v_trg[0 * Nt + t] += scal * v[0];
          v_trg[1 * Nt + t] += scal * v[1];
          v_trg[2 * Nt + t] += scal * v[2];
        }

        Profile::Add_FLOP(Ns * Nt * 30);
      }

      void BuildMatrix(const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& r_trg, Matrix<Real>& M) const {
      }

    private:
  };

  template <class Real> class Surface {
      static constexpr Integer COORD_DIM = 3;

    public:

      Surface(Long Nt, Long Np) : Nt_(Nt), Np_(Np) {
        X_.ReInit(COORD_DIM * Nt_ * Np_);
        for (Long t = 0; t < Nt_; t++) {
          for (Long p = 0; p < Np_; p++) {
            X_[(0 * Nt_ + t) * Np_ + p] = (cos(2.0 * const_pi<Real>() * p / Np_) + 2) * cos(2.0 * const_pi<Real>() * t / Nt_);
            X_[(1 * Nt_ + t) * Np_ + p] = (cos(2.0 * const_pi<Real>() * p / Np_) + 2) * sin(2.0 * const_pi<Real>() * t / Nt_);
            X_[(2 * Nt_ + t) * Np_ + p] = (sin(2.0 * const_pi<Real>() * p / Np_) + 0);
          }
        }

        F_.ReInit(ker.Dim(0) * Nt_ * Np_);
        for (Integer k = 0; k < ker.Dim(0); k++) {
          for (Long t = 0; t < Nt_; t++) {
            for (Long p = 0; p < Np_; p++) {
              F_[(k * Nt_ + t) * Np_ + p] = 1;
            }
          }
        }

        U_.ReInit(ker.Dim(1) * Nt_ * Np_);
        U_.SetZero();

        Setup();
        Evaluate();
      }

      Long NTor() const {return Nt_;}
      Long NPol() const {return Np_;}
      const Vector<Real>& Coord() const {return X_;}
      const Vector<Real>& TangentTor() const {return Xt_;}
      const Vector<Real>& TangentPol() const {return Xp_;}
      const Vector<Real>& AreaElem() const {return Xa_;}
      const Vector<Real>& Normal() const {return Xn_;}
      Vector<Real>& Potential() {return U_;}
      Vector<Real>& Density() {return F_;}

    private:

      void Setup() const {
        Xt_.ReInit(COORD_DIM * Nt_ * Np_);
        Xp_.ReInit(COORD_DIM * Nt_ * Np_);

        Vector<Real> FX;
        FFT<Real> fft_r2c, fft_c2r;
        StaticArray<Long, 2> fft_dim = {Nt_, Np_};
        fft_r2c.Setup(FFT_Type::R2C, COORD_DIM, Vector<Long>(2, fft_dim, false));
        fft_c2r.Setup(FFT_Type::C2R, COORD_DIM, Vector<Long>(2, fft_dim, false));
        PVFMM_ASSERT(X_.Dim() == COORD_DIM * Nt_ * Np_);
        fft_r2c.Execute(X_, FX);

        { // Compute Xt_, Xp_
          auto FX_ = FX;
          Long Nt = Nt_;
          Long Np = FX_.Dim() / (COORD_DIM * Nt * 2);
          PVFMM_ASSERT(FX_.Dim() == COORD_DIM * Nt * Np * 2);
          for (Integer k = 0; k < COORD_DIM; k++) {
            for (Long t = 0; t < Nt; t++) {
              for (Long p = 0; p < Np; p++) {
                Real real = FX_[((k * Nt + t) * Np + p) * 2 + 0];
                Real imag = FX_[((k * Nt + t) * Np + p) * 2 + 1];
                FX_[((k * Nt + t) * Np + p) * 2 + 0] =  imag * (t - (t > Nt / 2 ? Nt : 0));
                FX_[((k * Nt + t) * Np + p) * 2 + 1] = -real * (t - (t > Nt / 2 ? Nt : 0));
              }
            }
          }
          fft_c2r.Execute(FX_, Xt_);

          FX_ = FX;
          for (Integer k = 0; k < COORD_DIM; k++) {
            for (Long t = 0; t < Nt; t++) {
              for (Long p = 0; p < Np; p++) {
                Real real = FX_[((k * Nt + t) * Np + p) * 2 + 0];
                Real imag = FX_[((k * Nt + t) * Np + p) * 2 + 1];
                FX_[((k * Nt + t) * Np + p) * 2 + 0] =  imag * p;
                FX_[((k * Nt + t) * Np + p) * 2 + 1] = -real * p;
              }
            }
          }
          fft_c2r.Execute(FX_, Xp_);
        }
        { // Compute Xa_, Xn_
          Long N = Nt_ * Np_;
          Xa_.ReInit(N);
          Xn_.ReInit(COORD_DIM * N);
          for (Long i = 0; i < N; i++) {
            StaticArray<Real, COORD_DIM> xt = {Xt_[0*N+i], Xt_[1*N+i], Xt_[2*N+i]};
            StaticArray<Real, COORD_DIM> xp = {Xp_[0*N+i], Xp_[1*N+i], Xp_[2*N+i]};
            StaticArray<Real, COORD_DIM> xn;
            xn[0] = xp[1] * xt[2] - xt[1] * xp[2];
            xn[1] = xp[2] * xt[0] - xt[2] * xp[0];
            xn[2] = xp[0] * xt[1] - xt[0] * xp[1];
            Real xa = sqrt(xn[0] * xn[0] + xn[1] * xn[1] + xn[2] * xn[2]);
            Real xa_inv = 1.0 / xa;
            Xa_[i] = xa;
            Xn_[0 * N + i] = xn[0] * xa_inv;
            Xn_[1 * N + i] = xn[1] * xa_inv;
            Xn_[2 * N + i] = xn[2] * xa_inv;
          }
        }
      }

      void Evaluate() {
        FW_.ReInit(ker.Dim(0) * Nt_ * Np_);
        PVFMM_ASSERT(F_.Dim() == ker.Dim(0) * Nt_ * Np_);
        Real scal = (4 * const_pi<Real>() * const_pi<Real>()) / (Nt_ * Np_);
        for (Integer k = 0; k < ker.Dim(0); k++) {
          for (Long i = 0; i < Nt_ * Np_; i++) {
            FW_[k * Nt_ * Np_ + i] = F_[k * Nt_ * Np_ + i] * Xa_[i] * scal;
          }
        }
        ker(X_, Xn_, FW_, X_, U_);
      }

      Long Nt_, Np_;
      Vector<Real> X_, F_, U_;
      mutable Vector<Real> Xt_, Xp_, Xn_, Xa_, FW_;
      Kernel<Real> ker;
  };

  template <class Real> class SingularCorrection {
      static constexpr Integer COORD_DIM = 3;
      static constexpr Integer PATCH_DIM = 11;
      static constexpr Integer RAD_DIM = 10;
      static constexpr Integer ANG_DIM = 20;
      static constexpr Integer Ngrid = PATCH_DIM * PATCH_DIM;
      static constexpr Integer Npolar = RAD_DIM * ANG_DIM;
      static constexpr Integer KDIM0 = 3;
      static constexpr Integer KDIM1 = 3;

    public:

      SingularCorrection() {
        { // Set Gpou, Ppou
          auto pou = [&](Real r) {
            return (r < 1.0 ? exp<Real>(1.0 - 1.0 / (1.0 - r * r)) : 0.0);
          };
          for (Integer i = 0; i < PATCH_DIM; i++){
            for (Integer j = 0; j < PATCH_DIM; j++){
              Real dr[2] = {i / (PATCH_DIM * 0.5 - 0.5) - 1.0, j / (PATCH_DIM * 0.5 - 0.5) - 1.0};
              Real r = sqrt(dr[0] * dr[0] + dr[1] * dr[1]);
              Gpou_[i * PATCH_DIM + j] = pou(r);
            }
          }
          for (Integer i = 0; i < RAD_DIM; i++){
            for (Integer j = 0; j < ANG_DIM; j++){
              Real r = i / (RAD_DIM - 1.0);
              Ppou_[i * ANG_DIM + j]= pou(r);
            }
          }
        }

        G2P.ReInit(Ngrid, Npolar);
        G2P_grad0.ReInit(Ngrid, Npolar);
        G2P_grad1.ReInit(Ngrid, Npolar);

        PVFMM_ASSERT(KDIM0 == ker.Dim(0));
        PVFMM_ASSERT(KDIM1 == ker.Dim(1));
      }

      void operator()(Surface<Real>& S, Long t, Long p) {
        Vector<Real> G (Ngrid * COORD_DIM, G_ , false);
        Vector<Real> G0(Ngrid * COORD_DIM, G0_, false);
        Vector<Real> G1(Ngrid * COORD_DIM, G1_, false);
        Vector<Real> Gn(Ngrid * COORD_DIM, Gn_, false);
        Vector<Real> GF(Ngrid * KDIM0    , GF_, false);
        Vector<Real> Ga(Ngrid            , Ga_, false);

        SetPatch(G , t, p, S.Coord()     , S.NTor(), S.NPol());
        SetPatch(G0, t, p, S.TangentTor(), S.NTor(), S.NPol());
        SetPatch(G1, t, p, S.TangentPol(), S.NTor(), S.NPol());
        SetPatch(Gn, t, p, S.Normal()    , S.NTor(), S.NPol());
        SetPatch(GF, t, p, S.Density()   , S.NTor(), S.NPol());
        SetPatch(Ga, t, p, S.AreaElem()  , S.NTor(), S.NPol());
        for (Integer k = 0; k < KDIM0; k++) { // GF <-- GF * Ga * Gpou
          for (Integer i = 0; i < Ngrid; i++) {
            GF[k * Ngrid + i] *= Ga[i] * Gpou_[i];
          }
        }

        StaticArray<Real, KDIM1> U_;
        Vector<Real> U(KDIM1, U_, false);
        StaticArray<Real, COORD_DIM> TrgCoord_;
        Vector<Real> TrgCoord(COORD_DIM, TrgCoord_, false);
        for (Integer k = 0; k < COORD_DIM; k++) { // Set TrgCoord
          TrgCoord[k] = S.Coord()[(k * S.NTor() + t) * S.NPol() + p];
        }
        U.SetZero();

        ker(G, Gn, GF, TrgCoord, U);
        for (Integer k = 0; k < KDIM1; k++) { // Subtract singular part from S.Potential
          S.Potential()[(k * S.NTor() + t) * S.NPol() + p] -= U[k];
        }

      }

    private:

      void SetPatch(Vector<Real>& out, Long t0, Long p0, const Vector<Real>& in, Long Nt, Long Np) {
        PVFMM_ASSERT(Nt >= PATCH_DIM);
        PVFMM_ASSERT(Np >= PATCH_DIM);

        Long dof = in.Dim() / (Nt * Np);
        PVFMM_ASSERT(in.Dim() == dof * Nt * Np);
        PVFMM_ASSERT(out.Dim() == dof * PATCH_DIM * PATCH_DIM);
        for (Long k = 0; k < dof; k++) {
          for (Long i = 0; i < PATCH_DIM; i++) {
            for (Long j = 0; j < PATCH_DIM; j++) {
              Long t = t0 + i - (PATCH_DIM - 1) / 2;
              Long p = p0 + j - (PATCH_DIM - 1) / 2;
              t -= (t >= Nt ? Nt : 0);
              p -= (p >= Np ? Np : 0);
              t += (t < 0 ? Nt : 0);
              p += (p < 0 ? Np : 0);
              out[(k * PATCH_DIM + i) * PATCH_DIM + j] = in[(k * Nt + t) * Np + p];
            }
          }
        }
      }

      StaticArray<Real, Ngrid * COORD_DIM> G_;
      StaticArray<Real, Ngrid * COORD_DIM> GF_;
      StaticArray<Real, Ngrid * COORD_DIM> G0_;
      StaticArray<Real, Ngrid * COORD_DIM> G1_;
      StaticArray<Real, Ngrid * KDIM0> Gn_;
      StaticArray<Real, Ngrid> Gpou_;
      StaticArray<Real, Ngrid> Ga_;

      StaticArray<Real, Npolar * COORD_DIM> P_;
      StaticArray<Real, Npolar * COORD_DIM> P0_;
      StaticArray<Real, Npolar * COORD_DIM> P1_;
      StaticArray<Real, Npolar * COORD_DIM> Pn_;
      StaticArray<Real, Npolar * KDIM0> PF_;
      StaticArray<Real, Npolar> Ppou_;
      StaticArray<Real, Npolar> Pa_;

      Matrix<Real> G2P, G2P_grad0, G2P_grad1;
      Kernel<Real> ker;
  };

  template <class Real> void test() {
    Long Nt = 200, Np = 200;
    pvfmm::Profile::Tic("Trapezoidal");
    Surface<Real> S(Nt, Np);
    pvfmm::Profile::Toc();

    pvfmm::Profile::Tic("POU-Correction");
    SingularCorrection<Real> correction;
    correction(S, 0, 0);
    std::cout<<S.Potential()[0]<<'\n';
    pvfmm::Profile::Toc();
  }

}

int main(int argc, char** argv) {
#ifdef PVFMM_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  pvfmm::Profile::Enable(true);
  pvfmm::test<double>();
  pvfmm::Profile::print();

#ifdef PVFMM_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
