#ifndef _SCTL_SPH_HARM_HPP_
#define _SCTL_SPH_HARM_HPP_

#define SCTL_SHMAXDEG 1024

#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(fft_wrapper.hpp)
#include SCTL_INCLUDE(common.hpp)

namespace SCTL_NAMESPACE {

enum class SHCArrange {
  // (p+1) x (p+1) complex elements in row-major order.
  // A : { A(0,0), A(0,1), ... A(0,p), A(1,0), ... A(p,p) }
  // where, A(n,m) = { Ar(n,m), Ai(n,m) } (real and imaginary parts)
  ALL,

  // (p+1)(p+2)/2  complex elements in row-major order (lower triangular part)
  // A : { A(0,0), A(1,0), A(1,1), A(2,0), A(2,1), A(2,2), ... A(p,p) }
  // where, A(n,m) = { Ar(n,m), Ai(n,m) } (real and imaginary parts)
  ROW_MAJOR,

  // (p+1)(p+1) real elements in col-major order (non-zero lower triangular part)
  // A : { Ar(0,0), Ar(1,0), ... Ar(p,0), Ar(1,1), ... Ar(p,1), Ai(1,1), ... Ai(p,1), ..., Ar(p,p), Ai(p,p)
  // where, A(n,m) = { Ar(n,m), Ai(n,m) } (real and imaginary parts)
  COL_MAJOR_NONZERO
};

template <class Real> class SphericalHarmonics{
  static constexpr Integer COORD_DIM = 3;

  public:

    // Scalar Spherical Harmonics
    static void Grid2SHC(const Vector<Real>& X_in, Long Nt_in, Long Np_in, Long p_out, Vector<Real>& S_out, SHCArrange arrange_out);

    static void SHC2Grid(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, Long Nt_out, Long Np_out, Vector<Real>* X_out, Vector<Real>* X_theta_out=nullptr, Vector<Real>* X_phi_out=nullptr);

    static void SHCEval(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& theta_phi_in, Vector<Real>& X_out);

    static void SHC2Pole(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, Vector<Real>& P_out);

    static void WriteVTK(const char* fname, const Vector<Real>* S, const Vector<Real>* f_val, SHCArrange arrange, Long p_in, Long p_out, Real period=0, const Comm& comm = Comm::World());


    // Vector Spherical Harmonics
    static void Grid2VecSHC(const Vector<Real>& X_in, Long Nt_in, Long Np_in, Long p_out, Vector<Real>& S_out, SHCArrange arrange_out);

    static void VecSHC2Grid(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, Long Nt_out, Long Np_out, Vector<Real>& X_out);

    static void VecSHCEval(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& theta_phi_in, Vector<Real>& X_out);

    static void StokesEvalSL(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& theta_phi_in, Vector<Real>& X_out);

    static void StokesEvalDL(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& theta_phi_in, Vector<Real>& X_out);


    static void test3() {
      int k=0, n=1, m=0;
      for (int k=0;k<3;k++) {
        for (int n=0;n<3;n++) {
          for (int m=0;m<=n;m++) {

      int p=5;
      int dof = 1;
      int Nt = p+1;
      int Np = 2*p+2;
      int Ngrid = Nt * Np;
      int Ncoeff = (p + 1) * (p + 2);
      Vector<Real> sin_theta, cos_theta = LegendreNodes(Nt-1);
      for (Long i=0;i<cos_theta.Dim();i++) sin_theta.PushBack(sqrt<Real>(1-cos_theta[i]*cos_theta[i]));

      auto print_coeff = [&](Vector<Real> S) {
        Long idx=0;
        Long dof = S.Dim() / Ncoeff;
        for (Long k=0;k<dof;k++) {
          for (Long n=0;n<=p;n++) {
            std::cout<<Vector<Real>(2*n+2, S.begin()+idx);
            idx+=2*n+2;
          }
        }
        std::cout<<'\n';
      };

      Vector<Real> coeff(dof*Ncoeff); coeff=0;
      auto write_coeff = [&](Vector<Real>& coeff, Complex<Real> c, Long k, Long n, Long m) {
        if (0 <= m && m <= n && n <= p) {
          Long idx = k*Ncoeff + n*(n+1)+ 2*m;
          coeff[idx+0] = c.real;
          coeff[idx+1] = c.imag;
        }
      };
      write_coeff(coeff, Complex<Real>(1,1.3),0,n,m);
      //print_coeff(coeff);

      Vector<Real> Y, Yt, Yp;
      SHC2Grid(coeff, SHCArrange::ROW_MAJOR, p, Nt, Np, &Y, &Yt, &Yp);
      //std::cout<<Matrix<Real>(Nt, Np, Y.begin())<<'\n';
      //std::cout<<Matrix<Real>(Nt, Np, Yt.begin())<<'\n';
      //std::cout<<Matrix<Real>(Nt, Np, Yp.begin())<<'\n';

      Vector<Real> gradYt = Yt;
      Vector<Real> gradYp = Yp;
      for (Long i=0;i<Nt;i++) {
        for (Long j=0;j<Np;j++) {
          gradYp[i*Np+j] /= sin_theta[i];
        }
      }

      Vector<Real> Vr, Vt, Vp;
      Vector<Real> Wr, Wt, Wp;
      Vector<Real> Xr, Xt, Xp;

      Vr = Y*(-n-1);
      Vt = gradYt;
      Vp = gradYp;

      Wr = Y*n;
      Wt = gradYt;
      Wp = gradYp;

      Xr = Y*0;
      Xt = gradYp;
      Xp = gradYt * (-1);

      Vector<Real> SS(COORD_DIM * Ngrid);
      SS=0;
      if (k == 0) {
        Vector<Real>(Ngrid, SS.begin() + 0*Ngrid, false) = Vr;
        Vector<Real>(Ngrid, SS.begin() + 1*Ngrid, false) = Vt;
        Vector<Real>(Ngrid, SS.begin() + 2*Ngrid, false) = Vp;
      }
      if (k == 1) {
        Vector<Real>(Ngrid, SS.begin() + 0*Ngrid, false) = Wr;
        Vector<Real>(Ngrid, SS.begin() + 1*Ngrid, false) = Wt;
        Vector<Real>(Ngrid, SS.begin() + 2*Ngrid, false) = Wp;
      }
      if (k == 2) {
        Vector<Real>(Ngrid, SS.begin() + 0*Ngrid, false) = Xr;
        Vector<Real>(Ngrid, SS.begin() + 1*Ngrid, false) = Xt;
        Vector<Real>(Ngrid, SS.begin() + 2*Ngrid, false) = Xp;
      }

      Vector<Real> SSS;
      {
        Vector<Real> coeff(COORD_DIM*Ncoeff);
        coeff=0;
        write_coeff(coeff, Complex<Real>(1,1.3),k,n,m);
        //print_coeff(coeff);
        VecSHC2Grid(coeff, SHCArrange::ROW_MAJOR, p, Nt, Np, SSS);
      }

      //std::cout<<Matrix<Real>(COORD_DIM*Nt, Np, SS.begin())<<'\n';
      //std::cout<<Matrix<Real>(COORD_DIM*Nt, Np, SSS.begin())<<'\n';
      //std::cout<<Matrix<Real>(COORD_DIM*Nt, Np, SSS.begin()) - Matrix<Real>(COORD_DIM*Nt, Np, SS.begin())<<'\n';

      auto err=SSS-SS;
      Real max_err=0;
      for (auto x:err) max_err=std::max(max_err, fabs(x));
      std::cout<<max_err<<' ';

          }
          std::cout<<'\n';
        }
        std::cout<<'\n';
      }

      Clear();
    }

    static void test2() {
      int p = 6;
      int dof = 3;
      int Nt = p+1, Np = 2*p+2;

      auto print_coeff = [&](Vector<Real> S) {
        Long idx=0;
        for (Long k=0;k<dof;k++) {
          for (Long n=0;n<=p;n++) {
            std::cout<<Vector<Real>(2*n+2, S.begin()+idx);
            idx+=2*n+2;
          }
        }
        std::cout<<'\n';
      };

      Vector<Real> f(dof * Nt * Np);
      { // Set f
        Vector<Real> leg_nodes = LegendreNodes(Nt-1);
        for (Long i = 0; i < Nt; i++) {
          for (Long j = 0; j < Np; j++) {
            Real cos_theta = leg_nodes[i];
            Real sin_theta = sqrt<Real>(1-cos_theta*cos_theta);
            Real phi = 2 * const_pi<Real>() * j / Np;
            Real r = 1;

            Real x = r * cos_theta;
            Real y = r * sin_theta * cos<Real>(phi);
            Real z = r * sin_theta * sin<Real>(phi);

            // Unit vectors in polar coordinates
            Real xr = cos_theta;
            Real yr = sin_theta * cos<Real>(phi);
            Real zr = sin_theta * sin<Real>(phi);

            Real xt = - sin_theta;
            Real yt = cos_theta * cos<Real>(phi);
            Real zt = cos_theta * sin<Real>(phi);

            Real xp = 0;
            Real yp = - sin<Real>(phi);
            Real zp = cos<Real>(phi);

            ////////////////////////////////////

            Real fx = 0;
            Real fy = 0;
            Real fz = 1;

            f[(0 * Nt + i) * Np + j] = (fx * xr + fy * yr + fz * zr);
            f[(1 * Nt + i) * Np + j] = (fx * xt + fy * yt + fz * zt);
            f[(2 * Nt + i) * Np + j] = (fx * xp + fy * yp + fz * zp);

            f[(0 * Nt + i) * Np + j] = 0; //sin(phi) * sin_theta;
            f[(1 * Nt + i) * Np + j] = 1; //sin(phi) * cos_theta; // * sin_theta;
            f[(2 * Nt + i) * Np + j] = 1; //cos(phi)            ; // * sin_theta;
          }
        }
      }

      Vector<Real> f_coeff;
      Grid2VecSHC(f, Nt, Np, p, f_coeff, sctl::SHCArrange::ROW_MAJOR);

      if(0){
        Vector<Real> f_, f_coeff_, f__;
        SHC2Grid(f_coeff, sctl::SHCArrange::ROW_MAJOR, p, Nt, Np, &f_);
        Grid2SHC(f_, Nt, Np, p, f_coeff_, sctl::SHCArrange::ROW_MAJOR);
        SHC2Grid(f_coeff_, sctl::SHCArrange::ROW_MAJOR, p, Nt, Np, &f__);
        std::cout<<Matrix<Real>(dof*Nt, Np, f.begin())-Matrix<Real>(dof*Nt, Np, f_.begin())<<'\n';
        std::cout<<Matrix<Real>(dof*Nt, Np, f_.begin())-Matrix<Real>(dof*Nt, Np, f__.begin())<<'\n';
      }

      if(0)
      for (Long i = 0; i < 20; i++) { // Evaluate
        Vector<Real> r_cos_theta_phi;
        r_cos_theta_phi.PushBack(drand48() + (i<10?0:2)); // [1, inf)
        r_cos_theta_phi.PushBack(drand48() * 2 - 1); // [-1, 1]
        r_cos_theta_phi.PushBack(drand48() * 2 * const_pi<Real>()); // [0, 2*pi]

        Vector<Real> Df;
        StokesEvalDL(f_coeff, sctl::SHCArrange::ROW_MAJOR, p, r_cos_theta_phi, Df);
        //VecSHCEval(f_coeff, sctl::SHCArrange::ROW_MAJOR, p, r_cos_theta_phi, Df);
        //std::cout<<r_cos_theta_phi;
        std::cout<<Df[0]*Df[0] + Df[1]*Df[1] + Df[2]*Df[2]<<'\n';
      }

      print_coeff(f_coeff);
      Clear();
    }

    static void test() {
      int p = 4;
      int dof = 3;
      int Nt = p+1, Np = 2*p;

      auto print_coeff = [&](Vector<Real> S) {
        Long idx=0;
        for (Long k=0;k<dof;k++) {
          for (Long n=0;n<=p;n++) {
            std::cout<<Vector<Real>(2*n+2, S.begin()+idx);
            idx+=2*n+2;
          }
        }
        std::cout<<'\n';
      };

      Vector<Real> r_theta_phi, theta_phi;
      { // Set r_theta_phi, theta_phi
        Vector<Real> leg_nodes = LegendreNodes(Nt-1);
        for (Long i=0;i<Nt;i++) {
          for (Long j=0;j<Np;j++) {
            r_theta_phi.PushBack(1);
            r_theta_phi.PushBack(leg_nodes[i]);
            r_theta_phi.PushBack(j * 2 * const_pi<Real>() / Np);
            theta_phi.PushBack(leg_nodes[i]);
            theta_phi.PushBack(j * 2 * const_pi<Real>() / Np);
          }
        }
      }

      int Ncoeff = (p + 1) * (p + 1);
      Vector<Real> Xcoeff(dof * Ncoeff), Xgrid;
      for (int i=0;i<Xcoeff.Dim();i++) Xcoeff[i]=i+1;
      Xcoeff=0;
      Xcoeff[0]=1;
      //Xcoeff[12]=1;
      //for (int i=0*Ncoeff;i<1*Ncoeff;i++) Xcoeff[i]=i+1;

      VecSHC2Grid(Xcoeff, sctl::SHCArrange::COL_MAJOR_NONZERO, p, Nt, Np, Xgrid);
      std::cout<<"Y0_=[\n";
      std::cout<<Matrix<Real>(Nt, Np, Xgrid.begin()+Nt*Np*0)<<"];\n";
      std::cout<<"Y1_=[\n";
      std::cout<<Matrix<Real>(Nt, Np, Xgrid.begin()+Nt*Np*1)<<"];\n";
      std::cout<<"Y2_=[\n";
      std::cout<<Matrix<Real>(Nt, Np, Xgrid.begin()+Nt*Np*2)<<"];\n";

      if (0) {
        Vector<Real> val;
        VecSHCEval(Xcoeff, sctl::SHCArrange::COL_MAJOR_NONZERO, p, theta_phi, val);
        //StokesEvalSL(Xcoeff, sctl::SHCArrange::COL_MAJOR_NONZERO, p, r_theta_phi, val);
        Matrix<Real>(dof, val.Dim()/dof, val.begin(), false) = Matrix<Real>(val.Dim()/dof, dof, val.begin()).Transpose();
        std::cout<<Matrix<Real>(val.Dim()/Np, Np, val.begin())<<'\n';
        std::cout<<Matrix<Real>(val.Dim()/Np, Np, val.begin()) - Matrix<Real>(Nt*dof, Np, Xgrid.begin())<<'\n';
      }

      Grid2VecSHC(Xgrid, Nt, Np, p, Xcoeff, sctl::SHCArrange::ROW_MAJOR);
      print_coeff(Xcoeff);

      //SphericalHarmonics<Real>::WriteVTK("test", nullptr, &Xcoeff, sctl::SHCArrange::ROW_MAJOR, p, 32);
      Clear();
    }

    static void Clear() { MatrixStore().Resize(0); }

  private:

    // Probably don't work anymore, need to be updated :(
    static void SHC2GridTranspose(const Vector<Real>& X, Long p0, Long p1, Vector<Real>& S);
    static void RotateAll(const Vector<Real>& S, Long p0, Long dof, Vector<Real>& S_);
    static void RotateTranspose(const Vector<Real>& S_, Long p0, Long dof, Vector<Real>& S);
    static void StokesSingularInteg(const Vector<Real>& S, Long p0, Long p1, Vector<Real>* SLMatrix=nullptr, Vector<Real>* DLMatrix=nullptr);

    static void Grid2SHC_(const Vector<Real>& X, Long Nt, Long Np, Long p, Vector<Real>& B1);
    static void SHCArrange0(const Vector<Real>& B1, Long p, Vector<Real>& S, SHCArrange arrange);

    static void SHC2Grid_(const Vector<Real>& S, Long p, Long Nt, Long Np, Vector<Real>* X, Vector<Real>* X_theta=nullptr, Vector<Real>* X_phi=nullptr);
    static void SHCArrange1(const Vector<Real>& S_in, SHCArrange arrange_out, Long p, Vector<Real>& S_out);

    /**
     * \brief Computes all the Associated Legendre Polynomials (normalized) up to the specified degree.
     * \param[in] degree The degree up to which the Legendre polynomials have to be computed.
     * \param[in] X The input values for which the polynomials have to be computed.
     * \param[in] N The number of input points.
     * \param[out] poly_val The output array of size (degree+1)*(degree+2)*N/2 containing the computed polynomial values.
     * The output values are in the order:
     * P(n,m)[i] => {P(0,0)[0], P(0,0)[1], ..., P(0,0)[N-1], P(1,0)[0], ..., P(1,0)[N-1],
     * P(2,0)[0], ..., P(degree,0)[N-1], P(1,1)[0], ...,P(2,1)[0], ..., P(degree,degree)[N-1]}
     */
    static void LegPoly(Vector<Real>& poly_val, const Vector<Real>& X, Long degree);
    static void LegPolyDeriv(Vector<Real>& poly_val, const Vector<Real>& X, Long degree);

    static const Vector<Real>& LegendreNodes(Long p1);
    static const Vector<Real>& LegendreWeights(Long p1);
    static const Vector<Real>& SingularWeights(Long p1);

    static const Matrix<Real>& MatFourier(Long p0, Long p1);
    static const Matrix<Real>& MatFourierInv(Long p0, Long p1);
    static const Matrix<Real>& MatFourierGrad(Long p0, Long p1);

    static const FFT<Real>& OpFourier(Long Np);
    static const FFT<Real>& OpFourierInv(Long Np);

    static const std::vector<Matrix<Real>>& MatLegendre(Long p0, Long p1);
    static const std::vector<Matrix<Real>>& MatLegendreInv(Long p0, Long p1);
    static const std::vector<Matrix<Real>>& MatLegendreGrad(Long p0, Long p1);

    // Evaluate all Spherical Harmonic basis functions up to order p at (theta, phi) coordinates.
    static void SHBasisEval(Long p, const Vector<Real>& cos_theta_phi, Matrix<Real>& M);
    static void VecSHBasisEval(Long p, const Vector<Real>& cos_theta_phi, Matrix<Real>& M);

    static const std::vector<Matrix<Real>>& MatRotate(Long p0);

    template <bool SLayer, bool DLayer> static void StokesSingularInteg_(const Vector<Real>& X0, Long p0, Long p1, Vector<Real>& SL, Vector<Real>& DL);

    struct MatrixStorage{
      MatrixStorage(){
        const Long size = SCTL_SHMAXDEG;
        Resize(size);
      }

      void Resize(Long size){
        Qx_ .resize(size);
        Qw_ .resize(size);
        Sw_ .resize(size);
        Mf_ .resize(size*size);
        Mdf_.resize(size*size);
        Ml_ .resize(size*size);
        Mdl_.resize(size*size);
        Mr_ .resize(size);
        Mfinv_ .resize(size*size);
        Mlinv_ .resize(size*size);

        Mfft_.resize(size);
        Mfftinv_.resize(size);
      }

      std::vector<Vector<Real>> Qx_;
      std::vector<Vector<Real>> Qw_;
      std::vector<Vector<Real>> Sw_;
      std::vector<Matrix<Real>> Mf_ ;
      std::vector<Matrix<Real>> Mdf_;
      std::vector<std::vector<Matrix<Real>>> Ml_ ;
      std::vector<std::vector<Matrix<Real>>> Mdl_;
      std::vector<std::vector<Matrix<Real>>> Mr_;
      std::vector<Matrix<Real>> Mfinv_ ;
      std::vector<std::vector<Matrix<Real>>> Mlinv_ ;

      std::vector<FFT<Real>> Mfft_;
      std::vector<FFT<Real>> Mfftinv_;
    };
    static MatrixStorage& MatrixStore(){
      static MatrixStorage storage;
      return storage;
    }
};

template class SphericalHarmonics<double>;

}  // end namespace

#include SCTL_INCLUDE(sph_harm.txx)

#endif // _SCTL_SPH_HARM_HPP_

