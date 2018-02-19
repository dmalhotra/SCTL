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

    static void SHCEval(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& cos_theta_phi_in, Vector<Real>& X_out);

    static void SHC2Pole(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, Vector<Real>& P_out);

    static void WriteVTK(const char* fname, const Vector<Real>* S, const Vector<Real>* f_val, SHCArrange arrange, Long p_in, Long p_out, Real period=0, const Comm& comm = Comm::World());


    // Vector Spherical Harmonics
    static void Grid2VecSHC(const Vector<Real>& X_in, Long Nt_in, Long Np_in, Long p_out, Vector<Real>& S_out, SHCArrange arrange_out);

    static void VecSHC2Grid(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, Long Nt_out, Long Np_out, Vector<Real>& X_out);

    static void VecSHCEval(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& cos_theta_phi_in, Vector<Real>& X_out);

    static void StokesEvalSL(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& coord_in, Vector<Real>& X_out);

    static void StokesEvalDL(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, const Vector<Real>& coord_in, Vector<Real>& X_out);


    static void test_stokes() {
      int p = 4;
      int dof = 3;
      int Nt = p+1, Np = 2*p+1;

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
        for (Long i = 0; i < Nt; i++) {
          for (Long j = 0; j < Np; j++) {
            f[(0 * Nt + i) * Np + j] = 3;
            f[(1 * Nt + i) * Np + j] = 2;
            f[(2 * Nt + i) * Np + j] = 1;
          }
        }
      }

      Vector<Real> f_coeff;
      Grid2VecSHC(f, Nt, Np, p, f_coeff, sctl::SHCArrange::ROW_MAJOR);
      print_coeff(f_coeff);

      for (Long i = 0; i < 20; i++) { // Evaluate
        Vector<Real> Df;
        Vector<Real> x(3);
        x[0] = drand48();
        x[1] = drand48();
        x[2] = drand48();
        Real R = sqrt<Real>(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        x[0]/=R*(i+0.5)/10;
        x[1]/=R*(i+0.5)/10;
        x[2]/=R*(i+0.5)/10;

        StokesEvalDL(f_coeff, sctl::SHCArrange::ROW_MAJOR, p, x, Df);
        std::cout<<Df+1e-10;
      }
      Clear();
    }

    static void test() {
      int p = 3;
      int dof = 1;
      int Nt = p+1, Np = 2*p+1;

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

      SHC2Grid(Xcoeff, sctl::SHCArrange::COL_MAJOR_NONZERO, p, Nt, Np, &Xgrid);
      std::cout<<Matrix<Real>(Nt*dof, Np, Xgrid.begin())<<'\n';

      {
        Vector<Real> val;
        SHCEval(Xcoeff, sctl::SHCArrange::COL_MAJOR_NONZERO, p, theta_phi, val);
        Matrix<Real>(dof, val.Dim()/dof, val.begin(), false) = Matrix<Real>(val.Dim()/dof, dof, val.begin()).Transpose();
        std::cout<<Matrix<Real>(val.Dim()/Np, Np, val.begin()) - Matrix<Real>(Nt*dof, Np, Xgrid.begin())+1e-10<<'\n';
      }

      Grid2SHC(Xgrid, Nt, Np, p, Xcoeff, sctl::SHCArrange::ROW_MAJOR);
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

