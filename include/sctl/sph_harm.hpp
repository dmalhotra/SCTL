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

    /**
     * \brief Compute spherical harmonic coefficients from grid values.
     * \param[in] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), X(t1,p1), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     * \param[in] Nt Number of grid points \theta \in (1,pi).
     * \param[in] Np Number of grid points \phi \in (1,2*pi).
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[out] S Spherical harmonic coefficients.
     */
    static void Grid2SHC(const Vector<Real>& X, Long Nt, Long Np, Long p, Vector<Real>& S, SHCArrange arrange);

    /**
     * \brief Evaluate grid values from spherical harmonic coefficients.
     * \param[in] S Spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Nt Number of grid points \theta \in (1,pi).
     * \param[in] Np Number of grid points \phi \in (1,2*pi).
     * \param[out] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), X(t1,p1), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     * \param[out] X_theta \theta derivative of X evaluated at grid points.
     * \param[out] X_phi \phi derivative of X evaluated at grid points.
     */
    static void SHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p, Long Nt, Long Np, Vector<Real>* X, Vector<Real>* X_theta=nullptr, Vector<Real>* X_phi=nullptr);

    /**
     * \brief Evaluate point values from spherical harmonic coefficients.
     * \param[in] S Spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] cos_theta_phi Evaluation coordinates given as {cos(t0),p0, cos(t1),p1, ... }.
     * \param[out] X Evaluated values {X0, X1, ... }.
     */
    static void SHCEval(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& cos_theta_phi, Vector<Real>& X);

    static void SHC2Pole(const Vector<Real>& S, SHCArrange arrange, Long p, Vector<Real>& P);

    static void WriteVTK(const char* fname, const Vector<Real>* S, const Vector<Real>* f_val, SHCArrange arrange, Long p_in, Long p_out, Real period=0, const Comm& comm = Comm::World());


    // Vector Spherical Harmonics

    /**
     * \brief Compute vector spherical harmonic coefficients from grid values.
     * \param[in] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), ... , Y(t0,p0), ... , Z(t0,p0), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     * \param[in] Nt Number of grid points \theta \in (1,pi).
     * \param[in] Np Number of grid points \phi \in (1,2*pi).
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[out] S Vector spherical harmonic coefficients.
     */
    static void Grid2VecSHC(const Vector<Real>& X, Long Nt, Long Np, Long p, Vector<Real>& S, SHCArrange arrange);

    /**
     * \brief Evaluate grid values from vector spherical harmonic coefficients.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Nt Number of grid points \theta \in (1,pi).
     * \param[in] Np Number of grid points \phi \in (1,2*pi).
     * \param[out] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), X(t1,p1), ... , Y(t0,p0), ... , Z(t0,p0), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     */
    static void VecSHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p, Long Nt, Long Np, Vector<Real>& X);

    /**
     * \brief Evaluate point values from vector spherical harmonic coefficients.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] cos_theta_phi Evaluation coordinates given as {cos(t0),p0, cos(t1),p1, ... }.
     * \param[out] X Evaluated values {X0,Y0,Z0, X1,Y1,Z1, ... }.
     */
    static void VecSHCEval(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& cos_theta_phi, Vector<Real>& X);

    /**
     * \brief Evaluate Stokes single-layer operator at point values from the vector spherical harmonic coefficients for the density.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Evaluation coordinates given as {x0,y0,z0, x1,y1,z1, ... }.
     * \param[out] U Evaluated values {Ux0,Uy0,Uz0, Ux1,Uy1,Uz1, ... }.
     */
    static void StokesEvalSL(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, Vector<Real>& U);

    /**
     * \brief Evaluate Stokes double-layer operator at point values from the vector spherical harmonic coefficients for the density.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Evaluation coordinates given as {x0,y0,z0, x1,y1,z1, ... }.
     * \param[out] U Evaluated values {Ux0,Uy0,Uz0, Ux1,Uy1,Uz1, ... }.
     */
    static void StokesEvalDL(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, Vector<Real>& U);


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

    /**
     * \brief Clear all precomputed data. This must be done before the program exits to avoid memory leaks.
     */
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

