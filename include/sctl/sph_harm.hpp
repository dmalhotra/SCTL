#ifndef _SCTL_SPH_HARM_HPP_
#define _SCTL_SPH_HARM_HPP_

#include <vector>             // for vector

#include "sctl/common.hpp"    // for Long, Integer, sctl
#include "sctl/comm.hpp"      // for Comm
#include "sctl/comm.txx"      // for Comm::World
#include "sctl/iterator.hpp"  // for Iterator
#include "sctl/iterator.txx"  // for NullIterator
#include "sctl/mem_mgr.txx"   // for aligned_delete, aligned_new

#define SCTL_SHMAXDEG 1024

namespace sctl {

template <class ValueType> class FFT;
template <class ValueType> class Vector;
template <class ValueType> class Matrix;

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
     * Compute spherical harmonic coefficients from grid values.
     * \param[in] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), X(t1,p1), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     * \param[in] Nt Number of grid points \theta \in (0,pi).
     * \param[in] Np Number of grid points \phi \in (0,2*pi).
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[out] S Spherical harmonic coefficients.
     */
    static void Grid2SHC(const Vector<Real>& X, Long Nt, Long Np, Long p, Vector<Real>& S, SHCArrange arrange);

    /**
     * Evaluate grid values from spherical harmonic coefficients.
     * \param[in] S Spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Nt Number of grid points \theta \in (0,pi).
     * \param[in] Np Number of grid points \phi \in (0,2*pi).
     * \param[out] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), X(t1,p1), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     * \param[out] X_theta \theta derivative of X evaluated at grid points.
     * \param[out] X_phi \phi derivative of X evaluated at grid points.
     */
    static void SHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p, Long Nt, Long Np, Vector<Real>* X, Vector<Real>* X_theta=nullptr, Vector<Real>* X_phi=nullptr);

    /**
     * Evaluate point values from spherical harmonic coefficients.
     * \param[in] S Spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] theta_phi Evaluation coordinates given as {t0,p0, t1,p1, ... }.
     * \param[out] X Evaluated values {X0, X1, ... }.
     */
    static void SHCEval(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& theta_phi, Vector<Real>& X);

    static void SHC2Pole(const Vector<Real>& S, SHCArrange arrange, Long p, Vector<Real>& P);

    static void WriteVTK(const char* fname, const Vector<Real>* S, const Vector<Real>* f_val, SHCArrange arrange, Long p_in, Long p_out, Real period=0, const Comm& comm = Comm::World());


    // Vector Spherical Harmonics

    /**
     * Compute vector spherical harmonic coefficients from grid values.
     * \param[in] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), ... , Y(t0,p0), ... , Z(t0,p0), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     * \param[in] Nt Number of grid points \theta \in (0,pi).
     * \param[in] Np Number of grid points \phi \in (0,2*pi).
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[out] S Vector spherical harmonic coefficients.
     */
    static void Grid2VecSHC(const Vector<Real>& X, Long Nt, Long Np, Long p, Vector<Real>& S, SHCArrange arrange);

    /**
     * Evaluate grid values from vector spherical harmonic coefficients.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Nt Number of grid points \theta \in (0,pi).
     * \param[in] Np Number of grid points \phi \in (0,2*pi).
     * \param[out] X Grid values {X(t0,p0), X(t0,p1), ... , X(t1,p0), X(t1,p1), ... , Y(t0,p0), ... , Z(t0,p0), ... }, where, {cos(t0), cos(t1), ... } are the Gauss-Legendre nodes of order (Nt-1) in the interval [-1,1] and {p0, p1, ... } are equispaced in [0, 2*pi].
     */
    static void VecSHC2Grid(const Vector<Real>& S, SHCArrange arrange, Long p, Long Nt, Long Np, Vector<Real>& X);

    /**
     * Evaluate point values from vector spherical harmonic coefficients.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] theta_phi Evaluation coordinates given as {t0,p0, t1,p1, ... }.
     * \param[out] X Evaluated values {X0,Y0,Z0, X1,Y1,Z1, ... }.
     */
    static void VecSHCEval(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& theta_phi, Vector<Real>& X);

    static void LaplaceEvalSL(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, bool interior, Vector<Real>& U);
    static void LaplaceEvalDL(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, bool interior, Vector<Real>& U);

    /**
     * Evaluate Stokes single-layer operator at point values from the vector spherical harmonic coefficients for the density.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Evaluation coordinates given as {x0,y0,z0, x1,y1,z1, ... }.
     * \param[out] U Evaluated values {Ux0,Uy0,Uz0, Ux1,Uy1,Uz1, ... }.
     */
    static void StokesEvalSL(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, bool interior, Vector<Real>& U);

    /**
     * Evaluate Stokes double-layer operator at point values from the vector spherical harmonic coefficients for the density.
     * \param[in] S Vector spherical harmonic coefficients.
     * \param[in] arrange Arrangement of the coefficients.
     * \param[in] p Order of spherical harmonic expansion.
     * \param[in] Evaluation coordinates given as {x0,y0,z0, x1,y1,z1, ... }.
     * \param[out] U Evaluated values {Ux0,Uy0,Uz0, Ux1,Uy1,Uz1, ... }.
     */
    static void StokesEvalDL(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, bool interior, Vector<Real>& U);

    static void StokesEvalKL(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, const Vector<Real>& norm, bool interior, Vector<Real>& U);

    static void StokesEvalKSelf(const Vector<Real>& S, SHCArrange arrange, Long p, const Vector<Real>& coord, bool interior, Vector<Real>& U);

    /**
     * Nodes and weights for Gauss-Legendre quadrature rule
     */
    static const Vector<Real>& LegendreNodes(Long p1);
    static const Vector<Real>& LegendreWeights(Long p1);

    static void test_stokes();

    static void test();

    /**
     * Clear all precomputed data. This must be done before the program exits to avoid memory leaks.
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
     * Computes all the Associated Legendre Polynomials (normalized) up to the specified degree.
     * \param[in] degree The degree up to which the Legendre polynomials have to be computed.
     * \param[in] X The input values for which the polynomials have to be computed.
     * \param[in] N The number of input points.
     * \param[out] poly_val The output array of size (degree+1)*(degree+2)*N/2 containing the computed polynomial values.
     * The output values are in the order:
     * P(n,m)[i] => {P(0,0)[0], P(0,0)[1], ..., P(0,0)[N-1], P(1,0)[0], ..., P(1,0)[N-1],
     * P(2,0)[0], ..., P(degree,0)[N-1], P(1,1)[0], ...,P(2,1)[0], ..., P(degree,degree)[N-1]}
     */
    static void LegPoly(Vector<Real>& poly_val, const Vector<Real>& X, Long degree);
    static void LegPoly_(Vector<Real>& poly_val, const Vector<Real>& theta, Long degree);
    static void LegPolyDeriv(Vector<Real>& poly_val, const Vector<Real>& X, Long degree);
    static void LegPolyDeriv_(Vector<Real>& poly_val, const Vector<Real>& X, Long degree);

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
    static void SHBasisEval(Long p, const Vector<Real>& theta_phi, Matrix<Real>& M);
    static void VecSHBasisEval(Long p, const Vector<Real>& theta_phi, Matrix<Real>& M);

    static const std::vector<Matrix<Real>>& MatRotate(Long p0);

    template <bool SLayer, bool DLayer> static void StokesSingularInteg_(const Vector<Real>& X0, Long p0, Long p1, Vector<Real>& SL, Vector<Real>& DL);

    struct MatrixStorage{
      MatrixStorage() : Mfft_(NullIterator<FFT<Real>>()), Mfftinv_(NullIterator<FFT<Real>>()) {
        Resize(SCTL_SHMAXDEG);
      }
      ~MatrixStorage() {
        Resize(0);
      }
      MatrixStorage(const MatrixStorage&) = delete;
      MatrixStorage& operator=(const MatrixStorage&) = delete;

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

        aligned_delete(Mfft_);
        aligned_delete(Mfftinv_);
        if (size) {
          Mfft_ = aligned_new<FFT<Real>>(size);
          Mfftinv_ = aligned_new<FFT<Real>>(size);
        } else {
          Mfft_ = NullIterator<FFT<Real>>();
          Mfftinv_ = NullIterator<FFT<Real>>();
        }
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

      Iterator<FFT<Real>> Mfft_;
      Iterator<FFT<Real>> Mfftinv_;
    };
    static MatrixStorage& MatrixStore(){
      static MatrixStorage storage;
      if (!storage.Qx_.size()) storage.Resize(SCTL_SHMAXDEG);
      return storage;
    }
};

//template class SphericalHarmonics<double>;

}  // end namespace

#endif // _SCTL_SPH_HARM_HPP_
