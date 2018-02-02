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

    static void Grid2SHC(const Vector<Real>& X_in, Long Nt_in, Long Np_in, Long p_out, Vector<Real>& S_out, SHCArrange arrange_out);

    static void SHC2Grid(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, Long Nt_out, Long Np_out, Vector<Real>* X_out, Vector<Real>* X_theta_out=nullptr, Vector<Real>* X_phi_out=nullptr);

    static void SHC2Pole(const Vector<Real>& S_in, SHCArrange arrange_in, Long p_in, Vector<Real>& P_out);

    static void WriteVTK(const char* fname, const Vector<Real>* S, const Vector<Real>* f_val, SHCArrange arrange, Long p_in, Long p_out, Real period=0, const Comm& comm = Comm::World());

    static void test() {
      int p = 3;
      int dof = 2;

      int Ncoeff = (p + 1) * (p + 1);
      Vector<Real> Xcoeff(dof * Ncoeff);
      for (int i=0;i<Xcoeff.Dim();i++) Xcoeff[i]=i;

      Vector<Real> Xgrid;
      int Nt = p+1, Np = 2*p+1;
      SHC2Grid(Xcoeff, sctl::SHCArrange::COL_MAJOR_NONZERO, p, Nt, Np, &Xgrid);
      Grid2SHC(Xgrid, Nt, Np, p, Xcoeff, sctl::SHCArrange::ROW_MAJOR);

      int indx=0;
      for (int i=0;i<dof;i++) {
        for (int n=0;n<=p;n++){
          std::cout<<Vector<Real>(2*n+2,Xcoeff.begin()+indx);
          indx+=2*n+2;
        }
      }

      SphericalHarmonics<Real>::WriteVTK("test", nullptr, &Xcoeff, sctl::SHCArrange::ROW_MAJOR, p, 32);
    }

    static void Clear() { MatrixStore().Resize(0); }

  private:

    // Probably don't work anymore, need to be updated :(
    static void SHC2GridTranspose(const Vector<Real>& X, Long p0, Long p1, Vector<Real>& S);
    static void RotateAll(const Vector<Real>& S, Long p0, Long dof, Vector<Real>& S_);
    static void RotateTranspose(const Vector<Real>& S_, Long p0, Long dof, Vector<Real>& S);
    static void StokesSingularInteg(const Vector<Real>& S, Long p0, Long p1, Vector<Real>* SLMatrix=nullptr, Vector<Real>* DLMatrix=nullptr);

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

