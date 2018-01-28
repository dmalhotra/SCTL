#ifndef _SCTL_SPH_HARM_HPP_
#define _SCTL_SPH_HARM_HPP_

#define SCTL_SHMAXDEG 256

#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(fft_wrapper.hpp)
#include SCTL_INCLUDE(common.hpp)

namespace SCTL_NAMESPACE {

template <class Real> class SphericalHarmonics{
  static constexpr Integer COORD_DIM = 3;

  public:

    // TODO: Ynm *= sqrt(2)*(m==0?1:2);

    static void SHC2Grid(const Vector<Real>& S, Long p0, Long p1, Vector<Real>& X, Vector<Real>* X_theta=nullptr, Vector<Real>* X_phi=nullptr);

    static void Grid2SHC(const Vector<Real>& X, Long Nt, Long Np, Long p1, Vector<Real>& S);
    static void Grid2SHC(const Vector<Real>& X, Long          p0, Long p1, Vector<Real>& S);

    static void SHC2GridTranspose(const Vector<Real>& X, Long p0, Long p1, Vector<Real>& S);

    static void SHC2Pole(const Vector<Real>& S, Long p0, Vector<Real>& P);

    static void RotateAll(const Vector<Real>& S, Long p0, Long dof, Vector<Real>& S_);

    static void RotateTranspose(const Vector<Real>& S_, Long p0, Long dof, Vector<Real>& S);

    static void StokesSingularInteg(const Vector<Real>& S, Long p0, Long p1, Vector<Real>* SLMatrix=nullptr, Vector<Real>* DLMatrix=nullptr);

    static void WriteVTK(const char* fname, long p0, long p1, Real period=0, const Vector<Real>* S=nullptr, const Vector<Real>* f_val=nullptr, MPI_Comm comm=MPI_COMM_WORLD);

  private:

    static Vector<Real>& LegendreNodes(Long p1);

    static Vector<Real>& LegendreWeights(Long p1);

    static Vector<Real>& SingularWeights(Long p1);

    static Matrix<Real>& MatFourier(Long p0, Long p1);

    static Matrix<Real>& MatFourierInv(Long p0, Long p1);

    static Matrix<Real>& MatFourierGrad(Long p0, Long p1);

    static std::vector<Matrix<Real>>& MatLegendre(Long p0, Long p1);

    static std::vector<Matrix<Real>>& MatLegendreInv(Long p0, Long p1);

    static std::vector<Matrix<Real>>& MatLegendreGrad(Long p0, Long p1);

    static std::vector<Matrix<Real>>& MatRotate(Long p0);

    /**
     * \brief Computes all the Associated Legendre Polynomials (normalized) upto the specified degree.
     * \param[in] degree The degree upto which the legendre polynomials have to be computed.
     * \param[in] X The input values for which the polynomials have to be computed.
     * \param[in] N The number of input points.
     * \param[out] poly_val The output array of size (degree+1)*(degree+2)*N/2 containing the computed polynomial values.
     * The output values are in the order:
     * P(n,m)[i] => {P(0,0)[0], P(0,0)[1], ..., P(0,0)[N-1], P(1,0)[0], ..., P(1,0)[N-1],
     * P(2,0)[0], ..., P(degree,0)[N-1], P(1,1)[0], ...,P(2,1)[0], ..., P(degree,degree)[N-1]}
     */
    //static void LegPoly(Vector<Real>& poly_val, const Vector<Real>& X, Long degree);
    //static void LegPolyDeriv(Vector<Real>& poly_val, const Vector<Real>& X, Long degree);

    static void LegPoly(Real* poly_val, const Real* X, Long N, Long degree);
    static void LegPolyDeriv(Real* poly_val, const Real* X, Long N, Long degree);

    template <bool SLayer, bool DLayer> static void StokesSingularInteg_(const Vector<Real>& X0, Long p0, Long p1, Vector<Real>& SL, Vector<Real>& DL);

    struct MatrixStorage{
      MatrixStorage(Long size){
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
    };
    static MatrixStorage& MatrixStore(){
      static MatrixStorage storage(SCTL_SHMAXDEG);
      return storage;
    }
};

template class SphericalHarmonics<double>;

}  // end namespace

#include SCTL_INCLUDE(sph_harm.txx)

#endif // _SCTL_SPH_HARM_HPP_

