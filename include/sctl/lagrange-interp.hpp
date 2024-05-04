#ifndef _SCTL_LAGRANGE_INTERP_HPP_
#define _SCTL_LAGRANGE_INTERP_HPP_

#include <sctl/common.hpp>

namespace SCTL_NAMESPACE {

  template <class ValueType> class Vector;

  template <class Real> class LagrangeInterp {
    public:
      /**
       * Compute the Lagrange interpolation weights to interpolate from values
       * at src_nds to values at trg_nds.
       *
       * \param[out] wts the entries of the interpolation matrix with
       * dimensions Ns x Nt stored in row major order.
       * (where Ns = src_nds.Dim() and Nt = trg_nds.Dim())
       *
       * \praram[in] src_nds the vector of source node positions.
       *
       * \praram[in] trg_nds the vector of target node positions.
       */
      static void Interpolate(Vector<Real>& wts, const Vector<Real>& src_nds, const Vector<Real>& trg_nds);

      static void Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds);

      static void test() {
        Vector<Real> src, trg; // source and target interpolation nodes
        for (Long i = 0; i < 3; i++) src.PushBack(i);
        for (Long i = 0; i < 11; i++) trg.PushBack(i*0.2);

        Matrix<Real> f(1,3); // values at source nodes
        f[0][0] = 0; f[0][1] = 1; f[0][2] = 0.5;

	// Interpolate
        Vector<Real> wts;
        Interpolate(wts,src,trg);
        Matrix<Real> Mwts(src.Dim(), trg.Dim(), wts.begin(), false);
        Matrix<Real> ff = f * Mwts;
        std::cout<<ff<<'\n';

	// Compute derivative
        Vector<Real> df;
        Derivative(df, Vector<Real>(f.Dim(0)*f.Dim(1),f.begin()), src);
        std::cout<<df<<'\n';
      }

  };

}

#include SCTL_INCLUDE(lagrange-interp.txx)

#endif //_SCTL_LAGRANGE_INTERP_HPP_
