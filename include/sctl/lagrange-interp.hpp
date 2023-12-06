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

      static void test();
  };

}

#endif //_SCTL_LAGRANGE_INTERP_HPP_
