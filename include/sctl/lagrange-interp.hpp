#ifndef _SCTL_LAGRANGE_INTERP_HPP_
#define _SCTL_LAGRANGE_INTERP_HPP_

#include <sctl/common.hpp>

namespace SCTL_NAMESPACE {

  template <class ValueType> class Vector;

  template <class Real> class LagrangeInterp {
    public:
      static void Interpolate(Vector<Real>& wts, const Vector<Real>& src_nds, const Vector<Real>& trg_nds);

      static void Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds);

      static void test();
  };

}

#include SCTL_INCLUDE(lagrange-interp.txx)

#endif //_SCTL_LAGRANGE_INTERP_HPP_
