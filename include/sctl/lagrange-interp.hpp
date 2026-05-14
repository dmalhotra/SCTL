#ifndef _SCTL_LAGRANGE_INTERP_HPP_
#define _SCTL_LAGRANGE_INTERP_HPP_

#include "sctl/common.hpp"  // for sctl

namespace sctl {

  template <class ValueType> class Vector;

  /**
   * This class provides functionality for Lagrange interpolation,
   * including computing interpolation weights and derivatives.
   *
   * @tparam Real The type of the interpolation nodes and values.
   */
  template <class Real> class LagrangeInterp {
    public:
      /**
       * This function computes the interpolation weights to interpolate
       * values from the source nodes to the target nodes.
       *
       * @param[out] wts The interpolation weights stored in row-major order.
       *                 The dimensions are Ns x Nt, where Ns is the number of
       *                 source nodes and Nt is the number of target nodes.
       * @param[in] src_nds The vector of source node positions.
       * @param[in] trg_nds The vector of target node positions.
       */
      static void Interpolate(Vector<Real>& wts, const Vector<Real>& src_nds, const Vector<Real>& trg_nds);

      /**
       * Compute the derivative of the unique polynomial that interpolates
       * `f` through `nds`, evaluated at those same nodes. `df[i]` is the
       * derivative at `nds[i]` (so the source nodes and the target nodes are
       * one and the same).
       *
       * @param[out] df Derivative values, same layout and length as `f`:
       * `df.Dim() == f.Dim() == dof * nds.Dim()`. Reallocated if needed.
       *
       * @param[in] f The vector of function values at the interpolation nodes.
       * Multiple scalar functions may be passed as: f = [f1(x1), f1(x2), ...,
       * f1(xN), f2(x1), f2(x2), ..., f2(xN), ...], where x_i are the
       * interpolation nodes. `f.Dim()` must be a multiple of `nds.Dim()`.
       *
       * @param[in] nds The interpolation node positions. Must be the same
       * nodes at which `f` is sampled. Nodes must be distinct (the
       * barycentric form divides by `nds[i] - nds[j]`).
       */
      static void Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds);

      /**
       * This function performs a simple test of Lagrange interpolation
       * and derivative computation.
       */
      static void test();

  };

}

#endif // _SCTL_LAGRANGE_INTERP_HPP_
