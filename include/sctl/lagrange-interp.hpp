#ifndef _SCTL_LAGRANGE_INTERP_HPP_
#define _SCTL_LAGRANGE_INTERP_HPP_

#include "sctl/common.hpp"  // for SCTL_NAMESPACE

namespace SCTL_NAMESPACE {

  template <class ValueType> class Vector;

  /**
   * @brief A class for Lagrange interpolation.
   *
   * This class provides functionality for Lagrange interpolation,
   * including computing interpolation weights and derivatives.
   *
   * @tparam Real The type of the interpolation nodes and values.
   */
  template <class Real> class LagrangeInterp {
    public:
      /**
       * @brief Compute the Lagrange interpolation weights.
       *
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
       * @brief Compute the derivative of interpolated values.
       *
       * This function computes the derivative of interpolated values
       * at given nodes.
       *
       * @param[out] df The vector storing the derivative values.
       * @param[in] f The vector of function values at the interpolation nodes.
       * @param[in] nds The vector of node positions.
       */
      static void Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds);

      /**
       * @brief A test function for Lagrange interpolation.
       *
       * This function performs a simple test of Lagrange interpolation
       * and derivative computation.
       */
      static void test();

  };

}

#endif // _SCTL_LAGRANGE_INTERP_HPP_
