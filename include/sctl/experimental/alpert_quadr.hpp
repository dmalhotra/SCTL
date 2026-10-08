#ifndef _SCTL_ALPERT_QUADR_HPP_
#define _SCTL_ALPERT_QUADR_HPP_

#include "sctl/common.hpp"            // for Integer
#include "sctl/vector.hpp"            // for Vector

namespace sctl {

  /**
   * Alpert's hybrid Gauss-trapezoidal quadrature rules: endpoint corrections
   * of a uniform trapezoidal rule, for integrands that are smooth or have a
   * logarithmic singularity at the endpoint. Each function builds the
   * corrections of all orders on its first call and keeps them for the
   * lifetime of the program; the functions are safe to call from several
   * threads at once.
   */
  template <class Real> class AlpertQuadRule {
    public:

      /**
       * Endpoint correction of one order. On the grid a, a+h, a+2h, ..., it
       * removes the nskip grid points nearest the endpoint a, the endpoint
       * included, and adds the points a + nds[k]*h with weights wts[k]*h; the
       * remaining grid points keep weight h.
       */
      struct EndpointCorrection {
        Vector<Real> nds; // offsets from the endpoint, in units of h
        Vector<Real> wts; // weights, in units of h
        Integer nskip = 0; // grid points removed at the endpoint, the endpoint included
      };

      /**
       * Returns the endpoint correction for an integrand with a logarithmic
       * singularity at the endpoint.
       *
       * @param[in] order order of the corrected rule; one of {2, 3, 4, 5, 6, 8,
       * 10, 12, 14, 16}.
       *
       * @return const reference to the correction of that order.
       */
      static const EndpointCorrection& LogCorrection(const Integer order);

      /**
       * Returns the endpoint correction for an integrand that is smooth at the
       * endpoint.
       *
       * @param[in] order order of the corrected rule, at most 32. The
       * correction of the smallest tabulated order not less than it is
       * returned; the tabulated orders are {3, 4, 5, 6, 7, 8, 12, 16, 20, 24,
       * 28, 32}.
       *
       * @return const reference to the correction of that order.
       */
      static const EndpointCorrection& SmoothCorrection(const Integer order);
  };

}

#endif // _SCTL_ALPERT_QUADR_HPP_
