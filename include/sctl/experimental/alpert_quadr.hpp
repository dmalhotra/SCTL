#ifndef _SCTL_ALPERT_QUADR_HPP_
#define _SCTL_ALPERT_QUADR_HPP_

#include "sctl/common.hpp"            // for Integer
#include "sctl/vector.hpp"            // for Vector

namespace sctl {

  /**
   * Endpoint correction of a uniform trapezoidal rule (Alpert's hybrid
   * Gauss-trapezoidal rules). On the grid a, a+h, a+2h, ..., the correction
   * removes the NodesToSkip grid points nearest the endpoint a, the endpoint
   * included, and adds the points a + ExtraNodes[k]*h with weights
   * ExtraWeights[k]*h; the remaining grid points keep weight h.
   */
  template <class Real> struct ExtraPtResult {
    Vector<Real> ExtraNodes; // offsets from the endpoint, in units of h
    Vector<Real> ExtraWeights; // weights, in units of h
    Integer NodesToSkip = 0; // grid points removed at the endpoint, the endpoint included
  };

  /**
   * Returns the endpoint correction for an integrand with a logarithmic
   * singularity at the endpoint.
   *
   * @param[in] order order of the corrected rule; one of {2, 3, 4, 5, 6, 8,
   * 10, 12, 14, 16}.
   *
   * @return the correction nodes and weights, and the number of grid points
   * they replace.
   */
  template <class Real> ExtraPtResult<Real> QuadLogExtraPtNodes(const Integer order);

  /**
   * Returns the endpoint correction for an integrand that is smooth at the
   * endpoint.
   *
   * @param[in] order order of the corrected rule, at most 32. The rule of the
   * smallest tabulated order not less than it is returned; the tabulated
   * orders are {3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32}.
   *
   * @return the correction nodes and weights, and the number of grid points
   * they replace.
   */
  template <class Real> ExtraPtResult<Real> QuadSmoothExtraPtNodes(const Integer order);

}

#endif // _SCTL_ALPERT_QUADR_HPP_
