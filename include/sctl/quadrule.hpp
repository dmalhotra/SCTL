#ifndef _SCTL_QUADRULE_HPP_
#define _SCTL_QUADRULE_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(math_utils.hpp)

namespace SCTL_NAMESPACE {

  /**
   * Clenshaw-Curtis quadrature rules in the interval [0,1].
   */
  template <class Real> class ChebQuadRule {
    public:

      /**
       * @brief Precompute and keep in memory all quadrature nodes up to MAX_ORDER; and
       * return const reference to a vector containing quadrature nodes of order N.
       */
      template <Integer MAX_ORDER=50, class ValueType=Real> static const Vector<Real>& nds(Integer N);

      /**
       * @brief Precompute and keep in memory all quadrature weights up to MAX_ORDER; and
       * return const reference to a vector containing quadrature weights of order N.
       */
      template <Integer MAX_ORDER=50, class ValueType=Real> static const Vector<Real>& wts(Integer N);

      /**
       * @brief Return const reference to a vector containing quadrature nodes of order N.
       */
      template <Integer N, class ValueType=Real> static const Vector<Real>& nds();

      /**
       * @brief Return const reference to a vector containing quadrature weights of order N.
       */
      template <Integer N, class ValueType=Real> static const Vector<Real>& wts();

      /**
       * @brief Compute nodes and/or weights for quadrature of order N.
       *
       * @param[out] nds (optional )pointer to vector containing the quadrature nodes.
       *
       * @param[out] wts (optional )pointer to vector containing the quadrature weights.
       *
       * @param[in] N order of the quadrature.
       */
      template <class ValueType=Real> static void ComputeNdsWts(Vector<Real>* nds, Vector<Real>* wts, Integer N);
  };

  /**
   * Gauss-Legendre quadrature rules in the interval [0,1].
   */
  template <class Real> class LegQuadRule {
    public:

      /**
       * @brief Precompute and keep in memory all quadrature nodes up to MAX_ORDER; and
       * return const reference to a vector containing quadrature nodes of order N.
       */
      template <Integer MAX_ORDER=50, class ValueType=Real> static const Vector<Real>& nds(Integer N);

      /**
       * @brief Precompute and keep in memory all quadrature weights up to MAX_ORDER; and
       * return const reference to a vector containing quadrature weights of order N.
       */
      template <Integer MAX_ORDER=50, class ValueType=Real> static const Vector<Real>& wts(Integer N);

      /**
       * @brief Return const reference to a vector containing quadrature nodes of order N.
       */
      template <Integer N, class ValueType=Real> static const Vector<Real>& nds();

      /**
       * @brief Return const reference to a vector containing quadrature weights of order N.
       */
      template <Integer N, class ValueType=Real> static const Vector<Real>& wts();

      /**
       * @brief Compute nodes and/or weights for quadrature of order N.
       *
       * @param[out] nds (optional )pointer to vector containing the quadrature nodes.
       *
       * @param[out] wts (optional )pointer to vector containing the quadrature weights.
       *
       * @param[in] N order of the quadrature.
       */
      template <class ValueType=Real> static void ComputeNdsWts(Vector<Real>* nds, Vector<Real>* wts, Integer N);

      /**
       * @brief Compute Legendre polynomial and/or its first derivative on the interval [-1,1].
       *
       * @param[out] P (optional) pointer to vector containing the values of the Legendre polynomial.
       *
       * @param[out] dP (optional) pointer to vector containing the first derivative of the Legendre polynomial.
       *
       * @param[in] X vector containing the points in [-1,1] values where the polynomial will be evaluated.
       *
       * @param[in] degree the degree of the polynomial.
       */
      template <class ValueType> static void LegPoly(Vector<ValueType>* P, Vector<ValueType>* dP, const Vector<ValueType>& X, Long degree);
  };

  /**
   * Build generalize Chebyshev quadrature rules by first finding an orthonormal basis to a given set of integrands
   * using either column pivoted QR or SVD. Then finding a set of stable interpolation nodes that serve as the
   * quadrature nodes. The quadrature weights are then computed by solving a least-squares problem. (see
   * DOI:10.1137/080737046 for algorithm details).
   *
   * @tparam UseSVD use SVD to orthonormalize the set of integrands.
   */
  template <class Real, bool UseSVD=true> class InterpQuadRule {
    public:

      /**
       * @brief Build quadrature rule from a function pointer to the integrands, by using an adaptive composite panel
       * Gauss-Legendre quadrature rule to discretize the integrands.
       *
       * @param[out] quad_nds, quad_wts the output quadrature nodes and weights.
       *
       * @param[in] integrands function pointer to evaluate the integrand functions. It a vector of nodes {X[0], ...,
       * X[N-1]} (of type Vector<Real>) where the integrand functions must be evaluated and returns a Matrix<Real> of
       * integrand values. The output matrix M[i][j] is the j-th integrand evaluated at X[i].
       *
       * @param[in] interval_start, interval_end the integration interval.
       *
       * @param[in] eps (optional,default=1e-16) accuracy tolerance for discretizing the integrand functions and
       * determining the number of output quadrature nodes (i.e. truncation tolerance after orthogonalization).
       *
       * @param[in] order (optional) number of output quadrature nodes to use (overrides \p eps).
       *
       * @param[in] nds_interval_start, nds_interval_stop (optional) interval in which to pick the quadrature nodes.
       *
       * @return condition number of the interpolation matrix.
       */
      template <class BasisObj> static Real Build(Vector<Real>& quad_nds, Vector<Real>& quad_wts, const BasisObj& integrands, const Real interval_start, const Real interval_end, const Real eps = 1e-16, const Long order = 0, const Real nds_interval_start = 0, const Real nds_interval_end = 0);

      /**
       * @brief Build quadrature rule from a discretization of the integrand functions.
       *
       * @param[out] quad_nds, quad_wts the output quadrature nodes and weights.
       *
       * @param[in] M matrix containing the values of the integrand functions at a set of discretization nodes such that
       * M[i][j] is the j-th integrand evaluated at the i-th discretization node.
       *
       * @param[in] nds, wts nodes and weights of the quadrature used to discretize the integrand functions.
       *
       * @param[in] eps (optional,default=1e-16) accuracy tolerance which determines the number of output quadrature
       * nodes (i.e. truncation tolerance after orthogonalization).
       *
       * @param[in] order (optional) number of output quadrature nodes to use (overrides \p eps).
       *
       * @param[in] nds_interval_start, nds_interval_stop (optional) interval in which to pick the quadrature nodes.
       *
       * @return condition number of the interpolation matrix.
       */
      static Real Build(Vector<Real>& quad_nds, Vector<Real>& quad_wts, const Matrix<Real> M, const Vector<Real>& nds, const Vector<Real>& wts, const Real eps = 1e-16, const Long order = 0, const Real nds_interval_start = 0, const Real nds_interval_end = 0);

      /**
       * @brief Build a set of quadrature rules for different accuracies from a discretization of the integrand
       * functions.
       *
       * @param[out] quad_nds, quad_wts the output quadrature nodes and weights.
       *
       * @param[in] M matrix containing the values of the integrand functions at a set of discretization nodes such that
       * M[i][j] is the j-th integrand evaluated at the i-th discretization node.
       *
       * @param[in] nds, wts nodes and weights of the quadrature used to discretize the integrand functions.
       *
       * @param[in] eps_vec (optional) vector of accuracy tolerances (which determines the number of quadrature nodes)
       * for each output quadrature rule that is generated.
       *
       * @param[in] order_vec (optional) vector of number of output quadrature nodes for each quadrature rule that is
       * generated (overrides \p eps_vec).
       *
       * @param[in] nds_interval_start, nds_interval_stop (optional) interval in which to pick the quadrature nodes.
       *
       * @return vector of condition numbers of the interpolation matrix for each quadrature rule.
       */
      static Vector<Real> Build(Vector<Vector<Real>>& quad_nds, Vector<Vector<Real>>& quad_wts, const Matrix<Real>& M, const Vector<Real>& nds, const Vector<Real>& wts, const Vector<Real>& eps_vec = Vector<Real>(), const Vector<Long>& order_vec = Vector<Long>(), const Real nds_interval_start = 0, const Real nds_interval_end = 0);

      static void test() {
        const Integer order = 16;
        auto integrands = [order](const Vector<Real>& nds) { // p(x) + q(x) log(x)
          const Long N = nds.Dim();
          Matrix<Real> M(N, order);
          for (Long j = 0; j < N; j++) {
            for (Long i = 0; i < order/2; i++) { // p(x)
              M[j][i] = pow<Real>(nds[j],i);
            }
            for (Long i = order/2; i < order; i++) { // q(x) log(x)
              M[j][i] = pow<Real>(nds[j],i-order/2) * log<Real>(nds[j]);
            }
          }
          return M;
        };

        Vector<Real> nds, wts;
        InterpQuadRule::Build(nds, wts, integrands, 0.0, 1.0, 1e-16, 0, 1e-4, 1);
        for (Integer i = 0; i < nds.Dim(); i++) {
          std::cout<<std::scientific<<std::setprecision(20);
          std::cout<<std::setw(27)<<nds[i]<<' '<<std::setw(27)<<wts[i]<<'\n';
        }
        std::cout<<"\n";
      }

    private:

      template <class FnObj> static void adap_quad_rule(Vector<Real>& nds, Vector<Real>& wts, const FnObj& fn, const Real a, const Real b, const Real tol);
  };
}

#include SCTL_INCLUDE(quadrule.txx)

#endif // _SCTL_QUADRULE_HPP_
