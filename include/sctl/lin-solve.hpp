#ifndef _SCTL_LIN_SOLVE_HPP_
#define _SCTL_LIN_SOLVE_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)

#include <functional>
#include <list>

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;
template <class ValueType> class Matrix;

/**
 * @brief Preconditioner built from the Krylov-subspace constructed during GMRES solves.
 *
 * This class implements a preconditioner built from the Krylov-subspace constructed during GMRES solves.
 *
 * @tparam Real The data type of the values.
 */
template <class Real> class KrylovPrecond {
  public:

    /**
     * @brief Constructor.
     */
    KrylovPrecond();

    /**
     * @brief Get the size of the input vector to the operator.
     *
     * @return The length of the input vector.
     */
    Long Size() const;

    /**
     * @brief Get the cumulative size of the Krylov-subspaces.
     *
     * @return The cumulative size of the Krylov-subspaces.
     */
    Long Rank() const;

    /**
     * @brief Append a Krylov-subspace to the operator.
     *
     * The operator P is updated as:
     *   P = P * (I + U * Qt)
     *
     * @param[in] Qt The matrix Q transpose.
     * @param[in] U The matrix U.
     */
    void Append(const Matrix<Real>& Qt, const Matrix<Real>& U);

    /**
     * @brief Apply the preconditioner.
     *
     * @param[in,out] x The input vector which is updated by applying the preconditioner.
     */
    void Apply(Vector<Real>& x) const;

  private:

    Long N_; ///< Length of the input vector.
    std::list<Matrix<Real>> mat_lst; ///< List of matrices storing Krylov-subspaces.
};

/**
 * @brief Implements a distributed memory GMRES solver.
 *
 * This class implements a distributed memory GMRES solver.
 *
 * @tparam Real The data type of the values.
 */
template <class Real> class GMRES {

 public:

  using ParallelOp = std::function<void(Vector<Real>*, const Vector<Real>&)>; ///< Function type for linear operator.

  /**
   * @brief Constructor.
   *
   * @param[in] comm The communicator.
   * @param[in] verbose Verbosity flag.
   */
  GMRES(const Comm& comm = Comm::Self(), bool verbose = true) : comm_(comm), verbose_(verbose) {}

  /**
   * @brief Solve the linear system: A x = b.
   *
   * @param[out] x The solution vector.
   * @param[in] A The linear operator.
   * @param[in] b The right-hand-side vector.
   * @param[in] tol The accuracy tolerance.
   * @param[in] max_iter Maximum number of iterations (default -1 corresponds to no limit).
   * @param[in] use_abs_tol Whether to use absolute tolerance (default false).
   * @param[out] solve_iter Number of iterations.
   * @param[in,out] krylov_precond Krylov-subspace preconditioner. The preconditioner is updated.
   */
  void operator()(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, const Integer max_iter = -1, const bool use_abs_tol = false, Long* solve_iter=nullptr, KrylovPrecond<Real>* krylov_precond=nullptr);

  /**
   * @brief A test function for GMRES solver.
   *
   * @param[in] N Size of the test problem (default is 15).
   */
  static void test(Long N = 15);

 private:

  void GenericGMRES(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond);

  Comm comm_; ///< Communicator.
  bool verbose_; ///< Verbosity flag.
};

}  // end namespace

#include SCTL_INCLUDE(lin-solve.txx)

#endif  //_SCTL_LIN_SOLVE_HPP_
