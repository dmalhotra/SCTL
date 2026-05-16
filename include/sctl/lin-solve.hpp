#ifndef _SCTL_LIN_SOLVE_HPP_
#define _SCTL_LIN_SOLVE_HPP_

#include <functional>       // for function
#include <list>             // for list

#include "sctl/common.hpp"  // for Long, Integer, sctl
#include "sctl/comm.hpp"    // for Comm
#include "sctl/comm.txx"    // for Comm::Self, Comm::Comm
#include "sctl/vector.hpp"  // for Vector

namespace sctl {

template <class ValueType> class Matrix;

/**
 * This class implements a preconditioner built from the Krylov-subspace constructed during GMRES solves.
 *
 * @tparam Real The data type of the values.
 */
template <class Real> class KrylovPrecond {
  public:

    /**
     * Constructor.
     */
    KrylovPrecond();

    /**
     * Get the size of the input vector to the operator.
     *
     * @return The length of the input vector.
     */
    Long Size() const;

    /**
     * Get the cumulative size of the Krylov-subspaces.
     *
     * @return The cumulative size of the Krylov-subspaces.
     */
    Long Rank() const;

    /**
     * Append a Krylov-subspace to the operator.
     * The operator P is updated as:
     *   P = P * (I + U * Qt)
     *
     * @param[in] Qt The matrix Q transpose.
     * @param[in] U The matrix U.
     */
    void Append(const Matrix<Real>& Qt, const Matrix<Real>& U);

    /**
     * Apply the preconditioner.
     *
     * @param[in,out] x The input vector which is updated by applying the preconditioner.
     */
    void Apply(Vector<Real>& x, const Comm& comm) const;

  private:

    Long N_; ///< Length of the input vector.
    std::list<Matrix<Real>> mat_lst; ///< List of matrices storing Krylov-subspaces.
};

/**
 * This class implements a distributed memory GMRES solver.
 *
 * @tparam Real The data type of the values.
 */
template <class Real> class GMRES {

 public:

  using ParallelOp = std::function<void(Vector<Real>*, const Vector<Real>&)>; ///< Function type for linear operator.

  /**
   * Gram-Schmidt orthogonalization scheme used in the Arnoldi step.
   *
   *   - `MGS`: Modified Gram-Schmidt. One reduction per basis vector —
   *     (k+1) `Allreduce`s of size 1 per Arnoldi iteration plus one for
   *     the norm. Stable per step but Allreduce-latency-bound on
   *     distributed runs.
   *   - `CGS`: Classical Gram-Schmidt. All `k+1` basis dot products are
   *     batched into a single `Allreduce` of size `k+1`. Two reductions
   *     per iteration. Fastest in MPI; can lose orthogonality when the
   *     basis becomes ill-conditioned (use `num_reorth >= 1`).
   *
   * Both schemes support additional reorthogonalization passes (the
   * "twice is enough" rule). Per pass: MGS adds (k+1) more size-1
   * reductions, CGS adds one size-(k+1) reduction.
   *
   * Typical recommendations:
   *   - Serial / shared memory: `MGS, 0` (default).
   *   - Distributed, well-conditioned: `CGS, 0`.
   *   - Distributed, robust: `CGS, 1` — MGS-equivalent stability at ~k/3
   *     the latency cost.
   */
  enum class GramSchmidt { MGS, CGS };

  /**
   * Constructor.
   *
   * @param[in] comm The communicator.
   * @param[in] verbose Verbosity flag.
   * @param[in] gs Gram-Schmidt scheme used in the Arnoldi step (default
   *               `MGS` — matches the legacy behavior).
   * @param[in] num_reorth Number of additional reorthogonalization passes
   *               after the initial Gram-Schmidt pass. 0 = no reorth
   *               (default), 1 = "twice is enough", >1 = extra-paranoid.
   *               Each pass costs one more reduction round-trip per
   *               Arnoldi iteration (k+1 for MGS, 1 for CGS).
   */
  GMRES(const Comm& comm = Comm::Self(), bool verbose = true, GramSchmidt gs = GramSchmidt::MGS, Integer num_reorth = 0)
    : comm_(comm), verbose_(verbose), gs_strategy_(gs), num_reorth_(num_reorth) {}

  /**
   * Solve the linear system: A x = b.
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
  void operator()(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, const Integer max_iter = -1, const bool use_abs_tol = false, Long* solve_iter=nullptr, KrylovPrecond<Real>* krylov_precond=nullptr) const;

  /**
   * A test function for GMRES solver.
   *
   * @param[in] N Size of the test problem (default is 15).
   */
  static void test(Long N = 15);

 private:

  void GenericGMRES(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) const;

  Comm comm_; ///< Communicator.
  bool verbose_; ///< Verbosity flag.
  GramSchmidt gs_strategy_; ///< Arnoldi orthogonalization scheme.
  Integer num_reorth_; ///< Additional reorthogonalization passes after the initial Gram-Schmidt pass.
};

}  // end namespace

#endif // _SCTL_LIN_SOLVE_HPP_
