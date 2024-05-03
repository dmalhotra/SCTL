#ifndef _SCTL_LIN_SOLVE_HPP_
#define _SCTL_LIN_SOLVE_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)

#include <functional>
#include <list>

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;
template <class ValueType> class Matrix;

/**
 * Preconditioner built from the Krylov-subspace constructed during GMRES solves.
 */
template <class Real> class KrylovPrecond {
  public:

    /**
     * Constructor.
     */
    KrylovPrecond();

    /**
     * @return length of the input vector to the operator.
     */
    Long Size() const;

    /**
     * @return cumulative size of the Krylov-subspaces.
     */
    Long Rank() const;

    /**
     * Append a Krylov-subspace to the operator. The operator P is updated as:
     *   P = P * (I + U * Qt)
     */
    void Append(const Matrix<Real>& Qt, const Matrix<Real>& U);

    /**
     * Apply the preconditioner.
     *
     * @param[in,out] x input vector which is updated my applying the preconditioner.
     */
    void Apply(Vector<Real>& x) const;

  private:

    Long N_;
    std::list<Matrix<Real>> mat_lst;
};

/**
 * Implements a distributed memory GMRES solver.
 */
template <class Real> class GMRES {

 public:

  using ParallelOp = std::function<void(Vector<Real>*, const Vector<Real>&)>;

  /**
   * Constructor.
   *
   * @param[in] comm communicator.
   *
   * @param[in] verbose verbosity flag.
   */
  GMRES(const Comm& comm = Comm::Self(), bool verbose = true) : comm_(comm), verbose_(verbose) {}

  /**
   * Solve the linear system: A x = b
   *
   * @param[out] x the solution vector.
   *
   * @param[in] A the linear operator.
   *
   * @param[in] b the righ-hand-side vector.
   *
   * @param[in] tol the accuracy tolerance.
   *
   * @param[in] max_iter maximum number of iterations (default -1 corresponds to no-limit).
   *
   * @param[in] use_abs_tol whether touse absolute tolerance (default false).
   *
   * @param[out] solve_iter number of iterations.
   *
   * @param[in,out] krylov_precond Krylov-subspace preconditioner. The preconditioner is updated.
   */
  void operator()(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, const Integer max_iter = -1, const bool use_abs_tol = false, Long* solve_iter=nullptr, KrylovPrecond<Real>* krylov_precond=nullptr);

  static void test(Long N = 15) {
    srand48(0);
    Matrix<Real> A(N, N);
    Vector<Real> b(N), x;
    for (Long i = 0; i < N; i++) {
      b[i] = drand48();
      for (Long j = 0; j < N; j++) {
        A[i][j] = drand48();
      }
    }

    auto LinOp = [&A](Vector<Real>* Ax, const Vector<Real>& x) {
      const Long N = x.Dim();
      Ax->ReInit(N);
      Matrix<Real> Ax_(N, 1, Ax->begin(), false);
      Ax_ = A * Matrix<Real>(N, 1, (Iterator<Real>)x.begin(), false);
    };

    Long solve_iter;
    GMRES<Real> solver;
    solver(&x, LinOp, b, 1e-10, -1, false, &solve_iter);

    auto print_error = [N,&A,&b](const Vector<Real>& x) {
      Real max_err = 0;
      auto Merr = A*Matrix<Real>(N, 1, (Iterator<Real>)x.begin(), false) - Matrix<Real>(N, 1, b.begin(), false);
      for (const auto& a : Merr) max_err = std::max(max_err, fabs(a));
      std::cout<<"Maximum error = "<<max_err<<'\n';
    };
    print_error(x);
    std::cout<<"GMRES iterations = "<<solve_iter<<'\n';
  }

 private:

  void GenericGMRES(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond);

  Comm comm_;
  bool verbose_;
};

}  // end namespace

#include SCTL_INCLUDE(lin-solve.txx)

#endif  //_SCTL_LIN_SOLVE_HPP_
