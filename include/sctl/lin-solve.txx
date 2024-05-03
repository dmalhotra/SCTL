#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)
#include SCTL_INCLUDE(math_utils.hpp)

#include <functional>
#include <list>

namespace SCTL_NAMESPACE {

  template <class ValueType> class Vector;
  template <class ValueType> class Matrix;

  template <class Real> KrylovPrecond<Real>::KrylovPrecond() : N_(0) {}

  template <class Real> Long KrylovPrecond<Real>::Size() const {
    return N_;
  }

  template <class Real> Long KrylovPrecond<Real>::Rank() const {
    Long rank = 0;
    for (auto it = mat_lst.begin(); it != mat_lst.end(); it++) {
      rank += it->Dim(1);
      it++;
    }
    return rank;
  }

  template <class Real> void KrylovPrecond<Real>::Append(const Matrix<Real>& Qt, const Matrix<Real>& U) {
    SCTL_ASSERT(Qt.Dim(0) == U.Dim(1));
    SCTL_ASSERT(Qt.Dim(1) == U.Dim(0));
    if (Qt.Dim(0) != N_) { // clear
      mat_lst.clear();
      N_ = Qt.Dim(0);
    }

    mat_lst.push_front(U);
    mat_lst.push_front(Qt);
  }

  template <class Real> void KrylovPrecond<Real>::Apply(Vector<Real>& y) const {
    if (N_ != y.Dim()) return;

    Matrix<Real> y_Qt, y_(1, N_, y.begin(), false);
    for (auto it = mat_lst.begin(); it != mat_lst.end(); it++) {
      const auto& Qt = *it;
      it++;
      const auto& U = *it;

      //y_ += (y_ * Qt) * U;
      y_Qt.ReInit(1, Qt.Dim(1));
      Matrix<Real>::GEMM(y_Qt, y_, Qt);
      Matrix<Real>::GEMM(y_, y_Qt, U, (Real)1);
    }
  }



  template <class Real> static Real inner_prod(const Vector<Real>& x, const Vector<Real>& y, const Comm& comm) {
    Real x_dot_y = 0;
    Long N = x.Dim();
    SCTL_ASSERT(y.Dim() == N);
    for (Long i = 0; i < N; i++) x_dot_y += x[i] * y[i];

    Real x_dot_y_glb = 0;
    comm.Allreduce(Ptr2ConstItr<Real>(&x_dot_y, 1), Ptr2Itr<Real>(&x_dot_y_glb, 1), 1, Comm::CommOp::SUM);

    return x_dot_y_glb;
  }

  template <class Real> inline void GMRES<Real>::GenericGMRES(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, Real tol, Integer max_iter, bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) {
    const Long N = b.Dim();
    if (max_iter < 0) { // set max_iter
      StaticArray<Long,2> NN{N,0};
      comm_.Allreduce(NN+0, NN+1, 1, Comm::CommOp::SUM);
      max_iter = NN[1];
    }
    static constexpr Real ARRAY_RESIZE_FACTOR = 1.618;

    Vector<Real> Q_mat, H_mat;
    auto ResizeVector = [](Vector<Real>& v, const Long N0) {
      if (v.Dim() < N0) {
        Vector<Real> v_(N0);
        for (Long i = 0; i < v.Dim(); i++) v_[i] = v[i];
        for (Long i = v.Dim(); i < N0; i++) v_[i] = 0;
        v.Swap(v_);
      }
    };
    auto Q_row = [N,&Q_mat,&ResizeVector](Long i) -> Iterator<Real> {
      const Long idx = i*N;
      if (Q_mat.Dim() <= idx+N) {
        ResizeVector(Q_mat, (Long)((idx+N)*ARRAY_RESIZE_FACTOR));
      }
      return Q_mat.begin() + idx;
    };
    auto Q = [&Q_row](Long i, Long j) -> Real& {
      return Q_row(i)[j];
    };
    auto H_row = [&H_mat,&ResizeVector](Long i) -> Iterator<Real> {
      const Long idx = i*(i+1)/2;
      if (H_mat.Dim() <= idx+i+1) ResizeVector(H_mat, (Long)((idx+i+1)*ARRAY_RESIZE_FACTOR));
      return H_mat.begin() + idx;
    };
    auto H = [&H_row](Long i, Long j) -> Real& {
      return H_row(i)[j];
    };

    auto apply_givens_rotation = [](Vector<Real>& h, Real& cs_k, Real& sn_k, const Vector<Real>& cs, const Vector<Real>& sn, const Long k) {
      // apply for ith row
      for (Long i = 0; i < k; i++) {
        Real temp = cs[i] * h[i] + sn[i] * h[i+1];
        h[i+1]   = -sn[i] * h[i] + cs[i] * h[i+1];
        h[i]     = temp;
      }

      // update the next sin cos values for rotation
      const Real t = sqrt<Real>(h[k]*h[k] + h[k+1]*h[k+1]);
      cs_k = h[k] / t;
      sn_k = h[k+1] / t;

      // eliminate H(i + 1, i)
      h[k] = cs_k * h[k] + sn_k * h[k+1];
      h[k+1] = 0.0;
    };
    auto arnoldi = [this,N,&Q_row,&Q,&krylov_precond](Vector<Real>& h, Vector<Real>& q, const ParallelOp& A, const Long k) {
      Vector<Real> q_k(N, Q_row(k), krylov_precond?true:false);
      if (krylov_precond) krylov_precond->Apply(q_k);
      A(&q, q_k);

      for (Long i = 0; i < k+1; i++) { // Modified Gram-Schmidt, keeping the Hessenberg matrix
        h[i] = inner_prod(q, Vector<Real>(N, Q_row(i), false), comm_);
        for (Long j = 0; j < N; j++) {
          q[j] -= h[i] * Q(i,j);
        }
      }
      //for (Long i = 0; i < k+1; i++) { // re-orthogonalize (more stable)
      //  const Real h_ = inner_prod(q, Vector<Real>(N, Q_row(i), false), comm_);
      //  for (Long j = 0; j < N; j++) {
      //    q[j] -= h_ * Q(i,j);
      //  }
      //  h[i] += h_;
      //}
      h[k+1] = sqrt<Real>(inner_prod(q, q, comm_));
      q *= 1/h[k+1];
    };

    Vector<Real> r;
    if (x->Dim() == N) { // r = b - A * x;
      Vector<Real> Ax;
      A(&Ax, *x);
      r = b - Ax;
    } else {
      r = b;
      x->ReInit(N);
      x->SetZero();
    }

    const Real b_norm = sqrt<Real>(inner_prod(b, b, comm_));
    const Real abs_tol = tol * (use_abs_tol ? 1 : b_norm);

    const Real r_norm = sqrt<Real>(inner_prod(r, r, comm_));
    for (Long i = 0; i < N; i++) Q(0,i) = r[i] / r_norm;
    Vector<Real> beta(1); beta = r_norm;
    Vector<Real> sn, cs, h_k, q_k(N);

    Long k = 0;
    Real error = r_norm;
    for (; k < max_iter && error > abs_tol; k++) {
      if (verbose_ && !comm_.Rank()) printf("%3lld KSP Residual norm %.12e\n", (long long)k, (double)error);
      if (sn.Dim() <= k) ResizeVector(sn, (Long)((k+1)*ARRAY_RESIZE_FACTOR));
      if (cs.Dim() <= k) ResizeVector(cs, (Long)((k+1)*ARRAY_RESIZE_FACTOR));
      if (beta.Dim() <= k+1) ResizeVector(beta, (Long)((k+2)*ARRAY_RESIZE_FACTOR));
      if ( h_k.Dim() <= k+1) ResizeVector( h_k, (Long)((k+2)*ARRAY_RESIZE_FACTOR));

      arnoldi(h_k, q_k, A, k);
      apply_givens_rotation(h_k, cs[k], sn[k], cs, sn, k); // eliminate the last element in H ith row and update the rotation matrix
      for (Long i = 0; i < k+1; i++) H(k,i) = h_k[i];
      for (Long i = 0; i < N; i++) Q(k+1,i) = q_k[i];

      // update the residual vector
      beta[k+1] = -sn[k] * beta[k];
      beta[k]   = cs[k] * beta[k];
      error     = fabs(beta[k+1]);
    }
    if (verbose_ && !comm_.Rank()) printf("%3lld KSP Residual norm %.12e\n", (long long)k, (double)error);

    for (Long i = k-1; i >= 0; i--) { // beta <-- beta * inv(H); (through back substitution)
      beta[i] /= H(i,i);
      for (Long j = 0; j < i; j++) {
        beta[j] -= beta[i] * H(i,j);
      }
    }
    Vector<Real> x_(N); x_ = 0;
    for (Long i = 0; i < N; i++) { // x <-- beta * Q
      for (Long j = 0; j < k; j++) {
        x_[i] += beta[j] * Q(j,i);
      }
    }
    if (krylov_precond) krylov_precond->Apply(x_);
    (*x) += x_;

    if (solve_iter) (*solve_iter) = k;

    if (krylov_precond) {
      Matrix<Real> Qt(N, k), U(k, N);
      for (Long i = 0; i < N; i++) { // apply givens rotations to Qt
        for (Long j = 0; j < k; j++) Qt[i][j] = Q_mat[j*N+i];
        for (Long j = 0; j < k-1; j++) { // apply givens rotations to Qt
          Real temp = cs[j] * Qt[i][j] + sn[j] * Qt[i][j+1];
          Qt[i][j+1] = -sn[j] * Qt[i][j] + cs[j] * Qt[i][j+1];
          Qt[i][j] = temp;
        }
      }
      for (Long i = 0; i < N; i++) {
        Qt[i][k-1] = cs[k-1] * Qt[i][k-1] + sn[k-1] * Q_mat[k*N+i];
      }

      Matrix<Real> Hinv(k,k);
      for (Long l = 0; l < k; l++) {
        for (Long i = 0; i < k; i++) Hinv[l][i] = 0;
        Hinv[l][l] = 1;
        for (Long i = l; i >= 0; i--) {
          Hinv[l][i] /= H(i,i);
          for (Long j = 0; j < i; j++) {
            Hinv[l][j] -= Hinv[l][i] * H(i,j);
          }
        }
      }
      Matrix<Real>::GEMM(U, Hinv, Matrix<Real>(k, N, Q_mat.begin(), false));
      for (Long j = 0; j < N; j++) {
        for (Long i = 0; i < k; i++) {
          U[i][j] -= Qt[j][i];
        }
      }

      krylov_precond->Append(Qt, U);
    }
  }

  template <class Real> inline void GMRES<Real>::operator()(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, const Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) {
    GenericGMRES(x, A, b, tol, max_iter, use_abs_tol, solve_iter, krylov_precond);
  }

}  // end namespace


#ifdef SCTL_HAVE_PETSC

#include <petscksp.h>

namespace SCTL_NAMESPACE {

  template <class Real> int GMRESMatVec(Mat M_, ::Vec x_, ::Vec Mx_) {
    PetscErrorCode ierr;

    PetscInt N, N_;
    VecGetLocalSize(x_, &N);
    VecGetLocalSize(Mx_, &N_);
    SCTL_ASSERT(N == N_);

    void* data = nullptr;
    MatShellGetContext(M_, &data);
    auto& M = dynamic_cast<const typename GMRES<Real>::ParallelOp&>(*(typename GMRES<Real>::ParallelOp*)data);

    const PetscScalar* x_ptr;
    ierr = VecGetArrayRead(x_, &x_ptr);
    CHKERRQ(ierr);

    Vector<Real> x(N);
    for (Long i = 0; i < N; i++) x[i] = (Real)x_ptr[i];
    Vector<Real> Mx(N);
    M(&Mx, x);

    PetscScalar* Mx_ptr;
    ierr = VecGetArray(Mx_, &Mx_ptr);
    CHKERRQ(ierr);

    for (long i = 0; i < N; i++) Mx_ptr[i] = Mx[i];
    ierr = VecRestoreArray(Mx_, &Mx_ptr);
    CHKERRQ(ierr);

    return 0;
  }

  PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy) {
    Comm* comm = (Comm*)dummy;
    if (!comm->Rank()) printf("%3lld KSP Residual norm %.12e\n", (long long)n, (double)rnorm);
    //PetscPrintf(PETSC_COMM_WORLD,"iteration %D KSP Residual norm %14.12e \n",n,rnorm);

    //PetscViewerAndFormat *vf;
    //PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf);
    //KSPMonitorResidual(ksp, n, rnorm, vf);
    //PetscViewerAndFormatDestroy(&vf);
    return 0;
  }

  template <class Real> inline void PETScGMRES(Vector<Real>* x, const typename GMRES<Real>::ParallelOp& A, const Vector<Real>& b, const Real tol, Integer max_iter, const bool use_abs_tol, const bool verbose_, const Comm& comm_, Long* solve_iter) {
    PetscInt N = b.Dim();
    if (max_iter < 0) { // set max_iter
      StaticArray<Long,2> NN{N,0};
      comm_.Allreduce(NN+0, NN+1, 1, Comm::CommOp::SUM);
      max_iter = NN[1];
    }
    const MPI_Comm comm = comm_.GetMPI_Comm();
    PetscErrorCode ierr;

    Mat PetscA;
    {  // Create Matrix. PetscA
      MatCreateShell(comm, N, N, PETSC_DETERMINE, PETSC_DETERMINE, (void*)&A, &PetscA);
      MatShellSetOperation(PetscA, MATOP_MULT, (void (*)(void))GMRESMatVec<Real>);
    }

    // Create linear solver context
    KSP ksp;
    ierr = KSPCreate(comm, &ksp);
    CHKERRABORT(comm, ierr);

    ::Vec Petsc_x, Petsc_b;
    {  // Create vectors
      VecCreateMPI(comm, N, PETSC_DETERMINE, &Petsc_b);
      VecCreateMPI(comm, N, PETSC_DETERMINE, &Petsc_x);

      PetscScalar* b_ptr;
      ierr = VecGetArray(Petsc_b, &b_ptr);
      CHKERRABORT(comm, ierr);
      for (long i = 0; i < N; i++) b_ptr[i] = b[i];
      ierr = VecRestoreArray(Petsc_b, &b_ptr);
      CHKERRABORT(comm, ierr);

      if (x->Dim() != N) {
        x->ReInit(N);
      } else {
        PetscScalar* x_ptr;
        ierr = VecGetArray(Petsc_x, &x_ptr);
        CHKERRABORT(comm, ierr);
        for (long i = 0; i < N; i++) x_ptr[i] = (*x)[i];
        ierr = VecRestoreArray(Petsc_x, &x_ptr);
        CHKERRABORT(comm, ierr);

        ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
      }
    }

    // Set operators. Here the matrix that defines the linear system
    // also serves as the preconditioning matrix.
    ierr = KSPSetOperators(ksp, PetscA, PetscA);
    CHKERRABORT(comm, ierr);

    // Set runtime options
    KSPSetType(ksp, KSPGMRES);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    if (use_abs_tol) KSPSetTolerances(ksp, PETSC_DEFAULT, tol, PETSC_DEFAULT, max_iter);
    else KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_iter);
    KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization);
    if (verbose_) KSPMonitorSet(ksp, MyKSPMonitor, comm, nullptr);
    KSPGMRESSetRestart(ksp, max_iter);
    ierr = KSPSetFromOptions(ksp);
    CHKERRABORT(comm, ierr);

    // -------------------------------------------------------------------
    // Solve the linear system: Ax=b
    // -------------------------------------------------------------------
    ierr = KSPSolve(ksp, Petsc_b, Petsc_x);
    CHKERRABORT(comm, ierr);

    // View info about the solver
    // KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD); CHKERRABORT(comm, ierr);

    // Iterations
    PetscInt its;
    ierr = KSPGetIterationNumber(ksp,&its); CHKERRABORT(comm, ierr);
    // ierr = PetscPrintf(PETSC_COMM_WORLD,"Iterations %D\n",its); CHKERRABORT(comm, ierr);

    {  // Set x
      const PetscScalar* x_ptr;
      ierr = VecGetArrayRead(Petsc_x, &x_ptr);
      CHKERRABORT(comm, ierr);

      for (long i = 0; i < N; i++) (*x)[i] = (Real)x_ptr[i];
    }

    ierr = KSPDestroy(&ksp);
    CHKERRABORT(comm, ierr);
    ierr = MatDestroy(&PetscA);
    CHKERRABORT(comm, ierr);
    ierr = VecDestroy(&Petsc_x);
    CHKERRABORT(comm, ierr);
    ierr = VecDestroy(&Petsc_b);
    CHKERRABORT(comm, ierr);

    if (solve_iter) (*solve_iter) = its;
  }

  template <> inline void GMRES<double>::operator()(Vector<double>* x, const ParallelOp& A, const Vector<double>& b, const double tol, const Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) {
    PETScGMRES(x, A, b, tol, max_iter, use_abs_tol, verbose_, comm_, solve_iter);
  }

  template <> inline void GMRES<float>::operator()(Vector<float>* x, const ParallelOp& A, const Vector<float>& b, const float tol, const Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) {
    PETScGMRES(x, A, b, tol, max_iter, use_abs_tol, verbose_, comm_, solve_iter);
  }

}  // end namespace

#endif

