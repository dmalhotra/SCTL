#ifndef _SCTL_LIN_SOLVE_TXX_
#define _SCTL_LIN_SOLVE_TXX_

#include <stdio.h>                // for printf
#include <stdlib.h>               // for drand48, srand48
#include <algorithm>              // for max
#include <iostream>               // for basic_ostream, char_traits, operator<<
#include <iterator>               // for advance

#include "sctl/common.hpp"        // for Long, SCTL_ASSERT, Integer, SCTL_NA...
#include "sctl/lin-solve.hpp"     // for KrylovPrecond, GMRES
#include "sctl/comm.hpp"          // for Comm, CommOp
#include "sctl/comm.txx"          // for Comm::Allreduce, Comm::Rank
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator
#include "sctl/iterator.txx"      // for Iterator::Iterator<ValueType>, Iter...
#include "sctl/math_utils.hpp"    // for sqrt, fabs
#include "sctl/matrix.hpp"        // for Matrix
#include "sctl/ompUtils.txx"      // for omp_par::memcpy
#include "sctl/static-array.hpp"  // for StaticArray
#include "sctl/static-array.txx"  // for StaticArray::operator+, StaticArray...
#include "sctl/vector.hpp"        // for Vector

namespace sctl {

  template <class Real> KrylovPrecond<Real>::KrylovPrecond() : N_(0) {}

  template <class Real> Long KrylovPrecond<Real>::Size() const {
    return N_;
  }

  template <class Real> Long KrylovPrecond<Real>::Rank() const {
    Long rank = 0;
    for (auto it = mat_lst.begin(); it != mat_lst.end(); std::advance(it,2)) {
      rank += it->Dim(1);
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

  // Apply the stored low-rank correction in order: for each (Qt, U) pair,
  // y <- y + (y Qt) U. Distributed: one Allreduce of size Qt.Dim(1) per pair.
  template <class Real> void KrylovPrecond<Real>::Apply(Vector<Real>& y, const Comm& comm) const {
    if (N_ != y.Dim()) return;

    Matrix<Real> y_(1, N_, y.begin(), false);
    for (auto it = mat_lst.begin(); it != mat_lst.end(); it++) {
      const auto& Qt = *it;
      it++;
      const auto& U = *it;

      // y_ += (y_ * Qt) * U;
      ScratchBuf<Real> y_Qt_buf(Qt.Dim(1));
      Matrix<Real> y_Qt(1, Qt.Dim(1), y_Qt_buf.begin(), false);
      Matrix<Real>::GEMM(y_Qt, y_, Qt);

      if (comm.Size() > 1) {
        ScratchBuf<Real> y_Qt_glb_buf(Qt.Dim(1));
        Matrix<Real> y_Qt_glb(1, Qt.Dim(1), y_Qt_glb_buf.begin(), false);
        comm.Allreduce(y_Qt.begin(), y_Qt_glb.begin(), Qt.Dim(1), CommOp::SUM);
        Matrix<Real>::GEMM(y_, y_Qt_glb, U, (Real)1);
      } else {
        Matrix<Real>::GEMM(y_, y_Qt, U, (Real)1);
      }
    }
  }



  template <class Real> static Real inner_prod(const Vector<Real>& x, const Vector<Real>& y, const Comm& comm) {
    Real x_dot_y = 0;
    Long N = x.Dim();
    SCTL_ASSERT(y.Dim() == N);
    for (Long i = 0; i < N; i++) x_dot_y += x[i] * y[i];

    Real x_dot_y_glb = 0;
    comm.Allreduce(Ptr2ConstItr<Real>(&x_dot_y, 1), Ptr2Itr<Real>(&x_dot_y_glb, 1), 1, CommOp::SUM);

    return x_dot_y_glb;
  }

  // Right-preconditioned GMRES: builds an Arnoldi basis Q for K_k(A M^-1, r0)
  // and the upper-Hessenberg H = Q^T A M^-1 Q in-place via Givens-rotated QR,
  // then back-solves for y minimizing ||H y - beta e_1|| and forms x += M^-1 Q y.
  // Q is stored row-major (rows are basis vectors); H is stored as a packed
  // upper-triangle (row i has i+1 entries) since rotations zero subdiagonal entries.
  template <class Real> inline void GMRES<Real>::GenericGMRES(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, Real tol, Integer max_iter, bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) const {
    const Long N = b.Dim();
    if (max_iter < 0) { // set max_iter
      StaticArray<Long,2> NN{N,0};
      comm_.Allreduce(NN+0, NN+1, 1, CommOp::SUM);
      max_iter = NN[1];
    }
    static constexpr Real ARRAY_RESIZE_FACTOR = 1.618;

    Vector<Real> Q_mat, H_mat;
    auto ResizeVector = [](Vector<Real>& v, const Long N0) {
      if (v.Dim() < N0) {
        const Long old = v.Dim();
        Vector<Real> v_(N0);
        if (old > 0) omp_par::memcpy(v_.begin(), (ConstIterator<Real>)v.begin(), old);
        memset((Iterator<char>)v_.begin() + old * (Long)sizeof(Real), 0, (N0 - old) * (Long)sizeof(Real));
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
    auto H_row = [&H_mat,&ResizeVector](Long i) -> Iterator<Real> {
      const Long idx = i*(i+1)/2;
      if (H_mat.Dim() <= idx+i+1) ResizeVector(H_mat, (Long)((idx+i+1)*ARRAY_RESIZE_FACTOR));
      return H_mat.begin() + idx;
    };

    auto apply_givens_rotation = [](Vector<Real>& h, Real& cs_k, Real& sn_k, const Vector<Real>& cs, const Vector<Real>& sn, const Long k) {
      // apply for ith row
      for (Long i = 0; i < k; i++) {
        Real temp = cs[i] * h[i] + sn[i] * h[i+1];
        h[i+1]   = -sn[i] * h[i] + cs[i] * h[i+1];
        h[i]     = temp;
      }

      // update the next sin cos values for rotation (hypot avoids overflow)
      const Real t = hypot<Real>(h[k], h[k+1]);
      cs_k = h[k] / t;
      sn_k = h[k+1] / t;

      // eliminate H(i + 1, i)
      h[k] = cs_k * h[k] + sn_k * h[k+1];
      h[k+1] = 0.0;
    };
    auto arnoldi = [this,N,&Q_row,&krylov_precond](Vector<Real>& h, Vector<Real>& q, const ParallelOp& A, const Long k) {
      Iterator<Real> Qk = Q_row(k);
      if (krylov_precond) { // q_pc = M^-1 * Q_row(k)
        ScratchBuf<Real> q_pc_buf(N);
        Vector<Real> q_pc(q_pc_buf);
        omp_par::memcpy(q_pc.begin(), (ConstIterator<Real>)Qk, N);
        krylov_precond->Apply(q_pc, comm_);
        A(&q, q_pc);
      } else {
        Vector<Real> q_src(N, Qk, false, true);
        A(&q, q_src);
      }

      // One Gram-Schmidt pass: c[i] = q . Q_row(i), q -= sum c[i]*Q_row(i),
      // accumulating c into h[0..k]. Chained passes (initial + reorth) sum.
      auto gs_pass = [this, N, &Q_row, &q, &h, k]() {
        if (gs_strategy_ == GramSchmidt::MGS) {
          for (Long i = 0; i < k+1; i++) { // Modified Gram-Schmidt
            auto Q_row_i = Q_row(i);
            Real dot_local = 0;
            for (Long j = 0; j < N; j++) dot_local += q[j] * Q_row_i[j];
            Real dot_glb = 0;
            comm_.Allreduce(Ptr2ConstItr<Real>(&dot_local, 1), Ptr2Itr<Real>(&dot_glb, 1), 1, CommOp::SUM);
            h[i] += dot_glb;
            for (Long j = 0; j < N; j++) q[j] -= dot_glb * Q_row_i[j];
          }
        } else { // Classical Gram-Schmidt via BLAS GEMV — one batched Allreduce of size k+1
          ScratchBuf<Real> dots_local(k+1), dots_glb(k+1);
          Matrix<Real> Q_view(k+1, N, Q_row(0), false);
          Matrix<Real> q_col(N, 1, q.begin(), false);
          Matrix<Real> dots_col(k+1, 1, dots_local.begin(), false);
          Matrix<Real>::GEMM(dots_col, Q_view, q_col); // dots = Q . q

          comm_.Allreduce(dots_local.begin(), dots_glb.begin(), k+1, CommOp::SUM);
          for (Long i = 0; i < k+1; i++) { // accumulate then negate for the subtract
            h[i] += dots_glb[i];
            dots_glb[i] = -dots_glb[i];
          }

          Matrix<Real> q_row(1, N, q.begin(), false);
          Matrix<Real> neg_dots(1, k+1, dots_glb.begin(), false);
          Matrix<Real>::GEMM(q_row, neg_dots, Q_view, (Real)1); // q -= Q^T . dots_glb
        }
      };

      for (Long i = 0; i < k+1; i++) h[i] = 0;
      gs_pass();                                           // initial pass
      for (Integer p = 0; p < num_reorth_; p++) gs_pass(); // optional reorth passes

      h[k+1] = sqrt<Real>(inner_prod(q, q, comm_));
      // Lucky breakdown (h[k+1] = 0): skip normalization. Loop exits naturally
      // on the next iter: sn[k] = 0 -> beta[k+1] = 0 -> error = 0.
      if (h[k+1] > (Real)0) q *= 1/h[k+1];
    };

    Vector<Real> r;
    if (x->Dim() == N) { // r = b - A * x;
      ScratchBuf<Real> Ax_buf(N);
      Vector<Real> Ax(Ax_buf, false); // disable_reinit=false: user's callback may ReInit
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

    // Early termination: r already meets tolerance (also guards 0/0 below).
    if (r_norm <= abs_tol) {
      if (solve_iter) *solve_iter = 0;
      return;
    }

    {
      Iterator<Real> Q0 = Q_row(0);
      for (Long i = 0; i < N; i++) Q0[i] = r[i] / r_norm;
    }
    ScratchBuf<Real> sn_storage(max_iter), cs_storage(max_iter), beta_storage(max_iter+1), h_k_storage(max_iter+1), q_k_storage(N);
    Vector<Real> sn(sn_storage), cs(cs_storage), beta(beta_storage), h_k(h_k_storage), q_k(q_k_storage);
    beta[0] = r_norm;

    Long k = 0;
    Real error = r_norm;
    for (; k < max_iter && error > abs_tol; k++) {
      if (verbose_ && !comm_.Rank()) printf("%3lld KSP Residual norm %.12e\n", (long long)k, (double)error);
      // Pre-size Q_mat so every Q_row(i) below is a pure pointer return
      // (no iterator invalidation — required by arnoldi's GEMM path).
      if (Q_mat.Dim() < (k+2)*N) ResizeVector(Q_mat, (Long)(((k+2)*N)*ARRAY_RESIZE_FACTOR));

      arnoldi(h_k, q_k, A, k);
      apply_givens_rotation(h_k, cs[k], sn[k], cs, sn, k); // eliminate the last element in H ith row and update the rotation matrix
      {
        Iterator<Real> Hk = H_row(k);
        for (Long i = 0; i < k+1; i++) Hk[i] = h_k[i];
      }
      {
        Iterator<Real> Qk1 = Q_row(k+1);
        for (Long i = 0; i < N; i++) Qk1[i] = q_k[i];
      }

      // update the residual vector
      beta[k+1] = -sn[k] * beta[k];
      beta[k]   = cs[k] * beta[k];
      error     = fabs(beta[k+1]);
    }
    if (verbose_ && !comm_.Rank()) printf("%3lld KSP Residual norm %.12e\n", (long long)k, (double)error);

    for (Long i = k-1; i >= 0; i--) { // beta <-- beta * inv(H); (through back substitution)
      Iterator<Real> Hi = H_row(i);
      beta[i] /= Hi[i];
      for (Long j = 0; j < i; j++) {
        beta[j] -= beta[i] * Hi[j];
      }
    }
    // x_ = Q^T * beta[0..k]  (via GEMM beta=0)
    ScratchBuf<Real> x_buf(N);
    Vector<Real> x_(x_buf);
    if (k > 0) {
      Matrix<Real> x_row(1, N, x_.begin(), false);
      Matrix<Real> beta_row(1, k, beta.begin(), false);
      Matrix<Real> Q_top(k, N, Q_row(0), false);
      Matrix<Real>::GEMM(x_row, beta_row, Q_top, (Real)0);
    } else {
      for (Long i = 0; i < N; i++) x_[i] = 0;
    }
    if (krylov_precond) krylov_precond->Apply(x_, comm_);
    (*x) += x_;

    if (solve_iter) (*solve_iter) = k;

    if (krylov_precond && k > 0) {
      // Build a low-rank correction (Qt, U) so that I + U Qt^T approximates
      // A^-1 in the directions covered by this solve's Krylov subspace.
      // Qt = G^T Q (Q with the Givens rotations applied to its columns) and
      // U  = H^-1 Q - Qt. Future calls with a nearby RHS converge in few iters.
      ScratchBuf<Real>   Qt_buf((Long)N * k);
      ScratchBuf<Real>    U_buf((Long)k * N);
      ScratchBuf<Real> Hinv_buf((Long)k * k);
      Matrix<Real> Qt  (N, k,   Qt_buf.begin(), false);
      Matrix<Real> U   (k, N,    U_buf.begin(), false);
      Matrix<Real> Hinv(k, k, Hinv_buf.begin(), false);
      // Phase 1: Qt = Q_mat^T. Split out from the Givens pass so Q_mat is
      // read sequentially (full cache-line utilization, perfect prefetch)
      // rather than the original stride-N gather.
      for (Long j = 0; j < k; j++) {
        ConstIterator<Real> Q_col = Q_mat.begin() + j*N;
        for (Long i = 0; i < N; i++) Qt[i][j] = Q_col[i];
      }
      // Phase 2: apply Givens rotations row-by-row of Qt. Each row is k
      // contiguous elements and carries a serial data dependency between
      // adjacent j (Qt[i][j+1] is written then read as Qt[i][j+1] next iter).
      for (Long i = 0; i < N; i++) {
        for (Long j = 0; j < k-1; j++) {
          Real temp = cs[j] * Qt[i][j] + sn[j] * Qt[i][j+1];
          Qt[i][j+1] = -sn[j] * Qt[i][j] + cs[j] * Qt[i][j+1];
          Qt[i][j] = temp;
        }
        Qt[i][k-1] = cs[k-1] * Qt[i][k-1] + sn[k-1] * Q_mat[k*N+i];
      }

      for (Long l = 0; l < k; l++) {
        for (Long i = 0; i < k; i++) Hinv[l][i] = 0;
        Hinv[l][l] = 1;
        for (Long i = l; i >= 0; i--) {
          Iterator<Real> Hi = H_row(i);
          Hinv[l][i] /= Hi[i];
          for (Long j = 0; j < i; j++) {
            Hinv[l][j] -= Hinv[l][i] * Hi[j];
          }
        }
      }
      Matrix<Real>::GEMM(U, Hinv, Matrix<Real>(k, N, Q_mat.begin(), false));
      // Loop order swapped: inner j makes U sequential (stride 1, the long
      // dim) at the cost of stride-k reads from Qt (k is small, fits cache).
      for (Long i = 0; i < k; i++) {
        Iterator<Real> Ui = U[i];
        for (Long j = 0; j < N; j++) {
          Ui[j] -= Qt[j][i];
        }
      }

      krylov_precond->Append(Qt, U);
    }
  }

  template <class Real> inline void GMRES<Real>::operator()(Vector<Real>* x, const ParallelOp& A, const Vector<Real>& b, const Real tol, const Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) const {
    GenericGMRES(x, A, b, tol, max_iter, use_abs_tol, solve_iter, krylov_precond);
  }

  template <class Real> void GMRES<Real>::test(Long N) {
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
      if (Ax->Dim() != N) Ax->ReInit(N);
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

}  // end namespace

#ifdef SCTL_HAVE_PETSC

#include <petscksp.h>

namespace sctl {

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

  inline PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy) {
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
      comm_.Allreduce(NN+0, NN+1, 1, CommOp::SUM);
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

  template <> inline void GMRES<double>::operator()(Vector<double>* x, const ParallelOp& A, const Vector<double>& b, const double tol, const Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) const {
    PETScGMRES(x, A, b, tol, max_iter, use_abs_tol, verbose_, comm_, solve_iter);
  }

  template <> inline void GMRES<float>::operator()(Vector<float>* x, const ParallelOp& A, const Vector<float>& b, const float tol, const Integer max_iter, const bool use_abs_tol, Long* solve_iter, KrylovPrecond<Real>* krylov_precond) const {
    PETScGMRES(x, A, b, tol, max_iter, use_abs_tol, verbose_, comm_, solve_iter);
  }

}  // end namespace

#endif

#endif // _SCTL_LIN_SOLVE_TXX_
