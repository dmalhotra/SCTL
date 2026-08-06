/**
 * Green's identity convergence on the cubed sphere at FIXED geometric resolution
 * (order-12 elements, 12x12 panels per cube face by default), sweeping only the
 * requested quadrature tolerance. The geometry and the density are spectrally
 * resolved far below 1e-13 at this resolution, so the measured error is the
 * quadrature error plus the arithmetic floor: the sweep locates where each
 * precision stops converging.
 *
 * Interior representation identity for an exterior point source X0:
 *   (S[du/dn] - D[u]) - 0.5*u == u   on the surface.
 *
 * Build and run:
 *   make bin/test-greens-conv
 *   OMP_NUM_THREADS=60 OMP_PLACES=cores OMP_PROC_BIND=close ./bin/test-greens-conv sweep
 *
 * Modes:
 *   sweep  -- tol sweep in double and long double (default)
 *   parts  -- per-stage double vs long double comparison at a single tol
 * Env overrides: GC_ORDER, GC_PPF, GC_TOLS (comma list), GC_PREC (d|l|dl), GC_KER (lap|stk)
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <cstdlib>
#include <cstdio>
#if defined(__linux__)
#include <sched.h>
#endif

using namespace sctl;

namespace {

constexpr Integer COORD_DIM = 3;

// --- environment / reporting helpers -------------------------------------------

Long EnvLong(const char* name, const Long dflt) {
  if (const char* v = std::getenv(name)) { const Long x = std::atol(v); if (x > 0) return x; }
  return dflt;
}
std::string EnvStr(const char* name, const std::string& dflt) {
  if (const char* v = std::getenv(name)) return std::string(v);
  return dflt;
}
std::vector<double> ParseTols(const std::string& s) {
  std::vector<double> t;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    const std::string tok = s.substr(i, j - i);
    if (!tok.empty()) t.push_back(std::atof(tok.c_str()));
    i = j + 1;
  }
  return t;
}

// Duplicate cores mean the binding is wrong and the timings below are meaningless.
void CheckBinding() {
#if defined(__linux__) && defined(_OPENMP)
  const int p = omp_get_max_threads();
  std::vector<int> cpu(p, -1);
  #pragma omp parallel num_threads(p)
  { const int t = omp_get_thread_num(); if (t < p) cpu[t] = sched_getcpu(); }
  std::set<int> uniq(cpu.begin(), cpu.end());
  std::printf("threads = %d, distinct cores = %ld -> %s\n", p, (long)uniq.size(),
              ((int)uniq.size() == p) ? "OK" : "*** OVERSUBSCRIBED ***");
  std::printf("cpus:");
  for (int t = 0; t < p; t++) std::printf(" %d", cpu[t]);
  std::printf("\n");
#else
  std::printf("binding: sched_getcpu/OpenMP unavailable, not verified\n");
#endif
}

double WallTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// --- geometry -------------------------------------------------------------------

template <class Real> void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R) {
  switch (face) {
    case 0: x =  1; y =  a; z =  b; break;
    case 1: x = -1; y = -a; z =  b; break;
    case 2: x =  a; y =  1; z = -b; break;
    case 3: x =  a; y = -1; z =  b; break;
    case 4: x =  a; y =  b; z =  1; break;
    case 5: x = -a; y =  b; z = -1; break;
    default: SCTL_ASSERT(false);
  }
  const Real r = sqrt<Real>(x * x + y * y + z * z);
  x *= R / r; y *= R / r; z *= R / r;
}

// Cubed sphere of radius R: ppf^2 panels per cube face, `order` nodes per direction.
// Nodes are computed in QuadReal and rounded to Real so that the double and the
// long double element lists discretize the SAME surface to double precision --
// the only difference between the two runs is the arithmetic that follows.
template <class Real> Vector<Real> CubedSphereCoord(const Long order, const Long ppf, const double R) {
  using HP = QuadReal; // node coordinates are exact to ~33 digits, then rounded
  Vector<Real> X;
  const Vector<HP>& nds = QuadElemList<HP>::ParamNodes(order);
  for (Integer face = 0; face < 6; face++) {
    for (Long iu = 0; iu < ppf; iu++) {
      for (Long iv = 0; iv < ppf; iv++) {
        for (Long i = 0; i < order; i++) {
          const HP a = 2 * ((iu + nds[i]) / (HP)ppf) - 1;
          for (Long j = 0; j < order; j++) {
            const HP b = 2 * ((iv + nds[j]) / (HP)ppf) - 1;
            HP x, y, z;
            FacePoint<HP>(x, y, z, face, a, b, (HP)R);
            X.PushBack((Real)x); X.PushBack((Real)y); X.PushBack((Real)z);
          }
        }
      }
    }
  }
  return X;
}

template <class Real> QuadElemList<Real> BuildCubedSphere(const Long order, const Long ppf, const double R, const Comm& comm) {
  QuadElemList<Real> qel(order, CubedSphereCoord<Real>(order, ppf, R), comm);
  // Fork has 5 schemes (default Adaptive); upstream had only this one. Measure Duffy so the
  // tested operator and its higher-precision reference twins use the SAME quadrature scheme.
  qel.SetQuadScheme(QuadElemList<Real>::QuadScheme::Duffy);
  return qel;
}

// --- Green's identity -----------------------------------------------------------

// Surface data for the exterior point source X0 with unit strength:
//   Fd = u|_S (DL density), Fs = +du/dn (SL density, outward normal), Uref = u at Xtrg.
template <class Real, class KerSL, class KerGrad>
void GreensData(Vector<Real>& Fs, Vector<Real>& Fd, Vector<Real>& Uref,
                const Vector<Real>& X, const Vector<Real>& Xn, const Vector<Real>& Xtrg, const Vector<Real>& X0) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  KerSL ker_sl; KerGrad ker_grad;
  Vector<Real> Xn0{0,0,0}, F0(KDIM0), dU;
  for (Integer i = 0; i < KDIM0; i++) F0[i] = (Real)1 / (Real)(i + 1); // fixed, precision-independent

  ker_sl.Eval(Fd, X, X0, Xn0, F0);       // u at the surface nodes
  ker_grad.Eval(dU, X, X0, Xn0, F0);     // grad u at the surface nodes
  ker_sl.Eval(Uref, Xtrg, X0, Xn0, F0);  // u at the targets (reference)

  const Long N = X.Dim() / COORD_DIM;
  Fs.ReInit(N * KDIM0);
  for (Long i = 0; i < N; i++) {
    for (Integer j = 0; j < KDIM0; j++) {
      Real dn = 0;
      for (Integer k = 0; k < COORD_DIM; k++) dn += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
      Fs[i*KDIM0+j] = dn;
    }
  }
}

// max|Uerr| / max|Uref| for the on-surface interior identity.
template <class Real, class KerSL, class KerDL, class KerGrad>
double GreensError(const QuadElemList<Real>& elem_lst, const Real tol, const Comm& comm, double* t_setup = nullptr, double* t_eval = nullptr) {
  KerSL ker_sl; KerDL ker_dl;
  BoundaryIntegralOp<Real,KerSL> BIOpSL(ker_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BIOpDL(ker_dl, false, comm);
  BIOpSL.AddElemList(elem_lst); BIOpDL.AddElemList(elem_lst);
  BIOpSL.SetAccuracy(tol);      BIOpDL.SetAccuracy(tol);

  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2};
  GreensData<Real,KerSL,KerGrad>(Fs, Fd, Uref, X, Xn, X, X0);

  const double t0 = WallTime();
  BIOpSL.Setup(); BIOpDL.Setup();
  const double t1 = WallTime();
  BIOpSL.ComputePotential(Us, Fs);
  BIOpDL.ComputePotential(Ud, Fd);
  const double t2 = WallTime();
  if (t_setup) *t_setup = t1 - t0;
  if (t_eval)  *t_eval  = t2 - t1;

  Ud -= (Real)0.5 * Fd; // interior DL jump (on-surface targets)
  StaticArray<Real,2> err{0,0}, val{0,0};
  for (Long i = 0; i < Uref.Dim(); i++) {
    err[0] = std::max<Real>(err[0], fabs((Us[i] - Ud[i]) - Uref[i]));
    val[0] = std::max<Real>(val[0], fabs(Uref[i]));
  }
  comm.Allreduce(err+0, err+1, 1, CommOp::MAX);
  comm.Allreduce(val+0, val+1, 1, CommOp::MAX);

  // Where on the panel does the error live? Ring 0 = nodes on the panel boundary
  // (adjacent-panel near quadrature dominates), higher rings = panel interior (self
  // quadrature + far field). A flat profile points at geometry/density resolution.
  if (std::getenv("GC_RINGS")) {
    const Integer order = elem_lst.Order(), nring = (order+1)/2;
    Vector<Real> ring_err(nring); ring_err.SetZero();
    for (Long e = 0; e < elem_lst.Size(); e++) {
      for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
          const Long t = (e*order+i)*order + j;
          const Integer r = std::min(std::min(i, order-1-i), std::min(j, order-1-j));
          ring_err[r] = std::max<Real>(ring_err[r], fabs((Us[t] - Ud[t]) - Uref[t]));
        }
      }
    }
    if (!comm.Rank()) {
      std::printf("    ring max-err:");
      for (Integer r = 0; r < nring; r++) std::printf(" [%d]%8.2Le", (int)r, (long double)(ring_err[r]/val[1]));
      std::printf("\n");
    }
  }
  return (double)(err[1] / val[1]);
}

// Double-layer constant-density identity: the on-surface (principal-value) DL of a constant
// density q is exactly -q/2 on a closed outward-oriented surface. Density resolution drops
// out entirely, so this isolates the quadrature + geometry. Returns max|U/q + 1/2| / (1/2).
template <class Real, class KerDL>
double DLConstError(const QuadElemList<Real>& elem_lst, const Real tol, const Comm& comm, double* t_setup = nullptr, double* t_eval = nullptr) {
  constexpr Integer KDIM0 = KerDL::SrcDim();
  KerDL ker_dl;
  BoundaryIntegralOp<Real,KerDL> BIOp(ker_dl, false, comm);
  BIOp.AddElemList(elem_lst);
  BIOp.SetAccuracy(tol);

  Vector<Real> X, Xn, U;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim() / COORD_DIM;
  Vector<Real> q(N * KDIM0);
  for (Long i = 0; i < N; i++)
    for (Integer k = 0; k < KDIM0; k++) q[i*KDIM0+k] = (Real)(k + 1);

  const double t0 = WallTime();
  BIOp.Setup();
  const double t1 = WallTime();
  BIOp.ComputePotential(U, q);
  const double t2 = WallTime();
  if (t_setup) *t_setup = t1 - t0;
  if (t_eval)  *t_eval  = t2 - t1;

  const Real c_expect = (Real)-0.5;
  StaticArray<Real,2> err{0,0};
  Long argmax = 0;
  for (Long i = 0; i < N; i++)
    for (Integer k = 0; k < KDIM0; k++) {
      const Real e = fabs(U[i*KDIM0+k]/q[i*KDIM0+k] - c_expect);
      if (e > err[0]) { err[0] = e; argmax = i; }
    }
  comm.Allreduce(err+0, err+1, 1, CommOp::MAX);
  if (!comm.Rank()) {
    const Long nne = elem_lst.Order()*elem_lst.Order();
    std::printf("    worst target: node %ld (elem %ld, local %ld = row %ld col %ld of %ld)\n",
                (long)argmax, (long)(argmax/nne), (long)(argmax%nne),
                (long)((argmax%nne)/elem_lst.Order()), (long)((argmax%nne)%elem_lst.Order()),
                (long)elem_lst.Order());
  }
  return (double)(err[1] / fabs(c_expect));
}

template <class Real> const char* PrecName();
template <> const char* PrecName<double>() { return "double     "; }
template <> const char* PrecName<long double>() { return "long double"; }
template <> const char* PrecName<QuadReal>() { return "QuadReal   "; }

template <class Real> void RunDLConst(const Long order, const Long ppf, const std::vector<double>& tols, const Comm& comm) {
  auto elem_lst = BuildCubedSphere<Real>(order, ppf, 1.0, comm);
  if (!comm.Rank()) {
    std::printf("\n=== DL constant-density identity, %s : order=%ld ppf=%ld nelem=%ld nodes=%ld ===\n",
                PrecName<Real>(), (long)order, (long)ppf, (long)elem_lst.Size()*comm.Size(),
                (long)elem_lst.Size()*order*order*comm.Size());
    std::printf("%10s %14s %10s %10s\n", "tol", "dlconst_err", "setup[s]", "eval[s]");
  }
  for (const double tol_ : tols) {
    const Real tol = (Real)tol_;
    double ts = 0, te = 0;
    const double e = DLConstError<Real, Laplace3D_DxU>(elem_lst, tol, comm, &ts, &te);
    if (!comm.Rank()) {
      std::printf("%10.1e %14.4e %10.2f %10.2f\n", tol_, e, ts, te);
      std::fflush(stdout);
    }
  }
}

template <class Real> void RunSweep(const Long order, const Long ppf, const std::vector<double>& tols, const std::string& ker, const Comm& comm) {
  const double t0 = WallTime();
  auto elem_lst = BuildCubedSphere<Real>(order, ppf, 1.0, comm);
  if (!comm.Rank()) {
    std::printf("\n=== %s (eps = %.3e, %ld bytes) : order=%ld ppf=%ld nelem=%ld nodes=%ld  [build %.2fs] ===\n",
                PrecName<Real>(), (double)machine_eps<Real>(), (long)sizeof(Real),
                (long)order, (long)ppf, (long)elem_lst.Size()*comm.Size(),
                (long)elem_lst.Size()*order*order*comm.Size(), WallTime()-t0);
    std::printf("%10s %14s %10s %10s\n", "tol", "greens_err", "setup[s]", "eval[s]");
  }
  for (const double tol_ : tols) {
    const Real tol = (Real)tol_;
    double ts = 0, te = 0, e = 0;
#ifdef GC_WITH_STOKES // doubles the (already large) template instantiation set; opt in with -DGC_WITH_STOKES
    if (ker == "stk") e = GreensError<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, tol, comm, &ts, &te);
    else
#endif
                      e = GreensError<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, tol, comm, &ts, &te);
    if (!comm.Rank()) {
      std::printf("%10.1e %14.4e %10.2f %10.2f\n", tol_, e, ts, te);
      std::fflush(stdout);
    }
  }
}

// --- stage-by-stage double vs long double comparison ---------------------------
//
// Both element lists are built from the SAME (double-representable) nodal coordinates,
// so every difference below is arithmetic, not a difference of discretized surface.
// Stage outputs are collected as long double (lossless up-cast of the double pass) and
// compared. Where a stage consumes another stage's output (density, far weights) the
// double pass is fed the long double values cast down, so each row isolates the
// round-off of that stage's own arithmetic -- except that every stage that touches the
// element geometry inherits the round-off of the cached nodal derivatives (rows 1-2).

struct Stages {
  Vector<long double> Xn_node, wts_far, dist_far;   // geometry
  Vector<long double> Fs, Fd;                       // surface data
  Vector<long double> U_self_sl, U_self_dl;         // self operator applied to the density
  Vector<long double> U_near_sl, U_near_dl;         // near operator applied to the density
  Vector<long double> U_far_sl, U_far_dl;           // far-field N^2 sum
  long double Uref_max = 1;
};

template <class T, class S> Vector<T> Cast(const Vector<S>& v) {
  Vector<T> o(v.Dim());
  for (Long i = 0; i < v.Dim(); i++) o[i] = (T)v[i];
  return o;
}

// max_i |a_i - b_i|, and the same normalized by `scale`.
void ReportRow(const char* name, const Vector<long double>& a, const Vector<long double>& b, const long double scale) {
  SCTL_ASSERT(a.Dim() == b.Dim());
  long double d = 0, m = 0;
  for (Long i = 0; i < a.Dim(); i++) {
    d = std::max<long double>(d, fabs(a[i] - b[i]));
    m = std::max<long double>(m, fabs(b[i]));
  }
  std::printf("  %-28s  n=%-9ld  max|d|=%9.3Le   max|ref|=%9.3Le   max|d|/scale=%9.3Le\n",
              name, (long)a.Dim(), d, m, d / scale);
}

template <class Real>
void ComputeStages(Stages& out, const QuadElemList<Real>& elem, const Real tol,
                   const Vector<Long>& far_trg, const Vector<Long>& near_elem, const Vector<Long>& near_trg_dsp, const Vector<Long>& near_trg,
                   const Stages* shared) {
  using KerSL = Laplace3D_FxU; using KerDL = Laplace3D_DxU; using KerGrad = Laplace3D_FxdU;
  KerSL ker_sl; KerDL ker_dl;
  const Long nnode = elem.Size() * elem.Order() * elem.Order();

  { // geometry
    Vector<Real> X, Xn;
    Vector<Long> cnt_;
    elem.GetNodeCoord(&X, &Xn, nullptr);
    out.Xn_node = Cast<long double>(Xn);
    Vector<Real> Xf, Xnf, wf, df;
    elem.GetFarFieldNodes(Xf, Xnf, wf, df, cnt_, tol);
    out.wts_far = Cast<long double>(wf);
    out.dist_far = Cast<long double>(df);

    // surface data: Fs = du/dn uses Xn, so feed the shared Xn to isolate the kernel eval
    Vector<Real> Fs, Fd, Uref;
    const Vector<Real> Xn_in = (shared ? Cast<Real>(shared->Xn_node) : Xn);
    const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2};
    GreensData<Real,KerSL,KerGrad>(Fs, Fd, Uref, X, Xn_in, X, X0);
    out.Fs = Cast<long double>(Fs);
    out.Fd = Cast<long double>(Fd);
    out.Uref_max = 0;
    for (const auto& u : Uref) out.Uref_max = std::max<long double>(out.Uref_max, fabs((long double)u));
  }

  const Vector<Real> Fs = (shared ? Cast<Real>(shared->Fs) : Cast<Real>(out.Fs));
  const Vector<Real> Fd = (shared ? Cast<Real>(shared->Fd) : Cast<Real>(out.Fd));
  const Vector<Real> wts = (shared ? Cast<Real>(shared->wts_far) : Cast<Real>(out.wts_far));

  { // self operator: U[t] = sum_p f[p] * M[p][t], per element
    const Long nne = elem.Order() * elem.Order();
    for (Integer which = 0; which < 2; which++) {
      Vector<Matrix<Real>> M_lst(elem.Size()); // SelfInterac requires M_lst pre-sized to nelem
      if (which == 0) QuadElemList<Real>::template SelfInterac<KerSL>(M_lst, ker_sl, tol, false, &elem);
      else            QuadElemList<Real>::template SelfInterac<KerDL>(M_lst, ker_dl, tol, false, &elem);
      const Vector<Real>& f = (which == 0 ? Fs : Fd);
      Vector<long double> U(nnode); U.SetZero();
      for (Long e = 0; e < elem.Size(); e++) {
        for (Long t = 0; t < nne; t++) {
          Real acc = 0;
          for (Long p = 0; p < nne; p++) acc += f[e*nne+p] * M_lst[e][p][t];
          U[e*nne+t] = (long double)acc;
        }
      }
      if (which == 0) out.U_self_sl = U; else out.U_self_dl = U;
    }
  }

  { // near operator on the sampled (element, target) pairs
    const Long nne = elem.Order() * elem.Order();
    Vector<Real> X, Xn;
    elem.GetNodeCoord(&X, &Xn, nullptr);
    const Long npair = near_trg.Dim();
    for (Integer which = 0; which < 2; which++) {
      const Vector<Real>& f = (which == 0 ? Fs : Fd);
      Vector<long double> U(npair); U.SetZero();
      #pragma omp parallel for schedule(dynamic)
      for (Long k = 0; k < near_elem.Dim(); k++) {
        const Long e = near_elem[k];
        const Long i0 = near_trg_dsp[k], i1 = near_trg_dsp[k+1];
        if (i1 <= i0) continue;
        Vector<Real> Xt((i1-i0)*COORD_DIM);
        for (Long i = i0; i < i1; i++)
          for (Integer d = 0; d < COORD_DIM; d++) Xt[(i-i0)*COORD_DIM+d] = X[near_trg[i]*COORD_DIM+d];
        Matrix<Real> M;
        if (which == 0) QuadElemList<Real>::template NearInterac<KerSL>(M, Xt, Vector<Real>(), ker_sl, tol, e, &elem);
        else            QuadElemList<Real>::template NearInterac<KerDL>(M, Xt, Vector<Real>(), ker_dl, tol, e, &elem);
        for (Long i = i0; i < i1; i++) {
          Real acc = 0;
          for (Long p = 0; p < nne; p++) acc += f[e*nne+p] * M[p][i-i0];
          U[i] = (long double)acc;
        }
      }
      if (which == 0) out.U_near_sl = U; else out.U_near_dl = U;
    }
  }

  { // far-field N^2 direct sum at the sampled targets (this is what the FMM-less path does)
    Vector<Real> Xf, Xnf, wf, df; Vector<Long> cnt_;
    elem.GetFarFieldNodes(Xf, Xnf, wf, df, cnt_, tol);
    Vector<Real> Xt(far_trg.Dim()*COORD_DIM);
    for (Long i = 0; i < far_trg.Dim(); i++)
      for (Integer d = 0; d < COORD_DIM; d++) Xt[i*COORD_DIM+d] = Xf[far_trg[i]*COORD_DIM+d];
    for (Integer which = 0; which < 2; which++) {
      const Vector<Real>& f = (which == 0 ? Fs : Fd);
      Vector<Real> Fw(nnode), U;
      for (Long i = 0; i < nnode; i++) Fw[i] = f[i] * wts[i];
      if (which == 0) ker_sl.template Eval<Real,true>(U, Xt, Xf, Xnf, Fw);
      else            ker_dl.template Eval<Real,true>(U, Xt, Xf, Xnf, Fw);
      if (which == 0) out.U_far_sl = Cast<long double>(U); else out.U_far_dl = Cast<long double>(U);
    }
  }
}

void RunParts(const Long order, const Long ppf, const double tol_, const Comm& comm) {
  SCTL_ASSERT_MSG(comm.Size() == 1, "parts mode is serial (single MPI rank)");
  const Vector<double> Xin = CubedSphereCoord<double>(order, ppf, 1.0);
  QuadElemList<double> ed(order, Xin, comm);
  QuadElemList<long double> el(order, Cast<long double>(Xin), comm);
  // Duffy on both (fork default is Adaptive); the double/long-double passes must match schemes.
  ed.SetQuadScheme(QuadElemList<double>::QuadScheme::Duffy);
  el.SetQuadScheme(QuadElemList<long double>::QuadScheme::Duffy);
  const Long nne = order*order, nnode = ed.Size()*nne;

  // sample sets, chosen from the (identical) double geometry and shared by both passes
  Vector<Long> far_trg, near_elem, near_trg, near_trg_dsp;
  {
    const Long Nsamp = 2048;
    for (Long i = 0; i < Nsamp; i++) far_trg.PushBack((i * nnode) / Nsamp);

    Vector<double> X, Xn, wts, df; Vector<Long> cnt_;
    ed.GetNodeCoord(&X, &Xn, nullptr);
    ed.GetFarFieldNodes(X, Xn, wts, df, cnt_, tol_);
    const Long Nelem_samp = 8;
    near_trg_dsp.PushBack(0);
    for (Long k = 0; k < Nelem_samp; k++) {
      const Long e = (k * ed.Size()) / Nelem_samp;
      double r = 0;
      for (Long p = 0; p < nne; p++) r = std::max(r, df[e*nne+p]);
      for (Long j = 0; j < nnode; j++) {
        if (j/nne == e) continue;
        double d2min = 1e30;
        for (Long p = 0; p < nne; p++) {
          double d2 = 0;
          for (Integer c = 0; c < COORD_DIM; c++) { const double t = X[j*COORD_DIM+c]-X[(e*nne+p)*COORD_DIM+c]; d2 += t*t; }
          d2min = std::min(d2min, d2);
        }
        if (d2min < r*r) near_trg.PushBack(j);
      }
      near_elem.PushBack(e);
      near_trg_dsp.PushBack(near_trg.Dim());
    }
  }

  std::printf("\n=== stage-by-stage double vs long double : order=%ld ppf=%ld nelem=%ld nodes=%ld tol=%.1e ===\n",
              (long)order, (long)ppf, (long)ed.Size(), (long)nnode, tol_);
  std::printf("    samples: %ld far targets, %ld near (elem,target) pairs over %ld elements\n",
              (long)far_trg.Dim(), (long)near_trg.Dim(), (long)near_elem.Dim());

  Stages sl, sd;
  double t0 = WallTime();
  ComputeStages<long double>(sl, el, (long double)tol_, far_trg, near_elem, near_trg_dsp, near_trg, nullptr);
  std::printf("    long double pass: %.1fs\n", WallTime()-t0);
  t0 = WallTime();
  ComputeStages<double>(sd, ed, tol_, far_trg, near_elem, near_trg_dsp, near_trg, &sl);
  std::printf("    double pass:      %.1fs\n", WallTime()-t0);

  const long double scale = sl.Uref_max; // so max|d|/scale is directly comparable to the reported Green's error
  std::printf("    scale = max|Uref| = %.4Le\n", scale);
  ReportRow("geom: node normals Xn", sd.Xn_node, sl.Xn_node, 1);
  ReportRow("geom: far weights w", sd.wts_far, sl.wts_far, 1);
  ReportRow("geom: far dist_far", sd.dist_far, sl.dist_far, 1);
  ReportRow("data: Fs = du/dn", sd.Fs, sl.Fs, 1);
  ReportRow("data: Fd = u", sd.Fd, sl.Fd, 1);
  ReportRow("op:   self SL . Fs", sd.U_self_sl, sl.U_self_sl, scale);
  ReportRow("op:   self DL . Fd", sd.U_self_dl, sl.U_self_dl, scale);
  ReportRow("op:   near SL . Fs", sd.U_near_sl, sl.U_near_sl, scale);
  ReportRow("op:   near DL . Fd", sd.U_near_dl, sl.U_near_dl, scale);
  ReportRow("op:   far  SL . Fs", sd.U_far_sl, sl.U_far_sl, scale);
  ReportRow("op:   far  DL . Fd", sd.U_far_dl, sl.U_far_dl, scale);
}


// --- self/near/far residual isolation -------------------------------------------
//
// Constant-density DL identity: sum_e D_e[1](t) = -1/2 exactly, for every target t.
// Density interpolation plays no role, so any discrepancy is operator quadrature.
//
// For each sampled target we compute, per element:
//   ref   -- adaptive brute-force quadrature of D_e[1](t) (smooth for e != self)
//   far   -- the scheme's smooth far-field rule over the element's own nodes
//   near  -- the scheme's NearInterac for that (element, target)
// and for the self element the reference comes from the identity itself,
//   self_ref = -1/2 - sum_{e != e0} ref_e,
// which needs no singular reference. Comparing each against ref localizes the residual.

template <class RefT, class Ker>
RefT BruteForceDL(const QuadElemList<RefT>& elem, const Long e, const Vector<RefT>& Xt,
                  const Ker& ker, const RefT eps, RefT* err_est, long* ncell) {
  using Real = RefT;
  const Integer q = 32;
  Vector<Real> gn, gw;
  LegQuadRule<Real>::ComputeNdsWts(&gn, &gw, q);
  auto cellval = [&](const Real u0, const Real u1, const Real v0, const Real v1) {
    Vector<Real> up(q), vp(q);
    for (Integer i = 0; i < q; i++) { up[i] = u0 + (u1-u0)*gn[i]; vp[i] = v0 + (v1-v0)*gn[i]; }
    Vector<Real> Xs, Ns, Xa, U, F(q*q);
    elem.GetGeom(&Xs, &Ns, &Xa, nullptr, nullptr, up, vp, e);
    for (Integer a = 0; a < q; a++)
      for (Integer b = 0; b < q; b++) F[a*q+b] = Xa[a*q+b]*gw[a]*gw[b]*(u1-u0)*(v1-v0);
    ker.Eval(U, Xt, Xs, Ns, F);
    return U[0];
  };
  struct Cell { Real u0,u1,v0,v1; Integer depth; };
  std::vector<Cell> stack{{(Real)0,(Real)1,(Real)0,(Real)1,0}};
  Real acc = 0, err = 0;
  long nc = 0;
  const Real floor_err = 8*machine_eps<Real>();
  const Integer max_depth = 36;
  while (!stack.empty()) {
    const Cell c = stack.back(); stack.pop_back();
    const Real um = (c.u0+c.u1)/2, vm = (c.v0+c.v1)/2;
    const Real whole = cellval(c.u0,c.u1,c.v0,c.v1);
    const Real parts = cellval(c.u0,um,c.v0,vm) + cellval(um,c.u1,c.v0,vm)
                     + cellval(c.u0,um,vm,c.v1) + cellval(um,c.u1,vm,c.v1);
    const Real d = fabs(whole - parts);
    if (d <= std::max<Real>(eps*(c.u1-c.u0)*(c.v1-c.v0), floor_err) || c.depth >= max_depth || nc > 20000) {
      acc += parts; err += d; nc++;
    } else {
      stack.push_back({c.u0,um,c.v0,vm,(Integer)(c.depth+1)});
      stack.push_back({um,c.u1,c.v0,vm,(Integer)(c.depth+1)});
      stack.push_back({c.u0,um,vm,c.v1,(Integer)(c.depth+1)});
      stack.push_back({um,c.u1,vm,c.v1,(Integer)(c.depth+1)});
    }
  }
  if (err_est) *err_est = err;
  if (ncell) *ncell = nc;
  return acc;
}

template <class Real> void RunSplit(const Long order, const Long ppf, const Real tol, const Comm& comm) {
  using KerDL = Laplace3D_DxU;
  KerDL ker_dl;
  const Vector<Real> Xin = CubedSphereCoord<Real>(order, ppf, 1.0);
  auto elem = BuildCubedSphere<Real>(order, ppf, 1.0, comm);
  // Reference twin at QuadReal on the identical discrete surface (coords cast up, so the
  // order-p interpolant is bit-identical). D[1] = -1/2 holds exactly on it too.
  Vector<QuadReal> XinR(Xin.Dim());
  for (Long i = 0; i < Xin.Dim(); i++) XinR[i] = (QuadReal)Xin[i];
  QuadElemList<QuadReal> elemR(order, XinR, comm);
  elemR.SetQuadScheme(QuadElemList<QuadReal>::QuadScheme::Duffy);  // match `elem`'s scheme (Duffy)
  const Long nelem = elem.Size(), nne = order*order;
  Vector<Real> X, Xn;
  elem.GetNodeCoord(&X, &Xn, nullptr);
  Vector<Real> Xf, Xnf, wf, df;
  Vector<Long> cnt;
  elem.GetFarFieldNodes(Xf, Xnf, wf, df, cnt, tol);

  std::printf("\n=== self/near/far split, DL constant density (D[1] = -1/2) ===\n");
  std::printf("    order=%ld ppf=%ld nelem=%ld tol=%.1e  %s\n",
              (long)order, (long)ppf, (long)nelem, (double)tol, PrecName<Real>());

  Vector<Matrix<Real>> Mself(nelem);
  double t0 = WallTime();
  QuadElemList<Real>::template SelfInterac<KerDL>(Mself, ker_dl, tol, false, &elem);
  std::printf("    SelfInterac: %.1fs\n", WallTime()-t0);

  std::vector<Long> trg_list;
  { const std::string spec = EnvStr("GC_SPLIT_TRG", "");
    if (spec.empty()) { for (Long s = 0; s < 3; s++) trg_list.push_back((s*nelem/3)*nne + nne/2 + order/3); }
    else { size_t i = 0;
      while (i < spec.size()) { size_t j = spec.find(',', i); if (j == std::string::npos) j = spec.size();
        trg_list.push_back(std::atol(spec.substr(i, j-i).c_str())); i = j+1; } } }
  for (size_t s = 0; s < trg_list.size(); s++) {
    const Long t = trg_list[s];
    const Long e0 = t/nne, tloc = t%nne;
    const Vector<Real> Xt{X[t*COORD_DIM+0], X[t*COORD_DIM+1], X[t*COORD_DIM+2]};

    QuadReal sum_ref = 0, ref_err_tot = 0;
    Real sum_scheme = 0;
    const Vector<QuadReal> XtR{(QuadReal)Xt[0], (QuadReal)Xt[1], (QuadReal)Xt[2]};
    Real self_scheme = 0, err_far_max = 0, err_near_max = 0;
    Long n_near = 0;
        long cells_tot = 0;
    t0 = WallTime();
    for (Long e = 0; e < nelem; e++) {
      if (e == e0) {
        for (Long p = 0; p < nne; p++) self_scheme += Mself[e][p][tloc];
        continue;
      }
      QuadReal rerr = 0; long nc = 0;
      const QuadReal ref = BruteForceDL<QuadReal,KerDL>(elemR, e, XtR, ker_dl, (QuadReal)1e-32, &rerr, &nc);
      ref_err_tot += rerr; cells_tot += nc;
      sum_ref += ref;

      Real vfar = 0;                                     // scheme's smooth far rule
      { Vector<Real> Xs(nne*COORD_DIM), Ns(nne*COORD_DIM), F(nne), U;
        for (Long p = 0; p < nne; p++) {
          for (Integer k = 0; k < COORD_DIM; k++) {
            Xs[p*COORD_DIM+k] = Xf[(e*nne+p)*COORD_DIM+k];
            Ns[p*COORD_DIM+k] = Xnf[(e*nne+p)*COORD_DIM+k];
          }
          F[p] = wf[e*nne+p];
        }
        ker_dl.Eval(U, Xt, Xs, Ns, F);
        vfar = U[0]; }

      Matrix<Real> M;                                    // scheme's near rule
      QuadElemList<Real>::template NearInterac<KerDL>(M, Xt, Vector<Real>(), ker_dl, tol, e, &elem);
      Real vnear = 0;
      for (Long p = 0; p < nne; p++) vnear += M[p][0];

      // near/far classification, mirroring the dist_far criterion
      Real dmin = 1e30, dfmax = 0;
      for (Long p = 0; p < nne; p++) {
        Real d2 = 0;
        for (Integer k = 0; k < COORD_DIM; k++) { const Real z = X[(e*nne+p)*COORD_DIM+k]-Xt[k]; d2 += z*z; }
        dmin = std::min<Real>(dmin, sqrt<Real>(d2));
        dfmax = std::max<Real>(dfmax, df[e*nne+p]);
      }
      const bool is_near = (dmin < dfmax);
      if (is_near) { n_near++; err_near_max = std::max<Real>(err_near_max, (Real)fabs<QuadReal>((QuadReal)vnear-ref)); }
      else          err_far_max = std::max<Real>(err_far_max, (Real)fabs<QuadReal>((QuadReal)vfar-ref));
      sum_scheme += (is_near ? vnear : vfar);
    }
    const QuadReal self_ref = (QuadReal)-0.5 - sum_ref;
    const Real total_err = fabs(sum_scheme + self_scheme + (Real)0.5);

    std::printf("\n  target node %ld (elem %ld, local %ld)   [%.1fs, %ld brute cells, ref err est %.2Le]\n",
                (long)t, (long)e0, (long)tloc, WallTime()-t0, cells_tot, (long double)ref_err_tot);
    std::printf("    near elements                : %ld\n", (long)n_near);
    std::printf("    SELF  |scheme - ref|         : %.4Le\n", (long double)fabs<QuadReal>((QuadReal)self_scheme - self_ref));
    std::printf("    NEAR  max |scheme - ref|     : %.4Le\n", (long double)err_near_max);
    std::printf("    FAR   max |scheme - ref|     : %.4Le\n", (long double)err_far_max);
    std::printf("    TOTAL |sum(scheme) + 1/2|    : %.4Le\n", (long double)total_err);
    std::fflush(stdout);
  }
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const std::string mode = (argc > 1 ? argv[1] : "sweep");
    const Long order = EnvLong("GC_ORDER", 12);
    const Long ppf   = EnvLong("GC_PPF", 12);
    const std::string prec = EnvStr("GC_PREC", "dl");
    const std::string ker  = EnvStr("GC_KER", "lap");
    const std::vector<double> tols = ParseTols(EnvStr("GC_TOLS", "1e-4,1e-6,1e-8,1e-10,1e-12,1e-13,1e-14"));

    if (!comm.Rank()) {
      std::printf("==== Green's identity convergence: cubed sphere, mode=%s kernel=%s ====\n", mode.c_str(), ker.c_str());
      CheckBinding();
    }

    if (mode == "sweep") {
      if (prec.find('d') != std::string::npos) RunSweep<double>(order, ppf, tols, ker, comm);
      if (prec.find('l') != std::string::npos) RunSweep<long double>(order, ppf, tols, ker, comm);
      if (prec.find('q') != std::string::npos) RunSweep<QuadReal>(order, ppf, tols, ker, comm);
    } else if (mode == "dl") {
      if (prec.find('d') != std::string::npos) RunDLConst<double>(order, ppf, tols, comm);
      if (prec.find('l') != std::string::npos) RunDLConst<long double>(order, ppf, tols, comm);
      if (prec.find('q') != std::string::npos) RunDLConst<QuadReal>(order, ppf, tols, comm);
    } else if (mode == "split") {
      const double tl = (tols.empty() ? 1e-19 : tols[0]);
      if (prec.find('l') != std::string::npos) RunSplit<long double>(order, ppf, (long double)tl, comm);
      if (prec.find('q') != std::string::npos) RunSplit<QuadReal>(order, ppf, (QuadReal)tl, comm);
      if (prec.find('d') != std::string::npos) RunSplit<double>(order, ppf, tl, comm);
    } else if (mode == "parts") {
      for (const double tol : tols) RunParts(order, ppf, tol, comm);
    } else {
      if (!comm.Rank()) std::printf("unknown mode '%s'\n", mode.c_str());
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
