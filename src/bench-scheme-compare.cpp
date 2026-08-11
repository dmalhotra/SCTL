/**
 * Singular-quadrature scheme comparison on a twisted cubed sphere: RP / Adaptive / Hybrid / Duffy.
 *
 * Compares the four singular-quadrature SCHEMES against each other on one machine: for each scheme
 * both the on-surface Green's-identity error and the single-layer setup throughput are reported, so
 * accuracy and speed can be read side by side.
 *
 * The twist rotates each point about z by theta*z -- an isometry of the sphere, so the SURFACE is
 * the unit sphere at every twist and only the PARAMETRISATION shears. The exact answers do not
 * move, but the elements do, which is what makes it a clean stress test.
 *
 * Reported per configuration (per (kernel, scheme, twist, tol)):
 *   error       max|(S[du/dn] - D[u]) - u| / max|u|   the on-surface Green's identity (greens_sol)
 *   setup_sl    wall time of the single-layer BIOp Setup()
 *   pts/s/core  N_pts / setup_sl / (threads*ranks)     single-layer setup throughput
 *
 * tol -> (Nbeta, max_depth) ladder. There is no automatic map in the core (tol only feeds
 * SetAccuracy -> digits), so we encode one here. Nbeta (SetQuadScheme cov_order) is used by RP and
 * by Hybrid's self phase; max_depth by Adaptive and Hybrid's near phase; Duffy ignores both. The
 * ladder is therefore inert for Duffy and partially inert for the others -- by design.
 *
 *   tol     Nbeta   max_depth
 *   1e-3     48        4
 *   1e-5     48        4
 *   1e-7    100        8
 *   1e-9    200       12
 *   1e-11   400       30
 *
 * In addition to a human-readable table, each config emits a machine-readable tagged line for the
 * parser (scripts/parse-scheme-compare.sh). The same format serves both modes; in conv mode `thr`
 * is the fixed node width, in omp mode it is the swept thread count:
 *   @@ROW kernel=<k> scheme=<s> thr=<n> twist=<t> tol=<tol> error=<e> pps=<p> setup=<s>
 *
 * Threads default to OMP_NUM_THREADS and can be overridden by the last argument. Binding still
 * comes from OMP_PLACES / OMP_PROC_BIND, so set those too -- the run warns if the threads do not
 * land on distinct cores, which would make every pts/s below meaningless. Run:
 *     ./bin/bench-scheme-compare conv <laplace|stokes> <RP|Adaptive|Hybrid|Duffy> [nthreads]
 *     ./bin/bench-scheme-compare omp  <laplace|stokes> <RP|Adaptive|Hybrid|Duffy> [nthreads]
 *     OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close \
 *         ./bin/bench-scheme-compare conv laplace Duffy
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace sctl;

namespace {

using Real = double;
using QScheme = QuadElemList<Real>::QuadScheme;
constexpr Integer COORD_DIM = 3;

double WallTime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
}

// Warn if the threads share cores: pts/s/core is meaningless under oversubscription.
void CheckBinding(const Integer T) {
#if defined(__linux__) && defined(_OPENMP)
  std::vector<int> cpu(T, -1);
  #pragma omp parallel num_threads(T)
  { const Integer t = omp_get_thread_num(); if (t < T) cpu[t] = sched_getcpu(); }
  const std::set<int> uniq(cpu.begin(), cpu.end());
  if ((Integer)uniq.size() != T)
    std::printf("*** WARNING: %ld distinct cores for %d threads -- oversubscribed, pts/s/core is meaningless\n",
                (long)uniq.size(), (int)T);
#else
  (void)T;
#endif
}

Integer NumThreads() {
#ifdef _OPENMP
  return (Integer)omp_get_max_threads();
#else
  return 1;
#endif
}

// --- scheme + tolerance ladder ----------------------------------------------------------------

// "RP" is the doc shorthand for the RectPolar enumerator; the other three match the enum.
bool SchemeFromName(const std::string& s, QScheme& out) {
  if (s == "RP" || s == "RectPolar") { out = QScheme::RectPolar; return true; }
  if (s == "Adaptive")               { out = QScheme::Adaptive;  return true; }
  if (s == "Hybrid")                 { out = QScheme::Hybrid;    return true; }
  if (s == "Duffy")                  { out = QScheme::Duffy;     return true; }
  return false;
}

struct Ladder { Integer nbeta, max_depth; };

// tol -> (Nbeta, max_depth). max_depth must be one of {4,8,12,30} (asserted by SetQuadScheme).
// Compare against a slightly-loosened tol so exact 1e-k literals land in the intended rung.
Ladder TolLadder(const Real tol) {
  if (tol >= (Real)0.9e-3)  return {48,   4};   // 1e-3
  if (tol >= (Real)0.9e-5)  return {48,   4};   // 1e-5
  if (tol >= (Real)0.9e-7)  return {100,  8};   // 1e-7
  if (tol >= (Real)0.9e-9)  return {200, 12};   // 1e-9
  return {400, 30};                             // 1e-11 and tighter
}

void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R) {
  switch (face) {
    case 0: x =  1; y =  a; z =  b; break;
    case 1: x = -1; y = -a; z =  b; break;
    case 2: x =  a; y =  1; z = -b; break;
    case 3: x =  a; y = -1; z =  b; break;
    case 4: x =  a; y =  b; z =  1; break;
    default: x = -a; y =  b; z = -1; break;
  }
  const Real r = sqrt<Real>(x*x + y*y + z*z);
  x *= R/r; y *= R/r; z *= R/r;
}

// Cubed sphere, ppf^2 patches per face, twisted about z by theta*z, with the given scheme and its
// tol-derived Nbeta / max_depth. Nbeta and max_depth are ignored by the schemes that do not use
// them (Duffy ignores both, RectPolar ignores max_depth) -- SetQuadScheme just stores them.
QuadElemList<Real> BuildSphere(const Integer order, const Long ppf, const Real R, const Real twist,
                               const QScheme scheme, const Integer nbeta, const Integer max_depth, const Comm& comm) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (Integer f = 0; f < 6; f++) {
    for (Long iu = 0; iu < ppf; iu++) {
      for (Long iv = 0; iv < ppf; iv++) {
        for (Integer i = 0; i < order; i++) {
          const Real a = 2*((iu + nds[i]) / (Real)ppf) - 1;
          for (Integer j = 0; j < order; j++) {
            const Real b = 2*((iv + nds[j]) / (Real)ppf) - 1;
            Real x, y, z;
            FacePoint(x, y, z, f, a, b, R);
            const Real s = sin<Real>(twist*z), c = cos<Real>(twist*z);
            X.PushBack(x*c + y*s);
            X.PushBack(-x*s + y*c);
            X.PushBack(z);
          }
        }
      }
    }
  }
  QuadElemList<Real> qel(order, X, comm);
  qel.SetQuadScheme(scheme, /*q*/ 6, /*cov_order = Nbeta*/ nbeta, /*max_depth*/ max_depth);
  return qel;
}

// Surface data for an exterior point source: Fd = u|_S (DL density), Fs = du/dn (SL density),
// Uref = u at the targets. Lifted verbatim from test-greens-conv.
template <class KerSL, class KerGrad>
void GreensData(Vector<Real>& Fs, Vector<Real>& Fd, Vector<Real>& Uref,
                const Vector<Real>& X, const Vector<Real>& Xn, const Vector<Real>& Xtrg, const Vector<Real>& X0) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  KerSL ker_sl; KerGrad ker_grad;
  Vector<Real> Xn0{0,0,0}, F0(KDIM0), dU;   // unused: neither kernel reads the source normal
  for (Integer i = 0; i < KDIM0; i++) F0[i] = (Real)1 / (Real)(i + 1);

  ker_sl.Eval(Fd, X, X0, Xn0, F0);
  ker_grad.Eval(dU, X, X0, Xn0, F0);
  ker_sl.Eval(Uref, Xtrg, X0, Xn0, F0);

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

// On-surface interior identity S[du/dn] - D[u] = u, with the DL jump. Lifted from test-greens-conv.
// The two Setup() calls are the timed ones; t_setup_sl is the reported single-layer setup time.
template <class KerSL, class KerDL, class KerGrad>
double GreensSolError(const QuadElemList<Real>& qel, const Real tol, const Vector<Real>& X0, const Comm& comm,
                      double* t_setup_sl, double* t_setup_dl) {
  KerSL ker_sl; KerDL ker_dl;
  BoundaryIntegralOp<Real,KerSL> BSL(ker_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BDL(ker_dl, false, comm);
  BSL.AddElemList(qel); BDL.AddElemList(qel);
  BSL.SetAccuracy(tol); BDL.SetAccuracy(tol);

  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  GreensData<KerSL,KerGrad>(Fs, Fd, Uref, X, Xn, X, X0);

  const double t0 = WallTime(); BSL.Setup();
  const double t1 = WallTime(); BDL.Setup();
  const double t2 = WallTime();
  *t_setup_sl = t1 - t0; *t_setup_dl = t2 - t1;

  BSL.ComputePotential(Us, Fs);
  BDL.ComputePotential(Ud, Fd);
  Ud -= (Real)0.5 * Fd;   // interior DL jump

  StaticArray<Real,2> err{0,0}, val{0,0};
  for (Long i = 0; i < Uref.Dim(); i++) {
    err[0] = std::max<Real>(err[0], fabs<Real>((Us[i] - Ud[i]) - Uref[i]));
    val[0] = std::max<Real>(val[0], fabs<Real>(Uref[i]));
  }
  comm.Allreduce(err+0, err+1, 1, CommOp::MAX);
  comm.Allreduce(val+0, val+1, 1, CommOp::MAX);
  return (double)(err[1] / val[1]);
}

void Header() {
  std::printf("#%-7s %-9s %4s %5s %4s %8s %9s %6s %8s | %10s | %9s %10s\n",
              "kernel", "scheme", "thr", "order", "ppf", "twist", "tol", "Nbeta", "maxdep",
              "error", "setup_sl", "pps/c_sl");
}

// One (kernel, scheme, twist, tol) configuration. Warms the per-order static tables with an untimed
// throwaway Setup() so the timed Setup() inside GreensSolError does not pay the one-time first-touch
// cost (NearGradeTable / DuffyTable / ParamNodes / DiffMat), which is not per-element setup work.
template <class KerSL, class KerDL, class KerGrad>
void Run(const char* kname, const char* sname, const QScheme scheme, const Integer order, const Long ppf,
         const Real twist, const Real tol, const Comm& comm) {
  const Real R = 1;
  const Ladder L = TolLadder(tol);
  const QuadElemList<Real> qel = BuildSphere(order, ppf, R, twist, scheme, L.nbeta, L.max_depth, comm);
  const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2};   // exterior source

  // Untimed warm-up: first-touch the static per-order tables so the timed Setup() below is warm.
  { BoundaryIntegralOp<Real,KerSL> Bw(KerSL(), false, comm);
    Bw.SetAccuracy(tol); Bw.AddElemList(qel); Bw.Setup(); Bw.ClearSetup(); }

  double ts_sl = 0, ts_dl = 0;
  const double err = GreensSolError<KerSL,KerDL,KerGrad>(qel, tol, X0, comm, &ts_sl, &ts_dl);

  StaticArray<Long,2> n{qel.Size()*order*order, 0};
  comm.Allreduce(n+0, n+1, 1, CommOp::SUM);
  const double N = (double)n[1], T = (double)NumThreads()*comm.Size();
  const double pps = (ts_sl > 0 ? N/ts_sl/T : 0);
  if (!comm.Rank()) {
    std::printf(" %-7s %-9s %4d %5d %4ld %8.4f %9.0e %6d %8d | %10.2e | %9.3f %10.1f\n",
                kname, sname, (int)NumThreads(), (int)order, (long)ppf, (double)twist, (double)tol,
                (int)L.nbeta, (int)L.max_depth, err, ts_sl, pps);
    std::printf("@@ROW kernel=%s scheme=%s thr=%d twist=%.6f tol=%.0e error=%.6e pps=%.1f setup=%.6f\n",
                kname, sname, (int)NumThreads(), (double)twist, (double)tol, err, pps, ts_sl);
    std::fflush(stdout);
  }
}

// nthreads <= 0 leaves OMP_NUM_THREADS alone. Re-check binding whenever the width changes.
void SetWidth(const Integer nthreads, const Comm& comm) {
#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads((int)nthreads);
#endif
  static Integer checked = -1;
  if (NumThreads() != checked) { checked = NumThreads(); if (!comm.Rank()) CheckBinding(checked); }
}

void RunConvKernel(const std::string& k, const char* sname, const QScheme scheme, const Integer order,
                   const Long ppf, const Real twist, const Real tol, const Comm& comm) {
  if (k == "stokes") Run<Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT >("stokes",  sname, scheme, order, ppf, twist, tol, comm);
  else               Run<Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>("laplace", sname, scheme, order, ppf, twist, tol, comm);
}

// OMP strong-scaling: order 12, ppf 8, twist pi/6, tol 1e-9, for one (kernel, scheme). Widths are
// the full node then powers of two down to 1 -- comparable across machines with different core
// counts. Emits one @@ROW line per width for the parser.
void RunOmp(const std::string& k, const char* sname, const QScheme scheme, const Integer nt_max, const Comm& comm) {
  const Real pi = const_pi<Real>();
  std::vector<Integer> widths;
  widths.push_back(nt_max);
  for (Integer q = 1; q <= nt_max; q *= 2) widths.push_back(q);
  std::sort(widths.begin(), widths.end(), std::greater<Integer>());
  widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
  for (const Integer nt : widths) {
    SetWidth(nt, comm);
    RunConvKernel(k, sname, scheme, /*order*/ 12, /*ppf*/ 8, pi/6, (Real)1e-9, comm);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();

    if (argc < 4) {
      if (!comm.Rank()) {
        std::printf("usage: %s conv|omp <laplace|stokes> <RP|Adaptive|Hybrid|Duffy> [nthreads]\n", argv[0]);
      }
      Comm::MPI_Finalize();
      return 1;
    }
    const std::string mode = argv[1];
    const std::string kernel = argv[2];
    const std::string sname = argv[3];
    QScheme scheme;
    if (!SchemeFromName(sname, scheme)) {
      if (!comm.Rank()) std::printf("# unknown scheme '%s' (use RP|Adaptive|Hybrid|Duffy)\n", sname.c_str());
      Comm::MPI_Finalize();
      return 1;
    }
    Integer nthreads = 0;   // 0 => leave OMP_NUM_THREADS alone
    if (argc >= 5) {
      const long n = atol(argv[4]);
      if (n > 0) nthreads = (Integer)n;
      else if (!comm.Rank()) std::printf("# ignoring non-numeric thread count '%s'\n", argv[4]);
    }

    if (!comm.Rank()) {
      std::printf("# mode=%s kernel=%s scheme=%s threads/rank=%d ranks=%d\n",
                  mode.c_str(), kernel.c_str(), sname.c_str(),
                  (int)(nthreads > 0 ? nthreads : NumThreads()), (int)comm.Size());
    }

    if (mode == "conv") {
      SetWidth(nthreads, comm);
      if (!comm.Rank()) Header();
      const Real pi = const_pi<Real>();
      for (const Real twist : {pi/6, pi/2, pi})
        for (const double tol : {1e-3, 1e-5, 1e-7, 1e-9, 1e-11})
          RunConvKernel(kernel, sname.c_str(), scheme, /*order*/ 12, /*ppf*/ 12, twist, (Real)tol, comm);
    } else if (mode == "omp") {
      if (!comm.Rank()) {
        std::printf("# OpenMP strong scaling (order 12, ppf 8, twist pi/6, tol 1e-9)\n");
        Header();
      }
      const Integer nt_max = (nthreads > 0 ? nthreads : NumThreads());
      RunOmp(kernel, sname.c_str(), scheme, nt_max, comm);
    } else {
      if (!comm.Rank()) std::printf("# unknown mode '%s' (use conv|omp)\n", mode.c_str());
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
