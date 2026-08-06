/**
 * Accuracy and throughput of QuadElemList on a twisted cubed sphere.
 *
 * The twist rotates each point about z by theta*z. That is an isometry of the sphere, so the
 * SURFACE is the unit sphere at every twist and only the PARAMETRISATION shears -- which is what
 * makes it a clean stress test: the exact answers below do not move, but the elements do.
 *
 * Reported per configuration:
 *   geom_area   |sum(w) - 4 pi R^2| / (4 pi R^2)          quadrature weights vs the exact area
 *   geom_surf   max| |X(u,v)| - R | / R at OFF-NODE (u,v)  polynomial patch vs the true sphere
 *   SL[1]       spread of S[1] about its mean, and the mean against its exact value: S[1] is
 *               constant on a sphere, equal to R (Laplace) or 2R/3 (Stokes)
 *   DL[1]       max|D[1] + 1/2| / (1/2)                    exact identity, density drops out
 *   greens_den  max interpolation error of the Green's densities at off-node (u,v)
 *   greens_sol  max|(S[du/dn] - D[u]) - u| / max|u|        the on-surface identity
 *   setup, pts/s/core                                      per operator, SL and DL
 *
 * geom_* and greens_den are resolution indicators, not bounds on greens_sol. They are pointwise
 * errors BETWEEN nodes, while the identity is tested AT nodes, where the densities are exact and
 * the quadrature averages the between-node error -- measured, greens_sol can sit well below
 * greens_den. Read them together: greens_sol near geom_surf/greens_den means the discretisation
 * limits the result, greens_sol far above them means the quadrature does.
 *
 * Threads default to OMP_NUM_THREADS and can be overridden by the last argument. Binding still
 * comes from OMP_PLACES / OMP_PROC_BIND, so set those too -- the run warns if the threads do not
 * land on distinct cores, which would make every pts/s below meaningless. Run:
 *     ./bin/bench-cubed-sphere                                  full sweep
 *     ./bin/bench-cubed-sphere <nthreads>                       full sweep, given threads
 *     ./bin/bench-cubed-sphere <laplace|stokes> <order> <ppf> <twist> <tol> [nthreads]
 *     OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close ./bin/bench-cubed-sphere stokes 12 8 3.14159 1e-12
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

// Cubed sphere, ppf^2 patches per face, twisted about z by theta*z.
QuadElemList<Real> BuildSphere(const Integer order, const Long ppf, const Real R, const Real twist, const Comm& comm) {
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
  // Opt-in self-scheme override for the port comparison; default (unset) = Adaptive (centered-Alpert).
  if (const char* s = std::getenv("SCTL_SELF_SCHEME")) {
    if (!std::strcmp(s, "duffy")) qel.SetQuadScheme(QuadElemList<Real>::QuadScheme::Duffy);
  }
  return qel;
}

// Off-node parameters: midway between consecutive GL nodes, where the interpolant is least
// constrained. Node values are exact by construction, so sampling ON the grid measures nothing.
Vector<Real> MidNodes(const Integer order) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> m(order - 1);
  for (Integer i = 0; i + 1 < order; i++) m[i] = (nds[i] + nds[i+1]) / 2;
  return m;
}

struct Geom { double area, surf; };

// area: quadrature weights vs 4 pi R^2.  surf: how far the polynomial patch strays from the
// sphere between nodes -- the geometry error the operators actually see.
Geom GeomError(const QuadElemList<Real>& qel, const Real R, const Real tol, const Comm& comm) {
  Vector<Real> Xf, Xnf, wf, df; Vector<Long> cnt;
  qel.GetFarFieldNodes(Xf, Xnf, wf, df, cnt, tol);
  StaticArray<Real,2> a{0,0};
  for (Long i = 0; i < wf.Dim(); i++) a[0] += wf[i];
  comm.Allreduce(a+0, a+1, 1, CommOp::SUM);
  const Real exact = 4*const_pi<Real>()*R*R;

  const Vector<Real> mid = MidNodes(qel.Order());
  StaticArray<Real,2> s{0,0};
  for (Long e = 0; e < qel.Size(); e++) {
    Vector<Real> Xm;
    qel.GetGeom(&Xm, nullptr, nullptr, nullptr, nullptr, mid, mid, e);
    for (Long p = 0; p < Xm.Dim()/COORD_DIM; p++) {
      Real r2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) r2 += Xm[p*COORD_DIM+k]*Xm[p*COORD_DIM+k];
      s[0] = std::max<Real>(s[0], fabs<Real>(sqrt<Real>(r2) - R));
    }
  }
  comm.Allreduce(s+0, s+1, 1, CommOp::MAX);
  return {(double)(fabs<Real>(a[1] - exact)/exact), (double)(s[1]/R)};
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

// How well the order-p patch represents the Green's densities: interpolate the nodal values to
// off-node (u,v) and compare against the analytic field there.
template <class KerSL, class KerGrad>
double GreensDensityError(const QuadElemList<Real>& qel, const Vector<Real>& X0, const Comm& comm) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  const Integer order = qel.Order();
  const Long nnode = (Long)order*order;
  const Vector<Real> mid = MidNodes(order);
  const Long nm = mid.Dim();

  Vector<Real> X, Xn, Fs, Fd, Uref;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  GreensData<KerSL,KerGrad>(Fs, Fd, Uref, X, Xn, X, X0);

  // 1D interpolation from the order GL nodes to the midpoints; the 2D map is its tensor square.
  Matrix<Real> M(order, nm);
  { Vector<Real> v(order*nm, M.begin(), false);
    LagrangeInterp<Real>::Interpolate(v, QuadElemList<Real>::ParamNodes(order), mid); }

  StaticArray<Real,2> err{0,0}, val{0,0};
  for (Long e = 0; e < qel.Size(); e++) {
    Vector<Real> Xm, Xnm;
    qel.GetGeom(&Xm, &Xnm, nullptr, nullptr, nullptr, mid, mid, e);
    Vector<Real> Fs_ex, Fd_ex, U_ex;
    GreensData<KerSL,KerGrad>(Fs_ex, Fd_ex, U_ex, Xm, Xnm, Xm, X0);
    for (Long a = 0; a < nm; a++) {
      for (Long b = 0; b < nm; b++) {
        for (Integer k = 0; k < KDIM0; k++) {
          Real fs = 0, fd = 0;
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) {
              const Real w = M[i][a]*M[j][b];
              const Long t = (e*nnode + (Long)i*order + j)*KDIM0 + k;
              fs += w*Fs[t]; fd += w*Fd[t];
            }
          }
          const Long q = ((Long)a*nm + b)*KDIM0 + k;
          err[0] = std::max<Real>(err[0], fabs<Real>(fs - Fs_ex[q]));
          err[0] = std::max<Real>(err[0], fabs<Real>(fd - Fd_ex[q]));
          val[0] = std::max<Real>(val[0], std::max<Real>(fabs<Real>(Fs_ex[q]), fabs<Real>(Fd_ex[q])));
        }
      }
    }
  }
  comm.Allreduce(err+0, err+1, 1, CommOp::MAX);
  comm.Allreduce(val+0, val+1, 1, CommOp::MAX);
  return (double)(err[1] / val[1]);
}

// Constant density q: on a sphere S[q] is spatially constant, so the spread about the mean is
// pure error, and the mean itself has an exact value -- R for Laplace, 2R/3 for the Stokeslet.
template <class Ker>
void ConstSL(const QuadElemList<Real>& qel, const Real tol, const Real expect, const Comm& comm,
             double* rel_spread, double* abs_err) {
  constexpr Integer KDIM0 = Ker::SrcDim();
  Vector<Real> X, Xn; qel.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim()/COORD_DIM;
  Vector<Real> F(N*KDIM0); F.SetZero();
  for (Long i = 0; i < N; i++) F[i*KDIM0] = 1;

  BoundaryIntegralOp<Real,Ker> B(Ker(), false, comm);
  B.SetAccuracy(tol); B.AddElemList(qel); B.Setup();
  Vector<Real> U; B.ComputePotential(U, F);

  StaticArray<Real,2> sum{0,0}, mn{0,0}, mx{0,0};
  StaticArray<Long,2> cnt{0,0};
  mn[0] = 1e300; mx[0] = -1e300;
  for (Long i = 0; i < N; i++) {
    const Real v = U[i*KDIM0];
    sum[0] += v; mn[0] = std::min<Real>(mn[0], v); mx[0] = std::max<Real>(mx[0], v);
  }
  cnt[0] = N;
  comm.Allreduce(sum+0, sum+1, 1, CommOp::SUM);
  comm.Allreduce(cnt+0, cnt+1, 1, CommOp::SUM);
  comm.Allreduce(mn+0, mn+1, 1, CommOp::MIN);
  comm.Allreduce(mx+0, mx+1, 1, CommOp::MAX);
  const double mean = (double)sum[1] / (double)cnt[1];
  const double spread = std::max(std::fabs((double)mx[1] - mean), std::fabs(mean - (double)mn[1]));
  *rel_spread = spread / std::max(1e-300, std::fabs(mean));
  *abs_err = std::fabs(mean - (double)expect) / (double)expect;
}

// D[q] = -q/2 on a closed outward-oriented surface, for any constant q.
template <class Ker>
double ConstDL(const QuadElemList<Real>& qel, const Real tol, const Comm& comm) {
  constexpr Integer KDIM0 = Ker::SrcDim();
  Vector<Real> X, Xn; qel.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim()/COORD_DIM;
  Vector<Real> q(N*KDIM0);
  for (Long i = 0; i < N; i++) for (Integer k = 0; k < KDIM0; k++) q[i*KDIM0+k] = (Real)(k+1);

  BoundaryIntegralOp<Real,Ker> B(Ker(), false, comm);
  B.SetAccuracy(tol); B.AddElemList(qel); B.Setup();
  Vector<Real> U; B.ComputePotential(U, q);

  StaticArray<Real,2> err{0,0};
  for (Long i = 0; i < N; i++)
    for (Integer k = 0; k < KDIM0; k++)
      err[0] = std::max<Real>(err[0], fabs<Real>(U[i*KDIM0+k]/q[i*KDIM0+k] + (Real)0.5));
  comm.Allreduce(err+0, err+1, 1, CommOp::MAX);
  return (double)(err[1] / 0.5);
}

// On-surface interior identity S[du/dn] - D[u] = u, with the DL jump. Lifted from test-greens-conv.
//
// TIMING DEPENDS ON CALL ORDER. The two Setup() calls below are timed, and they are warm only
// because ConstSL and ConstDL have already run a full Setup() with these same two kernels. The
// first Setup() in a process is ~60% slower (0.90 vs 0.56 s, Laplace, order 12 / ppf 4 / 64
// threads) -- it first-touches the static tables (NearGradeTable's rung ladder, DuffyTable,
// ParamNodes, DiffMat), a one-time cost that is not per-element setup work and does not belong
// in the reported throughput. Keep this call last in Run(), or add an untimed warm-up here.
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
  std::printf("#%-7s %4s %5s %4s %8s %9s | %9s %9s | %9s %9s %9s | %10s %10s | %8s %8s %10s %10s\n",
              "kernel", "thr", "order", "ppf", "twist", "tol",
              "geom_area", "geom_surf", "SL[1]sprd", "SL[1]abs", "DL[1]",
              "greens_den", "greens_sol", "setup_sl", "setup_dl", "pps/c_sl", "pps/c_dl");
}

template <class KerSL, class KerDL, class KerGrad>
void Run(const char* name, const Real sl_scale, const Integer order, const Long ppf, const Real twist, const Real tol, const Comm& comm) {
  const Real R = 1;
  const QuadElemList<Real> qel = BuildSphere(order, ppf, R, twist, comm);
  const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2};   // exterior source

  const Geom g = GeomError(qel, R, tol, comm);
  const double den = GreensDensityError<KerSL,KerGrad>(qel, X0, comm);

  double sl_sprd = 0, sl_abs = 0;
  ConstSL<KerSL>(qel, tol, sl_scale*R, comm, &sl_sprd, &sl_abs);
  const double dl = ConstDL<KerDL>(qel, tol, comm);

  // Last: its Setup() calls are the timed ones and rely on the two above to have warmed the
  // per-order static tables. See the note on GreensSolError.
  double ts_sl = 0, ts_dl = 0;
  const double sol = GreensSolError<KerSL,KerDL,KerGrad>(qel, tol, X0, comm, &ts_sl, &ts_dl);

  StaticArray<Long,2> n{qel.Size()*order*order, 0};
  comm.Allreduce(n+0, n+1, 1, CommOp::SUM);
  const double N = (double)n[1], T = (double)NumThreads()*comm.Size();
  if (!comm.Rank()) {
    std::printf(" %-7s %4d %5d %4ld %8.4f %9.0e | %9.2e %9.2e | %9.2e %9.2e %9.2e | %10.2e %10.2e | %8.3f %8.3f %10.1f %10.1f\n",
                name, (int)NumThreads(), (int)order, (long)ppf, (double)twist, (double)tol,
                g.area, g.surf, sl_sprd, sl_abs, dl, den, sol,
                ts_sl, ts_dl, N/ts_sl/T, N/ts_dl/T);
    std::fflush(stdout);
  }
}

// nthreads <= 0 leaves OMP_NUM_THREADS alone. Set here rather than once in main so a sweep can
// vary the width per configuration; the binding is re-checked whenever the width changes.
void RunKernel(const std::string& k, const Integer order, const Long ppf, const Real twist, const Real tol, const Integer nthreads, const Comm& comm) {
#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads((int)nthreads);
#endif
  static Integer checked = -1;
  if (NumThreads() != checked) { checked = NumThreads(); if (!comm.Rank()) CheckBinding(checked); }

  // S[1] on a sphere of radius R: R for Laplace, 2R/3 for the Stokeslet.
  if (k == "stokes") Run<Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT >("stokes",  (Real)2/3, order, ppf, twist, tol, comm);
  else               Run<Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>("laplace", (Real)1,   order, ppf, twist, tol, comm);
}

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();

    // Threads: last positional argument if present, else OMP_NUM_THREADS. RunKernel applies it.
    const bool single = (argc >= 6);
    const char* nthr_arg = (single && argc >= 7) ? argv[6] : (!single && argc == 2 ? argv[1] : nullptr);
    Integer nthreads = 0;   // 0 => leave OMP_NUM_THREADS alone
    if (nthr_arg) {
      const long n = atol(nthr_arg);
      if (n > 0) nthreads = (Integer)n;
      else if (!comm.Rank()) std::printf("# ignoring non-numeric thread count '%s'\n", nthr_arg);
    }

    if (!comm.Rank()) {
      std::printf("threads/rank = %d, ranks = %d\n", (int)(nthreads > 0 ? nthreads : NumThreads()), (int)comm.Size());
      std::printf("SL[1]abs compares S[1] against its exact value: R (Laplace), 2R/3 (Stokes).\n\n");
    }
    Header();

    if (argc >= 6) {
      RunKernel(argv[1], (Integer)atol(argv[2]), (Long)atol(argv[3]), (Real)atof(argv[4]), (Real)atof(argv[5]), nthreads, comm);
    } else {
      if (argc > 1 && argc != 2 && !comm.Rank()) std::printf("# ignoring partial arguments; running the full sweep\n");
      const Real pi = const_pi<Real>();
      for (const char* k : {"laplace", "stokes"})
        for (const Integer order : {12})
          for (const Real twist : {pi/6, pi/2, pi})
            for (const double tol : {1e-3, 1e-6, 1e-9, 1e-12})
              RunKernel(k, order, /*ppf*/ 12, twist, (Real)tol, nthreads, comm);

      // Strong scaling from the widest available width down to 1. Smaller mesh than the sweep
      // above: the 1-thread points dominate the runtime, and GreensDensityError is serial so it
      // does not shrink with width at all.
      const Integer nt_max = (nthreads > 0 ? nthreads : NumThreads());
      if (!comm.Rank()) std::printf("# OpenMP strong scaling (order 12, ppf 8, twist pi/6, tol 1e-9)\n");
      // Widths: the full node, then powers of two down to 1. Halving from nt_max instead
      // would give 96, 48, 24, ... on a 96-core node, which is not comparable across machines.
      std::vector<Integer> widths;
      widths.push_back(nt_max);
      for (Integer q = 1; q <= nt_max; q *= 2) widths.push_back(q);
      std::sort(widths.begin(), widths.end(), std::greater<Integer>());
      widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
      for (const Integer nt : widths) RunKernel("laplace", /*order*/ 12, /*ppf*/ 8, pi/6, (Real)1e-9, nt, comm);
      for (const Integer nt : widths) RunKernel("stokes",  /*order*/ 12, /*ppf*/ 8, pi/6, (Real)1e-9, nt, comm);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}