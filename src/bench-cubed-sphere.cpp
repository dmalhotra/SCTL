/**
 * Accuracy and throughput of QuadElemList on a twisted cubed sphere, for every quadrature scheme.
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
 * At a target on the surface, TensorProduct and Duffy return the principal value of the double
 * layer and Hedgehog the limit from outside, larger by half the density; DL[1] and greens_sol
 * remove that difference, so every scheme is scored against the same exact values.
 *
 * geom_* and greens_den are resolution indicators, not bounds on greens_sol. They are pointwise
 * errors BETWEEN nodes, while the identity is tested AT nodes, where the densities are exact and
 * the quadrature averages the between-node error -- measured, greens_sol can sit well below
 * greens_den. Read them together: greens_sol near geom_surf/greens_den means the discretisation
 * limits the result, greens_sol far above them means the quadrature does.
 *
 * The setup times are the minimum of BENCH_SETUP_REPS setups (environment variable, default 2),
 * after an untimed setup of the same operator that builds the static tables. Threads default to
 * OMP_NUM_THREADS and can be overridden by the last argument. Binding still comes from OMP_PLACES /
 * OMP_PROC_BIND, so set those too -- the run warns if the threads do not land on distinct cores,
 * which would make every pts/s below meaningless. Run:
 *     ./bin/bench-cubed-sphere [nthreads]                        sweep, then strong scaling
 *     ./bin/bench-cubed-sphere sweep|scaling [nthreads]          one of the two
 *     ./bin/bench-cubed-sphere <laplace|stokes> <order> <ppf> <twist> <tol> [nthreads]
 *     OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close ./bin/bench-cubed-sphere stokes 12 8 3.14159 1e-12
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
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

struct SchemeInfo {
  const char* name;
  QuadElemList<Real>::QuadScheme scheme;
  Real dl_jump; // on-surface double layer minus its principal value, per unit density
};

const std::array<SchemeInfo,3> Schemes{{
  {"TensorProduct", QuadElemList<Real>::QuadScheme::TensorProduct, 0},
  {"Duffy", QuadElemList<Real>::QuadScheme::Duffy, 0},
  {"Hedgehog", QuadElemList<Real>::QuadScheme::Hedgehog, (Real)0.5}}};

double WallTime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
}

// The number of distinct cores under the T threads of every process; warn if they share cores: pts/s/core is
// meaningless under oversubscription. Collective.
void CheckBinding(const Integer T, const Comm& comm) {
#if defined(__linux__) && defined(_OPENMP)
  Vector<Integer> cpu(T), all(T * comm.Size());
  #pragma omp parallel num_threads(T)
  {
    const Integer t = omp_get_thread_num();
    if (t < T) cpu[t] = (Integer)sched_getcpu();
  }
  comm.Allgather(cpu.begin(), T, all.begin(), T);
  const std::set<Integer> uniq(all.begin(), all.end());
  if (!comm.Rank()) {
    std::printf("# distinct cores: %ld for %d processes x %d threads\n", (long)uniq.size(), (int)comm.Size(), (int)T);
    if ((Long)uniq.size() != T * comm.Size()) std::printf("*** WARNING: %ld distinct cores for %d processes x %d threads -- oversubscribed, pts/s/core is meaningless\n", (long)uniq.size(), (int)comm.Size(), (int)T);
  }
#else
  (void)T;
  (void)comm;
#endif
}

Integer NumThreads() {
#ifdef _OPENMP
  return (Integer)omp_get_max_threads();
#else
  return 1;
#endif
}

// Cubed sphere of radius R, ppf^2 patches per face, twisted about z by twist*z. Each process
// builds its contiguous range of the elements.
QuadElemList<Real> BuildSphere(const Integer order, const Long ppf, const Real R, const Real twist, const Comm& comm) {
  static constexpr Integer face_map[6][3][3] = { // cube-face point (x,y,z) as coefficients of (1, a, b)
    {{ 1,  0, 0}, { 0, 1, 0}, { 0, 0,  1}},
    {{-1,  0, 0}, { 0, -1, 0}, { 0, 0,  1}},
    {{ 0,  1, 0}, { 1, 0, 0}, { 0, 0, -1}},
    {{ 0,  1, 0}, {-1, 0, 0}, { 0, 0,  1}},
    {{ 0,  1, 0}, { 0, 0, 1}, { 1, 0,  0}},
    {{ 0, -1, 0}, { 0, 0, 1}, {-1, 0,  0}}};
  const Long Nelem = 6 * ppf * ppf;
  const Long elem0 = Nelem * comm.Rank() / comm.Size();
  const Long elem1 = Nelem * (comm.Rank() + 1) / comm.Size();
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> X;
  for (Long elem = elem0; elem < elem1; elem++) {
    const Integer f = (Integer)(elem / (ppf * ppf));
    const Long iu = elem / ppf % ppf, iv = elem % ppf;
    for (Integer i = 0; i < order; i++) {
      const Real a = 2 * ((iu + nds[i]) / (Real)ppf) - 1;
      for (Integer j = 0; j < order; j++) {
        const Real b = 2 * ((iv + nds[j]) / (Real)ppf) - 1;
        std::array<Real,3> p;
        for (Integer k = 0; k < COORD_DIM; k++) p[k] = face_map[f][k][0] + face_map[f][k][1] * a + face_map[f][k][2] * b;
        const Real r = R / sqrt<Real>(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        const Real x = p[0] * r, y = p[1] * r, z = p[2] * r;
        const Real s = sin<Real>(twist * z), c = cos<Real>(twist * z);
        X.PushBack(x * c + y * s);
        X.PushBack(-x * s + y * c);
        X.PushBack(z);
      }
    }
  }
  return QuadElemList<Real>(order, X);
}

// Off-node parameters: midway between consecutive GL nodes, where the interpolant is least
// constrained. Node values are exact by construction, so sampling ON the grid measures nothing.
Vector<Real> MidNodes(const Integer order) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> m(order - 1);
  for (Integer i = 0; i + 1 < order; i++) m[i] = (nds[i] + nds[i + 1]) / 2;
  return m;
}

struct Geom {
  double area, surf;
};

// area: quadrature weights vs 4 pi R^2.  surf: how far the polynomial patch strays from the
// sphere between nodes -- the geometry error the operators actually see.
Geom GeomError(const QuadElemList<Real>& qel, const Real R, const Real tol, const Comm& comm) {
  Vector<Real> Xf, Xnf, wf, df;
  Vector<Long> cnt;
  qel.GetFarFieldNodes(Xf, Xnf, wf, df, cnt, tol);
  StaticArray<Real,2> a{0, 0};
  for (Long i = 0; i < wf.Dim(); i++) a[0] += wf[i];
  comm.Allreduce(a + 0, a + 1, 1, CommOp::SUM);
  const Real exact = 4 * const_pi<Real>() * R * R;

  const Vector<Real> mid = MidNodes(qel.Order());
  StaticArray<Real,2> s{0, 0};
  for (Long e = 0; e < qel.Size(); e++) {
    Vector<Real> Xm;
    qel.GetGeom(&Xm, nullptr, nullptr, nullptr, nullptr, mid, mid, e);
    for (Long p = 0; p < Xm.Dim() / COORD_DIM; p++) {
      Real r2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) r2 += Xm[p * COORD_DIM + k] * Xm[p * COORD_DIM + k];
      s[0] = std::max<Real>(s[0], fabs<Real>(sqrt<Real>(r2) - R));
    }
  }
  comm.Allreduce(s + 0, s + 1, 1, CommOp::MAX);
  return {(double)(fabs<Real>(a[1] - exact) / exact), (double)(s[1] / R)};
}

// Surface data for an exterior point source: Fd = u|_S (DL density), Fs = du/dn (SL density),
// Uref = u at the targets.
template <class KerSL, class KerGrad> void GreensData(Vector<Real>& Fs, Vector<Real>& Fd, Vector<Real>& Uref, const Vector<Real>& X, const Vector<Real>& Xn, const Vector<Real>& Xtrg, const Vector<Real>& X0) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  const KerSL ker_sl;
  const KerGrad ker_grad;
  Vector<Real> Xn0{0, 0, 0}, F0(KDIM0), dU; // neither kernel reads the source normal
  for (Integer i = 0; i < KDIM0; i++) F0[i] = (Real)1 / (Real)(i + 1);

  ker_sl.Eval(Fd, X, X0, Xn0, F0);
  ker_grad.Eval(dU, X, X0, Xn0, F0);
  ker_sl.Eval(Uref, Xtrg, X0, Xn0, F0);

  const Long N = X.Dim() / COORD_DIM;
  Fs.ReInit(N * KDIM0);
  for (Long i = 0; i < N; i++) {
    for (Integer j = 0; j < KDIM0; j++) {
      Real dn = 0;
      for (Integer k = 0; k < COORD_DIM; k++) dn += dU[(i * KDIM0 + j) * COORD_DIM + k] * Xn[i * COORD_DIM + k];
      Fs[i * KDIM0 + j] = dn;
    }
  }
}

// How well the order-p patch represents the Green's densities: interpolate the nodal values to
// off-node (u,v) and compare against the analytic field there.
template <class KerSL, class KerGrad> double GreensDensityError(const QuadElemList<Real>& qel, const Vector<Real>& X0, const Comm& comm) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  const Integer order = qel.Order();
  const Long nnode = (Long)order * order;
  const Vector<Real> mid = MidNodes(order);
  const Long nm = mid.Dim();

  Vector<Real> X, Xn, Fs, Fd, Uref;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  GreensData<KerSL,KerGrad>(Fs, Fd, Uref, X, Xn, X, X0);

  Matrix<Real> M(order, nm); // 1D interpolation from the GL nodes to the midpoints; the 2D map is its tensor square
  {
    Vector<Real> v(order * nm, M.begin(), false);
    LagrangeInterp<Real>::Interpolate(v, QuadElemList<Real>::ParamNodes(order), mid);
  }

  StaticArray<Real,2> err{0, 0}, val{0, 0};
  for (Long e = 0; e < qel.Size(); e++) {
    Vector<Real> Xm, Xnm, Fs_ex, Fd_ex, U_ex;
    qel.GetGeom(&Xm, &Xnm, nullptr, nullptr, nullptr, mid, mid, e);
    GreensData<KerSL,KerGrad>(Fs_ex, Fd_ex, U_ex, Xm, Xnm, Xm, X0);
    for (Long a = 0; a < nm; a++) {
      for (Long b = 0; b < nm; b++) {
        for (Integer k = 0; k < KDIM0; k++) {
          Real fs = 0, fd = 0;
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) {
              const Real w = M[i][a] * M[j][b];
              const Long t = (e * nnode + (Long)i * order + j) * KDIM0 + k;
              fs += w * Fs[t];
              fd += w * Fd[t];
            }
          }
          const Long q = (a * nm + b) * KDIM0 + k;
          err[0] = std::max<Real>(err[0], std::max<Real>(fabs<Real>(fs - Fs_ex[q]), fabs<Real>(fd - Fd_ex[q])));
          val[0] = std::max<Real>(val[0], std::max<Real>(fabs<Real>(Fs_ex[q]), fabs<Real>(Fd_ex[q])));
        }
      }
    }
  }
  comm.Allreduce(err + 0, err + 1, 1, CommOp::MAX);
  comm.Allreduce(val + 0, val + 1, 1, CommOp::MAX);
  return (double)(err[1] / val[1]);
}

// Constant density q: on a sphere S[q] is spatially constant, so the spread about the mean is
// pure error, and the mean itself has an exact value -- R for Laplace, 2R/3 for the Stokeslet.
template <class Ker> void ConstSL(const QuadElemList<Real>& qel, const Real tol, const Real expect, const Comm& comm, double* rel_spread, double* abs_err) {
  constexpr Integer KDIM0 = Ker::SrcDim();
  Vector<Real> X, Xn;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim() / COORD_DIM;
  Vector<Real> F(N * KDIM0), U;
  F.SetZero();
  for (Long i = 0; i < N; i++) F[i * KDIM0] = 1;

  BoundaryIntegralOp<Real,Ker> B(Ker(), false, comm);
  B.SetAccuracy(tol);
  B.AddElemList(qel);
  B.ComputePotential(U, F);

  StaticArray<Real,2> sum{0, 0}, mn{1e300, 0}, mx{-1e300, 0};
  StaticArray<Long,2> cnt{N, 0};
  for (Long i = 0; i < N; i++) {
    const Real v = U[i * KDIM0];
    sum[0] += v;
    mn[0] = std::min<Real>(mn[0], v);
    mx[0] = std::max<Real>(mx[0], v);
  }
  comm.Allreduce(sum + 0, sum + 1, 1, CommOp::SUM);
  comm.Allreduce(cnt + 0, cnt + 1, 1, CommOp::SUM);
  comm.Allreduce(mn + 0, mn + 1, 1, CommOp::MIN);
  comm.Allreduce(mx + 0, mx + 1, 1, CommOp::MAX);
  const double mean = (double)sum[1] / (double)cnt[1];
  const double spread = std::max(std::fabs((double)mx[1] - mean), std::fabs(mean - (double)mn[1]));
  *rel_spread = spread / std::max(1e-300, std::fabs(mean));
  *abs_err = std::fabs(mean - (double)expect) / (double)expect;
}

// D[q] = -q/2 on a closed outward-oriented surface, for any constant q, after removing the
// scheme's dl_jump.
template <class Ker> double ConstDL(const QuadElemList<Real>& qel, const Real tol, const Real dl_jump, const Comm& comm) {
  constexpr Integer KDIM0 = Ker::SrcDim();
  Vector<Real> X, Xn;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim() / COORD_DIM;
  Vector<Real> q(N * KDIM0), U;
  for (Long i = 0; i < N; i++) {
    for (Integer k = 0; k < KDIM0; k++) q[i * KDIM0 + k] = (Real)(k + 1);
  }

  BoundaryIntegralOp<Real,Ker> B(Ker(), false, comm);
  B.SetAccuracy(tol);
  B.AddElemList(qel);
  B.ComputePotential(U, q);
  U -= dl_jump * q;

  StaticArray<Real,2> err{0, 0};
  for (Long i = 0; i < N * KDIM0; i++) err[0] = std::max<Real>(err[0], fabs<Real>(U[i] / q[i] + (Real)0.5));
  comm.Allreduce(err + 0, err + 1, 1, CommOp::MAX);
  return (double)(err[1] / 0.5);
}

// On-surface interior identity S[du/dn] - D[u] = u, with the DL jump and the scheme's dl_jump.
// The setup times are the minimum of BENCH_SETUP_REPS (default 2), taken after ConstSL and ConstDL
// have set up the same kernels at the same tolerance: the first setup in a process also builds the
// static tables, a one-time cost that is not per-element setup work.
template <class KerSL, class KerDL, class KerGrad> double GreensSolError(const QuadElemList<Real>& qel, const Real tol, const Real dl_jump, const Vector<Real>& X0, const Comm& comm, double* t_setup_sl, double* t_setup_dl) {
  const KerSL ker_sl;
  const KerDL ker_dl;
  BoundaryIntegralOp<Real,KerSL> BSL(ker_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BDL(ker_dl, false, comm);
  BSL.AddElemList(qel);
  BDL.AddElemList(qel);
  BSL.SetAccuracy(tol);
  BDL.SetAccuracy(tol);

  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  GreensData<KerSL,KerGrad>(Fs, Fd, Uref, X, Xn, X, X0);

  const char* reps_env = std::getenv("BENCH_SETUP_REPS");
  const Integer reps = (reps_env ? std::max(1, atoi(reps_env)) : 2);
  *t_setup_sl = *t_setup_dl = 1e300;
  for (Integer rep = 0; rep < reps; rep++) {
    BSL.ClearSetup();
    BDL.ClearSetup();
    comm.Barrier();
    const double t0 = WallTime();
    BSL.Setup();
    comm.Barrier();
    const double t1 = WallTime();
    BDL.Setup();
    comm.Barrier();
    const double t2 = WallTime();
    *t_setup_sl = std::min(*t_setup_sl, t1 - t0);
    *t_setup_dl = std::min(*t_setup_dl, t2 - t1);
  }

  BSL.ComputePotential(Us, Fs);
  BDL.ComputePotential(Ud, Fd);
  Ud -= ((Real)0.5 + dl_jump) * Fd;

  StaticArray<Real,2> err{0, 0}, val{0, 0};
  for (Long i = 0; i < Uref.Dim(); i++) {
    err[0] = std::max<Real>(err[0], fabs<Real>((Us[i] - Ud[i]) - Uref[i]));
    val[0] = std::max<Real>(val[0], fabs<Real>(Uref[i]));
  }
  comm.Allreduce(err + 0, err + 1, 1, CommOp::MAX);
  comm.Allreduce(val + 0, val + 1, 1, CommOp::MAX);
  return (double)(err[1] / val[1]);
}

void Header() {
  std::printf("#%-7s %-13s %4s %5s %4s %8s %9s | %9s %9s | %9s %9s %9s | %10s %10s | %8s %8s %10s %10s\n", "kernel", "scheme", "thr", "order", "ppf", "twist", "tol", "geom_area", "geom_surf", "SL[1]sprd", "SL[1]abs", "DL[1]", "greens_den", "greens_sol", "setup_sl", "setup_dl", "pps/c_sl", "pps/c_dl");
}

template <class KerSL, class KerDL, class KerGrad> void Run(const char* name, const Real sl_scale, const Integer order, const Long ppf, const Real twist, const Real tol, const Comm& comm) {
  const Real R = 1;
  QuadElemList<Real> qel = BuildSphere(order, ppf, R, twist, comm);
  const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2}; // exterior source

  const Geom g = GeomError(qel, R, tol, comm);
  const double den = GreensDensityError<KerSL,KerGrad>(qel, X0, comm);
  StaticArray<Long,2> n{qel.Size() * order * order, 0};
  comm.Allreduce(n + 0, n + 1, 1, CommOp::SUM);
  const double N = (double)n[1], T = (double)NumThreads() * comm.Size();

  for (const SchemeInfo& sc : Schemes) {
    qel.SetQuadScheme(sc.scheme);
    double sl_sprd = 0, sl_abs = 0, ts_sl = 0, ts_dl = 0;
    ConstSL<KerSL>(qel, tol, sl_scale * R, comm, &sl_sprd, &sl_abs);
    const double dl = ConstDL<KerDL>(qel, tol, sc.dl_jump, comm);
    const double sol = GreensSolError<KerSL,KerDL,KerGrad>(qel, tol, sc.dl_jump, X0, comm, &ts_sl, &ts_dl);
    if (!comm.Rank()) {
      std::printf(" %-7s %-13s %4d %5d %4ld %8.4f %9.0e | %9.2e %9.2e | %9.2e %9.2e %9.2e | %10.2e %10.2e | %8.3f %8.3f %10.1f %10.1f\n", name, sc.name, (int)NumThreads(), (int)order, (long)ppf, (double)twist, (double)tol, g.area, g.surf, sl_sprd, sl_abs, dl, den, sol, ts_sl, ts_dl, N / ts_sl / T, N / ts_dl / T);
      std::fflush(stdout);
    }
  }
}

// nthreads <= 0 leaves OMP_NUM_THREADS alone. Set here rather than once in main so a sweep can
// vary the width per configuration; the binding is re-checked whenever the width changes.
void RunKernel(const std::string& k, const Integer order, const Long ppf, const Real twist, const Real tol, const Integer nthreads, const Comm& comm) {
#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads((int)nthreads);
#endif
  static Integer checked = -1;
  if (NumThreads() != checked) {
    checked = NumThreads();
    CheckBinding(checked, comm);
  }
  // S[1] on a sphere of radius R: R for Laplace, 2R/3 for the Stokeslet.
  if (k == "stokes") {
    Run<Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>("stokes", (Real)2 / 3, order, ppf, twist, tol, comm);
  } else {
    Run<Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>("laplace", (Real)1, order, ppf, twist, tol, comm);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const bool single = (argc >= 6);
    const std::string mode = (!single && argc >= 2 && (std::string(argv[1]) == "sweep" || std::string(argv[1]) == "scaling") ? argv[1] : "");

    // Threads: last positional argument if present, else OMP_NUM_THREADS. RunKernel applies it.
    const char* nthr_arg = nullptr;
    if (single && argc >= 7) nthr_arg = argv[6];
    if (!single && argc == (mode.empty() ? 2 : 3)) nthr_arg = argv[argc - 1];
    Integer nthreads = 0; // 0: leave OMP_NUM_THREADS alone
    if (nthr_arg) {
      const long n = atol(nthr_arg);
      if (n > 0) nthreads = (Integer)n;
      else if (!comm.Rank()) std::printf("# ignoring non-numeric thread count '%s'\n", nthr_arg);
    }

    if (!comm.Rank()) {
      std::printf("threads/rank = %d, ranks = %d\n", (int)(nthreads > 0 ? nthreads : NumThreads()), (int)comm.Size());
      std::printf("SL[1]abs compares S[1] against its exact value: R (Laplace), 2R/3 (Stokes).\n\n");
      Header();
    }

    if (single) {
      RunKernel(argv[1], (Integer)atol(argv[2]), (Long)atol(argv[3]), (Real)atof(argv[4]), (Real)atof(argv[5]), nthreads, comm);
    } else {
      const Real pi = const_pi<Real>();
      if (mode != "scaling") {
        for (const char* k : {"laplace", "stokes"}) {
          for (const Real twist : {pi / 6, pi / 2, pi}) {
            for (const double tol : {1e-3, 1e-6, 1e-9, 1e-12}) RunKernel(k, 12, 12, twist, (Real)tol, nthreads, comm);
          }
        }
      }
      if (mode != "sweep") {
        // Strong scaling from the widest available width down to 1: the full node, then powers
        // of two, which compare across machines with different core counts. Smaller mesh than the
        // sweep: the 1-thread points dominate the runtime, and GreensDensityError is serial.
        const Integer nt_max = (nthreads > 0 ? nthreads : NumThreads());
        if (!comm.Rank()) std::printf("# OpenMP strong scaling (order 12, ppf 8, twist pi/6, tol 1e-9)\n");
        std::vector<Integer> widths{nt_max};
        for (Integer q = 1; q <= nt_max; q *= 2) widths.push_back(q);
        std::sort(widths.begin(), widths.end(), std::greater<Integer>());
        widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
        for (const char* k : {"laplace", "stokes"}) {
          for (const Integer nt : widths) RunKernel(k, 12, 8, pi / 6, (Real)1e-9, nt, comm);
        }
      }
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
