/**
 * On-surface Green's-identity accuracy of the SELF scheme on twisted cubed spheres.
 *
 * Isolates the singular self-quadrature: every scheme below shares the SAME split-at-foot near
 * path, so the only thing that varies is the self operator. Compares
 *   adaptive  = split near + centered graded-u x Alpert-v self (current default)
 *   rp        = split near + RectPolar (Bruno 2018) self       (Hybrid; the scheme that plateaus
 *               on sheared panels -- its v-rule / COV caps the attainable accuracy under twist)
 *   duffy     = split near + Duffy edge-collapsed sinh self     (ported from upstream)
 *
 * Reports ONLY the on-surface interior identity  S[du/dn] - D[u] = u  (DL jump included), for both
 * Laplace and the Stokeslet, over a twist sweep at fixed order/ppf. A finely-resolved sphere
 * (ppf 12) pushes the geometry/near floor well below the self-scheme error, so the plateau (or
 * lack of one) is the self scheme's.
 *
 * Threads default to OMP_NUM_THREADS; bind with OMP_PLACES=cores OMP_PROC_BIND=close so pts land
 * on distinct physical cores. Run:
 *     OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close ./bin/sweep-twist-selfscheme
 *     ... ./bin/sweep-twist-selfscheme <adaptive|rp|duffy> [nbeta]
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

#include <cstdio>
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

Integer NumThreads() {
#ifdef _OPENMP
  return (Integer)omp_get_max_threads();
#else
  return 1;
#endif
}

// Warn if threads share cores -- the run would not be on distinct physical CPUs.
void CheckBinding(const Integer T) {
#if defined(__linux__) && defined(_OPENMP)
  std::vector<int> cpu(T, -1);
  #pragma omp parallel num_threads(T)
  { const Integer t = omp_get_thread_num(); if (t < T) cpu[t] = sched_getcpu(); }
  const std::set<int> uniq(cpu.begin(), cpu.end());
  std::printf("# thread->cpu map:");
  for (Integer t = 0; t < T; t++) std::printf(" %d", cpu[t]);
  std::printf("\n");
  if ((Integer)uniq.size() != T)
    std::printf("*** WARNING: %ld distinct cores for %d threads -- NOT mapped to distinct physical CPUs\n",
                (long)uniq.size(), (int)T);
  else
    std::printf("# OK: %d threads on %d distinct cores\n", (int)T, (int)T);
#else
  (void)T;
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
  for (Integer f = 0; f < 6; f++)
    for (Long iu = 0; iu < ppf; iu++)
      for (Long iv = 0; iv < ppf; iv++)
        for (Integer i = 0; i < order; i++) {
          const Real a = 2*((iu + nds[i]) / (Real)ppf) - 1;
          for (Integer j = 0; j < order; j++) {
            const Real b = 2*((iv + nds[j]) / (Real)ppf) - 1;
            Real x, y, z; FacePoint(x, y, z, f, a, b, R);
            const Real s = sin<Real>(twist*z), c = cos<Real>(twist*z);
            X.PushBack(x*c + y*s); X.PushBack(-x*s + y*c); X.PushBack(z);
          }
        }
  return QuadElemList<Real>(order, X, comm);
}

// Fd = u|_S (DL density), Fs = du/dn (SL density), Uref = u at targets (here the surface nodes).
template <class KerSL, class KerGrad>
void GreensData(Vector<Real>& Fs, Vector<Real>& Fd, Vector<Real>& Uref,
                const Vector<Real>& X, const Vector<Real>& Xn, const Vector<Real>& X0) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  KerSL ker_sl; KerGrad ker_grad;
  Vector<Real> Xn0{0,0,0}, F0(KDIM0), dU;
  for (Integer i = 0; i < KDIM0; i++) F0[i] = (Real)1 / (Real)(i + 1);
  ker_sl.Eval(Fd, X, X0, Xn0, F0);
  ker_grad.Eval(dU, X, X0, Xn0, F0);
  ker_sl.Eval(Uref, X, X0, Xn0, F0);
  const Long N = X.Dim() / COORD_DIM;
  Fs.ReInit(N * KDIM0);
  for (Long i = 0; i < N; i++)
    for (Integer j = 0; j < KDIM0; j++) {
      Real dn = 0;
      for (Integer k = 0; k < COORD_DIM; k++) dn += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
      Fs[i*KDIM0+j] = dn;
    }
}

// On-surface interior identity S[du/dn] - D[u] = u, with the DL jump. Returns max rel error.
template <class KerSL, class KerDL, class KerGrad>
double GreensSolError(const QuadElemList<Real>& qel, const Real tol, const Vector<Real>& X0, const Comm& comm) {
  KerSL ker_sl; KerDL ker_dl;
  BoundaryIntegralOp<Real,KerSL> BSL(ker_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BDL(ker_dl, false, comm);
  BSL.AddElemList(qel); BDL.AddElemList(qel);
  BSL.SetAccuracy(tol); BDL.SetAccuracy(tol);

  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  GreensData<KerSL,KerGrad>(Fs, Fd, Uref, X, Xn, X0);

  BSL.Setup(); BDL.Setup();
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

enum class Scheme { Adaptive, RP, Duffy };

void ApplyScheme(QuadElemList<Real>& qel, Scheme s, Integer nbeta) {
  using QS = QuadElemList<Real>::QuadScheme;
  if (s == Scheme::RP)         qel.SetQuadScheme(QS::Hybrid, /*q=*/6, /*cov_order=*/nbeta, /*max_depth=*/30);
  else if (s == Scheme::Duffy) qel.SetQuadScheme(QS::Duffy);
  // Adaptive = default; leave as-is.
}

double WallTime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
}

void RunSweep(Scheme s, const char* sname, Integer nbeta, const Integer order, const Long ppf, const Comm& comm) {
  const Real R = 1, pi = const_pi<Real>();
  const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2};   // exterior source
  const std::vector<std::pair<const char*,Real>> twists = {
    {"0", 0}, {"pi/6", pi/6}, {"pi/4", pi/4}, {"pi/3", pi/3}, {"pi/2", pi/2}, {"pi", pi}};
  const std::vector<double> tols = {1e-9, 1e-11};

  if (!comm.Rank())
    std::printf("# scheme=%s  order=%d  ppf=%ld  Nelem=%ld  nbeta=%d(rp only)\n#%-6s %-6s %-9s %12s %12s %9s %9s\n",
                sname, (int)order, (long)ppf, (long)(6*ppf*ppf), (int)nbeta,
                "twist", "tol", "kernel", "greens_sol", "", "t_setup", "");
  for (const auto& tw : twists) {
    QuadElemList<Real> qel = BuildSphere(order, ppf, R, tw.second, comm);
    ApplyScheme(qel, s, nbeta);
    for (const double tol : tols) {
      const double t0 = WallTime();
      const double e_lap = GreensSolError<Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(qel, (Real)tol, X0, comm);
      const double t1 = WallTime();
      const double e_stk = GreensSolError<Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(qel, (Real)tol, X0, comm);
      const double t2 = WallTime();
      if (!comm.Rank()) {
        std::printf(" %-6s %-6.0e laplace   %12.3e              %9.2f\n", tw.first, tol, e_lap, t1-t0);
        std::printf(" %-6s %-6.0e stokes    %12.3e              %9.2f\n", tw.first, tol, e_stk, t2-t1);
        std::fflush(stdout);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const Integer order = 12; const Long ppf = 12;
    Integer nbeta = (argc >= 3 ? (Integer)atol(argv[2]) : 300);

    if (!comm.Rank()) {
      std::printf("threads/rank = %d, ranks = %d\n", (int)NumThreads(), (int)comm.Size());
      CheckBinding(NumThreads());
      std::printf("\n");
    }

    auto run = [&](Scheme s, const char* n) { RunSweep(s, n, nbeta, order, ppf, comm); if (!comm.Rank()) std::printf("\n"); };
    if (argc >= 2) {
      if      (!std::strcmp(argv[1], "duffy"))    run(Scheme::Duffy, "duffy");
      else if (!std::strcmp(argv[1], "rp"))       run(Scheme::RP, "rp(Hybrid)");
      else if (!std::strcmp(argv[1], "adaptive")) run(Scheme::Adaptive, "adaptive");
      else if (!comm.Rank()) std::printf("unknown scheme '%s' (use adaptive|rp|duffy)\n", argv[1]);
    } else {
      run(Scheme::RP, "rp(Hybrid)");
      run(Scheme::Duffy, "duffy");
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
