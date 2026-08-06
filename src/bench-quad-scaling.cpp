/**
 * OpenMP strong-scaling of the QuadElemList setup stages.
 *
 * Unlike bench-quad-interac (single-element, single-thread cost attribution),
 * this driver exercises the two OpenMP regions that actually exist:
 *
 *   SELF  QuadElemList::SelfInteracHelper  `omp parallel for schedule(static)`
 *         over elements. Measured directly -- SelfInterac is exactly the call
 *         BoundaryIntegralOp::SetupSelf makes.
 *
 *   NEAR  BoundaryIntegralOp::SetupNear    `omp parallel for schedule(dynamic)`
 *         over (element, near-target) pairs, each calling NearInterac with a
 *         single target. QuadElemList itself is serial here, so the stage is
 *         only reachable through the BIOp; it is measured via the profiler's
 *         nested SetupNear / BuildNearLst labels.
 *
 * Per thread count the driver reports machine-readable `#SCALE` rows plus the
 * full profile tree, and verifies that the OpenMP threads land on distinct
 * cores before timing anything.
 *
 * Build and run (must compile on the target node: -march=native):
 *     make bin/bench-quad-scaling
 *     OMP_PLACES=cores OMP_PROC_BIND=close ./bin/bench-quad-scaling
 *
 * Args: [order] [patch_per_face] [tol] [thread_list]
 *     ./bin/bench-quad-scaling 8 16 1e-6 1,2,4,8,16,32,64,128
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

namespace sctl {
  // QuadElemList declares `template<typename> friend struct QuadElemTestAccess`, so this
  // gives the bench read access to the private self-quadrature rule sizes. Each target is
  // integrated over four Duffy triangles with an ns x nt rule; ns and nt are the same for
  // every target, so the node count per target is exactly 4*ns*nt.
  template <class Real> struct QuadElemTestAccess {
    template <Integer order> static void SelfRuleSizes(const Integer digits, const Integer kdim0) {
      const Integer ns = QuadElemList<Real>::template DuffyTable<order>().ns;
      const Integer nt = QuadElemList<Real>::DuffyTOrder(digits, order, kdim0);
      std::printf("self quadrature (Duffy, order=%d, digits=%d, kdim0=%d):\n",
                  (int)order, (int)digits, (int)kdim0);
      std::printf("  radial GL nodes ns: %d      edge nodes nt: %d\n", (int)ns, (int)nt);
      std::printf("  nodes per target  = 4*ns*nt = %ld\n", (long)(4*ns*nt));
      std::printf("  nodes per element = sum over %d targets = %ld\n\n",
                  (int)(order*order), (long)(4*ns*nt)*order*order);
    }
  };
}

using namespace sctl;

namespace {

using Real = double;
constexpr Integer COORD_DIM = 3;

void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R) {
  switch (face) {
    case 0: x =  1; y =  a; z =  b; break;
    case 1: x = -1; y = -a; z =  b; break;
    case 2: x =  a; y =  1; z = -b; break;
    case 3: x =  a; y = -1; z =  b; break;
    case 4: x =  a; y =  b; z =  1; break;
    case 5: x = -a; y =  b; z = -1; break;
    default: SCTL_ASSERT(false);
  }
  const Real r = sqrt<Real>(x*x + y*y + z*z);
  x *= R/r; y *= R/r; z *= R/r;
}

// Cubed-sphere of radius R, PatchPerFace^2 patches/face, twisted about z. The twist
// makes the patches non-axis-aligned so the adaptive near tree does realistic work.
QuadElemList<Real> BuildTwistedSphere(Integer ElemOrder, Long PatchPerFace, Real Radius, Real theta_twist) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(ElemOrder);
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PatchPerFace; iu++)
      for (Long iv = 0; iv < PatchPerFace; iv++)
        for (Integer i = 0; i < ElemOrder; i++) {
          const Real a = 2*((iu + nds[i]) / (Real)PatchPerFace) - 1;
          for (Integer j = 0; j < ElemOrder; j++) {
            const Real b = 2*((iv + nds[j]) / (Real)PatchPerFace) - 1;
            Real x, y, z;
            FacePoint(x, y, z, face, a, b, Radius);
            const Real s = sin<Real>(theta_twist*z), c = cos<Real>(theta_twist*z);
            X.PushBack(x*c + y*s);
            X.PushBack(-x*s + y*c);
            X.PushBack(z);
          }
        }
  QuadElemList<Real> qel(ElemOrder, X);
  // Fork has 5 schemes (default Adaptive); upstream had only this one. Measure Duffy
  // (upstream self + split-at-foot near) to reproduce the numbers this bench targets.
  qel.SetQuadScheme(QuadElemList<Real>::QuadScheme::Duffy);
  return qel;
}

double Wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
}

// Run an empty parallel region at width p and report where the threads landed.
// Duplicate cores mean the binding is wrong and every timing below is suspect.
bool CheckBinding(const Integer p) {
#if defined(__linux__) && defined(_OPENMP)
  std::vector<int> cpu(p, -1);
  #pragma omp parallel num_threads(p)
  {
    const int t = omp_get_thread_num();
    if (t < p) cpu[t] = sched_getcpu();
  }
  std::set<int> uniq(cpu.begin(), cpu.end());
  const bool ok = ((Integer)uniq.size() == p);
  std::printf("  binding: %ld distinct cores for %d threads -> %s\n",
              (long)uniq.size(), (int)p, ok ? "OK" : "*** OVERSUBSCRIBED ***");
  std::printf("  cpus:");
  for (Integer t = 0; t < p; t++) std::printf(" %d", cpu[t]);
  std::printf("\n");
  return ok;
#else
  (void)p;
  std::printf("  binding: sched_getcpu/OpenMP unavailable, not verified\n");
  return true;
#endif
}

// Element shape at the nodal grid: |dX/du| : |dX/dv| aspect ratio and the skew angle
// between the tangents (the twist shears patches, it does not stretch them).
void ReportElementShape(const QuadElemList<Real>& qel, const Integer order) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Real ar_max = 0, ar_min = 1e30, skew_max = 0, len_min = 1e30, len_max = 0;
  for (Long e = 0; e < qel.Size(); e++) {
    Vector<Real> X, Xn, Xa, dXu, dXv;
    qel.GetGeom(&X, &Xn, &Xa, &dXu, &dXv, nds, nds, e);
    for (Long i = 0; i < X.Dim()/COORD_DIM; i++) {
      Real lu = 0, lv = 0, uv = 0;
      for (Integer k = 0; k < COORD_DIM; k++) {
        lu += dXu[i*COORD_DIM+k]*dXu[i*COORD_DIM+k];
        lv += dXv[i*COORD_DIM+k]*dXv[i*COORD_DIM+k];
        uv += dXu[i*COORD_DIM+k]*dXv[i*COORD_DIM+k];
      }
      lu = sqrt<Real>(lu); lv = sqrt<Real>(lv);
      const Real ar = (lu > lv ? lu/lv : lv/lu);
      const Real ang = acos<Real>(std::max<Real>(-1, std::min<Real>(1, uv/(lu*lv)))) * 180 / const_pi<Real>();
      const Real skew = std::fabs(ang - 90);
      if (ar > ar_max) ar_max = ar;
      if (ar < ar_min) ar_min = ar;
      if (skew > skew_max) skew_max = skew;
      len_min = std::min(len_min, std::min(lu, lv));
      len_max = std::max(len_max, std::max(lu, lv));
    }
  }
  std::printf("element shape (over all %ld elements x %d nodes):\n", qel.Size(), (int)(order*order));
  std::printf("  |dX/du|:|dX/dv| aspect ratio : %.4f .. %.4f\n", (double)ar_min, (double)ar_max);
  std::printf("  tangent skew from 90 deg     : up to %.2f deg\n", (double)skew_max);
  std::printf("  tangent length range         : %.4f .. %.4f\n\n", (double)len_min, (double)len_max);
}

// Laplace double-layer constant-density identity. With sigma = 1 on a closed surface the
// principal-value integral is -1/2 for outward normals (+1/2 inward). Any deviation is
// quadrature + polynomial-geometry error.
template <class Kernel>
void ReportDLIdentity(const QuadElemList<Real>& qel, const Kernel& ker, const Real tol, const Comm& comm) {
  Vector<Real> Xs, Xns;
  qel.GetNodeCoord(&Xs, &Xns, nullptr);
  const Long N = Xs.Dim()/COORD_DIM;
  Real xdotn = 0;
  for (Long i = 0; i < N; i++)
    for (Integer k = 0; k < COORD_DIM; k++) xdotn += Xs[i*COORD_DIM+k]*Xns[i*COORD_DIM+k];
  const bool outward = (xdotn > 0);
  const Real c = (outward ? (Real)-0.5 : (Real)0.5);

  BoundaryIntegralOp<Real, Kernel> BIOp(ker, /*trg_normal_dot_prod=*/false, comm);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(qel);
  Vector<Real> F(N), U;
  for (Long i = 0; i < N; i++) F[i] = 1;
  BIOp.ComputePotential(U, F);
  SCTL_ASSERT(U.Dim() == N);

  Real emax = 0, e2 = 0, umean = 0;
  for (Long i = 0; i < N; i++) {
    const Real d = std::fabs(U[i] - c);
    if (d > emax) emax = d;
    e2 += d*d; umean += U[i];
  }
  std::printf("Laplace double-layer identity (sigma = 1, tol = %.0e):\n", (double)tol);
  std::printf("  normal orientation : %s (sum x.n = %.4g)\n", outward ? "outward" : "inward", (double)xdotn);
  std::printf("  exact D[1]         : %+.1f\n", (double)c);
  std::printf("  mean computed      : %+.12f\n", (double)(umean/N));
  std::printf("  max |U - exact|    : %.3e\n", (double)emax);
  std::printf("  rms |U - exact|    : %.3e\n\n", (double)sqrt<Real>(e2/N));
}

std::vector<Integer> ParseThreadList(const char* s, const Integer pmax) {
  std::vector<Integer> lst;
  if (s) {
    for (const char* q = s; *q; ) {
      lst.push_back((Integer)strtol(q, (char**)&q, 10));
      while (*q == ',' || *q == ' ') q++;
    }
  } else {
    for (Integer p = 1; p <= pmax; p *= 2) lst.push_back(p);
    if (lst.empty() || lst.back() != pmax) lst.push_back(pmax);
  }
  return lst;
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    SCTL_ASSERT_MSG(comm.Size() == 1, "bench-quad-scaling measures OpenMP scaling; run with one MPI rank.");

    const Integer order      = (argc > 1 ? (Integer)atol(argv[1]) : 8);
    const Long PatchPerFace  = (argc > 2 ? atol(argv[2]) : 16);
    const Real tol           = (argc > 3 ? atof(argv[3]) : 1e-6);
    const Integer pmax       = SCTL_GET_MAX_THREADS();
    const std::vector<Integer> threads = ParseThreadList(argc > 4 ? argv[4] : nullptr, pmax);
    const Real twist         = (argc > 5 ? atof(argv[5]) : (double)(const_pi<Real>()/6));

    using Kernel = Laplace3D_DxU;
    const Kernel ker;

    QuadElemList<Real> qel = BuildTwistedSphere(order, PatchPerFace, 1.0, twist);
    const Long nelem = qel.Size();
    const Long nnode = nelem * order * order;

    std::printf("==== QuadElemList OpenMP scaling ====\n");
    std::printf("kernel=%s  order=%d  patch/face=%ld  nelem=%ld  nodes=%ld  tol=%.0e  omp_max=%d\n\n",
                Kernel::Name().c_str(), (int)order, PatchPerFace, nelem, nnode, tol, (int)pmax);
    std::printf("twist = %.6f rad\n\n", (double)twist);
    // Diagnostics cost a full BIOp setup; skip them when sweeping one width per process.
    if (!(std::getenv("SKIP_DIAG") && atoi(std::getenv("SKIP_DIAG")))) {
      SCTL_ASSERT_MSG(order == 8 && tol == 1e-6, "self-rule size probe is instantiated for order=8, digits=6 only");
      QuadElemTestAccess<Real>::template SelfRuleSizes<8>(6, Kernel::SrcDim());
      ReportElementShape(qel, order);
      ReportDLIdentity(qel, ker, tol, comm);
    }
    // Repeats guard against one-off jitter. Report the MIN: the fastest run is the one
    // least perturbed by stray OS/daemon activity.
    const int reps = (std::getenv("REPS") ? std::max(1, atoi(std::getenv("REPS"))) : 1);

    std::printf("SCALE row format: '#SCALE <threads> <t_self[s]> <t_setup_total[s]>'\n");
    std::printf("per-stage near/far split: read the profile tree printed under each block\n\n");

    Vector<Matrix<Real>> M_lst(nelem);

    for (const Integer p : threads) {
      if (p < 1 || p > pmax) {
        std::printf("-- skipping p=%d (omp_get_max_threads()=%d)\n", (int)p, (int)pmax);
        continue;
      }
#ifdef _OPENMP
      omp_set_num_threads(p);
#endif
      std::printf("======================== threads = %d ========================\n", (int)p);
      CheckBinding(p);

      // Warm-up at this width: creates the thread team, faults in each new thread's
      // thread_local scratch in IntegrateBlock, and pays any first-touch static rule
      // init. Timed runs below are all warm.
      QuadElemList<Real>::SelfInterac<Kernel>(M_lst, ker, tol, false, &qel);

      // ---- SELF: the omp-parallel-for over elements, called exactly as SetupSelf does.
      double t_self = 1e30;
      for (int r = 0; r < reps; r++) {
        const double t0 = Wtime();
        QuadElemList<Real>::SelfInterac<Kernel>(M_lst, ker, tol, false, &qel);
        const double dt = Wtime() - t0;
        std::printf("    self rep%d = %.4f s\n", r, dt);
        t_self = std::min(t_self, dt);
      }

      // ---- NEAR (+far/basic): full BIOp setup. SetupSelf runs again inside; the
      // profile tree separates SetupSingular / BuildNearLst / SetupNear / SetupFarField.
      double t_setup = 0;
      { // Warm-up: without this the timed Setup below is a COLD first call, and the near
        // numbers absorb malloc-arena growth and first-touch faults on K_near rather than
        // measuring the parallel loops. Scoped so its K_near is freed before the timed run.
        BoundaryIntegralOp<Real, Kernel> BIOp_warm(ker, /*trg_normal_dot_prod=*/false, comm);
        BIOp_warm.SetAccuracy(tol);
        BIOp_warm.AddElemList(qel);
        BIOp_warm.Setup();
      }
      t_setup = 1e30;
      for (int r = 0; r < reps; r++) {
        BoundaryIntegralOp<Real, Kernel> BIOp(ker, /*trg_normal_dot_prod=*/false, comm);
        BIOp.SetAccuracy(tol);
        BIOp.AddElemList(qel);

        Profile::Enable(true);
        Profile::reset();
        const double t0 = Wtime();
        BIOp.Setup();
        const double dt = Wtime() - t0;
        std::printf("    setup rep%d = %.4f s\n", r, dt);
        t_setup = std::min(t_setup, dt);
      }

      std::printf("#SCALE %4d %12.5e %12.5e\n", (int)p, t_self, t_setup);
      std::printf("---- profile tree (threads=%d) ----\n", (int)p);
      Profile::print(&comm, {"t_max"});
      std::printf("\n");
      std::fflush(stdout);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
