/**
 * Microbenchmark for the QuadElemList on-surface singular (SelfInterac) and
 * off-surface near-singular (NearInterac) setup, to localize where setup time
 * goes and why the reported flops/s looks low.
 *
 * Two views are produced per configuration:
 *   1. Coarse: wall time, per-element / per-target cost, and the profiler's
 *      counted-FLOP throughput (which counts ONLY kernel evaluations).
 *   2. Fine: a per-phase breakdown (interp build / geometry tensor GEMMs /
 *      assembly / kernel eval / kernel weight / projection / quadtree build)
 *      plus the TRUE GEMM flops actually issued, gated on -DBENCH_QUAD.
 *
 * Build (uninstrumented, coarse only):
 *     make bin/bench-quad-interac
 * Build (instrumented, full phase breakdown):
 *     make CXXFLAGS+=" -DBENCH_QUAD" bin/bench-quad-interac
 * Run single-threaded first for clean attribution, then scale threads:
 *     OMP_NUM_THREADS=1 ./bin/bench-quad-interac
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <sctl/experimental/bench_quad.hpp>
#include <iomanip>
#include <string>

using namespace sctl;

namespace {

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

// Cubed-sphere of radius R: PatchPerFace^2 patches/face, optionally twisted about z.
template <class Real>
QuadElemList<Real> BuildTwistedSphere(Long ElemOrder, Long PatchPerFace, Real Radius, Real theta_twist = 0) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(ElemOrder);
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PatchPerFace; iu++)
      for (Long iv = 0; iv < PatchPerFace; iv++)
        for (Long i = 0; i < ElemOrder; i++) {
          const Real a = 2 * ((iu + nds[i]) / (Real)PatchPerFace) - 1;
          for (Long j = 0; j < ElemOrder; j++) {
            const Real b = 2 * ((iv + nds[j]) / (Real)PatchPerFace) - 1;
            Real x, y, z;
            FacePoint(x, y, z, face, a, b, Radius);
            const Real s = sin<Real>(theta_twist * z), c = cos<Real>(theta_twist * z);
            X.PushBack(x * c + y * s);
            X.PushBack(-x * s + y * c);
            X.PushBack(z);
          }
        }
  return QuadElemList<Real>(ElemOrder, X);
}

// Single curved element z = u*v on [0,1]^2 (matches unit-test get_testsurf).
template <class Real> Vector<Real> get_testsurf(const Integer order) {
  Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
  for (Long i = 0; i < coord0.Dim() / 3; i++)
    coord0[i * 3 + 2] = coord0[i * 3 + 0] * coord0[i * 3 + 1];
  return coord0;
}

// ---- Self-interaction --------------------------------------------------------
// A single curved element (order^2 on-surface targets) isolates the per-call cost
// cheaply; set nface>0 to instead use an nface-per-side twisted sphere when you
// want OpenMP scaling across many elements.
template <class Kernel>
void bench_self(const Kernel& ker, Integer order, double tol, Long PatchPerFace = 0, Long ncall = 6) {
  using Real = double;
  QuadElemList<Real> qel = (PatchPerFace > 0)
      ? BuildTwistedSphere<Real>(order, PatchPerFace, 1.0, const_pi<Real>() / 6)
      : QuadElemList<Real>(order, get_testsurf<Real>(order));
  const Long nelem = qel.Size();
  const Long ntrg = nelem * (Long)order * order; // one on-surface target per node

  Vector<Matrix<Real>> M_lst(nelem);

  // No throwaway warm-up: call #0 is the COLD (cache-warming) SelfInterac -- it pays
  // the one-time first-touch build of the static singular-rule caches. Calls #1.. are
  // WARM (cache hits). Each call gets its OWN Profile label so it lands on its own row,
  // where t_avg = that call's wall time and f_avg = that call's counted (kernel-eval)
  // FLOPs -- so identical f_avg across rows confirms identical work while t_avg drops.
  char base[220];
  std::snprintf(base, sizeof(base), "self %-11s ord=%2d tol=%.0e",
                Kernel::Name().c_str(), (int)order, tol);
  std::printf("\n%s   (%ld targets, %ld elem)\n", base, ntrg, nelem);

  double t_cold = 0, t_warm_sum = 0; Long nwarm = 0;
  for (Long k = 0; k < ncall; k++) {
    char lbl[256];
    std::snprintf(lbl, sizeof(lbl), "%s #%ld%s", base, k, (k == 0 ? " [COLD]" : ""));
    const double t0 = bench::Wtime();
    Profile::Tic(lbl);
    QuadElemList<Real>::template SelfInterac<Kernel>(M_lst, ker, tol, false, &qel);
    Profile::Toc();
    const double dt = bench::Wtime() - t0;
    if (k == 0) { t_cold = dt; bench::Reset(); } else { t_warm_sum += dt; nwarm++; }
    std::printf("    call #%ld%-8s wall=%.4g s  | %.3g us/target\n",
                k, (k == 0 ? " COLD" : ""), dt, 1e6 * dt / ntrg);
  }
  const double t_warm_avg = (nwarm ? t_warm_sum / nwarm : 0);
  std::printf("    --> cold=%.4g s, warm_avg=%.4g s  =>  cold/warm = %.1fx slower\n",
              t_cold, t_warm_avg, (t_warm_avg > 0 ? t_cold / t_warm_avg : 0));
  bench::Report(base, t_warm_sum); // per-phase breakdown of the warm calls (Reset after cold above)
}

// ---- Near-interaction: single element, one off-surface target --------------
template <class Kernel>
void bench_near(const Kernel& ker, Integer order, double tol, Long nrep, Long ncall = 6,
                Long PatchPerFace = 0, double theta_twist = 0, double d = 0.01) {
  using Real = double;
  const Integer COORD_DIM = 3;
  const Long elem_idx = 0;
  // Default: single flat curved element. PatchPerFace>0 switches to a (possibly twisted)
  // cubed-sphere so we can compare adaptive-near leaf counts on sheared vs flat geometry.
  QuadElemList<Real> qel = (PatchPerFace > 0)
      ? BuildTwistedSphere<Real>(order, PatchPerFace, 1.0, (Real)theta_twist)
      : QuadElemList<Real>(order, get_testsurf<Real>(order));

  // Near-singular target: offset `d` (function arg) along the normal at an interior point.
  const Real u0 = 0.4, v0 = 0.6;
  Vector<Real> up{u0}, vp{v0}, Xsurf, Nsurf;
  qel.GetGeom(&Xsurf, &Nsurf, nullptr, nullptr, nullptr, up, vp, elem_idx);
  Vector<Real> Xt(COORD_DIM);
  for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = Xsurf[k] + d * Nsurf[k];

  Matrix<Real> M;
  Vector<Real> normal_trg; // empty: no target-normal contraction

  // As in bench_self: no throwaway warm-up. Each block of `nrep` calls gets its own
  // Profile label; block #0 is COLD (first touch of any cache this scheme needs that
  // isn't already resident). NOTE: self runs before near in main(), so DiffMat/ParamNodes
  // and the digit-/Nbeta-keyed rules are typically ALREADY warm -- near's cold penalty is
  // therefore small (only whatever this exact (order,tol) has not yet touched).
  char base[220];
  std::snprintf(base, sizeof(base), "near %-11s ord=%2d tol=%.0e d=%.0e%s",
                Kernel::Name().c_str(), (int)order, tol, d,
                (PatchPerFace > 0 ? (theta_twist != 0 ? " [twist]" : " [sphere]") : ""));
  std::printf("\n%s   (%ld reps/call)\n", base, nrep);

  double t_cold = 0, t_warm_sum = 0; Long nwarm = 0;
  for (Long k = 0; k < ncall; k++) {
    char lbl[256];
    std::snprintf(lbl, sizeof(lbl), "%s #%ld%s", base, k, (k == 0 ? " [COLD]" : ""));
    const double t0 = bench::Wtime();
    Profile::Tic(lbl);
    for (Long r = 0; r < nrep; r++)
      QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, normal_trg, ker, tol, elem_idx, &qel);
    Profile::Toc();
    const double dt = bench::Wtime() - t0;
    if (k == 0) { t_cold = dt; bench::Reset(); } else { t_warm_sum += dt; nwarm++; }
    std::printf("    call #%ld%-8s wall=%.4g s  | %.3g us/target\n",
                k, (k == 0 ? " COLD" : ""), dt, 1e6 * dt / nrep);
  }
  const double t_warm_avg = (nwarm ? t_warm_sum / nwarm : 0);
  std::printf("    --> cold=%.4g s, warm_avg=%.4g s  =>  cold/warm = %.1fx slower\n",
              t_cold, t_warm_avg, (t_warm_avg > 0 ? t_cold / t_warm_avg : 0));
  bench::Report(base, t_warm_sum); // per-phase breakdown of the warm calls (Reset after cold above)
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    SCTL_ASSERT_MSG(comm.Size() == 1, "bench-quad-interac is sequential (run with one MPI rank).");
    Profile::Enable(true);

    const Laplace3D_DxU ker_lap;  // scalar kernel: interpolation-dominated case
    const Stokes3D_DxU  ker_stk;  // matrix kernel: heavier KernelEval

    // Self-rule node counts grow with order and tolerance, so the self sweep is on a single
    // curved element and bounded; near is cheap (one off-surface target) so it runs the full
    // order x tolerance matrix.
    std::printf("==================== SELF-INTERACTION (single curved element) ====================\n");
    for (const Integer order : {4, 8, 16}) {
      bench_self(ker_lap, order, 1e-6);   // scalar, interpolation-dominated
      bench_self(ker_stk, order, 1e-6);   // matrix kernel: heavier KernelEval
    }
    bench_self(ker_lap, 8, 1e-10);        // tol sweep at fixed order

    std::printf("\n==================== NEAR-INTERACTION ====================\n");
    for (const Integer order : {4, 8, 16}) {
      bench_near(ker_lap, order, 1e-6,  /*nrep=*/100);
      bench_near(ker_lap, order, 1e-10, 100);
      bench_near(ker_stk, order, 1e-10, 100);
    }
    // One twisted-patch comparison: the same near path on a pi/2-sheared cubed-sphere patch.
    // Contrast leaves/target + phase split vs the flat case above to see whether shear inflates
    // the near cost uniformly (leaf-count growth) or hits a specific phase.
    bench_near(ker_lap, 8, 1e-10, /*nrep=*/100, /*ncall=*/6,
               /*PatchPerFace=*/1, /*theta_twist=*/const_pi<double>() / 2);

    // Deep-near regime (d=1e-4): each call refines to ~11 levels, so nrep is lowered.
    std::printf("\n---- deep-near (d=1e-4) ----\n");
    for (const Integer order : {8, 16}) {
      bench_near(ker_lap, order, 1e-6, /*nrep=*/50, 6, 0, 0, 1e-4);
      bench_near(ker_stk, order, 1e-6, /*nrep=*/50, 6, 0, 0, 1e-4);
      bench_near(ker_lap, order, 1e-7, /*nrep=*/50, 6, 0, 0, 1e-4);
      bench_near(ker_stk, order, 1e-7, /*nrep=*/50, 6, 0, 0, 1e-4);
    }

    std::printf("\n==== Profiler view: per-call rows (ALL counted FLOPs: kernel + Matrix GEMMs + elementwise) ====\n");
    std::printf("     each '#k' row is one setup call; #0 is COLD (cache-warming), #1.. WARM.\n");
    std::printf("     t_avg = that call's wall time; f_avg = that call's counted FLOPs; f/s_avg = GFLOP/s.\n");
    std::printf("     Read WITH the per-config '[bench]' phase tables above:\n");
    std::printf("       - Profile f/s_avg here = comprehensive throughput (kernel + all tensor GEMMs + elementwise).\n");
    std::printf("       - '[bench] gemm_f/s'    = tensor-GEMM-only GFLOP/s; the gap vs Profile f/s is elementwise + kernel.\n");
    std::printf("       - Low Profile f/s together with high 'leaves/target' and large QuadtreeBuild/ClosestNode/\n");
    std::printf("         Assembly/KernelWeight time share ==> cost is small-GEMM + uncounted quadtree/search work,\n");
    std::printf("         NOT a FLOP mis-count. Those uncounted-scalar phases are the ops that drag f/s down most.\n");
    Profile::print(&comm, {"t_avg", "t_max", "f_avg", "f/s_avg"});
  }
  Comm::MPI_Finalize();
  return 0;
}
