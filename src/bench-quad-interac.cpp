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

const char* SchemeName(typename QuadElemList<double>::QuadScheme s) {
  switch (s) {
    case QuadElemList<double>::QuadScheme::RectPolar: return "RectPolar";
    case QuadElemList<double>::QuadScheme::Hybrid:    return "Hybrid";
    default:                                          return "Adaptive";
  }
}

// ---- Self-interaction --------------------------------------------------------
// A single curved element (order^2 on-surface targets) isolates the per-call cost
// cheaply; set nface>0 to instead use an nface-per-side twisted sphere when you
// want OpenMP scaling across many elements.
template <class Kernel>
void bench_self(const Kernel& ker, typename QuadElemList<double>::QuadScheme scheme,
                Integer order, double tol, Integer q, Integer cov_order, Long PatchPerFace = 0) {
  using Real = double;
  QuadElemList<Real> qel = (PatchPerFace > 0)
      ? BuildTwistedSphere<Real>(order, PatchPerFace, 1.0, const_pi<Real>() / 6)
      : QuadElemList<Real>(order, get_testsurf<Real>(order));
  qel.SetQuadScheme(scheme, q, cov_order);
  const Long nelem = qel.Size();
  const Long ntrg = nelem * (Long)order * order; // one on-surface target per node

  Vector<Matrix<Real>> M_lst(nelem);
  // Warm-up: preloads the static singular-rule caches so they don't pollute timing.
  QuadElemList<Real>::template SelfInterac<Kernel>(M_lst, ker, tol, false, &qel);

  bench::Reset();
  char lbl[256];
  std::snprintf(lbl, sizeof(lbl), "self  %-13s %-12s order=%2d tol=%.0e nelem=%ld",
                Kernel::Name().c_str(), SchemeName(scheme), (int)order, tol, nelem);
  const double t0 = bench::Wtime();
  Profile::Tic(lbl);
  QuadElemList<Real>::template SelfInterac<Kernel>(M_lst, ker, tol, false, &qel);
  Profile::Toc();
  const double dt = bench::Wtime() - t0;

  std::printf("\n%s\n", lbl);
  std::printf("  wall=%.4g s  | %ld targets | %.3g us/target | %.3g ms/elem\n",
              dt, ntrg, 1e6 * dt / ntrg, 1e3 * dt / nelem);
  bench::Report("self phase breakdown", dt);
}

// ---- Near-interaction: single element, one off-surface target --------------
template <class Kernel>
void bench_near(const Kernel& ker, typename QuadElemList<double>::QuadScheme scheme,
                Integer order, double tol, Integer q, Integer cov_order, Long nrep) {
  using Real = double;
  const Integer COORD_DIM = 3;
  const Long elem_idx = 0;
  QuadElemList<Real> qel(order, get_testsurf<Real>(order));
  qel.SetQuadScheme(scheme, q, cov_order);

  // Near-singular target: small offset along the normal at an interior point.
  const Real u0 = 0.4, v0 = 0.6, d = 0.01;
  Vector<Real> up{u0}, vp{v0}, Xsurf, Nsurf;
  qel.GetGeom(&Xsurf, &Nsurf, nullptr, nullptr, nullptr, up, vp, elem_idx);
  Vector<Real> Xt(COORD_DIM);
  for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = Xsurf[k] + d * Nsurf[k];

  Matrix<Real> M;
  Vector<Real> normal_trg; // empty: no target-normal contraction
  QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, normal_trg, ker, tol, elem_idx, &qel); // warm-up

  bench::Reset();
  char lbl[256];
  std::snprintf(lbl, sizeof(lbl), "near  %-13s %-12s order=%2d tol=%.0e nrep=%ld",
                Kernel::Name().c_str(), SchemeName(scheme), (int)order, tol, nrep);
  const double t0 = bench::Wtime();
  Profile::Tic(lbl);
  for (Long r = 0; r < nrep; r++)
    QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, normal_trg, ker, tol, elem_idx, &qel);
  Profile::Toc();
  const double dt = bench::Wtime() - t0;

  std::printf("\n%s\n", lbl);
  std::printf("  wall=%.4g s  | %ld targets | %.3g us/target\n", dt, nrep, 1e6 * dt / nrep);
  bench::Report("near phase breakdown", dt);
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    SCTL_ASSERT_MSG(comm.Size() == 1, "bench-quad-interac is sequential (run with one MPI rank).");
    Profile::Enable(true);

    using QS = QuadElemList<double>::QuadScheme;
    const Laplace3D_DxU ker_lap;  // scalar kernel: interpolation-dominated case
    const Stokes3D_DxU  ker_stk;  // matrix kernel: heavier KernelEval

    // Adaptive self-rule node counts explode with order and tolerance, so the
    // self sweep is on a single curved element and bounded; near is cheap (one
    // off-surface target) so it runs the full order x scheme matrix.
    std::printf("==================== SELF-INTERACTION (single curved element) ====================\n");
    for (const Integer order : {4, 8, 16}) {
      bench_self(ker_lap, QS::Adaptive,  order, 1e-6, 10, 0);    // scalar, interpolation-dominated
      bench_self(ker_stk, QS::Adaptive,  order, 1e-6, 10, 0);    // matrix kernel: heavier KernelEval
      bench_self(ker_lap, QS::RectPolar, order, 1e-7, 10, 256);  // fixed Nbeta tensor rule
      bench_self(ker_lap, QS::Hybrid,    order, 1e-7, 10, 256);  // self uses RectPolar (near unused here)
    }
    bench_self(ker_lap, QS::Adaptive, 8, 1e-10, 10, 0);          // tol sweep at fixed order

    std::printf("\n==================== NEAR-INTERACTION ====================\n");
    for (const Integer order : {4, 8, 16}) {
      bench_near(ker_lap, QS::Adaptive,  order, 1e-6,  10, 0,   /*nrep=*/100);
      bench_near(ker_lap, QS::Adaptive,  order, 1e-10, 10, 0,   100);
      bench_near(ker_stk, QS::Adaptive,  order, 1e-10, 10, 0,   100);
      bench_near(ker_lap, QS::RectPolar, order, 1e-7,  10, 256, 100);
      bench_near(ker_stk, QS::RectPolar, order, 1e-7,  10, 256, 100);
      bench_near(ker_lap, QS::Hybrid,    order, 1e-10, 10, 0,   100);  // near uses adaptive (tol-driven)
    }

    std::printf("\n==== Profiler view (counts kernel-eval FLOPs only -> understates true work) ====\n");
    Profile::print(&comm, {"t_max", "f_max", "f/s_avg"});
  }
  Comm::MPI_Finalize();
  return 0;
}
