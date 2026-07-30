/**
 * Crossover study: Adaptive vs hedgehog / Line-QBX near-singular quadrature at a panel-INTERIOR
 * off-surface target, sweeping target distance d and the Adaptive accuracy knob tol (max_depth=30
 * fixed). Finds where (a) Adaptive can no longer resolve the near target as d -> 0 while hedgehog
 * holds its accuracy, and (b) Adaptive's cost (grows ~log2(1/d) with refinement depth) crosses
 * hedgehog's depth-independent cost -- at MATCHED accuracy.
 *
 * Geometry: a PI-TWISTED cubed-sphere (theta_twist = pi), operating on ONE sheared interior element
 * (the shear at the foot is printed as a diagnostic). This stresses the schemes on anisotropic
 * panels -- the twist inflates the Adaptive quadtree leaf count and can under-resolve the sheared
 * v-direction, so it is the harder case vs a flat/curved single panel.
 *
 * Accuracy is measured against an RP-Nbeta=512 gold; an RP-400 vs RP-512 self-consistency number is
 * printed per row so a row where the gold itself is unreliable (deep d / shear / fp cancellation) is
 * visible. Timing is wall time per NearInterac call. hedgehog uses SetLineQBXParams defaults.
 *
 * Self-contained; does NOT touch test-quad-elem.cpp / unit-test-quad-element.cpp.
 *
 * Build & run:
 *     . ./sctl_source
 *     make hedgehog                       (-> bin/test-hedgehog-near)
 *     OMP_NUM_THREADS=1 ./bin/test-hedgehog-near     (1 thread for clean per-call timing)
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <omp.h>
#include <cstdio>
#include <utility>
#include <vector>

using namespace sctl;

namespace {

using QS = QuadElemList<double>::QuadScheme;

enum SchemeKind { GOLD, HEDGEHOG, ADAPTIVE, RECTPOLAR };

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
  const Real r = sqrt<Real>(x*x + y*y + z*z);
  x *= R/r; y *= R/r; z *= R/r;
}

// Node coords of a cubed-sphere of radius R: PatchPerFace^2 patches/face, twisted about z by theta.
template <class Real> Vector<Real> twisted_sphere_coord(Integer order, Long PatchPerFace, Real R, Real theta) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PatchPerFace; iu++)
      for (Long iv = 0; iv < PatchPerFace; iv++)
        for (Integer i = 0; i < order; i++) {
          const Real a = 2 * ((iu + nds[i]) / (Real)PatchPerFace) - 1;
          for (Integer j = 0; j < order; j++) {
            const Real b = 2 * ((iv + nds[j]) / (Real)PatchPerFace) - 1;
            Real x, y, z;
            FacePoint(x, y, z, face, a, b, R);
            const Real s = sin<Real>(theta*z), c = cos<Real>(theta*z);
            X.PushBack(x*c + y*s);
            X.PushBack(-x*s + y*c);
            X.PushBack(z);
          }
        }
  return X;
}

// Build the near operator for source element elem_idx vs target Xt with the given scheme/knobs,
// time nrep calls, apply to the density. Returns (potential, seconds-per-call).
template <class Real, class Kernel>
std::pair<Vector<Real>, double>
eval_time(const Vector<Real>& coord, const Integer order, const Long elem_idx, const SchemeKind kind,
          const Integer Nbeta, const Integer max_depth, const Kernel& ker,
          const Vector<Real>& Xt, const Vector<Real>& nt, const Vector<Real>& sigma,
          const Real ctol, const Integer nrep) {
  const Integer KDIM0 = Kernel::SrcDim(), KDIM1 = Kernel::TrgDim();
  const Long nnode = (Long)order*order;
  QuadElemList<Real> q(order, coord);
  switch (kind) {
    case GOLD:      q.SetQuadScheme(QS::RectPolar, 10, 512, 30); break;
    case HEDGEHOG:  q.SetQuadScheme(QS::LineQBX);  q.SetLineQBXParams(); break;
    case ADAPTIVE:  q.SetQuadScheme(QS::Adaptive,  10, 0, max_depth); break;
    case RECTPOLAR: q.SetQuadScheme(QS::RectPolar, 10, Nbeta, 30); break;
  }
  Matrix<Real> M;
  QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, nt, ker, ctol, elem_idx, &q); // warm-up
  const double t0 = omp_get_wtime();
  for (Integer r = 0; r < nrep; r++)
    QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, nt, ker, ctol, elem_idx, &q);
  const double dt = (omp_get_wtime() - t0) / nrep;

  Vector<Real> u(KDIM1); u.SetZero();
  for (Long rr = 0; rr < nnode*KDIM0; rr++)
    for (Integer k1 = 0; k1 < KDIM1; k1++) u[k1] += sigma[rr]*M[rr][k1];
  return {u, dt};
}

// One (kernel, distance) block on element elem_idx: RP-512 gold (+ RP-400 self-consistency),
// hedgehog (defaults), and Adaptive (max_depth=30) at each tol. Prints rel-err vs the gold + us/call.
template <class Real, class Kernel>
void block(const Kernel& ker, const char* kname, const Vector<Real>& coord, const Integer order,
           const Long elem_idx, const Real u0, const Real v0, const Real d,
           const std::vector<double>& adatols, const Integer nrep) {
  static constexpr Integer COORD_DIM = 3;
  const Integer KDIM0 = Kernel::SrcDim(), KDIM1 = Kernel::TrgDim();
  const Long nnode = (Long)order*order;

  // Panel-interior off-surface target: point at (u0,v0) on elem_idx, pushed INWARD by d.
  QuadElemList<Real> q0(order, coord);
  Vector<Real> up{u0}, vp{v0}, Xs, Ns;
  q0.GetGeom(&Xs, &Ns, nullptr, nullptr, nullptr, up, vp, elem_idx);
  Vector<Real> Xt(COORD_DIM);
  for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = Xs[k] - d*Ns[k];

  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> sigma(nnode*KDIM0);
  for (Integer i = 0; i < order; i++)
    for (Integer j = 0; j < order; j++)
      for (Integer k0 = 0; k0 < KDIM0; k0++)
        sigma[(i*order+j)*KDIM0+k0] = cos<Real>(nds[i] + 2*nds[j] + (Real)0.5*k0);

  Vector<Real> nt;
  auto reldiff = [&](const Vector<Real>& a, const Vector<Real>& b) {
    Real e = 0, n = 0;
    for (Integer k = 0; k < KDIM1; k++) { e += (a[k]-b[k])*(a[k]-b[k]); n += b[k]*b[k]; }
    return (double)(sqrt<Real>(e) / sqrt<Real>(n));
  };

  const auto gold = eval_time<Real>(coord, order, elem_idx, GOLD,      512, 30, ker, Xt, nt, sigma, (Real)1e-13, nrep);
  const auto rpck = eval_time<Real>(coord, order, elem_idx, RECTPOLAR, 400, 30, ker, Xt, nt, sigma, (Real)1e-13, nrep);
  const auto hh   = eval_time<Real>(coord, order, elem_idx, HEDGEHOG,  0,   0,  ker, Xt, nt, sigma, (Real)1e-13, nrep);

  std::printf("== %-11s d=%.0e  (gold RP512 %.0f us; gold self-consistency |RP400-RP512|=%.1e) ==\n",
              kname, (double)d, gold.second*1e6, reldiff(rpck.first, gold.first));
  std::printf("   %-26s %-11s %9s\n", "scheme", "rel_err", "us/call");
  std::printf("   %-26s %.3e %9.0f\n", "hedgehog (defaults)", reldiff(hh.first, gold.first), hh.second*1e6);
  for (const double tol : adatols) {
    char lbl[48];
    const auto ad = eval_time<Real>(coord, order, elem_idx, ADAPTIVE, 0, 30, ker, Xt, nt, sigma, (Real)tol, nrep);
    std::snprintf(lbl, sizeof(lbl), "adaptive tol=%.0e md=30", tol);
    std::printf("   %-26s %.3e %9.0f\n", lbl, reldiff(ad.first, gold.first), ad.second*1e6);
  }
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Integer order = 12, nrep = 3;
    const Long PatchPerFace = 3;
    const Long elem_idx = 4; // center patch of face 0 (spans a z-range, so the pi-twist shears it)
    const double u0 = 0.4, v0 = 0.6; // panel-interior, away from all edges/seams
    const std::vector<double> adatols = {1e-6, 1e-9, 1e-12};
    const std::vector<double> dists   = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};

    const Vector<double> coord = twisted_sphere_coord<double>(order, PatchPerFace, 1.0, const_pi<double>());

    // Shear diagnostic at the foot (u0,v0) of elem_idx: tangent-length ratio + angle (90deg = orthogonal).
    {
      QuadElemList<double> q(order, coord);
      Vector<double> up{u0}, vp{v0}, Xs, Ns, dXu, dXv;
      q.GetGeom(&Xs, &Ns, nullptr, &dXu, &dXv, up, vp, elem_idx);
      double lu = 0, lv = 0, dot = 0;
      for (Integer k = 0; k < 3; k++) { lu += dXu[k]*dXu[k]; lv += dXv[k]*dXv[k]; dot += dXu[k]*dXv[k]; }
      lu = std::sqrt(lu); lv = std::sqrt(lv);
      const double ang = std::acos(dot/(lu*lv)) * 180.0/M_PI;
      std::printf("PI-TWISTED cubed-sphere (order %d, %ld patch/face), element %ld @ (%.2f,%.2f):\n"
                  "  |dX/du|=%.3f  |dX/dv|=%.3f  ratio=%.2f  tangent angle=%.1f deg (90=orthogonal)\n"
                  "hedgehog=SetLineQBXParams defaults; adaptive=max_depth 30, tol swept. Accuracy vs RP512 gold.\n",
                  (int)order, PatchPerFace, elem_idx, u0, v0, lu, lv, lu/lv, ang);
    }

    const Laplace3D_FxU lap_sl; const Laplace3D_DxU lap_dl;
    const Stokes3D_FxU  stk_sl; const Stokes3D_DxU  stk_dl;
    for (const double d : dists) {
      std::printf("\n");
      block<double>(lap_sl, "Laplace-FxU", coord, order, elem_idx, u0, v0, d, adatols, nrep);
      block<double>(lap_dl, "Laplace-DxU", coord, order, elem_idx, u0, v0, d, adatols, nrep);
      block<double>(stk_sl, "Stokes-FxU",  coord, order, elem_idx, u0, v0, d, adatols, nrep);
      block<double>(stk_dl, "Stokes-DxU",  coord, order, elem_idx, u0, v0, d, adatols, nrep);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
