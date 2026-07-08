/**
 * Diagnostic driver for the Gauss-Newton closest-point search in
 * QuadElemList::GetClosestPoint. For each patch and near target it prints the
 * number of Newton iterations executed and whether the grid-search fallback ran,
 * so we can confirm the solver converges in only a few iterations on realistic
 * geometry.
 *
 * Patches are drawn from two cubed-spheres: the regular sphere (theta_twist=0)
 * and the pi-twisted sphere (theta_twist=pi, the hardest sheared case we test).
 *
 * Build & run:
 *     make bin/investigate-newton-closest
 *     OMP_NUM_THREADS=1 ./bin/investigate-newton-closest
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <cstdio>

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

// Drive GetClosestPoint over a handful of patches x near targets, printing the
// Newton iteration count for each call.
template <class Real> void investigate(const char* label, const QuadElemList<Real>& qel, Long n_patch) {
  const Integer COORD_DIM = 3;
  // (u,v) seeds: interior, edge-adjacent and corner-adjacent parameter locations.
  const Real uv[] = {(Real)0.5, (Real)0.37, (Real)0.05, (Real)0.95, (Real)0.62};
  const Integer nuv = sizeof(uv) / sizeof(uv[0]);
  const Real dists[] = {(Real)0.05, (Real)0.005}; // near-surface offsets along the normal

  n_patch = std::min<Long>(n_patch, qel.Size());
  std::printf("\n==== %s: %ld patches (of %ld) ====\n", label, n_patch, qel.Size());
  std::printf("  %-5s %-13s %-8s %5s %-9s %11s %11s\n",
              "elem", "(u,v) trg", "d", "iters", "status", "|seed-uv*|", "|uv*-uv|");

  Integer max_iters = 0;
  Long n_fallback = 0, n_call = 0;
  // Correlate the fallback with how good the nearest-node seed was: mean seed->solution
  // distance and mean iteration count, split by converged vs fallback.
  Real seederr_fb = 0, seederr_ok = 0;
  Long iters_fb = 0, iters_ok = 0;
  for (Long e = 0; e < n_patch; e++) {
    for (Integer iu = 0; iu < nuv; iu++) {
      for (Integer iv = 0; iv < nuv; iv++) {
        const Real u0 = uv[iu], v0 = uv[iv];
        Vector<Real> up{u0}, vp{v0}, Xsurf, Nsurf;
        qel.GetGeom(&Xsurf, &Nsurf, nullptr, nullptr, nullptr, up, vp, e);
        for (Integer id = 0; id < (Integer)(sizeof(dists) / sizeof(dists[0])); id++) {
          const Real d = dists[id];
          Vector<Real> Xtrg(COORD_DIM);
          for (Integer k = 0; k < COORD_DIM; k++) Xtrg[k] = Xsurf[k] + d * Nsurf[k];

          // Seed the same way GetClosestPoint does (nearest node), so we can measure how
          // close the initial guess already was to the converged foot.
          Real useed, vseed;
          qel.GetClosestNode(useed, vseed, e, Xtrg);

          Real ustar, vstar;
          Integer n_iter = 0;
          bool used_fallback = false;
          const Real dist = qel.GetClosestPoint(ustar, vstar, e, Xtrg, &n_iter, &used_fallback);
          const Real duv = sqrt<Real>((ustar - u0) * (ustar - u0) + (vstar - v0) * (vstar - v0));
          const Real seed_err = sqrt<Real>((useed - ustar) * (useed - ustar) + (vseed - vstar) * (vseed - vstar));

          std::printf("  %-5ld (%.2f,%.2f)   %-8.4f %5d %-9s %11.3e %11.3e\n",
                      e, (double)u0, (double)v0, (double)d, (int)n_iter,
                      used_fallback ? "FALLBACK" : "converged", (double)seed_err, (double)duv);

          max_iters = std::max<Integer>(max_iters, n_iter);
          n_fallback += used_fallback ? 1 : 0;
          if (used_fallback) { seederr_fb += seed_err; iters_fb += n_iter; }
          else               { seederr_ok += seed_err; iters_ok += n_iter; }
          n_call++;
        }
      }
    }
  }
  const Long n_ok = n_call - n_fallback;
  std::printf("  -> %ld calls: max Newton iters = %d, fallbacks = %ld\n",
              n_call, (int)max_iters, n_fallback);
  std::printf("     converged: mean |seed-uv*| = %.3e, mean iters = %.2f  (n=%ld)\n",
              n_ok       ? (double)(seederr_ok / n_ok)       : 0.0, n_ok       ? (double)iters_ok / n_ok       : 0.0, n_ok);
  std::printf("     FALLBACK : mean |seed-uv*| = %.3e, mean iters = %.2f  (n=%ld)\n",
              n_fallback ? (double)(seederr_fb / n_fallback) : 0.0, n_fallback ? (double)iters_fb / n_fallback : 0.0, n_fallback);
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    using Real = double;
    const Long order = 8, PatchPerFace = 2, n_patch = 6;
    const Real R = 1.0;

    QuadElemList<Real> sph_reg   = BuildTwistedSphere<Real>(order, PatchPerFace, R, (Real)0);
    QuadElemList<Real> sph_twist = BuildTwistedSphere<Real>(order, PatchPerFace, R, const_pi<Real>());

    investigate<Real>("regular sphere (theta=0)",     sph_reg,   n_patch);
    investigate<Real>("pi-twisted sphere (theta=pi)", sph_twist, n_patch);
  }
  Comm::MPI_Finalize();
  return 0;
}
