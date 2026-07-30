/**
 * Adaptive-vs-hedgehog crossover via the off-surface Green's representation identity
 * (S[du/dn] - D[u] = u at interior targets), which is SELF-VALIDATING: the reference u is the exact
 * potential of a known exterior point source, so the reported error is the TRUE error -- no
 * scheme-based "gold" is used or needed.
 *
 * Geometry: PI-TWISTED cubed-sphere (sheared panels). Panel-interior targets (one per element at
 * (0.5,0.5), pushed inward by trg_dist). Sweeps trg_dist and, for Adaptive, the accuracy knob tol
 * (max_depth=30). hedgehog uses SetLineQBXParams defaults (no tol knob; its SetAccuracy tol only sets
 * the far field, kept tight at 1e-12 so the near scheme -- not the far field -- is what is measured).
 *
 * Timing: warm-up -> ClearSetup -> timed second ComputePotential(SL)+(DL), wall time via omp_get_wtime
 * (same warm-cache pattern as test-quad-elem.cpp). Self-contained; does NOT touch test-quad-elem.cpp.
 *
 * Build & run:
 *     . ./sctl_source
 *     make hedgehog-greens
 *     OMP_NUM_THREADS=4 ./bin/test-hedgehog-greens
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
  const Real r = sqrt<Real>(x*x + y*y + z*z);
  x *= R/r; y *= R/r; z *= R/r;
}

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

// Interior Green's identity S[du/dn] - D[u] = u(x) at panel-interior off-surface targets, self-
// validated against the exact point-source potential. Returns {true rel error, wall seconds for the
// timed (warm) SL+DL rebuild+eval}.
template <class Real, class KerSL, class KerDL, class KerGrad>
std::pair<double,double>
greens(const QuadElemList<Real>& elem_lst, const Comm& comm, const Real tol, const Vector<Real> X0, const Real trg_dist) {
  static constexpr Integer COORD_DIM = 3;
  srand48(42); // fixed density across every scheme/tol/distance run

  KerSL kernel_sl; KerDL kernel_dl; KerGrad kernel_grad;
  BoundaryIntegralOp<Real,KerSL> BIOpSL(kernel_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BIOpDL(kernel_dl, false, comm);
  BIOpSL.AddElemList(elem_lst); BIOpDL.AddElemList(elem_lst);
  BIOpSL.SetAccuracy(tol);      BIOpDL.SetAccuracy(tol);

  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud, Xtrg;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  {
    const Long Ne = elem_lst.Size();
    Xtrg.ReInit(Ne*COORD_DIM);
    const Vector<Real> up05{(Real)0.5}, vp05{(Real)0.5};
    for (Long e = 0; e < Ne; e++) {
      Vector<Real> Xc, Nc;
      elem_lst.GetGeom(&Xc, &Nc, nullptr, nullptr, nullptr, up05, vp05, e);
      for (Integer k = 0; k < COORD_DIM; k++) Xtrg[e*COORD_DIM+k] = Xc[k] - trg_dist*Nc[k];
    }
  }
  {
    Vector<Real> Xn0{0,0,0}, F0(KerSL::SrcDim()), dU, Usurf;
    for (auto& x : F0) x = drand48() - 0.5;
    kernel_sl.Eval(Usurf, X, X0, Xn0, F0);
    kernel_grad.Eval(dU, X, X0, Xn0, F0);
    kernel_sl.Eval(Uref, Xtrg, X0, Xn0, F0);
    Fd = Usurf;
    constexpr Integer KDIM0 = KerSL::SrcDim();
    const Long N = X.Dim()/COORD_DIM;
    Fs.ReInit(N*KDIM0);
    for (Long i = 0; i < N; i++)
      for (Integer j = 0; j < KDIM0; j++) {
        Real g = 0;
        for (Long k = 0; k < COORD_DIM; k++) g += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
        Fs[i*KDIM0+j] = g;
      }
  }
  BIOpSL.SetTargetCoord(Xtrg); BIOpDL.SetTargetCoord(Xtrg);

  BIOpSL.ComputePotential(Us, Fs);            // warm-up
  BIOpDL.ComputePotential(Ud, Fd);
  BIOpSL.ClearSetup(); BIOpDL.ClearSetup();
  Us = 0; Ud = 0;
  const double t0 = omp_get_wtime();          // timed second run
  BIOpSL.ComputePotential(Us, Fs);
  BIOpDL.ComputePotential(Ud, Fd);
  const double dt = omp_get_wtime() - t0;

  const Vector<Real> Uerr = (Us - Ud) - Uref;
  StaticArray<Real,2> me{0,0}, mv{0,0};
  for (auto x : Uerr) me[0] = std::max<Real>(me[0], fabs(x));
  for (auto x : Uref) mv[0] = std::max<Real>(mv[0], fabs(x));
  comm.Allreduce(me+0, me+1, 1, CommOp::MAX);
  comm.Allreduce(mv+0, mv+1, 1, CommOp::MAX);
  return {(double)(me[1]/mv[1]), dt};
}

using QS = QuadElemList<double>::QuadScheme;

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const Long ElemOrder = 12, PatchPerFace = 3;
    const Real theta = (Real)0;                          // regular (untwisted) sphere
    const Vector<Real> X0{1.3, 1.2, 0.2};
    const Vector<Real> coord = twisted_sphere_coord<Real>(ElemOrder, PatchPerFace, (Real)1.0, theta);

    const std::vector<Real>   dists   = {1e-3, 1e-4, 1e-5};
    const std::vector<double> adatols = {1e-6, 1e-9, 1e-12};

    QuadElemList<Real> qhh(ElemOrder, coord, comm); qhh.SetQuadScheme(QS::LineQBX); qhh.SetLineQBXParams();
    QuadElemList<Real> qad(ElemOrder, coord, comm); qad.SetQuadScheme(QS::Adaptive, 6, 0, 30);

    if (!comm.Rank())
      std::printf("REGULAR (untwisted) cubed-sphere (order %ld, %ld patch/face, %ld elems), panel-interior targets.\n"
                  "Accuracy = TRUE error from the analytic Green's identity (no reference scheme). Time = warm SL+DL rebuild+eval.\n",
                  ElemOrder, PatchPerFace, 6*PatchPerFace*PatchPerFace);

    for (const Real d : dists) {
      if (!comm.Rank()) std::printf("\n#################### trg_dist = %.0e ####################\n", (double)d);
      // hedgehog (far tight at 1e-12; near = LineQBX defaults)
      const auto hL = greens<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(qhh, comm, (Real)1e-12, X0, d);
      const auto hS = greens<Real, Stokes3D_FxU,  Stokes3D_DxU,  Stokes3D_FxT >(qhh, comm, (Real)1e-12, X0, d);
      if (!comm.Rank()) {
        std::printf("  %-26s | Laplace err=%.3e t=%.2fs | Stokes err=%.3e t=%.2fs\n",
                    "hedgehog (defaults)", hL.first, hL.second, hS.first, hS.second);
        for (const double tol : adatols) {
          const auto aL = greens<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(qad, comm, (Real)tol, X0, d);
          const auto aS = greens<Real, Stokes3D_FxU,  Stokes3D_DxU,  Stokes3D_FxT >(qad, comm, (Real)tol, X0, d);
          char lbl[48]; std::snprintf(lbl, sizeof(lbl), "adaptive tol=%.0e md=30", tol);
          std::printf("  %-26s | Laplace err=%.3e t=%.2fs | Stokes err=%.3e t=%.2fs\n",
                      lbl, aL.first, aL.second, aS.first, aS.second);
        }
      } else { // non-root ranks still participate in the collective ComputePotential calls
        for (const double tol : adatols) {
          greens<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(qad, comm, (Real)tol, X0, d);
          greens<Real, Stokes3D_FxU,  Stokes3D_DxU,  Stokes3D_FxT >(qad, comm, (Real)tol, X0, d);
        }
      }
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
