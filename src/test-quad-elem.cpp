/**
 * This demo code shows how to use the class sctl::QuadElemList to build a
 * cubed-sphere geometry and write it to VTK for visualization.
 *
 * To compile and run the code, start in the SCTL root directory and run:
 * make bin/test-quad-elem && export OMP_NUM_THREADS=4 && ./bin/test-quad-elem
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

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
  x *= R / r;
  y *= R / r;
  z *= R / r;
}

// Cubed-sphere of radius Radius: PatchPerFace^2 quad patches per cube face, ElemOrder nodes/direction. 
// twisted about z: at height z, {x,y} rotated by theta_twist*z.
// Regular sphere: theta_twist = 0.
template <class Real>
QuadElemList<Real> BuildTwistedSphere(Long ElemOrder, Long PatchPerFace, Real Radius, Real theta_twist = 0.) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(ElemOrder);
  for (Integer face = 0; face < 6; face++) {
    for (Long iu = 0; iu < PatchPerFace; iu++) {
      for (Long iv = 0; iv < PatchPerFace; iv++) {
        for (Long i = 0; i < ElemOrder; i++) {
          const Real u = (iu + nds[i]) / (Real)PatchPerFace;
          const Real a = 2 * u - 1;
          for (Long j = 0; j < ElemOrder; j++) {
            const Real v = (iv + nds[j]) / (Real)PatchPerFace;
            const Real b = 2 * v - 1;

            Real x, y, z;
            FacePoint(x, y, z, face, a, b, Radius);
            const Real sin_theta = sin<Real>(theta_twist * z);
            const Real cos_theta = cos<Real>(theta_twist * z);
            X.PushBack(x * cos_theta + y * sin_theta);
            X.PushBack(-x * sin_theta + y * cos_theta);
            X.PushBack(z);
          }
        }
      }
    }
  }
  return QuadElemList<Real>(ElemOrder, X);
}

// Quadrature weights must sum to the analytic sphere area 4 pi R^2.
template <class Real> void test_SurfaceArea(const QuadElemList<Real>& elem_lst, Real Radius) {
  Vector<Real> wts, Xtemp, Xntemp, dist_far;
  Vector<Long> elem_wise_temp;
  elem_lst.GetFarFieldNodes(Xtemp, Xntemp, wts, dist_far, elem_wise_temp, 1);
  Real Area = 0.;
  for (int i = 0; i < wts.Dim(); i++) Area += wts[i];
  const Real Area_exact = 4. * const_pi<Real>() * Radius * Radius;
  const Real relerr = std::fabs(Area - Area_exact) / Area_exact;
  std::cout << "Area from Jacobian compared to exact: " << std::setprecision(8) << relerr << std::endl;
}

// Stokes DL constant-density identity on a closed sphere: D[q] = c*q, |c|=1/2.
// Sign convention: this kernel (r = x_trg-x_src, source normal) gives c = -1/2 for
// an outward normal, +1/2 for inward; verify magnitude and sign vs. orientation.
template <class Real, class KerDL> void test_DLIdentity(const QuadElemList<Real>& elem_lst, const Comm& comm, const Real quad_tol = 1e-8) {
  const KerDL kernel_dl;
  BoundaryIntegralOp<Real, KerDL> BIOp(kernel_dl, false, comm);
  BIOp.SetAccuracy(quad_tol);
  BIOp.AddElemList(elem_lst);

  const Real c_expect = -0.5; // Elem_lst always have outward surface normals.

  // Constant density q at every node.
  const Long KDIM0 = KerDL::SrcDim();
  Vector<Real> X, Xn;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  const Long Nnode = X.Dim() / 3;
  Vector<Real> q(Nnode * KDIM0), U;
  for (Long i = 0; i < Nnode; i++) { 
    for (Long k=0; k<KDIM0; k++) {
      q[i*KDIM0 + k] = k+1; // arbitrary constant density {1,2,3}.
    }
  }
  BIOp.ComputePotential(U, q);

  // D[q] should equal c*q: measure mean U_x and max deviation.
  Vector<Real> cx_maxerr(KDIM0);
  cx_maxerr = 0.;
  for (Long i = 0; i < Nnode; i++) {
    for (Long k=0; k<KDIM0; k++) {
      cx_maxerr[k] = std::max(cx_maxerr[k], std::fabs(U[i*KDIM0+k] / q[i*KDIM0+k] - c_expect));
    }
  } 
  Vector<Real> cx_relerr = cx_maxerr / std::fabs(c_expect);
  Real cx_relerr_avg = 0.;
  for (Long k=0; k<KDIM0; k++) {
    cx_relerr_avg += cx_relerr[k];
  }
  cx_relerr_avg /= KDIM0;
  std::cout << std::setprecision(8) << "DL constant-density identity: max relative error = " << cx_relerr_avg << std::endl;
}

// Interior Green's representation identity on a closed surface (Laplace or Stokes):
// for a source X0 OUTSIDE the surface (u harmonic in the interior),
// (S[Fs] - D[Fd]) - 0.5*Fd == u|_S, with Fd = u, Fs = +du/dn (outward normal).
//
// trg_dist == 0 : on-surface (self-eval) targets = surface nodes; apply the -0.5 DL jump.
// trg_dist  > 0 : off-surface INTERIOR targets, pushed in by trg_dist along the inward normal
//                 (exercises the NEAR-interaction path). The near-singular quadrature returns the
//                 true off-surface D[u] (interior limit included), so no manual jump is applied and
//                 (S[Fs] - D[Fd]) == u at the interior targets directly.
template <class Real, class KerSL, class KerDL, class KerGrad> void test_greens_identity(const QuadElemList<Real>& elem_lst, const Comm& comm,
                          const Real tol, const Vector<Real> X0, const Real trg_dist = 0) {
  static constexpr Integer COORD_DIM = 3;
  const Long pid = comm.Rank();

  KerSL kernel_sl;
  KerDL kernel_dl;
  KerGrad kernel_grad;
  BoundaryIntegralOp<Real,KerSL> BIOpSL(kernel_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BIOpDL(kernel_dl, false, comm);
  BIOpSL.AddElemList(elem_lst);
  BIOpDL.AddElemList(elem_lst);
  BIOpSL.SetAccuracy(tol);
  BIOpDL.SetAccuracy(tol);

  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud, Xtrg;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  { // Targets: interior offset for the near test (push inward along the outward normal), else on-surface.
    if (trg_dist > 0) {
      const Long N = X.Dim()/COORD_DIM;
      Xtrg.ReInit(X.Dim());
      for (Long i = 0; i < N; i++)
        for (Integer k = 0; k < COORD_DIM; k++)
          Xtrg[i*COORD_DIM+k] = X[i*COORD_DIM+k] - trg_dist*Xn[i*COORD_DIM+k];
    } else {
      Xtrg = X;
    }
  }
  {
    Vector<Real> Xn0{0,0,0}, F0(KerSL::SrcDim()), dU, Usurf;
    for (auto& x : F0) x = drand48()-0.5;
    kernel_sl.Eval(Usurf, X, X0, Xn0, F0);    // u at source surface nodes (DL density)
    kernel_grad.Eval(dU, X, X0, Xn0, F0);     // grad u at source surface nodes
    kernel_sl.Eval(Uref, Xtrg, X0, Xn0, F0);  // u at the targets (reference)

    Fd = Usurf;
    { // Set Fs <-- +dot_prod(dU, Xn)  (= +du/dn; CSBQ utils.cpp free-function convention)
      constexpr Integer KDIM0 = KerSL::SrcDim();
      const Long N = X.Dim()/COORD_DIM;
      Fs.ReInit(N * KDIM0);
      for (Long i = 0; i < N; i++) {
        for (Integer j = 0; j < KDIM0; j++) {
          Real dU_dot_Xn = 0;
          for (Long k = 0; k < COORD_DIM; k++) {
            dU_dot_Xn += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
          }
          Fs[i*KDIM0+j] = dU_dot_Xn;
        }
      }
    }
  }

  // Off-surface targets exercise the near-interaction path (targets != surface nodes).
  if (trg_dist > 0) {
    BIOpSL.SetTargetCoord(Xtrg);
    BIOpDL.SetTargetCoord(Xtrg);
  }

  // Warm-up run
  BIOpSL.ComputePotential(Us,Fs);
  BIOpDL.ComputePotential(Ud,Fd);
  BIOpSL.ClearSetup();
  BIOpDL.ClearSetup();
  Us = 0; Ud = 0;

  sctl::Profile::Enable(true);
  Profile::Tic("Setup+Eval", &comm);
  BIOpSL.ComputePotential(Us,Fs);
  BIOpDL.ComputePotential(Ud,Fd);
  Profile::Toc();

  if (trg_dist == 0) Ud -= 0.5*Fd; // DL jump condition, on-surface only (off-surface D[u] already includes it)
  Vector<Real> Uerr = (Us - Ud) - Uref;
  // elem_lst.WriteVTK("Uerr", Uerr, comm);
  { // Print error
    StaticArray<Real,2> max_err{0,0};
    StaticArray<Real,2> max_val{0,0};
    for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
    for (auto x : Uref) max_val[0] = std::max<Real>(max_val[0], fabs(x));
    comm.Allreduce(max_err+0, max_err+1, 1, CommOp::MAX);
    comm.Allreduce(max_val+0, max_val+1, 1, CommOp::MAX);
    if (!pid) std::cout<<"Green's identity error = "<<max_err[1]/max_val[1]<<'\n';
  }

  sctl::Profile::print(&comm, {"t_avg", "f/s_avg"});
  sctl::Profile::reset();
  sctl::Profile::Enable(false);
}

// Nbeta (RectPolar cov_order) sweep on the regular sphere.
// For fixed ElemOrder/PatchPerFace, march Nbeta over the compile-time templated
// values {48, 100, 200, 300, 400, 512} (the only values GLRuleNbetaDispatch /
// RPSelfRuleDispatch accept) and record the active accuracy tests (surface area,
// DL constant-density identity, Green's identity) for Laplace and Stokes.
template <class Real> void test_NbetaSweep(const Comm& comm,
                     const std::vector<Long>& NbetaList = {48, 100, 200, 300, 400, 512},
                     const Long PatchPerFace = 8,
                     const Long ElemOrder = 16,
                     const Real quad_tol = 1e-13) {
  const Real Radius = 1.0;
  const Vector<Real> X0{1.3, 1.2, 0.2}; // Exterior source for Green's Id test on interior.
  const bool root = !comm.Rank();

  if (root) {
    std::cout << "\nNbeta sweep (RectPolar, regular sphere)\n";
    std::cout << "  ElemOrder=" << ElemOrder << ", PatchPerFace=" << PatchPerFace
              << ", quad_tol=" << quad_tol << "\n";
  }

  for (const Long Nbeta : NbetaList) {
    if (root) std::cout << "\n === Nbeta = " << Nbeta << " ===" << std::endl;

    QuadElemList<Real> elem_lst = BuildTwistedSphere<Real>(ElemOrder, PatchPerFace, Radius, /*theta_twist=*/ 0.);
    elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::RectPolar, 6, Nbeta);

    // test_SurfaceArea<Real>(elem_lst, Radius);
    // std::cout << "DL constant density test, Laplace. " << std::endl;
    // test_DLIdentity<Real, Laplace3D_DxU>(elem_lst, comm, quad_tol);
    std::cout << "DL constant density test, Stokes. " << std::endl;
    test_DLIdentity<Real, Stokes3D_DxU>(elem_lst, comm, quad_tol);
    // std::cout << "Green's identity test, Laplace. " << std::endl;
    // test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, quad_tol, X0);
    std::cout << "Green's identity test, Stokes. " << std::endl;
    test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, quad_tol, X0);
  }
  if (root) std::cout << "\nNbeta sweep: DONE" << std::endl;
}


// // Helpers validating cubed-sphere BIOs against the SH reference on the unit sphere.
// // Target placement relative to the unit-radius surface.
// enum class TgtType { OnSurface, Near, Far };
// // Sample a SCALAR density on the SH grid; return ROW_MAJOR scalar SH coeffs.
// template <class Real, class DensityFn> Vector<Real> SphereScalarSHC(Long p, DensityFn density) {
//   const Long Nt = p + 1, Np = 2 * p + 2;
//   const Vector<Real>& CosTheta = SphericalHarmonics<Real>::LegendreNodes(Nt - 1);
//   Vector<Real> Xgrid(Nt * Np); // theta-major: Xgrid[i*Np + j]
//   for (Long i = 0; i < Nt; i++) {
//     const Real ct = CosTheta[i], st = sqrt(1 - ct * ct);
//     for (Long j = 0; j < Np; j++) {
//       const Real phi = 2 * const_pi<Real>() * j / Np;
//       Real out[1];
//       density(st * cos(phi), st * sin(phi), ct, out);
//       Xgrid[i * Np + j] = out[0];
//     }
//   }
//   Vector<Real> S;
//   SphericalHarmonics<Real>::Grid2SHC(Xgrid, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
//   return S;
// }

// // Sample a VECTOR density on the SH grid (component-major SoA); return vector SH coeffs.
// template <class Real, class DensityFn> Vector<Real> SphereVecSHC(Long p, DensityFn density) {
//   const Long Nt = p + 1, Np = 2 * p + 2, Ngrid = Nt * Np;
//   const Vector<Real>& CosTheta = SphericalHarmonics<Real>::LegendreNodes(Nt - 1);
//   Vector<Real> Xgrid(3 * Ngrid);
//   for (Long i = 0; i < Nt; i++) {
//     const Real ct = CosTheta[i], st = sqrt(1 - ct * ct);
//     for (Long j = 0; j < Np; j++) {
//       const Real phi = 2 * const_pi<Real>() * j / Np;
//       Real out[3];
//       density(st * cos(phi), st * sin(phi), ct, out);
//       Xgrid[0 * Ngrid + i * Np + j] = out[0];
//       Xgrid[1 * Ngrid + i * Np + j] = out[1];
//       Xgrid[2 * Ngrid + i * Np + j] = out[2];
//     }
//   }
//   Vector<Real> S;
//   SphericalHarmonics<Real>::Grid2VecSHC(Xgrid, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
//   return S;
// }

// // Compare BIO vs. SH reference for one kernel across all three target placements.
// // is_DL: the discontinuous Real-layer PV equals the mean of the interior/exterior limits.
// template <class Real, class Kernel, class DensityFn, class RefEvalFn>
// void TestSphereBIOvsSH(const QuadElemList<Real>& elem_lst, const Comm& comm,
//                        const Kernel& ker, const char* kername, bool is_DL,
//                        DensityFn density, RefEvalFn ref_eval, const Real tol=1e-9) {
//   static constexpr Integer KDIM0 = Kernel::SrcDim();

//   Vector<Real> Xnodes;
//   elem_lst.GetNodeCoord(&Xnodes, nullptr, nullptr);
//   const Long Nnode = Xnodes.Dim() / 3;

//   // Density at the cubed-sphere nodes (AoS).
//   Vector<Real> F(Nnode * KDIM0);
//   for (Long i = 0; i < Nnode; i++) density(Xnodes[i*3+0], Xnodes[i*3+1], Xnodes[i*3+2], &F[i*KDIM0]);

//   BoundaryIntegralOp<Real, Kernel> BIOp(ker, /*trg_normal_dot_prod=*/false, comm);
//   BIOp.SetAccuracy(tol);
//   BIOp.AddElemList(elem_lst);

//   struct Cfg { const char* name; TgtType type; Real scale; };
//   const Cfg cfgs[3] = {
//     {"on-surface", TgtType::OnSurface, 1.00}, // singular self-interaction
//     {"near",       TgtType::Near,      1.02}, // near-singular correction
//     {"far",        TgtType::Far,       2.00}, // smooth far-field
//   };

//   for (const auto& c : cfgs) {
//     // Targets: surface nodes (on-surface) or nodes pushed radially outward (off-surface).
//     Vector<Real> Xtrg;
//     if (c.type != TgtType::OnSurface) {
//       Xtrg.ReInit(Nnode * 3);
//       for (Long i = 0; i < Nnode * 3; i++) Xtrg[i] = Xnodes[i] * c.scale;
//       BIOp.SetTargetCoord(Xtrg);
//     }

//     Vector<Real> U_quad;
//     BIOp.ComputePotential(U_quad, F);

//     // SH reference at the same targets.
//     const Vector<Real>& coord = (c.type == TgtType::OnSurface) ? Xnodes : Xtrg;
//     Vector<Real> U_ref;
//     if (c.type == TgtType::OnSurface && is_DL) {
//       Vector<Real> U_in, U_out; // PV = mean of the two one-sided limits
//       ref_eval(coord, /*interior=*/true,  U_in);
//       ref_eval(coord, /*interior=*/false, U_out);
//       U_ref.ReInit(U_in.Dim());
//       for (Long i = 0; i < U_ref.Dim(); i++) U_ref[i] = 0.5 * (U_in[i] + U_out[i]);
//     } else {
//       ref_eval(coord, /*interior=*/false, U_ref);
//     }

//     SCTL_ASSERT(U_quad.Dim() == U_ref.Dim());
//     Real err2 = 0, ref2 = 0;
//     for (Long i = 0; i < U_ref.Dim(); i++) {
//       const Real e = U_quad[i] - U_ref[i];
//       err2 += e * e; ref2 += U_ref[i] * U_ref[i];
//     }
//     const Real rel_l2 = sqrt(err2 / ref2);
//     std::cout << "  " << kername << " / " << c.name << " : rel L2 error = " << rel_l2 << std::endl;
//     SCTL_ASSERT(rel_l2 < 1e-5); // geometry/quadrature-limited (~1e-7 observed)
//   }
// }


// // BIOs vs. SH reference over {Laplace/Stokes x SL/DL} and {on-surface,near,far}.
// // On the unit sphere the layer operators are diagonalized by SH, giving a spectral
// // reference for a smooth non-polynomial density.
// template <class Real> void test_BIOvsSH(const QuadElemList<Real>& elem_lst, const Comm& comm, bool write_vtk = false, const Real tol = 1e-9) {
//   const Long p = 30; // SH truncation order (captures exp densities to ~eps)

//   // Non-polynomial densities (analytic -> fast SH decay).
//   auto lap_density = [](Real x, Real, Real, Real* o) { o[0] = std::exp(x); };
//   auto sto_density = [](Real x, Real y, Real z, Real* o) {
//     o[0] = std::exp(x); o[1] = std::exp(y); o[2] = std::exp(z);
//   };

//   // Density SH coefficients.
//   const Vector<Real> Slap = SphereScalarSHC<Real>(p, lap_density);
//   const Vector<Real> Ssto = SphereVecSHC<Real>(p, sto_density);

//   std::cout << "BIO vs. spherical-harmonics reference (density = exp):" << std::endl;

//   // Visualize the (kernel-independent) near/self refinement on one element; tol matches
//   // the SetAccuracy(1e-9) used in TestSphereBIOvsSH.
//   if (write_vtk) {
//     const Long evis = 0;                       // element to visualize
//     const Integer ord = elem_lst.Order();
//     // Near: a surface point pushed off along its outward normal.
//     Vector<Real> us{0.5}, vs{0.5}, Xc, Xn;
//     elem_lst.GetGeom(&Xc, &Xn, nullptr, nullptr, nullptr, us, vs, evis);
//     Vector<Real> Xtrg(3);
//     for (int k = 0; k < 3; k++) Xtrg[k] = Xc[k] + 0.02 * Xn[k];
//     if (elem_lst.NearUsesRectPolar()) {
//       elem_lst.WriteNearInteracRPVTK("near-interac-elem0", evis, Xtrg);
//     } else {
//       elem_lst.WriteNearInteracVTK("near-interac-elem0", evis, Xtrg, 1e-9, comm); // tol = 1e-9 just for plotting
//     }
    
//     // Self: an interior node parameter.
//     const auto& nds = QuadElemList<Real>::ParamNodes(ord);
//     const Real u0 = nds[ord/2], v0 = nds[ord/2];
//     if (elem_lst.SelfUsesRectPolar()) {
//       elem_lst.WriteSelfInteracRPVTK("self-interac-elem0", evis, u0, v0);
//     } else {
//       elem_lst.WriteSelfInteracVTK("self-interac-elem0", evis, u0, v0, 1e-9, comm);
//     }
    
//     std::cout << "  wrote near-interac-elem0-* and self-interac-elem0-* VTK files" << std::endl;
//   }

//   TestSphereBIOvsSH<Real>(elem_lst, comm, Laplace3D_FxU(), "Laplace3D_FxU", /*is_DL=*/false, lap_density,
//     [&](const Vector<Real>& c, bool in, Vector<Real>& U) {
//       SphericalHarmonics<Real>::LaplaceEvalSL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol);

//   TestSphereBIOvsSH<Real>(elem_lst, comm, Laplace3D_DxU(), "Laplace3D_DxU", /*is_DL=*/true, lap_density,
//     [&](const Vector<Real>& c, bool in, Vector<Real>& U) {
//       SphericalHarmonics<Real>::LaplaceEvalDL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol);

//   std::cout << "BIO vs. SH reference Laplace: PASSED" << std::endl;

//   TestSphereBIOvsSH<Real>(elem_lst, comm, Stokes3D_FxU(), "Stokes3D_FxU", /*is_DL=*/false, sto_density,
//     [&](const Vector<Real>& c, bool in, Vector<Real>& U) {
//       SphericalHarmonics<Real>::StokesEvalSL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol);

//   TestSphereBIOvsSH<Real>(elem_lst, comm, Stokes3D_DxU(), "Stokes3D_DxU", /*is_DL=*/true, sto_density,
//     [&](const Vector<Real>& c, bool in, Vector<Real>& U) {
//       SphericalHarmonics<Real>::StokesEvalDL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol);

//   std::cout << "BIO vs. SH reference Stokes: PASSED" << std::endl;
//   SphericalHarmonics<Real>::Clear();
// }

// // Manufactured-solution interior/exterior Dirichlet test via combined-field BIE + GMRES.
// // Point sources are placed on the side OPPOSITE the solution domain so their field u_e
// // is exact on the domain; sample u_e on the surface and solve
// //      ( c*I + SL_scal*S + DL_scal*D ) sigma = u_e|_surface,
// // with jump c = +-1/2*DL_scal (sign from outward normal and interior/exterior).
// // SIGN REQUIREMENT: CFIE is uniquely solvable only for SAME sign (exterior) /
// // OPPOSITE sign (interior) of (SL_scal, DL_scal); else the interior operator has a
// // null space. quadr_tol also sets GMRES tol.
// // Returns one rel-L2 per entry of eval_radii. The CFIE is solved ONCE (the dominant cost);
// // only the post-solve evaluation is repeated per radius, so e.g. near+far share a single solve.
// template <class KerSL, class KerDL, class Real>
// std::vector<Real> TestManufactured(const QuadElemList<Real>& elem_lst, const Comm& comm,
//                         const KerSL& ker_sl, const KerDL& ker_dl, const char* name,
//                         const Vector<Real>& Xsrc, const Vector<Real>& Fsrc,
//                         bool interior, const std::vector<Real>& eval_radii, const Real quadr_tol = 1e-9,
//                         Real SL_scal = 1.0, const Real DL_scal = 1.0) {
//   static constexpr Integer KDIM = KerSL::SrcDim(); // 1 (Laplace) or 3 (Stokes)

//   // Surface nodes/normals; orientation sets the DL jump sign.
//   Vector<Real> Xs, Xns;
//   elem_lst.GetNodeCoord(&Xs, &Xns, nullptr);
//   const Long Nnode = Xs.Dim() / 3;
//   Real xdotn = 0;
//   for (Long i = 0; i < Nnode; i++)
//     xdotn += Xs[i*3+0]*Xns[i*3+0] + Xs[i*3+1]*Xns[i*3+1] + Xs[i*3+2]*Xns[i*3+2];
//   // +1/2 for outward normal exterior trace; interior flips the sign.
//   const Real sgn = interior ? -1.0 : 1.0;
//   const Real jump = (xdotn > 0 ? 0.5 : -0.5) * DL_scal * sgn;

//   // Interior with same-sign SL/DL has a null space; flip SL sign.
//   if (interior && SL_scal*DL_scal > 0.) {
//     std::cout << "Warning: Interior problem has artificial null space when SL and DL same sign. Flipping SL sign. " << std::endl;
//     SL_scal = -1.*SL_scal;
//   }

//   // Dirichlet data: point-source field at surface nodes (SL kernel ignores src normal).
//   Vector<Real> bc;
//   ker_sl.Eval(bc, Xs, Xsrc, Xsrc, Fsrc);

//   // Combined-field operator pieces (on-surface PV).
//   BoundaryIntegralOp<Real, KerSL> SLOp(ker_sl, /*trg_normal_dot_prod=*/false, comm);
//   BoundaryIntegralOp<Real, KerDL> DLOp(ker_dl, /*trg_normal_dot_prod=*/false, comm);
//   SLOp.SetAccuracy(quadr_tol); DLOp.SetAccuracy(quadr_tol);
//   SLOp.AddElemList(elem_lst); DLOp.AddElemList(elem_lst);

//   const auto ApplyK = [&](Vector<Real>* U, const Vector<Real>& sigma) {
//     Vector<Real> Us, Ud;
//     SLOp.ComputePotential(Us, sigma);
//     DLOp.ComputePotential(Ud, sigma);
//     if (U->Dim() != sigma.Dim()) U->ReInit(sigma.Dim());
//     (*U) = SL_scal*Us + DL_scal*Ud + jump*sigma;
//   };

//   GMRES<Real> solver(comm, false);
//   Vector<Real> sigma;
//   Long iter = 0;
//   const Real gmres_tol = quadr_tol * 10.;
//   const Long gmres_max_iter = 100;

//   // Profile::reset();
//   // Profile::Tic("gmres solve");
//   solver(&sigma, ApplyK, bc, gmres_tol, gmres_max_iter, false, &iter);
//   // Profile::Toc();
//   // Profile::print(&comm, {"t_avg", "f_avg", "f/s_avg"});

//   // Evaluate the recovered potential at each target sphere (radius-1 nodes scaled by the
//   // radius). The solve above is reused; only target placement + the eval matvec change.
//   std::vector<Real> rel_l2s;
//   rel_l2s.reserve(eval_radii.size());
//   for (const Real eval_radius : eval_radii) {
//     Vector<Real> Xtrg(Nnode * 3);
//     for (Long i = 0; i < Nnode * 3; i++) Xtrg[i] = Xs[i] * eval_radius;
//     SLOp.SetTargetCoord(Xtrg); DLOp.SetTargetCoord(Xtrg);

//     // Profile::reset();
//     // Profile::Tic("eval");
//     Vector<Real> Us, Ud;
//     SLOp.ComputePotential(Us, sigma);
//     DLOp.ComputePotential(Ud, sigma);
//     Vector<Real> U = SL_scal * Us + DL_scal * Ud;
//     // Profile::Toc();
//     // Profile::print(&comm, {"t_avg", "f_avg", "f/s_avg"});

//     // Reference: point-source field evaluated directly at the targets.
//     Vector<Real> Uref;
//     ker_sl.Eval(Uref, Xtrg, Xsrc, Xsrc, Fsrc);

//     Real err2 = 0, ref2 = 0;
//     for (Long i = 0; i < U.Dim(); i++) { const Real e = U[i] - Uref[i]; err2 += e*e; ref2 += Uref[i]*Uref[i]; }
//     const Real rel_l2 = sqrt(err2 / ref2);
//     std::cout << "  " << name << " (R=" << eval_radius << ", GMRES iters = " << iter
//               << ") : rel L2 error = " << rel_l2 << std::endl;
//     rel_l2s.push_back(rel_l2);
//   }
//   return rel_l2s;
// }

// // h-refinement convergence: fixed ElemOrder, increasing PatchPerFace, report
// // manufactured-solution rel-L2 at near/far targets per resolution.
// template <class Real> void test_ManufacturedConvergence(const Comm& comm,
//                                   bool interior = false,
//                                   bool rect_polar = false,
//                                   const Real theta_twist = 0.,
//                                   const std::vector<Long>& PatchPerFaceList = {1, 2, 3, 4, 5},
//                                   Long ElemOrder = 16
//                                   ) {
//   const Real Radius = 1.0;
//   const Real base_tol = 1e-13; // should be this level for ElemOrder = 12, maybe allowed higher if lower order..

//   // Laplace charges / Stokeslets outside the sphere (interior problem).
//   const Vector<Real> Fsrc_lap{1.0, -0.7};
//   const Vector<Real> Fsrc_sto{1.0, 0.5, -0.3,  -0.4, 0.2, 0.1};
//   const Vector<Real> src_ext{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10}; // for ext-erior problem
//   const Vector<Real> src_int{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60}; // for int-erior problem
//   const Real Rint_far = 0.5;
//   const Real Rint_near = 0.999;
//   const Real Rext_far = 2.;
//   const Real Rext_near = 1.001;
//   Vector<Real> Xsrc_lap, Xsrc_sto;
//   Real R_far, R_near, SL_scal, DL_scal;
//   std::string name;
//   if (!interior) { // exterior problem, DL+SL
//     Xsrc_lap = src_ext;
//     Xsrc_sto = src_ext;
//     R_far = Rext_far;
//     R_near = Rext_near;
//     SL_scal = 1.0;
//     DL_scal = 1.0;
//     name = "DL+SL";
//   } else { // interior problem, DL only
//     Xsrc_lap = src_int;
//     Xsrc_sto = src_int;
//     R_far = Rint_far;
//     R_near = Rint_near;
//     SL_scal = 0.0;
//     DL_scal = 1.0;
//     name = "DL";
//   }

//   std::cout << "\nManufactured-solution convergence study (ElemOrder = " << ElemOrder << "):\n";
//   std::cout << std::scientific;
//   std::cout << "  kernel    PatchPerFace  Nelem   rel-L2 (near R=" << R_near <<")   rel-L2 (far R="<<R_far<<")\n"; 
//   for (const Long PatchPerFace : PatchPerFaceList) {
//     QuadElemList<Real> elem_lst = BuildTwistedSphere<Real>(ElemOrder, PatchPerFace, Radius, theta_twist);
//     if (rect_polar) {
//       elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::RectPolar);
//     }
//     const Long Nelem = 6 * PatchPerFace * PatchPerFace;

//     Real quadr_tol = base_tol;
//     if (PatchPerFace > 5) {
//       quadr_tol *= 0.0001;
//     } else if (PatchPerFace > 3) {
//       quadr_tol *= 0.01;
//     }

//     // Single solve per kernel, evaluated at both radii (near first, then far).
//     const std::vector<Real> el = TestManufactured<Real>(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
//                              ("Laplace "+name).c_str(), Xsrc_lap, Fsrc_lap, interior, {R_near, R_far}, quadr_tol, SL_scal, DL_scal);
//     const Real el_near = el[0], el_far = el[1];
//     std::cout << "  Laplace   " << std::setw(12) << PatchPerFace << "  " << std::setw(5) << Nelem
//               << "   " << el_near << "        " << el_far << "\n";

//     const std::vector<Real> es = TestManufactured<Real>(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
//                              ("Stokes "+name).c_str(), Xsrc_sto, Fsrc_sto, interior, {R_near, R_far}, quadr_tol, SL_scal, DL_scal);
//     const Real es_near = es[0], es_far = es[1];
//     std::cout << "  Stokes    " << std::setw(12) << PatchPerFace << "  " << std::setw(5) << Nelem
//               << "   " << es_near << "        " << es_far << "\n";
//   }
//   std::cout << "Manufactured-solution convergence study: DONE" << std::endl;
// }

}



int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;

  {
    const Comm comm = Comm::World();

    // Nbeta sweep over the compile-time templated RectPolar values on the regular sphere.
    // test_NbetaSweep<Real>(comm);

// #if 0 // previous scheme-comparison / tolerance sweep tests
    const Long PatchPerFace = 8;
    const Vector<Real> X0{1.3,1.2,0.2}; // Exterior source for Green's Id test on interior.
    const Real Radius = 1.0;
    const Vector<Long> ElemOrderList{8, 16};
    const Vector<Real> tolList{1e-6, 1e-7, 1e-9, 1e-11};
    const Vector<Long> NbetaList{48, 100, 200, 400};
    const Vector<Long> MaxDepthList{4, 8, 12, 30};

    for (Long idx=0; idx < tolList.Dim(); idx++) { 
      const Real tol = tolList[idx];
      const Long Nbeta = NbetaList[idx];
      const Long max_depth = MaxDepthList[idx];

      std::cout << "\n === Nbeta = " << Nbeta << ", tol = " << tol << ", max depth = " << max_depth << ". ===" << std::endl;

      for (Long ElemOrder : ElemOrderList) {

        // Regular sphere
        std::cout << "\n === ElemOrder = " << ElemOrder <<", Regular sphere. " << std::endl;
        QuadElemList<Real> elem_lst = BuildTwistedSphere<Real>(ElemOrder, PatchPerFace, Radius, /*theta_twist=*/ 0.);
        elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, Nbeta, max_depth);
        test_SurfaceArea<Real>(elem_lst, Radius);
        std::cout << "DL constant density test, Laplace. " << std::endl;
        test_DLIdentity<Real, Laplace3D_DxU>(elem_lst, comm, tol);
        std::cout << "DL constant density test, Stokes. " << std::endl;
        test_DLIdentity<Real, Stokes3D_DxU>(elem_lst, comm, tol);
        // On-surface (self-eval) Green's identity -- disabled for this near-only sweep.
        std::cout << "Green's identity test (on-surface), Laplace. " << std::endl;
        test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0);
        std::cout << "Green's identity test (on-surface), Stokes. " << std::endl;
        test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0);
        // Off-surface INTERIOR near targets (trg_dist=1e-4) -- exercises the near-interaction path.
        std::cout << "==== Near evaluation (ADAPTIVE near), Regular sphere. ====" << std::endl;
        std::cout << "Green's identity test (near, interior 1e-4), Laplace. " << std::endl;
        test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0, /*trg_dist=*/1e-4);
        std::cout << "Green's identity test (near, interior 1e-4), Stokes. " << std::endl;
        test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0, /*trg_dist=*/1e-4);

        // pi/2 Twisted sphere
        std::cout << "\n === ElemOrder = " << ElemOrder <<", pi/2 twisted sphere. " << std::endl;
        elem_lst = BuildTwistedSphere<Real>(ElemOrder, PatchPerFace, Radius, /*theta_twist=*/ const_pi<Real>()/2.);
        elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, Nbeta, max_depth);
        test_SurfaceArea<Real>(elem_lst, Radius);
        std::cout << "DL constant density test, Laplace. " << std::endl;
        test_DLIdentity<Real, Laplace3D_DxU>(elem_lst, comm, tol);
        std::cout << "DL constant density test, Stokes. " << std::endl;
        test_DLIdentity<Real, Stokes3D_DxU>(elem_lst, comm, tol);
        std::cout << "Green's identity test, Laplace. " << std::endl;
        test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0);
        std::cout << "Green's identity test, Stokes. " << std::endl;
        test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0);

        // pi Twisted sphere
        std::cout << "\n === ElemOrder = " << ElemOrder <<", pi twisted sphere. " << std::endl;
        elem_lst = BuildTwistedSphere<Real>(ElemOrder, PatchPerFace, Radius, /*theta_twist=*/ const_pi<Real>());
        elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, Nbeta, max_depth);
        test_SurfaceArea<Real>(elem_lst, Radius);
        std::cout << "DL constant density test, Laplace. " << std::endl;
        test_DLIdentity<Real, Laplace3D_DxU>(elem_lst, comm, tol);
        std::cout << "DL constant density test, Stokes. " << std::endl;
        test_DLIdentity<Real, Stokes3D_DxU>(elem_lst, comm, tol);
        std::cout << "Green's identity test, Laplace. " << std::endl;
        test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0);
        std::cout << "Green's identity test, Stokes. " << std::endl;
        test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0);

      }
    }

  }

  Comm::MPI_Finalize();
  return 0;
}
