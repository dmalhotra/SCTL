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
#include <sctl/experimental/gmsh_reader.cpp>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

using namespace sctl;

namespace {

// --- Distributed-memory helpers -------------------------------------------------
// Under MPI each rank owns only a slice of the geometry (see BuildTwistedSphere),
// so scalar norms/areas accumulated over local nodes must be reduced across ranks
// before they are compared or printed, and result prints are emitted on rank 0 only.
inline double GlobalReduce(double x, const Comm& comm, CommOp op) {
  StaticArray<double,2> buf; buf[0] = x; buf[1] = 0;
  comm.Allreduce(buf+0, buf+1, 1, op);
  return buf[1];
}
inline Long GlobalReduce(Long x, const Comm& comm, CommOp op) {
  StaticArray<Long,2> buf; buf[0] = x; buf[1] = 0;
  comm.Allreduce(buf+0, buf+1, 1, op);
  return buf[1];
}

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
// Every rank builds the full node array X, then the QuadElemList constructor keeps
// only this rank's contiguous element slice (replicate-then-slice partitioning).
template <class Real>
QuadElemList<Real> BuildTwistedSphere(Long ElemOrder, Long PatchPerFace, Real Radius, Real theta_twist = 0., const Comm& comm = Comm::Self()) {
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
  return QuadElemList<Real>(ElemOrder, X, comm);
}

// Helpers validating cubed-sphere BIOs against the SH reference on the unit sphere.

// Target placement relative to the unit-radius surface.
enum class TgtType { OnSurface, Near, Far };

// Sample a SCALAR density on the SH grid; return ROW_MAJOR scalar SH coeffs.
template <class DensityFn> Vector<double> SphereScalarSHC(Long p, DensityFn density) {
  const Long Nt = p + 1, Np = 2 * p + 2;
  const Vector<double>& CosTheta = SphericalHarmonics<double>::LegendreNodes(Nt - 1);
  Vector<double> Xgrid(Nt * Np); // theta-major: Xgrid[i*Np + j]
  for (Long i = 0; i < Nt; i++) {
    const double ct = CosTheta[i], st = sqrt(1 - ct * ct);
    for (Long j = 0; j < Np; j++) {
      const double phi = 2 * const_pi<double>() * j / Np;
      double out[1];
      density(st * cos(phi), st * sin(phi), ct, out);
      Xgrid[i * Np + j] = out[0];
    }
  }
  Vector<double> S;
  SphericalHarmonics<double>::Grid2SHC(Xgrid, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
  return S;
}

// Sample a VECTOR density on the SH grid (component-major SoA); return vector SH coeffs.
template <class DensityFn> Vector<double> SphereVecSHC(Long p, DensityFn density) {
  const Long Nt = p + 1, Np = 2 * p + 2, Ngrid = Nt * Np;
  const Vector<double>& CosTheta = SphericalHarmonics<double>::LegendreNodes(Nt - 1);
  Vector<double> Xgrid(3 * Ngrid);
  for (Long i = 0; i < Nt; i++) {
    const double ct = CosTheta[i], st = sqrt(1 - ct * ct);
    for (Long j = 0; j < Np; j++) {
      const double phi = 2 * const_pi<double>() * j / Np;
      double out[3];
      density(st * cos(phi), st * sin(phi), ct, out);
      Xgrid[0 * Ngrid + i * Np + j] = out[0];
      Xgrid[1 * Ngrid + i * Np + j] = out[1];
      Xgrid[2 * Ngrid + i * Np + j] = out[2];
    }
  }
  Vector<double> S;
  SphericalHarmonics<double>::Grid2VecSHC(Xgrid, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
  return S;
}

// Compare BIO vs. SH reference for one kernel across all three target placements.
// is_DL: the discontinuous double-layer PV equals the mean of the interior/exterior limits.
template <class Kernel, class DensityFn, class RefEvalFn>
void TestSphereBIOvsSH(const QuadElemList<double>& elem_lst, const Comm& comm,
                       const Kernel& ker, const char* kername, bool is_DL,
                       DensityFn density, RefEvalFn ref_eval, const double tol=1e-9, const double rel_tol=1e-5) {
  static constexpr Integer KDIM0 = Kernel::SrcDim();

  Vector<double> Xnodes;
  elem_lst.GetNodeCoord(&Xnodes, nullptr, nullptr);
  const Long Nnode = Xnodes.Dim() / 3;

  // Density at the cubed-sphere nodes (AoS).
  Vector<double> F(Nnode * KDIM0);
  for (Long i = 0; i < Nnode; i++) density(Xnodes[i*3+0], Xnodes[i*3+1], Xnodes[i*3+2], &F[i*KDIM0]);

  BoundaryIntegralOp<double, Kernel> BIOp(ker, /*trg_normal_dot_prod=*/false, comm);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(elem_lst);

  struct Cfg { const char* name; TgtType type; double scale; };
  const Cfg cfgs[3] = {
    {"on-surface", TgtType::OnSurface, 1.00}, // singular self-interaction
    {"near",       TgtType::Near,      1.02}, // near-singular correction
    {"far",        TgtType::Far,       2.00}, // smooth far-field
  };

  for (const auto& c : cfgs) {
    // Targets: surface nodes (on-surface) or nodes pushed radially outward (off-surface).
    Vector<double> Xtrg;
    if (c.type != TgtType::OnSurface) {
      Xtrg.ReInit(Nnode * 3);
      for (Long i = 0; i < Nnode * 3; i++) Xtrg[i] = Xnodes[i] * c.scale;
      BIOp.SetTargetCoord(Xtrg);
    }

    Vector<double> U_quad;
    BIOp.ComputePotential(U_quad, F);

    // SH reference at the same targets.
    const Vector<double>& coord = (c.type == TgtType::OnSurface) ? Xnodes : Xtrg;
    Vector<double> U_ref;
    if (c.type == TgtType::OnSurface && is_DL) {
      Vector<double> U_in, U_out; // PV = mean of the two one-sided limits
      ref_eval(coord, /*interior=*/true,  U_in);
      ref_eval(coord, /*interior=*/false, U_out);
      U_ref.ReInit(U_in.Dim());
      for (Long i = 0; i < U_ref.Dim(); i++) U_ref[i] = 0.5 * (U_in[i] + U_out[i]);
    } else {
      ref_eval(coord, /*interior=*/false, U_ref);
    }

    SCTL_ASSERT(U_quad.Dim() == U_ref.Dim());
    double err2 = 0, ref2 = 0;
    for (Long i = 0; i < U_ref.Dim(); i++) {
      const double e = U_quad[i] - U_ref[i];
      err2 += e * e; ref2 += U_ref[i] * U_ref[i];
    }
    err2 = GlobalReduce(err2, comm, CommOp::SUM); // targets are distributed across ranks
    ref2 = GlobalReduce(ref2, comm, CommOp::SUM);
    const double rel_l2 = sqrt(err2 / ref2);
    if (!comm.Rank()) std::cout << "  " << kername << " / " << c.name << " : rel L2 error = " << rel_l2 << std::endl;
    SCTL_ASSERT(rel_l2 < rel_tol); // geometry/quadrature-limited (~1e-7 observed on the high-order cubed-sphere)
  }
}

// Far-field quadrature weights must sum to the analytic sphere area 4 pi R^2.
void test_SurfaceArea(const QuadElemList<double>& elem_lst, double Radius, const Comm& comm = Comm::Self()) {
  Vector<double> wts, Xtemp, Xntemp, dist_far;
  Vector<Long> elem_wise_temp;
  elem_lst.GetFarFieldNodes(Xtemp, Xntemp, wts, dist_far, elem_wise_temp, 1);
  double Area = 0.;
  for (int i = 0; i < wts.Dim(); i++) Area += wts[i];
  Area = GlobalReduce(Area, comm, CommOp::SUM); // weights are distributed across ranks
  const double Area_exact = 4. * const_pi<double>() * Radius * Radius;
  if (!comm.Rank()) std::cout << "Area from Jacobian: " << Area << ", from formula: " << Area_exact << std::endl;
  SCTL_ASSERT(std::fabs(Area - Area_exact) / Area_exact < 1e-6);
  if (!comm.Rank()) std::cout << "Surface area test: PASSED" << std::endl;
}

// Stokes DL constant-density identity on a closed sphere: D[q] = c*q, |c|=1/2.
// Sign convention: this kernel (r = x_trg-x_src, source normal) gives c = -1/2 for
// an outward normal, +1/2 for inward; verify magnitude and sign vs. orientation.
void test_StokesDLIdentity(const QuadElemList<double>& elem_lst, const Comm& comm, bool check = true) {
  const Stokes3D_DxU ker_dl;
  BoundaryIntegralOp<double, Stokes3D_DxU> BIOp(ker_dl, /*trg_normal_dot_prod=*/false, comm);
  BIOp.SetAccuracy(1e-8);
  BIOp.AddElemList(elem_lst);
  BIOp.Setup();

  // Surface nodes/normals; orientation from sign(x.n) (x from sphere center).
  Vector<double> Xs, Xns;
  elem_lst.GetNodeCoord(&Xs, &Xns, nullptr);
  const Long Nnode = Xs.Dim() / 3;
  double xdotn = 0;
  for (Long i = 0; i < Nnode; i++) {
    xdotn += Xs[i*3+0]*Xns[i*3+0] + Xs[i*3+1]*Xns[i*3+1] + Xs[i*3+2]*Xns[i*3+2];
  }
  xdotn = GlobalReduce(xdotn, comm, CommOp::SUM); // nodes are distributed across ranks
  const bool outward = (xdotn > 0);
  const double c_expect = outward ? -0.5 : 0.5;

  // Constant density q = (1, 0, 0) at every node.
  Vector<double> q(Nnode * 3), U;
  for (Long i = 0; i < Nnode; i++) { q[i*3+0] = 1; q[i*3+1] = 0; q[i*3+2] = 0; }
  BIOp.ComputePotential(U, q);

  // D[q] should equal c*q = (c, 0, 0): measure mean U_x and max deviation.
  double sum_Ux = 0;
  for (Long i = 0; i < Nnode; i++) sum_Ux += U[i*3+0];
  const double cx_mean = GlobalReduce(sum_Ux, comm, CommOp::SUM) / GlobalReduce((double)Nnode, comm, CommOp::SUM);

  double max_dev = 0, max_perp = 0;
  for (Long i = 0; i < Nnode; i++) {
    max_dev  = std::max(max_dev,  std::fabs(U[i*3+0] - c_expect));
    max_perp = std::max(max_perp, std::max(std::fabs(U[i*3+1]), std::fabs(U[i*3+2])));
  }
  max_dev  = GlobalReduce(max_dev,  comm, CommOp::MAX);
  max_perp = GlobalReduce(max_perp, comm, CommOp::MAX);

  if (!comm.Rank())
    std::cout << "Stokes double-layer constant-density identity:\n"
              << "  normal orientation : " << (outward ? "outward" : "inward")
              << " (sum x.n = " << xdotn << ")\n"
              << "  mean U_x           : " << cx_mean << "  (expected " << c_expect << ")\n"
              << "  max |U_x - c|      : " << max_dev  << "\n"
              << "  max |U_perp|       : " << max_perp << std::endl;

  const double rel_tol = 1e-3; // dominated by the polynomial sphere-geometry error
  if (!check) return; // diagnostic mode: print only
  SCTL_ASSERT(std::fabs(cx_mean - c_expect) < rel_tol);
  SCTL_ASSERT(max_dev  < rel_tol);
  SCTL_ASSERT(max_perp < rel_tol);
  if (!comm.Rank())
    std::cout << "Stokes double-layer identity: PASSED (|c| = 1/2, sign tracks "
              << "outward normal -> -1/2)" << std::endl;
}

// BIOs vs. SH reference over {Laplace/Stokes x SL/DL} and {on-surface,near,far}.
// On the unit sphere the layer operators are diagonalized by SH, giving a spectral
// reference for a smooth non-polynomial density.
void test_BIOvsSH(const QuadElemList<double>& elem_lst, const Comm& comm, bool write_vtk = false, const double tol = 1e-9, const double rel_tol = 1e-5) {
  const Long p = 30; // SH truncation order (captures exp densities to ~eps)

  // Non-polynomial densities (analytic -> fast SH decay).
  auto lap_density = [](double x, double, double, double* o) { o[0] = std::exp(x); };
  auto sto_density = [](double x, double y, double z, double* o) {
    o[0] = std::exp(x); o[1] = std::exp(y); o[2] = std::exp(z);
  };

  // Density SH coefficients.
  const Vector<double> Slap = SphereScalarSHC(p, lap_density);
  const Vector<double> Ssto = SphereVecSHC(p, sto_density);

  if (!comm.Rank()) std::cout << "BIO vs. spherical-harmonics reference (density = exp):" << std::endl;

  // Visualize the (kernel-independent) near/self refinement on one element; tol matches
  // the SetAccuracy(1e-9) used in TestSphereBIOvsSH. Diagnostic only: rank 0 visualizes its
  // local element 0 with a self-communicator so the writers stay independent of other ranks.
  if (write_vtk && !comm.Rank()) {
    const Long evis = 0;                       // element to visualize
    const Integer ord = elem_lst.Order();
    // Near: a surface point pushed off along its outward normal.
    Vector<double> us{0.5}, vs{0.5}, Xc, Xn;
    elem_lst.GetGeom(&Xc, &Xn, nullptr, nullptr, nullptr, us, vs, evis);
    Vector<double> Xtrg(3);
    for (int k = 0; k < 3; k++) Xtrg[k] = Xc[k] + 0.02 * Xn[k];
    if (elem_lst.NearUsesRectPolar()) {
      elem_lst.WriteNearInteracRPVTK("near-interac-elem0", evis, Xtrg);
    } else {
      elem_lst.WriteNearInteracVTK("near-interac-elem0", evis, Xtrg, 1e-9, Comm::Self()); // tol = 1e-9 just for plotting
    }

    // Self: an interior node parameter.
    const auto& nds = QuadElemList<double>::ParamNodes(ord);
    const double u0 = nds[ord/2], v0 = nds[ord/2];
    if (elem_lst.SelfUsesRectPolar()) {
      elem_lst.WriteSelfInteracRPVTK("self-interac-elem0", evis, u0, v0);
    } else {
      elem_lst.WriteSelfInteracVTK("self-interac-elem0", evis, u0, v0, 1e-9, Comm::Self());
    }

    std::cout << "  wrote near-interac-elem0-* and self-interac-elem0-* VTK files" << std::endl;
  }

  Profile::Tic("Lap SL");
  TestSphereBIOvsSH(elem_lst, comm, Laplace3D_FxU(), "Laplace3D_FxU", /*is_DL=*/false, lap_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::LaplaceEvalSL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol, rel_tol);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "t_max", "f_avg", "f_max"});
  Profile::reset();

  Profile::Tic("Lap DL");
  TestSphereBIOvsSH(elem_lst, comm, Laplace3D_DxU(), "Laplace3D_DxU", /*is_DL=*/true, lap_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::LaplaceEvalDL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol, rel_tol);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "t_max", "f_avg", "f_max"});
  Profile::reset();

  if (!comm.Rank()) std::cout << "BIO vs. SH reference Laplace: PASSED" << std::endl;

  Profile::Tic("Stk SL");
  TestSphereBIOvsSH(elem_lst, comm, Stokes3D_FxU(), "Stokes3D_FxU", /*is_DL=*/false, sto_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::StokesEvalSL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol, rel_tol);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "t_max", "f_avg", "f_max"});
  Profile::reset();

  Profile::Tic("Stk DL");
  TestSphereBIOvsSH(elem_lst, comm, Stokes3D_DxU(), "Stokes3D_DxU", /*is_DL=*/true, sto_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::StokesEvalDL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); }, tol, rel_tol);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "t_max", "f_avg", "f_max"});
  Profile::reset();

  if (!comm.Rank()) std::cout << "BIO vs. SH reference Stokes: PASSED" << std::endl;
  SphericalHarmonics<double>::Clear();
}

// Surface area = sum of far-field quadrature weights (= integral of 1 dS).
double SurfaceAreaOf(const QuadElemList<double>& elem_lst, const Comm& comm = Comm::Self()) {
  Vector<double> wts, Xt, Xnt, dist_far;
  Vector<Long> ewt;
  elem_lst.GetFarFieldNodes(Xt, Xnt, wts, dist_far, ewt, 1);
  double A = 0.;
  for (Long i = 0; i < wts.Dim(); i++) A += wts[i];
  return GlobalReduce(A, comm, CommOp::SUM); // weights are distributed across ranks
}

// Largest nodal distance from the origin.
double MaxNodalRadius(const QuadElemList<double>& elem_lst, const Comm& comm = Comm::Self()) {
  Vector<double> X;
  elem_lst.GetNodeCoord(&X, nullptr, nullptr);
  double rmax = 0;
  for (Long i = 0; i < X.Dim() / 3; i++) {
    const double r = std::sqrt(X[i*3+0]*X[i*3+0] + X[i*3+1]*X[i*3+1] + X[i*3+2]*X[i*3+2]);
    rmax = std::max(rmax, r);
  }
  return GlobalReduce(rmax, comm, CommOp::MAX); // nodes are distributed across ranks
}

// Compare a gmsh-imported sphere against the analytic cubed-sphere TwistSphere(theta=0).
// The two meshes have entirely different node layouts, so we compare geometry-invariant
// quantities (surface area, bounding radius) and the BIO-vs-spherical-harmonics reference.
void test_GmshVsTwistSphere(const Comm& comm, const char* fname = "./sphere", const Long GmshOrder = 4, const double Radius = 1.0) {
  { std::ifstream is(fname); if (!is.good()) { if (!comm.Rank()) std::cout << "test_GmshVsTwistSphere: SKIPPED (mesh '" << fname << "' not found)\n"; return; } }

  const Long TwistOrder = 16, PatchPerFace = 5;
  QuadElemList<double> qel_gmsh  = GmshReader<double>::LoadQuadElemList(fname, GmshOrder, comm);
  qel_gmsh.SetQuadScheme(QuadElemList<double>::QuadScheme::RectPolar, 6, 512);
  
  Vector<double> Xtwist, Xntwist;
  qel_gmsh.GetNodeCoord(&Xtwist, &Xntwist, nullptr);
  qel_gmsh.WriteVTK("gmsh_sphere_mpi", Xntwist, comm);
  
  QuadElemList<double> qel_twist = BuildTwistedSphere<double>(TwistOrder, PatchPerFace, Radius, /*theta_twist=*/0., comm);
  qel_twist.SetQuadScheme(QuadElemList<double>::QuadScheme::RectPolar, 6, 512);
  const Long n_gmsh  = GlobalReduce(qel_gmsh.Size(),  comm, CommOp::SUM); // Size() is per-rank
  const Long n_twist = GlobalReduce(qel_twist.Size(), comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << "test_GmshVsTwistSphere: gmsh elems=" << n_gmsh << " (order " << GmshOrder << ")"
              << ", TwistSphere(theta=0) elems=" << n_twist << " (order " << TwistOrder << ")\n";

  double gmsh_tol = 3e-2;
  if (GmshOrder > 4) {
    gmsh_tol = 1e-6;
  }

  // --- Geometric invariant 1: surface area (integral of 1 dS) ---
  const double A_exact = 4. * const_pi<double>() * Radius * Radius;
  const double A_gmsh  = SurfaceAreaOf(qel_gmsh, comm);
  const double A_twist = SurfaceAreaOf(qel_twist, comm);
  if (!comm.Rank()) std::cout << "  surface area: gmsh=" << A_gmsh << ", TwistSphere=" << A_twist << ", exact(4 pi R^2)=" << A_exact << "\n";
  SCTL_ASSERT(std::fabs(A_twist - A_exact) / A_exact < 1e-8);   // high-order cubed-sphere
  SCTL_ASSERT(std::fabs(A_gmsh  - A_exact) / A_exact < gmsh_tol);   // linear (Q1) gmsh mesh, O(h^2)

  // --- Geometric invariant 2: bounding radius ---
  const double rmax_gmsh  = MaxNodalRadius(qel_gmsh, comm);
  const double rmax_twist = MaxNodalRadius(qel_twist, comm);
  if (!comm.Rank()) std::cout << "  max nodal radius: gmsh=" << rmax_gmsh << ", TwistSphere=" << rmax_twist << " (R=" << Radius << ")\n";
  SCTL_ASSERT(std::fabs(rmax_twist - Radius) < 1e-9);            // cubed-sphere nodes lie exactly on the sphere
  SCTL_ASSERT(rmax_gmsh <= Radius * (1 + 1e-9));                 // flat Q1 chords stay inside the sphere
  SCTL_ASSERT(std::fabs(rmax_gmsh - Radius) < gmsh_tol);

  // --- BIO vs spherical-harmonics reference on both geometries ---
  if (!comm.Rank()) std::cout << "  TwistSphere(theta=0) BIO-vs-SH:\n";
  test_BIOvsSH(qel_twist, comm, /*write_vtk=*/false, /*tol=*/1e-9, /*rel_tol=*/1e-5);
  if (!comm.Rank()) std::cout << "  gmsh-sphere BIO-vs-SH:\n";
  test_BIOvsSH(qel_gmsh, comm, /*write_vtk=*/false, /*tol=*/gmsh_tol, /*rel_tol=*/gmsh_tol);

  if (!comm.Rank()) std::cout << "test_GmshVsTwistSphere: PASSED" << std::endl;
}

// Manufactured-solution interior/exterior Dirichlet test via combined-field BIE + GMRES.
// Point sources are placed on the side OPPOSITE the solution domain so their field u_e
// is exact on the domain; sample u_e on the surface and solve
//      ( c*I + SL_scal*S + DL_scal*D ) sigma = u_e|_surface,
// with jump c = +-1/2*DL_scal (sign from outward normal and interior/exterior).
// SIGN REQUIREMENT: CFIE is uniquely solvable only for SAME sign (exterior) /
// OPPOSITE sign (interior) of (SL_scal, DL_scal); else the interior operator has a
// null space. quadr_tol also sets GMRES tol.
// Returns one rel-L2 per entry of eval_radii. The CFIE is solved ONCE (the dominant cost);
// only the post-solve evaluation is repeated per radius, so e.g. near+far share a single solve.
template <class KerSL, class KerDL>
std::vector<double> TestManufactured(const QuadElemList<double>& elem_lst, const Comm& comm,
                        const KerSL& ker_sl, const KerDL& ker_dl, const char* name,
                        const Vector<double>& Xsrc, const Vector<double>& Fsrc,
                        bool interior, const std::vector<double>& eval_radii, const double quadr_tol = 1e-9,
                        double SL_scal = 1.0, const double DL_scal = 1.0) {
  static constexpr Integer KDIM = KerSL::SrcDim(); // 1 (Laplace) or 3 (Stokes)

  // Surface nodes/normals; orientation sets the DL jump sign.
  Vector<double> Xs, Xns;
  elem_lst.GetNodeCoord(&Xs, &Xns, nullptr);
  const Long Nnode = Xs.Dim() / 3;
  double xdotn = 0;
  for (Long i = 0; i < Nnode; i++)
    xdotn += Xs[i*3+0]*Xns[i*3+0] + Xs[i*3+1]*Xns[i*3+1] + Xs[i*3+2]*Xns[i*3+2];
  // +1/2 for outward normal exterior trace; interior flips the sign.
  const double sgn = interior ? -1.0 : 1.0;
  const double jump = (xdotn > 0 ? 0.5 : -0.5) * DL_scal * sgn;

  // Interior with same-sign SL/DL has a null space; flip SL sign.
  if (interior && SL_scal*DL_scal > 0.) {
    if (!comm.Rank()) std::cout << "Warning: Interior problem has artificial null space when SL and DL same sign. Flipping SL sign. " << std::endl;
    SL_scal = -1.*SL_scal;
  }

  // Dirichlet data: point-source field at surface nodes (SL kernel ignores src normal).
  Vector<double> bc;
  ker_sl.Eval(bc, Xs, Xsrc, Xsrc, Fsrc);

  // Combined-field operator pieces (on-surface PV).
  BoundaryIntegralOp<double, KerSL> SLOp(ker_sl, /*trg_normal_dot_prod=*/false, comm);
  BoundaryIntegralOp<double, KerDL> DLOp(ker_dl, /*trg_normal_dot_prod=*/false, comm);
  SLOp.SetAccuracy(quadr_tol); DLOp.SetAccuracy(quadr_tol);
  SLOp.AddElemList(elem_lst); DLOp.AddElemList(elem_lst);

  const auto ApplyK = [&](Vector<double>* U, const Vector<double>& sigma) {
    Vector<double> Us, Ud;
    SLOp.ComputePotential(Us, sigma);
    DLOp.ComputePotential(Ud, sigma);
    if (U->Dim() != sigma.Dim()) U->ReInit(sigma.Dim());
    (*U) = SL_scal*Us + DL_scal*Ud + jump*sigma;
  };

  GMRES<double> solver(comm, false);
  Vector<double> sigma;
  Long iter = 0;
  const double gmres_tol = quadr_tol * 10.;
  const Long gmres_max_iter = 100;

  Profile::reset();
  Profile::Tic("gmres solve");
  solver(&sigma, ApplyK, bc, gmres_tol, gmres_max_iter, false, &iter);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "f_avg", "f/s_avg"});

  // Evaluate the recovered potential at each target sphere (radius-1 nodes scaled by the
  // radius). The solve above is reused; only target placement + the eval matvec change.
  std::vector<double> rel_l2s;
  rel_l2s.reserve(eval_radii.size());
  for (const double eval_radius : eval_radii) {
    Vector<double> Xtrg(Nnode * 3);
    for (Long i = 0; i < Nnode * 3; i++) Xtrg[i] = Xs[i] * eval_radius;
    SLOp.SetTargetCoord(Xtrg); DLOp.SetTargetCoord(Xtrg);

    Profile::reset();
    Profile::Tic("eval");
    Vector<double> Us, Ud;
    SLOp.ComputePotential(Us, sigma);
    DLOp.ComputePotential(Ud, sigma);
    Vector<double> U = SL_scal * Us + DL_scal * Ud;
    Profile::Toc();
    Profile::print(&comm, {"t_avg", "f_avg", "f/s_avg"});

    // Reference: point-source field evaluated directly at the targets.
    Vector<double> Uref;
    ker_sl.Eval(Uref, Xtrg, Xsrc, Xsrc, Fsrc);

    double err2 = 0, ref2 = 0;
    for (Long i = 0; i < U.Dim(); i++) { const double e = U[i] - Uref[i]; err2 += e*e; ref2 += Uref[i]*Uref[i]; }
    err2 = GlobalReduce(err2, comm, CommOp::SUM); // targets are distributed across ranks
    ref2 = GlobalReduce(ref2, comm, CommOp::SUM);
    const double rel_l2 = sqrt(err2 / ref2);
    if (!comm.Rank()) std::cout << "  " << name << " (R=" << eval_radius << ", GMRES iters = " << iter
              << ") : rel L2 error = " << rel_l2 << std::endl;
    rel_l2s.push_back(rel_l2);
  }
  return rel_l2s;
}

// Laplace CFIE Dirichlet manufactured solution: recover point-charge potential,
// for both exterior (charges inside) and interior (charges outside) problems.
void test_LaplaceManufactured(const QuadElemList<double>& elem_lst, const Comm& comm) {
  // const Vector<double> Fsrc{1.0, -0.7};
  const Vector<double> Fsrc{1.0, -1.0};

  // Exterior: charges inside the sphere; verify at near and far radius > 1.
  const Vector<double> Xsrc_ext{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10};
  if (!comm.Rank()) std::cout << "Manufactured solution (Laplace, exterior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radii=*/{1.001})[0] < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radii=*/{2.000})[0] < 1e-5);

  // Interior: charges outside the sphere; verify at near/far radius < 1.
  // Interior CFIE needs opposite-sign SL/DL, so SL_scal = -1.
  const Vector<double> Xsrc_int{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60};
  if (!comm.Rank()) std::cout << "Manufactured solution (Laplace, interior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radii=*/{0.999},
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0)[0] < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radii=*/{0.500},
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0)[0] < 1e-5);

  if (!comm.Rank()) std::cout << "Laplace manufactured-solution test: PASSED" << std::endl;
}

// Stokes CFIE Dirichlet manufactured solution: recover Stokeslet velocity field,
// for both exterior (Stokeslets inside) and interior (outside); net force nonzero.
void test_StokesManufactured(const QuadElemList<double>& elem_lst, const Comm& comm) {
  // const Vector<double> Fsrc{1.0, 0.5, -0.3,  -0.4, 0.2, 0.1};
  const Vector<double> Fsrc{1.0, 0.5, -0.3,  -1.0, -0.5, 0.3};

  // Exterior: Stokeslets inside the sphere.
  const Vector<double> Xsrc_ext{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10};
  if (!comm.Rank()) std::cout << "Manufactured solution (Stokes, exterior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radii=*/{1.001})[0] < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radii=*/{2.000})[0] < 1e-5);

  // Interior: Stokeslets outside; interior CFIE needs opposite-sign SL/DL, so SL_scal = -1.
  const Vector<double> Xsrc_int{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60};
  if (!comm.Rank()) std::cout << "Manufactured solution (Stokes, interior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radii=*/{0.999},
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0)[0] < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radii=*/{0.500},
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0)[0] < 1e-5);

  if (!comm.Rank()) std::cout << "Stokes manufactured-solution test: PASSED" << std::endl;
}

// h-refinement convergence: fixed ElemOrder, increasing PatchPerFace, report
// manufactured-solution rel-L2 at near/far targets per resolution.
void test_ManufacturedConvergence(const Comm& comm,
                                  bool interior = false,
                                  int scheme_ = 0, // 0: adaptive, 1: rect-polar, 2: hybrid
                                  const double theta_twist = 0.,
                                  const std::vector<Long>& PatchPerFaceList = {1, 2, 3, 4, 5},
                                  Long ElemOrder = 16
                                  ) {
  const double Radius = 1.0;
  const double base_tol = 1e-13; // should be this level for ElemOrder = 12, maybe allowed higher if lower order..

  // Laplace charges / Stokeslets outside the sphere (interior problem).
  const Vector<double> Fsrc_lap{1.0, -0.7};
  const Vector<double> Fsrc_sto{1.0, 0.5, -0.3,  -0.4, 0.2, 0.1};
  const Vector<double> src_ext{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10}; // for ext-erior problem
  const Vector<double> src_int{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60}; // for int-erior problem
  const double Rint_far = 0.5;
  const double Rint_near = 0.999;
  const double Rext_far = 2.;
  const double Rext_near = 1.001;
  Vector<double> Xsrc_lap, Xsrc_sto;
  double R_far, R_near, SL_scal, DL_scal;
  std::string name;
  if (!interior) { // exterior problem, DL+SL
    Xsrc_lap = src_ext;
    Xsrc_sto = src_ext;
    R_far = Rext_far;
    R_near = Rext_near;
    SL_scal = 1.0;
    DL_scal = 1.0;
    name = "DL+SL";
  } else { // interior problem, DL only
    Xsrc_lap = src_int;
    Xsrc_sto = src_int;
    R_far = Rint_far;
    R_near = Rint_near;
    SL_scal = 0.0;
    DL_scal = 1.0;
    name = "DL";
  }

  if (!comm.Rank()) {
    std::cout << "\nManufactured-solution convergence study (ElemOrder = " << ElemOrder << "):\n";
    std::cout << std::scientific;
    std::cout << "  kernel    PatchPerFace  Nelem   rel-L2 (near R=" << R_near <<")   rel-L2 (far R="<<R_far<<")\n";
  }
  for (const Long PatchPerFace : PatchPerFaceList) {
    QuadElemList<double> elem_lst = BuildTwistedSphere<double>(ElemOrder, PatchPerFace, Radius, theta_twist, comm);
    if (scheme_ == 1) {
      elem_lst.SetQuadScheme(QuadElemList<double>::QuadScheme::RectPolar);
    } else if (scheme_ == 2) {
      elem_lst.SetQuadScheme(QuadElemList<double>::QuadScheme::Hybrid);
    }
    const Long Nelem = 6 * PatchPerFace * PatchPerFace;

    double quadr_tol = base_tol;
    if (PatchPerFace > 5) {
      quadr_tol *= 0.0001;
    } else if (PatchPerFace > 3) {
      quadr_tol *= 0.01;
    }

    // Single solve per kernel, evaluated at both radii (near first, then far).
    const std::vector<double> el = TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                             ("Laplace "+name).c_str(), Xsrc_lap, Fsrc_lap, interior, {R_near, R_far}, quadr_tol, SL_scal, DL_scal);
    const double el_near = el[0], el_far = el[1];
    if (!comm.Rank()) std::cout << "  Laplace   " << std::setw(12) << PatchPerFace << "  " << std::setw(5) << Nelem
              << "   " << el_near << "        " << el_far << "\n";

    const std::vector<double> es = TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                             ("Stokes "+name).c_str(), Xsrc_sto, Fsrc_sto, interior, {R_near, R_far}, quadr_tol, SL_scal, DL_scal);
    const double es_near = es[0], es_far = es[1];
    if (!comm.Rank()) std::cout << "  Stokes    " << std::setw(12) << PatchPerFace << "  " << std::setw(5) << Nelem
              << "   " << es_near << "        " << es_far << "\n";
  }
  if (!comm.Rank()) std::cout << "Manufactured-solution convergence study: DONE" << std::endl;
}

// Nbeta (RectPolar cov_order) sweep on the maximally twisted sphere, Stokes kernel only.
// For fixed ElemOrder/twist, increase Nbeta (GL points per direction in the rectangular-polar
// COV) at each surface resolution (PatchPerFace) and record manufactured-solution rel-L2
// (near + far) plus wall-clock solve+eval time. Writes a formatted table to Nbeta_benchmark.txt.
void test_NbetaSweep(const Comm& comm,
                     const std::vector<Long>& NbetaList = {32, 64, 96, 128, 192, 256, 384, 512},
                     const std::vector<Long>& PatchPerFaceList = {5, 10},
                     Long ElemOrder = 16,
                     double theta_twist = const_pi<double>()) {
  const double Radius = 1.0;
  const double quadr_tol = 1e-13; // tight, so Nbeta (not adaptive/GMRES tol) is the bottleneck

  // Exterior Stokes DL+SL manufactured solution (Stokeslets inside the sphere).
  const Vector<double> Fsrc_sto{1.0, 0.5, -0.3,  -0.4, 0.2, 0.1};
  const Vector<double> src_ext{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10};
  const double R_near = 1.001, R_far = 2.0;
  const double SL_scal = 1.0, DL_scal = 1.0;
  const bool interior = false;

  const bool root = !comm.Rank();
  std::ofstream ofs; // only rank 0 writes the table file (avoids a multi-rank write race)
  if (root) {
    ofs.open("Nbeta_sweep.txt");
    ofs << std::scientific;
    ofs << "# Nbeta (RectPolar cov_order) sweep: Stokes DL+SL exterior manufactured solution\n";
    ofs << "# ElemOrder=" << ElemOrder << ", theta_twist=" << theta_twist
        << " (pi=" << const_pi<double>() << "), quadr_tol=" << quadr_tol << "\n";
    ofs << "# columns: PatchPerFace  Nbeta  Nelem  rel-L2(near R=" << R_near
        << ")  rel-L2(far R=" << R_far << ")  t_solve+eval(s)\n";

    std::cout << "\nNbeta sweep (RectPolar, Stokes, twisted sphere) -> Nbeta_benchmark.txt\n";
    std::cout << std::scientific;
  }

  for (const Long PatchPerFace : PatchPerFaceList) {
    const Long Nelem = 6 * PatchPerFace * PatchPerFace;
    if (root) {
      ofs << "# --- PatchPerFace = " << PatchPerFace << " (Nelem = " << Nelem << ") ---\n";
      std::cout << "# --- PatchPerFace = " << PatchPerFace << " (Nelem = " << Nelem << ") ---\n";
    }
    for (const Long Nbeta : NbetaList) {
      QuadElemList<double> elem_lst = BuildTwistedSphere<double>(ElemOrder, PatchPerFace, Radius, theta_twist, comm);
      elem_lst.SetQuadScheme(QuadElemList<double>::QuadScheme::RectPolar, /*q=*/6, /*cov_order=*/Nbeta);

      const auto t0 = std::chrono::high_resolution_clock::now();
      const std::vector<double> es = TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                               "Stokes DL+SL", src_ext, Fsrc_sto, interior, {R_near, R_far},
                               quadr_tol, SL_scal, DL_scal);
      const auto t1 = std::chrono::high_resolution_clock::now();
      const double elapsed = std::chrono::duration<double>(t1 - t0).count();
      const double es_near = es[0], es_far = es[1];

      if (root) {
        ofs << "  " << std::setw(12) << PatchPerFace << "  " << std::setw(5) << Nbeta
            << "  " << std::setw(6) << Nelem << "   " << es_near << "   " << es_far
            << "   " << elapsed << std::endl; // flush each row in case a heavy case crashes
        std::cout << "  Nbeta=" << std::setw(5) << Nbeta << "  rel-L2(near)=" << es_near
                  << "  rel-L2(far)=" << es_far << "  t=" << elapsed << "s\n";
      }
    }
  }
  if (root) std::cout << "Nbeta sweep: DONE" << std::endl;
}

}

//  ============= Timing ===================
void test_timing_StkSL(const QuadElemList<double>& elem_lst, const Comm& comm, const double tol = 1e-9) {
  // Non-polynomial densities (analytic -> fast SH decay).
  auto sto_density = [](double x, double y, double z, double* o) {
    o[0] = std::exp(x); o[1] = std::exp(y); o[2] = std::exp(z);
  };

  const Stokes3D_FxU ker_FxU;

  static constexpr Integer KDIM0 = Stokes3D_FxU::SrcDim();

  Vector<double> Xnodes, Xnnodes;
  elem_lst.GetNodeCoord(&Xnodes, &Xnnodes, nullptr);
  const Long Nnode = Xnodes.Dim() / 3;

  // Density at the cubed-sphere nodes (AoS).
  Vector<double> F(Nnode * KDIM0);
  for (Long i = 0; i < Nnode; i++) sto_density(Xnodes[i*3+0], Xnodes[i*3+1], Xnodes[i*3+2], &F[i*KDIM0]);

  BoundaryIntegralOp<double, Stokes3D_FxU> BIOp(ker_FxU, /*trg_normal_dot_prod=*/false, comm);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(elem_lst);

  Vector<double> Xtrg = Xnodes;
  Xtrg += 1e-6 * Xnnodes;
  BIOp.SetTargetCoord(Xtrg);

  Vector<double> U_quad;
  Profile::Tic("BIO eval near");
  BIOp.ComputePotential(U_quad, F);
  Profile::Toc();
  Profile::print(&comm, {"t_max", "f_max", "f/s_avg"});

}


// Double-layer constant-density identity on a closed surface (Laplace or Stokes):
// D[q] = c*q for constant q, with c = -1/2 for the outward-normal convention used here.
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

  // D[q] should equal c*q: measure max deviation.
  Vector<Real> cx_maxerr(KDIM0);
  cx_maxerr = 0.;
  for (Long i = 0; i < Nnode; i++) {
    for (Long k=0; k<KDIM0; k++) {
      cx_maxerr[k] = std::max(cx_maxerr[k], std::fabs(U[i*KDIM0+k] / q[i*KDIM0+k] - c_expect));
    }
  }
  Real cx_relerr_avg = 0.;
  for (Long k=0; k<KDIM0; k++) cx_relerr_avg += (cx_maxerr[k] / std::fabs(c_expect));
  cx_relerr_avg /= KDIM0;
  cx_relerr_avg = GlobalReduce(cx_relerr_avg, comm, CommOp::MAX);
  if (!comm.Rank()) std::cout << std::setprecision(8) << "DL constant-density identity: max relative error = " << cx_relerr_avg << std::endl;
}

// Interior Green's representation identity on a closed surface (Laplace or Stokes):
// for a source X0 OUTSIDE the surface (u harmonic/Stokeslet in the interior),
// (S[Fs] - D[Fd]) - 0.5*Fd == u|_S, with Fd = u, Fs = +du/dn (outward normal).
//
// trg_dist == 0 : on-surface (self-eval) targets = surface nodes; apply the -0.5 DL jump.
// trg_dist  > 0 : off-surface INTERIOR targets, pushed in by trg_dist along the inward normal
//                 (exercises the NEAR-interaction path). The near-singular quadrature returns the
//                 true off-surface D[u] (interior limit included), so no manual jump is applied and
//                 (S[Fs] - D[Fd]) == u at the interior targets directly.
template <class Real, class KerSL, class KerDL, class KerGrad> void test_greens_identity(const QuadElemList<Real>& elem_lst, const Comm& comm,
                          const Real tol, const Vector<Real> X0, const Real trg_dist = 0, const bool center_only = false) {
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
    if (center_only && trg_dist > 0) {
      // One target per panel at its parametric center (0.5,0.5), pushed inward. Being far from all
      // panel edges, these avoid the edge-near adjacent-panel regime -- isolates the panel-interior
      // near accuracy of the scheme.
      const Long Ne = elem_lst.Size();
      Xtrg.ReInit(Ne*COORD_DIM);
      const Vector<Real> up05{(Real)0.5}, vp05{(Real)0.5};
      for (Long e = 0; e < Ne; e++) {
        Vector<Real> Xc, Nc;
        elem_lst.GetGeom(&Xc, &Nc, nullptr, nullptr, nullptr, up05, vp05, e);
        for (Integer k = 0; k < COORD_DIM; k++) Xtrg[e*COORD_DIM+k] = Xc[k] - trg_dist*Nc[k];
      }
    } else if (trg_dist > 0) {
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

  // Warm-up (builds the near/self operators), then a timed rebuild+eval via the sctl profiler so
  // the near-interaction assembly cost (which differs per scheme) is captured under "Greens-SetupEval".
  BIOpSL.ComputePotential(Us,Fs);
  BIOpDL.ComputePotential(Ud,Fd);
  BIOpSL.ClearSetup(); BIOpDL.ClearSetup();
  Us = 0; Ud = 0;
  sctl::Profile::Enable(true);
  Profile::Tic("Greens-SetupEval", &comm);
  BIOpSL.ComputePotential(Us,Fs);
  BIOpDL.ComputePotential(Ud,Fd);
  Profile::Toc();

  if (trg_dist == 0) Ud -= 0.5*Fd; // DL jump condition, on-surface only (off-surface D[u] already includes it)
  Vector<Real> Uerr = (Us - Ud) - Uref;
  { // Print error
    StaticArray<Real,2> max_err{0,0};
    StaticArray<Real,2> max_val{0,0};
    for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
    for (auto x : Uref) max_val[0] = std::max<Real>(max_val[0], fabs(x));
    comm.Allreduce(max_err+0, max_err+1, 1, CommOp::MAX);
    comm.Allreduce(max_val+0, max_val+1, 1, CommOp::MAX);
    if (!pid) std::cout<<"Green's identity error = "<<max_err[1]/max_val[1]<<'\n';
  }
  sctl::Profile::print(&comm, {"t_max"});
  sctl::Profile::reset();
  sctl::Profile::Enable(false);
}


int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;

  {
    const Comm comm = Comm::World();
    // Distributed run: every rank builds the full geometry and BuildTwistedSphere /
    // GmshReader::LoadQuadElemList keep only this rank's contiguous element slice
    // (replicate-then-slice). BoundaryIntegralOp handles all cross-rank communication.

    // Profile::Enable(true);

    // gmsh import pipeline vs. analytic cubed-sphere (geometry invariants + BIO-vs-SH).
    // Coarse ./sphere mesh: low resolution, so resample its panels to QuadOrder 4.
    // test_GmshVsTwistSphere(comm, "./sphere", 4);
    // test_GmshVsTwistSphere(comm, "./sphere_ord9", 16);

    // test_NbetaSweep(comm);

// #if 0
    const Long ElemOrder = 16;
    const Long PatchPerFace = 5;
    const double Radius = 1.0;

    // === Near-quadrature schemes: on-surface DL + Green's identity, and off-surface interior near ===
    // On-surface (targets = nodes) exercises the self path + adjacent-panel EDGE near; off-surface
    // center targets exercise the panel-INTERIOR near path. Adaptive (closest-point near) and
    // RectPolar match at the surface-resolution floor (~5e-8 on-surface, ~5e-7 interior). hedgehog
    // matches on interior targets but is inaccurate near seams, so it is shown for interior only.
    {
      using QS = QuadElemList<double>::QuadScheme;
      const Vector<double> X0{1.3, 1.2, 0.2}; // exterior source for the interior Green's identity
      const Long EO = 12, PPF = 2; const double ctol = 1e-7; // 24-patch order-12 sphere
      auto make = [&](QS s) {
        auto e = BuildTwistedSphere<double>(EO, PPF, Radius, 0., comm);
        if      (s == QS::RectPolar) e.SetQuadScheme(QS::RectPolar, 6, 512, 30);
        else if (s == QS::LineQBX)   { e.SetQuadScheme(QS::LineQBX); e.SetLineQBXParams(); } // R=r=0.02L,p=16,eta=2,up=72
        else                         e.SetQuadScheme(QS::Adaptive, 6, 0, 30);
        return e;
      };

      if (!comm.Rank()) std::cout << "\n=== On-surface DL + Green's identity (order-" << EO << ", " << PPF << " patch/face) ===\n";
      for (const auto& c : {std::make_pair("Adaptive", QS::Adaptive), std::make_pair("RectPolar", QS::RectPolar)}) {
        auto elem = make(c.second);
        if (!comm.Rank()) std::cout << "\n--- " << c.first << " (on-surface) ---\n";
        if (!comm.Rank()) std::cout << "[Laplace DL] "; test_DLIdentity<double, Laplace3D_DxU>(elem, comm, ctol);
        if (!comm.Rank()) std::cout << "[Stokes  DL] "; test_DLIdentity<double, Stokes3D_DxU>(elem, comm, ctol);
        if (!comm.Rank()) std::cout << "[Laplace Green's] ";
        test_greens_identity<double, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem, comm, ctol, X0, /*trg_dist=*/0.);
        if (!comm.Rank()) std::cout << "[Stokes  Green's] ";
        test_greens_identity<double, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem, comm, ctol, X0, /*trg_dist=*/0.);
      }

      if (!comm.Rank()) std::cout << "\n=== Off-surface interior near: Stokes Green's identity @ trg_dist=1e-4 ===\n";
      for (const auto& c : {std::make_pair("Adaptive", QS::Adaptive), std::make_pair("RectPolar", QS::RectPolar), std::make_pair("hedgehog", QS::LineQBX)}) {
        auto elem = make(c.second);
        if (!comm.Rank()) std::cout << "\n--- " << c.first << " (interior near) ---\n";
        test_greens_identity<double, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem, comm, ctol, X0, /*trg_dist=*/1e-4, /*center_only=*/true);
      }
    }
    // QuadElemList<double> elem_lst = BuildTwistedSphere<double>(ElemOrder, PatchPerFace, Radius, 0., comm);

    // if (!comm.Rank()) std::cout << "\n=== Scheme 1: Adaptive subdivision of panels ===" << std::endl;
    // if (!comm.Rank()) std::cout << "------ Quadr and BIO tests for regular sphere -------" << std::endl;
    // test_SurfaceArea(elem_lst, Radius, comm);
    // test_StokesDLIdentity(elem_lst, comm);
    // test_BIOvsSH(elem_lst, comm, true);
    // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Exterior] ------" << std::endl;
    // test_ManufacturedConvergence(comm);
    // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Interior] ------" << std::endl;
    // test_ManufacturedConvergence(comm, true);
    // if (!comm.Rank()) std::cout << "------ Profile BIO compute potential at near target, regular sphere. ------ " << std::endl;
    // test_timing_StkSL(elem_lst, comm, 1e-12);

    // std::cout << "------ Quadr and BIO tests for twisted sphere -------" << std::endl;
    const Long ElemOrder_twisted = 16;
    const Long PatchPerFace_twisted = 5;
    // Small twist
    double theta_twist = const_pi<double>() / 6.;
    // QuadElemList<double> elem_lst_twist = BuildTwistedSphere<double>(ElemOrder_twisted, PatchPerFace_twisted, Radius, theta_twist);
    // test_SurfaceArea(elem_lst_twist, Radius);
    // test_StokesDLIdentity(elem_lst_twist, comm);
    // test_BIOvsSH(elem_lst_twist, comm, false, 1e-14);
    // std::cout << "------- Manufactured solutions test [Exterior] ------" << std::endl;
    // test_ManufacturedConvergence(comm, false, 0, theta_twist, {1,2,3}, 12);

    // // Moderate twist
    // theta_twist = const_pi<double>() / 2.;
    // QuadElemList<double> elem_lst_twist2 = BuildTwistedSphere<double>(ElemOrder_twisted, PatchPerFace_twisted, Radius, theta_twist);
    // test_SurfaceArea(elem_lst_twist2, Radius);
    // test_StokesDLIdentity(elem_lst_twist2, comm);
    // test_BIOvsSH(elem_lst_twist2, comm, false, 1e-14);
    // std::cout << "------- Manufactured solutions test [Exterior] ------" << std::endl;
    // test_ManufacturedConvergence(comm, false, 0, theta_twist, {1,2,3,4,5}, 12);

    // // Large twist
    // theta_twist = const_pi<double>();
    // QuadElemList<double> elem_lst_twist3 = BuildTwistedSphere<double>(ElemOrder_twisted, PatchPerFace_twisted, Radius, theta_twist, comm);
    // Vector<double> Xtwist, Xntwist;
    // elem_lst_twist3.GetNodeCoord(&Xtwist, &Xntwist, nullptr);
    // elem_lst_twist3.WriteVTK("twisted3_sphere_mpi", Xntwist, comm); // one .vtu per rank + rank-0 .pvtu master
    // test_SurfaceArea(elem_lst_twist3, Radius, comm);
    // test_StokesDLIdentity(elem_lst_twist3, comm);
    // test_BIOvsSH(elem_lst_twist3, comm, false, 1e-14);
    // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Exterior] ------" << std::endl;
    // test_ManufacturedConvergence(comm, false, 0, theta_twist, {3,4,5}, 16);


    // // --- Scheme 2: rectangular-polar COV (Bruno 2018) for near/self interactions ---
    // if (!comm.Rank()) std::cout << "\n=== Scheme 2: rectangular-polar change of variable ===" << std::endl;
    // // Profile::Enable(true);
    // QuadElemList<double> elem_lst_rp = BuildTwistedSphere<double>(ElemOrder, PatchPerFace, Radius, 0., comm);
    // elem_lst_rp.SetQuadScheme(QuadElemList<double>::QuadScheme::RectPolar, 6, 256);
    // test_SurfaceArea(elem_lst_rp, Radius, comm);
    // test_StokesDLIdentity(elem_lst_rp, comm);
    // test_BIOvsSH(elem_lst_rp, comm);
    // test_LaplaceManufactured(elem_lst_rp, comm);
    // test_StokesManufactured(elem_lst_rp, comm);
    // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Exterior] ------" << std::endl;
    // test_ManufacturedConvergence(comm, false, 1);
    // // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Interior] ------" << std::endl;
    // // test_ManufacturedConvergence(comm, true, 1);
    // if (!comm.Rank()) std::cout << "------ Profile BIO compute potential at near target, R-P scheme, regular sphere. ------ " << std::endl;
    // test_timing_StkSL(elem_lst_rp, comm, 1e-12);

    // elem_lst_rp = BuildTwistedSphere<double>(ElemOrder_twisted, PatchPerFace_twisted, Radius, theta_twist, comm);
    // elem_lst_rp.SetQuadScheme(QuadElemList<double>::QuadScheme::RectPolar);
    // test_SurfaceArea(elem_lst_rp, Radius, comm);
    // test_StokesDLIdentity(elem_lst_rp, comm);
    // test_BIOvsSH(elem_lst_rp, comm);
    // test_LaplaceManufactured(elem_lst_rp, comm);
    // test_StokesManufactured(elem_lst_rp, comm);
    // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Exterior] ------" << std::endl;
    // test_ManufacturedConvergence(comm, false, 1, theta_twist);
    // // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Interior] ------" << std::endl;
    // // test_ManufacturedConvergence(comm, true, 1, theta_twist);
    // if (!comm.Rank()) std::cout << "------ Profile BIO compute potential at near target, R-P scheme, twisted sphere. ------ " << std::endl;
    // test_timing_StkSL(elem_lst_rp, comm, 1e-12);

    // if (!comm.Rank()) std::cout << "\n=== Scheme 3: Hybrid ===" << std::endl;
    // if (!comm.Rank()) std::cout << "------ Quadr and BIO tests for regular sphere -------" << std::endl;
    // elem_lst.SetQuadScheme(QuadElemList<double>::QuadScheme::Hybrid, 6, 512);
    // test_SurfaceArea(elem_lst, Radius, comm);
    // test_StokesDLIdentity(elem_lst, comm);
    // test_BIOvsSH(elem_lst, comm, true);
    // if (!comm.Rank()) std::cout << "------- Manufactured solutions test [Exterior] ------" << std::endl;
    // test_ManufacturedConvergence(comm, false, 2);

// #endif

  }

  Comm::MPI_Finalize();
  return 0;
}
