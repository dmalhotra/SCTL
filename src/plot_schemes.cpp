// plot_schemes.cpp -- dump the self- AND near-interaction singular quadrature layout of the
// Adaptive, RectPolar and Duffy schemes on a single flat order-8 panel, for inspection in ParaView.
//
// One tolerance (1e-5) fixes the scheme knobs via the bench TolLadder preset: Nbeta = 48
// (RectPolar GL points per direction) and max_depth = 4 (Adaptive centered-graded u-depth cap).
// All the geometry / VTK writers used here already exist on QuadElemList.
//
// The singular point (u0,v0) is one interior tensor node of the panel, shared by both schemes.
// Per scheme it writes:
//   <scheme>-self-elem0            self quadrature nodes  (VTK_VERTEX cloud / warped QUAD mesh)
//   <scheme>-self-elem0-singpt     the on-surface singular point
// plus once, the bare panel grid:  panel-grid            (order-8 tensor GL mesh)
//
// Writer used per scheme's self path:
//   Adaptive  -> WriteSelfInteracVTK    (centered graded-GL u x Alpert log-singular v; uses tol/max_depth)
//   RectPolar -> WriteSelfInteracRPVTK  (rectangular-polar COV grid clustered at the singular point; uses Nbeta)

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <iostream>

using namespace sctl;

template <class Real> void plot_schemes() {
  using QS = typename QuadElemList<Real>::QuadScheme;
  const Integer order = 8;
  const Long evis = 0;

  // tol=1e-5 preset (bench-scheme-compare TolLadder): Nbeta=48, max_depth=4.
  const Real tol = (Real)1e-5;
  const Integer Nbeta = 48, max_depth = 4;

  // Flat panel z = 0, order 8, single element.
  Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);

  // Singular point at an interior tensor node (4th x-node, 5th y-node, 1-based), shared by both.
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Real u0 = nds[3], v0 = nds[4];

  // Near target: 0.02 off the surface along the normal at (u0,v0) -- shared by both schemes.
  Vector<Real> Xtrg(3);
  {
    QuadElemList<Real> qel(order, coord0);
    Vector<Real> up{u0}, vp{v0}, Xc, Xn;
    qel.GetGeom(&Xc, &Xn, nullptr, nullptr, nullptr, up, vp, evis);
    for (Integer k = 0; k < 3; k++) Xtrg[k] = Xc[k] + (Real)0.02 * Xn[k];
    qel.WriteVTK("panel-grid", Vector<Real>(), Comm::Self());
  }

  {
    QuadElemList<Real> qel(order, coord0);
    qel.SetQuadScheme(QS::Adaptive, /*q=*/6, /*cov_order=Nbeta*/ Nbeta, /*max_depth*/ max_depth);
    // Paneled self (GL panels + Alpert singular v-row), matching the near paneling.
    qel.WriteSelfInteracGradedVTK("adaptive-self-elem0", evis, u0, v0, tol, Comm::Self());
    // Production adaptive near = foot-graded separable tensor grid (BuildNearTensorRule),
    // NOT the superseded isotropic quadtree of WriteNearInteracVTK.
    qel.WriteNearInteracGradedVTK("adaptive-near-elem0", evis, Xtrg, tol, Comm::Self());
    std::cout << "  wrote adaptive-self-elem0-* and adaptive-near-elem0-* VTK files\n";
  }
  {
    QuadElemList<Real> qel(order, coord0);
    qel.SetQuadScheme(QS::RectPolar, /*q=*/6, /*cov_order=Nbeta*/ Nbeta, /*max_depth*/ max_depth);
    qel.WriteSelfInteracRPVTK("rectpolar-self-elem0", evis, u0, v0, Nbeta, Comm::Self());
    qel.WriteNearInteracRPVTK("rectpolar-near-elem0", evis, Xtrg, Nbeta, Comm::Self());
    std::cout << "  wrote rectpolar-self-elem0-* and rectpolar-near-elem0-* VTK files\n";
  }
  {
    QuadElemList<Real> qel(order, coord0);
    qel.SetQuadScheme(QS::Duffy, /*q=*/6, /*cov_order=Nbeta*/ Nbeta, /*max_depth*/ max_depth);
    // Duffy self = edge-collapsed (sinh) rule on four target-anchored triangles fanning from (u0,v0);
    // Duffy near = split-at-foot cells with the anisotropic u/v refinement ladder. Both take tol
    // (same 1e-5 preset), which sets the sinh t-order and the near QuadOrder/b_ellipse respectively.
    qel.WriteSelfInteracDuffyVTK("duffy-self-elem0", evis, u0, v0, tol, Comm::Self());
    qel.WriteNearInteracDuffyVTK("duffy-near-elem0", evis, Xtrg, tol, Comm::Self());
    std::cout << "  wrote duffy-self-elem0-* and duffy-near-elem0-* VTK files\n";
  }
  std::cout << "  wrote panel-grid-* VTK files\n";
}

int main() {
  std::cout << "plot_schemes: self+near quadrature, order-8 flat panel, tol=1e-5 (Nbeta=48, max_depth=4)\n";
  plot_schemes<double>();
  return 0;
}
