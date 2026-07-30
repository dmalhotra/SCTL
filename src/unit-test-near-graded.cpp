/**
 * Unit tests for QuadElemList's ADAPTIVE near quadrature (QuadScheme::Adaptive, and the near phase
 * of Hybrid): a separable FOOT-GRADED tensor rule. Each side of [0,1] is split AT the closest point
 * (foot) and graded geometrically outward at ratio r = b/(1+b) down to an innermost width of
 * dist/(b*L_phys); the quadrature is the full tensor product, so the interpolation is one large
 * tensor multiply per side rather than a small GEMM set per cell of a 2D tree.
 *
 * What is checked:
 *   1. rule structure   -- nodes in [0,1], positive weights, sum(w) == 1 per direction, and every
 *                          segment admissible under the OFF-SURFACE effective distance
 *                          rho = sqrt(pdist^2 + (dist/L_phys)^2). The foot-touching segment has
 *                          pdist = 0, so it satisfies no purely in-surface criterion -- which is
 *                          exactly why this rule is valid only for off-surface targets, and why
 *                          this test is the one that would catch it being reused for the self path.
 *   2. vs upsampled ref -- direct nsub=100 quadrature at moderate near distance d=1e-2, on a flat
 *                          and a curved patch, for Stokes FxU / Stokes DxU / Laplace FxU.
 *   3. deep near        -- flat panel at d in {1e-2,1e-3,1e-4} vs a RectPolar-512 gold, with the
 *                          RP-400-vs-RP-512 self-consistency printed so a row where the gold itself
 *                          is unreliable is visible.
 *   4. convergence      -- in tol (digits) and in max_depth.
 *
 * Self-contained; does NOT touch src/unit-test-quad-element.cpp or src/test-quad-elem.cpp.
 *
 * Build & run:
 *     . ./sctl_source
 *     make near-graded                                  (-> bin/unit-test-near-graded)
 *     OMP_NUM_THREADS=1 ./bin/unit-test-near-graded
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <cstdio>
#include <utility>
#include <vector>

using namespace sctl;

// Friend shim: forwards to QuadElemList's private near helpers (must live in namespace sctl).
namespace sctl {
template <class Real> struct QuadElemTestAccess {
  // NearFootAndDepth with the off-surface parameter distance h = dist/L_phys exposed.
  static Integer NearFootAndDepth(Real& us, Real& vs, Real& dist, Real& h, const QuadElemList<Real>& qel,
                                  const Long e, const Vector<Real>& Xt, const Real b, const Integer md) {
    return QuadElemList<Real>::NearFootAndDepth(us, vs, dist, qel, e, Xt, b, md, &h);
  }
  static Integer BuildNearTensorRule(Vector<Real>& up, Vector<Real>& wu, Vector<Real>& vp, Vector<Real>& wv,
                                     Vector<Real>* useg, Vector<Long>* ud, Vector<Real>* vseg, Vector<Long>* vd,
                                     const QuadElemList<Real>& qel, const Long e, const Vector<Real>& Xt,
                                     const Real b, const Vector<Real>& qn, const Vector<Real>& qw, const Integer md) {
    return QuadElemList<Real>::BuildNearTensorRule(up, wu, vp, wv, useg, ud, vseg, vd, qel, e, Xt, b, qn, qw, md);
  }
  template <Integer digits> static Integer DigitsQuadOrder() { return QuadElemList<Real>::template DigitsQuadOrder<digits>(); }
  template <Integer digits> static Real DigitsBEllipse() { return QuadElemList<Real>::template DigitsBEllipse<digits>(); }
  template <Integer digits> static const std::pair<Vector<Real>, Vector<Real>>& DigitsGLRule() { return QuadElemList<Real>::template DigitsGLRule<digits>(); }
};
}

namespace {

using QS = QuadElemList<double>::QuadScheme;

// Curved test patch z = u*v on the order x order GL grid.
template <class Real> Vector<Real> get_testsurf(const Integer order) {
  Vector<Real> c = QuadElemList<Real>::ParamGrid(order, 1);
  for (Long i = 0; i < c.Dim()/3; i++) c[i*3+2] = c[i*3+0]*c[i*3+1];
  return c;
}

// Smooth nodal density used by every potential comparison below (AoS).
template <class Real> Vector<Real> make_density(const Integer order, const Integer KDIM0) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> sigma((Long)order*order*KDIM0);
  for (Integer i = 0; i < order; i++)
    for (Integer j = 0; j < order; j++)
      for (Integer k = 0; k < KDIM0; k++)
        sigma[((Long)i*order+j)*KDIM0+k] = cos<Real>(nds[i] + 2*nds[j] + (Real)0.5*k);
  return sigma;
}

// u = sigma^T M for a single near target.
template <class Real, class Kernel> Vector<Real> apply_near(const QuadElemList<Real>& qel, const Long elem_idx,
    const Vector<Real>& Xt, const Vector<Real>& sigma, const Kernel& ker, const Real tol) {
  static constexpr Integer KDIM1 = Kernel::TrgDim();
  Matrix<Real> M; Vector<Real> nt;
  QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, nt, ker, tol, elem_idx, &qel);
  Vector<Real> u(KDIM1); u.SetZero();
  for (Long r = 0; r < M.Dim(0); r++)
    for (Integer k = 0; k < KDIM1; k++) u[k] += sigma[r]*M[r][k];
  return u;
}

template <class Real> Real rel_l2(const Vector<Real>& a, const Vector<Real>& b) {
  Real e = 0, n = 0;
  for (Long i = 0; i < a.Dim(); i++) { e += (a[i]-b[i])*(a[i]-b[i]); n += b[i]*b[i]; }
  return sqrt<Real>(e) / sqrt<Real>(n);
}

// Target `d` off the surface along the normal at (u0,v0) of elem_idx.
template <class Real> Vector<Real> offsurf_target(const QuadElemList<Real>& qel, const Long elem_idx,
                                                  const Real u0, const Real v0, const Real d) {
  Vector<Real> up{u0}, vp{v0}, Xs, Ns;
  qel.GetGeom(&Xs, &Ns, nullptr, nullptr, nullptr, up, vp, elem_idx);
  Vector<Real> Xt(3);
  for (Integer k = 0; k < 3; k++) Xt[k] = Xs[k] + d*Ns[k];
  return Xt;
}

// Reference near potential: uniform nsub x nsub refinement with an order-GL rule per subpanel,
// integrating the SAME order-order Lagrange interpolant of the density that NearInterac does.
template <class Real, class Kernel> Vector<Real> direct_upsampled_potential(
    const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& sigma,
    const Vector<Real>& Xt, const Kernel& ker, const Long nsub) {
  const Integer order = qel.Order();
  static constexpr Integer KDIM0 = Kernel::SrcDim();
  static constexpr Integer KDIM1 = Kernel::TrgDim();
  const Long nq = (Long)order*order;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Vector<Real>& wts = LegQuadRule<Real>::wts(order);

  Vector<Real> u(KDIM1); u.SetZero();
  Vector<Real> u_param(order), v_param(order);
  for (Long pi = 0; pi < nsub; pi++) {
    for (Long pj = 0; pj < nsub; pj++) {
      for (Integer a = 0; a < order; a++) u_param[a] = (nds[a] + pi) / (Real)nsub;
      for (Integer b = 0; b < order; b++) v_param[b] = (nds[b] + pj) / (Real)nsub;

      Vector<Real> X, Xn, Xa;
      qel.GetGeom(&X, &Xn, &Xa, nullptr, nullptr, u_param, v_param, elem_idx);

      Vector<Real> Lu(order*order), Lv(order*order);
      LagrangeInterp<Real>::Interpolate(Lu, nds, u_param);
      LagrangeInterp<Real>::Interpolate(Lv, nds, v_param);

      Vector<Real> sigma_q(nq*KDIM0); sigma_q.SetZero();
      for (Integer a = 0; a < order; a++)
        for (Integer b = 0; b < order; b++) {
          const Long q = a*order + b;
          for (Integer i = 0; i < order; i++)
            for (Integer j = 0; j < order; j++) {
              const Real L = Lu[i*order+a]*Lv[j*order+b];
              for (Integer k = 0; k < KDIM0; k++) sigma_q[q*KDIM0+k] += sigma[((Long)i*order+j)*KDIM0+k]*L;
            }
        }

      Matrix<Real> Mker;
      ker.template KernelMatrix<Real,false>(Mker, Xt, X, Xn);
      for (Integer a = 0; a < order; a++)
        for (Integer b = 0; b < order; b++) {
          const Long q = a*order + b;
          const Real wq = Xa[q]*wts[a]*wts[b] / ((Real)nsub*(Real)nsub);
          for (Integer k0 = 0; k0 < KDIM0; k0++)
            for (Integer k1 = 0; k1 < KDIM1; k1++) u[k1] += Mker[q*KDIM0+k0][k1]*sigma_q[q*KDIM0+k0]*wq;
        }
    }
  }
  return u;
}

// ---------------------------------------------------------------------------------------------
// 1. Rule structure + off-surface admissibility.
//
// sum(w) == 1 per direction proves the segments tile [0,1] exactly. The admissibility check is the
// substantive one: every segment must satisfy rho >= b*width with rho = sqrt(pdist^2 + h^2),
// h = dist/L_phys. The innermost segment touches the foot (pdist = 0) so it passes only via h --
// i.e. only because the target is OFF the surface. If this rule were ever wired into the self path
// (h = 0) this assert is what would fire.
// ---------------------------------------------------------------------------------------------
template <class Real> void test_rule_structure() {
  std::cout << "--- 1. foot-graded rule: partition of unity + off-surface admissibility ---\n";
  const Integer order = 12;
  Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
  QuadElemList<Real> qel(order, coord0);

  const Real b = QuadElemTestAccess<Real>::template DigitsBEllipse<9>();
  const auto& gl = QuadElemTestAccess<Real>::template DigitsGLRule<9>();
  const Integer Q = QuadElemTestAccess<Real>::template DigitsQuadOrder<9>();
  std::cout << "   QuadOrder=" << Q << " b_ellipse=" << (double)b << "\n";
  std::cout << "   (u*,v*)        d      L   nseg_u nseg_v      nq   |sum(wu)-1|  |sum(wv)-1|\n";

  Long nseg_checked = 0;
  for (const auto& uv : std::vector<std::pair<Real,Real>>{{0.4,0.6},{0.5,0.5},{0.0,0.5},{1.0,1.0},{0.03,0.97}}) {
    for (const Real d : {(Real)1e-1, (Real)1e-2, (Real)1e-4, (Real)1e-7}) {
      const Vector<Real> Xt = offsurf_target(qel, 0, uv.first, uv.second, d);
      Vector<Real> up, wu, vp, wv, useg, vseg; Vector<Long> ud, vd;
      const Integer L = QuadElemTestAccess<Real>::BuildNearTensorRule(up, wu, vp, wv, &useg, &ud, &vseg, &vd,
                            qel, 0, Xt, b, gl.first, gl.second, 30);
      Real us_, vs_, dist_, h_;
      QuadElemTestAccess<Real>::NearFootAndDepth(us_, vs_, dist_, h_, qel, 0, Xt, b, 30);

      Real su = 0, sv = 0;
      for (Long i = 0; i < up.Dim(); i++) { SCTL_ASSERT(up[i] > -1e-14 && up[i] < 1+1e-14); SCTL_ASSERT(wu[i] > 0); su += wu[i]; }
      for (Long i = 0; i < vp.Dim(); i++) { SCTL_ASSERT(vp[i] > -1e-14 && vp[i] < 1+1e-14); SCTL_ASSERT(wv[i] > 0); sv += wv[i]; }
      SCTL_ASSERT(std::fabs((double)su - 1) < 1e-13);
      SCTL_ASSERT(std::fabs((double)sv - 1) < 1e-13);

      // Off-surface admissibility, per direction.
      auto admissible = [&](const Vector<Real>& seg, const Real c) {
        for (Long i = 0; i < seg.Dim()/2; i++) {
          const Real a0 = seg[i*2+0], a1 = seg[i*2+1];
          const Real pd = std::fabs(std::min<Real>(a1, std::max<Real>(a0, c)) - c);
          if (!(sqrt<Real>(pd*pd + h_*h_) >= b*(a1-a0)*(Real)0.999)) return false;
        }
        return true;
      };
      SCTL_ASSERT(admissible(useg, us_));
      SCTL_ASSERT(admissible(vseg, vs_));
      nseg_checked += ud.Dim() + vd.Dim();

      printf("   (%.2f,%.2f)  %7.0e  %3d   %5ld  %5ld  %7ld   %8.1e     %8.1e\n",
             (double)uv.first, (double)uv.second, (double)d, (int)L, (long)ud.Dim(), (long)vd.Dim(),
             (long)(up.Dim()*vp.Dim()), std::fabs((double)su-1), std::fabs((double)sv-1));
    }
  }
  std::cout << "   " << nseg_checked << " segments all admissible under rho = sqrt(pdist^2 + (dist/L_phys)^2)\n";
  std::cout << "test_rule_structure: PASSED\n";
}

// ---------------------------------------------------------------------------------------------
// 2. Near potential vs a direct upsampled reference at moderate near distance (d = 1e-2).
// ---------------------------------------------------------------------------------------------
template <class Real, class Kernel> void test_vs_upsampled(const Kernel& ker, const bool curved, const char* label, const Real rel_tol = 1e-7) {
  static constexpr Integer KDIM0 = Kernel::SrcDim();
  const Integer order = 24;
  const Real d = 1e-2, tol = 1e-10;

  Vector<Real> coord0 = curved ? get_testsurf<Real>(order) : QuadElemList<Real>::ParamGrid(order, 1);
  const Vector<Real> sigma = make_density<Real>(order, KDIM0);

  QuadElemList<Real> qel(order, coord0);
  qel.SetQuadScheme(QS::Adaptive, 10, 0, 30);
  const Vector<Real> Xt = offsurf_target(qel, 0, (Real)0.4, (Real)0.6, d);
  const Vector<Real> u_ref = direct_upsampled_potential<Real, Kernel>(qel, 0, sigma, Xt, ker, 100);

  const Real e = rel_l2(apply_near(qel, 0, Xt, sigma, ker, tol), u_ref);
  printf("   %-30s rel_err = %.3e\n", label, (double)e);
  SCTL_ASSERT(e < rel_tol);
}

// ---------------------------------------------------------------------------------------------
// 3. Deep near vs a RectPolar-512 gold on the FLAT panel (RP is not trustworthy under shear, so
//    this is deliberately the unsheared case; sheared geometry is validated downstream by the
//    self-validating Green's identity instead).
// ---------------------------------------------------------------------------------------------
template <class Real, class Kernel> void test_deep_near_vs_gold(const Kernel& ker, const char* label, const Real tol_fail) {
  static constexpr Integer KDIM0 = Kernel::SrcDim();
  const Integer order = 24;
  Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
  const Vector<Real> sigma = make_density<Real>(order, KDIM0);

  std::cout << "   " << label << "\n";
  std::cout << "        d      |RP400-RP512|   |adaptive-gold|\n";
  for (const Real d : {(Real)1e-2, (Real)1e-3, (Real)1e-4}) {
    QuadElemList<Real> q0(order, coord0);
    const Vector<Real> Xt = offsurf_target(q0, 0, (Real)0.4, (Real)0.6, d);

    QuadElemList<Real> qg(order, coord0); qg.SetQuadScheme(QS::RectPolar, 10, 512, 30);
    QuadElemList<Real> q2(order, coord0); q2.SetQuadScheme(QS::RectPolar, 10, 400, 30);
    const Vector<Real> u_gold = apply_near(qg, 0, Xt, sigma, ker, (Real)1e-13);
    const Vector<Real> u_rp2  = apply_near(q2, 0, Xt, sigma, ker, (Real)1e-13);

    QuadElemList<Real> qel(order, coord0);
    qel.SetQuadScheme(QS::Adaptive, 10, 0, 30);
    const Real e = rel_l2(apply_near(qel, 0, Xt, sigma, ker, (Real)1e-12), u_gold);
    printf("     %7.0e     %9.2e      %9.2e\n", (double)d, (double)rel_l2(u_rp2, u_gold), (double)e);
    SCTL_ASSERT(e < tol_fail);
  }
}

// ---------------------------------------------------------------------------------------------
// 4. Convergence in tol and in max_depth vs the upsampled reference.
// ---------------------------------------------------------------------------------------------
template <class Real, class Kernel> void test_convergence(const Kernel& ker) {
  static constexpr Integer KDIM0 = Kernel::SrcDim();
  const Integer order = 24;
  const Real d = 1e-2;
  Vector<Real> coord0 = get_testsurf<Real>(order);
  const Vector<Real> sigma = make_density<Real>(order, KDIM0);
  QuadElemList<Real> q0(order, coord0);
  const Vector<Real> Xt = offsurf_target(q0, 0, (Real)0.4, (Real)0.6, d);
  const Vector<Real> u_ref = direct_upsampled_potential<Real, Kernel>(q0, 0, sigma, Xt, ker, 100);

  std::cout << "   tol sweep (curved patch, d=1e-2, max_depth=30):\n        tol       rel_err\n";
  for (const Real tol : {(Real)1e-4, (Real)1e-6, (Real)1e-8, (Real)1e-10, (Real)1e-12}) {
    QuadElemList<Real> qel(order, coord0);
    qel.SetQuadScheme(QS::Adaptive, 10, 0, 30);
    printf("     %8.0e    %9.2e\n", (double)tol, (double)rel_l2(apply_near(qel, 0, Xt, sigma, ker, tol), u_ref));
  }

  std::cout << "   max_depth sweep (curved patch, d=1e-2, tol=1e-10):\n     max_depth   rel_err\n";
  for (const Integer md : {4, 8, 12, 30}) {
    QuadElemList<Real> qel(order, coord0);
    qel.SetQuadScheme(QS::Adaptive, 10, 0, md);
    printf("     %8d    %9.2e\n", (int)md, (double)rel_l2(apply_near(qel, 0, Xt, sigma, ker, (Real)1e-10), u_ref));
  }
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Stokes3D_FxU ker_sFxU;
    const Stokes3D_DxU ker_sDxU;
    const Laplace3D_FxU ker_lFxU;

    test_rule_structure<Real>();

    std::cout << "--- 2. near potential vs direct upsampled reference (d=1e-2, nsub=100) ---\n";
    test_vs_upsampled<Real>(ker_sFxU, false, "Stokes3D_FxU / plane");
    test_vs_upsampled<Real>(ker_sFxU, true,  "Stokes3D_FxU / testsurf");
    test_vs_upsampled<Real>(ker_sDxU, false, "Stokes3D_DxU / plane");
    test_vs_upsampled<Real>(ker_sDxU, true,  "Stokes3D_DxU / testsurf");
    test_vs_upsampled<Real>(ker_lFxU, false, "Laplace3D_FxU / plane");
    test_vs_upsampled<Real>(ker_lFxU, true,  "Laplace3D_FxU / testsurf");
    std::cout << "test_vs_upsampled: PASSED\n";

    std::cout << "--- 3. deep near vs RectPolar-512 gold (flat panel) ---\n";
    test_deep_near_vs_gold<Real>(ker_sFxU, "Stokes3D_FxU", (Real)1e-7);
    test_deep_near_vs_gold<Real>(ker_sDxU, "Stokes3D_DxU", (Real)1e-6);
    std::cout << "test_deep_near_vs_gold: PASSED\n";

    std::cout << "--- 4. convergence in tol and max_depth ---\n";
    test_convergence<Real>(ker_sFxU);

    std::cout << "\nALL near-graded tests PASSED\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
