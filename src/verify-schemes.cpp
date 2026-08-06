/**
 * Post-core-fold-in regression check: confirm all five singular-quadrature schemes still RUN and
 * converge after the SCTL core library was updated to upstream `dmal/4-quadrature-heuristics`.
 *
 * For each scheme it builds a mildly twisted cubed sphere and reports the on-surface Laplace
 * Green's-identity error  max|(S[du/dn] - D[u]) - u| / max|u|. The point is not peak accuracy per
 * scheme (each has its own knobs and validity range) but that every scheme completes without
 * crash/assert and lands in its expected ballpark against the new core.
 *
 * Expected, on order=12 ppf=2 twist=0.3 tol=1e-6 (mild shear, well inside every scheme's range):
 *   Adaptive / Hybrid / RectPolar / Duffy : ~1e-6 or better
 *   LineQBX                               : larger (~1e-3..1e-2); its near is panel-INTERIOR only,
 *                                           so the cubed-sphere panel seams floor it -- EXPECTED.
 *
 *   ./bin/verify-schemes [order] [ppf] [twist] [tol]
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    default: x = -a; y =  b; z = -1; break;
  }
  const Real r = sqrt<Real>(x*x + y*y + z*z);
  x *= R/r; y *= R/r; z *= R/r;
}

// Twisted cubed sphere. Scheme is set by the caller so one geometry serves all five.
QuadElemList<Real> BuildSphere(const Integer order, const Long ppf, const Real R, const Real twist) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (Integer f = 0; f < 6; f++)
    for (Long iu = 0; iu < ppf; iu++)
      for (Long iv = 0; iv < ppf; iv++)
        for (Integer i = 0; i < order; i++) {
          const Real a = 2*((iu + nds[i]) / (Real)ppf) - 1;
          for (Integer j = 0; j < order; j++) {
            const Real b = 2*((iv + nds[j]) / (Real)ppf) - 1;
            Real x, y, z;
            FacePoint(x, y, z, f, a, b, R);
            const Real s = sin<Real>(twist*z), c = cos<Real>(twist*z);
            X.PushBack(x*c + y*s);
            X.PushBack(-x*s + y*c);
            X.PushBack(z);
          }
        }
  return QuadElemList<Real>(order, X);
}

// Exterior point-source surface data: Fd = u|_S (DL density), Fs = du/dn (SL density), Uref = u.
template <class KerSL, class KerGrad>
void GreensData(Vector<Real>& Fs, Vector<Real>& Fd, Vector<Real>& Uref,
                const Vector<Real>& X, const Vector<Real>& Xn, const Vector<Real>& X0) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  KerSL ker_sl; KerGrad ker_grad;
  Vector<Real> Xn0{0,0,0}, F0(KDIM0), dU;
  for (Integer i = 0; i < KDIM0; i++) F0[i] = (Real)1 / (Real)(i + 1);
  ker_sl.Eval(Fd, X, X0, Xn0, F0);
  ker_grad.Eval(dU, X, X0, Xn0, F0);
  ker_sl.Eval(Uref, X, X0, Xn0, F0);
  const Long N = X.Dim() / COORD_DIM;
  Fs.ReInit(N * KDIM0);
  for (Long i = 0; i < N; i++)
    for (Integer j = 0; j < KDIM0; j++) {
      Real dn = 0;
      for (Integer k = 0; k < COORD_DIM; k++) dn += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
      Fs[i*KDIM0+j] = dn;
    }
}

// On-surface Laplace Green's identity error for the given element-list (scheme already set).
double GreensSolError(const QuadElemList<Real>& qel, const Real tol, const Vector<Real>& X0, const Comm& comm) {
  using KerSL = Laplace3D_FxU; using KerDL = Laplace3D_DxU; using KerGrad = Laplace3D_FxdU;
  KerSL ker_sl; KerDL ker_dl;
  BoundaryIntegralOp<Real,KerSL> BSL(ker_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BDL(ker_dl, false, comm);
  BSL.AddElemList(qel); BDL.AddElemList(qel);
  BSL.SetAccuracy(tol); BDL.SetAccuracy(tol);

  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  GreensData<KerSL,KerGrad>(Fs, Fd, Uref, X, Xn, X0);

  BSL.Setup(); BDL.Setup();
  BSL.ComputePotential(Us, Fs);
  BDL.ComputePotential(Ud, Fd);
  Ud -= (Real)0.5 * Fd;   // interior DL jump

  Real err = 0, val = 0;
  for (Long i = 0; i < Uref.Dim(); i++) {
    err = std::max<Real>(err, fabs<Real>((Us[i] - Ud[i]) - Uref[i]));
    val = std::max<Real>(val, fabs<Real>(Uref[i]));
  }
  return (double)(err / val);
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const Integer order = (argc > 1 ? (Integer)atol(argv[1]) : 12);
    const Long    ppf   = (argc > 2 ? atol(argv[2]) : 2);
    const Real    twist = (argc > 3 ? (Real)atof(argv[3]) : 0.3);
    const Real    tol   = (argc > 4 ? (Real)atof(argv[4]) : 1e-6);
    const Vector<Real> X0{(Real)1.7, (Real)0.3, (Real)-0.5};  // exterior source

    using QS = QuadElemList<Real>::QuadScheme;
    const Integer Nbeta = 200;  // RectPolar/Hybrid self needs a large Nbeta (see memory)

    std::printf("==== verify-schemes: order=%d ppf=%ld twist=%.3f tol=%.0e ====\n",
                (int)order, ppf, (double)twist, (double)tol);
    std::printf("%-10s  %-12s  %s\n", "scheme", "greens_sol", "status");

    struct Cfg { const char* name; QS scheme; Integer q, cov, depth; };
    const Cfg cfgs[] = {
      {"Adaptive",  QS::Adaptive,  6, 0,     30},
      {"RectPolar", QS::RectPolar, 6, Nbeta, 30},
      {"Hybrid",    QS::Hybrid,    6, Nbeta, 30},
      {"LineQBX",   QS::LineQBX,   6, 0,     30},
      {"Duffy",     QS::Duffy,     6, 0,     30},
    };

    for (const Cfg& c : cfgs) {
      QuadElemList<Real> qel = BuildSphere(order, ppf, 1.0, twist);
      qel.SetQuadScheme(c.scheme, c.q, c.cov, c.depth);
      if (c.scheme == QS::LineQBX) qel.SetLineQBXParams();  // defaults
      const double e = GreensSolError(qel, tol, X0, comm);
      const bool ok = std::isfinite(e) && e < 1e-1;         // loose "ran and is sane" gate
      std::printf("%-10s  %-12.3e  %s\n", c.name, e, ok ? "OK" : "*** CHECK ***");
      std::fflush(stdout);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
