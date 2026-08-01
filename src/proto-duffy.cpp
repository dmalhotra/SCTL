/**
 * Standalone prototype + validation harness for the Duffy (edge-collapsed)
 * self-interaction scheme for QuadElemList -- duffy.txt sections 1-6, the
 * section-7 kernel-order gate, and validation V1 (section 10).
 *
 * No library changes: everything here is local except ParamNodes and
 * BuildCenteredGraded1D (reached through QuadElemTestAccess).
 *
 * Build:
 *   g++ -std=c++17 -fopenmp -O3 -march=native -mno-avx512fp16 -DNDEBUG \
 *     -DSCTL_GLOBAL_MEM_BUFF=0 -DSCTL_QUAD_T=__float128 -I./include \
 *     src/proto-duffy.cpp -o /tmp/proto-duffy \
 *     -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread \
 *     -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK -lquadmath
 *
 * Usage: proto-duffy <mode> [double|long|quad] [order] [node-stride]
 *   gate     small-r order of the four kernels (section-7 gate)
 *   ref      Duffy reference vs an independent polar rule vs the flat analytic value
 *   v1       error vs n_s at fixed t-rules (the flatness check)
 *   tune     (q_s x t-rule) grid, worst error over target nodes + point counts
 *   final    candidate rules over ALL p^2 nodes, metric-aware vs parameter-space t*
 *   diag     how far parameter-space (t*, d/L) is from the metric values
 *   adjoint  exact-adjointness of the projection + varying-density reference
 *   base     point counts of the shipped Adaptive self rule, for comparison
 *
 * Run with OMP_NUM_THREADS=1: every GEMM here is tiny and sctl's generic
 * (non-BLAS) gemm threads them, which costs ~10x in wall time.
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

#include <limits>
#include <omp.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace sctl {
  // quad_element.hpp declares this a friend; used only for BuildCenteredGraded1D.
  template <typename Real> struct QuadElemTestAccess {
    static void CenteredGraded1D(Vector<Real>& delta, Vector<Real>& w, const Real c, const Integer levels, const Vector<Real>& qn, const Vector<Real>& qw) {
      QuadElemList<Real>::BuildCenteredGraded1D(delta, w, c, levels, qn, qw);
    }
    // Baseline (shipped Adaptive) self rule sizes, for the point-count comparison.
    static void LogSingV(Vector<Real>& delta, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder) {
      QuadElemList<Real>::LogSingularQuad1DCentered(delta, w, v0, Lvl, QuadOrder);
    }
    static Integer VLevels(const Integer digits) { return QuadElemList<Real>::VLevelsForDigits(digits); }
    static Integer BaseQuadOrder(const Real tol) { Real b; Integer q; QuadElemList<Real>::QuadParams(tol, b, q); return q; }
  };
}

using namespace sctl;

static constexpr Integer DIM = 3;

// ----------------------------------------------------------------- helpers

template <class Real> Real Asinh_(const Real x) { return log<Real>(x + sqrt<Real>(x*x + (Real)1)); }
template <class Real> Real Sinh_ (const Real x) { const Real e = exp<Real>(x); return (e - (Real)1/e)/(Real)2; }
template <class Real> Real Cosh_ (const Real x) { const Real e = exp<Real>(x); return (e + (Real)1/e)/(Real)2; }
static double D_(const double x) { return x; }
static double D_(const long double x) { return (double)x; }
#ifdef SCTL_QUAD_T
static double D_(const QuadReal x) { return (double)x; }
#endif

// D[i][a] = L_i'(nds[a]) on the order-p GL nodes.
template <class Real> const Matrix<Real>& DMat(const Integer p) {
  static std::map<Integer, Matrix<Real>> cache;
  auto it = cache.find(p);
  if (it == cache.end()) {
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
    Vector<Real> f((Long)p*p); f.SetZero();
    for (Integer i = 0; i < p; i++) f[i*p+i] = 1;
    Vector<Real> df;
    LagrangeInterp<Real>::Derivative(df, f, nds);
    Matrix<Real> D(p, p);
    for (Integer i = 0; i < p; i++) for (Integer a = 0; a < p; a++) D[i][a] = df[i*p+a];
    it = cache.emplace(p, D).first;
  }
  return it->second;
}

// M[i][a] = L_i(x[a]) for the order-p GL basis.
template <class Real> void LagMat(Matrix<Real>& M, const Integer p, const Vector<Real>& x) {
  const Long N = x.Dim();
  M.ReInit(p, N);
  Vector<Real> v(p*N, M.begin(), false);
  LagrangeInterp<Real>::Interpolate(v, QuadElemList<Real>::ParamNodes(p), x);
}

// Matrix::Transpose is `#pragma omp parallel for`, which is a net loss on the
// small matrices here, so transpose by hand.
template <class Real> void TransposeTo(Matrix<Real>& dst, const Matrix<Real>& src) {
  const Long d0 = src.Dim(0), d1 = src.Dim(1);
  if (dst.Dim(0) != d1 || dst.Dim(1) != d0) dst.ReInit(d1, d0);
  for (Long i = 0; i < d0; i++) for (Long j = 0; j < d1; j++) dst[j][i] = src[i][j];
}

template <class Real> void GLRule(Vector<Real>& n, Vector<Real>& w, const Integer q) {
  LegQuadRule<Real>::ComputeNdsWts(&n, &w, q);
}

// ----------------------------------------------------------- model elements

// Nodal coords of the order-p interpolant of a smooth map; F[k*p*p + iu*p + iv].
template <class Real> struct ModelElem {
  Integer p = 0;
  Vector<Real> F;
  std::string name;
};

template <class Real> void MapPoint(Real* X, const Real u, const Real v, const std::string& g) {
  if (g == "flat") { X[0] = u; X[1] = v; X[2] = 0; return; }
  if (g == "skew") { // corner skew sin(theta) = 0.385
    X[0] = u + (Real)0.92289*v; X[1] = (Real)0.38500*v; X[2] = 0; return;
  }
  if (g == "poly") {
    X[0] = u; X[1] = v; X[2] = (Real)0.5*u*u - (Real)0.3*v*v + (Real)0.7*u*v + (Real)0.2*u*u*u*v; return;
  }
  if (g == "sphere" || g == "twist" || g == "sph4" || g == "twist4") {
    // face 0 of a cubed sphere, R=1. "*4": patch (1,1) of a 4-per-face split
    // (a size closer to production meshes than a whole face).
    const Integer npf = (g == "sph4" || g == "twist4" ? 4 : 1);
    const Integer iu = (npf == 4 ? 1 : 0), iv = (npf == 4 ? 1 : 0);
    const Real a = 2*((iu+u)/(Real)npf)-1, b = 2*((iv+v)/(Real)npf)-1;
    Real x = 1, y = a, z = b;
    const Real r = sqrt<Real>(x*x + y*y + z*z);
    x /= r; y /= r; z /= r;
    if (g == "twist" || g == "twist4") {
      const Real th = const_pi<Real>()/2;
      const Real s = sin<Real>(th*z), c = cos<Real>(th*z);
      X[0] = x*c + y*s; X[1] = -x*s + y*c; X[2] = z;
    } else { X[0] = x; X[1] = y; X[2] = z; }
    return;
  }
  SCTL_ASSERT_MSG(false, "unknown geometry");
}

template <class Real> void BuildModelElem(ModelElem<Real>& el, const Integer p, const std::string& g) {
  el.p = p; el.name = g;
  el.F.ReInit(DIM*(Long)p*p);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  for (Integer i = 0; i < p; i++)
    for (Integer j = 0; j < p; j++) {
      Real X[DIM];
      MapPoint<Real>(X, nds[i], nds[j], g);
      for (Integer k = 0; k < DIM; k++) el.F[k*(Long)p*p + i*p + j] = X[k];
    }
}

// Nodal coords shifted so the target node (ti,tj) sits exactly at the origin.
template <class Real> void ShiftToTarget(Vector<Real>& Fsh, const ModelElem<Real>& el, const Integer ti, const Integer tj) {
  const Integer p = el.p;
  Fsh.ReInit(DIM*(Long)p*p);
  for (Integer k = 0; k < DIM; k++) {
    const Real ok = el.F[k*(Long)p*p + ti*p + tj];
    for (Long q = 0; q < (Long)p*p; q++) Fsh[k*(Long)p*p + q] = el.F[k*(Long)p*p + q] - ok;
  }
}

// ------------------------------------- direct (unfactorized) geometry eval

// X/dXu/dXv (SoA, DIM x N) and optionally an interpolated nodal field, at an
// arbitrary (not tensor) list of parameter points. Independent of the Duffy
// tables -- used by the reference integrator.
template <class Real> void EvalGeomDirect(Vector<Real>& X, Vector<Real>& dXu, Vector<Real>& dXv, Vector<Real>* fout,
                                          const Integer p, const Vector<Real>& Fsh, const Vector<Real>* fnod,
                                          const Vector<Real>& uq, const Vector<Real>& vq) {
  const Long N = uq.Dim();
  Matrix<Real> Lu, Lv, dLu, dLv;
  LagMat(Lu, p, uq); LagMat(Lv, p, vq);
  dLu.ReInit(p, N); dLv.ReInit(p, N);
  Matrix<Real>::GEMM(dLu, DMat<Real>(p), Lu);
  Matrix<Real>::GEMM(dLv, DMat<Real>(p), Lv);
  X.ReInit(DIM*N); dXu.ReInit(DIM*N); dXv.ReInit(DIM*N);
  if (fout && fnod) fout->ReInit(N);
  Vector<Real> tv(p), tdv(p);
  for (Long a = 0; a < N; a++) {
    for (Integer k = 0; k < DIM; k++) {
      for (Integer i = 0; i < p; i++) {
        Real s = 0, sd = 0;
        for (Integer j = 0; j < p; j++) { const Real f = Fsh[k*(Long)p*p + i*p + j]; s += f*Lv[j][a]; sd += f*dLv[j][a]; }
        tv[i] = s; tdv[i] = sd;
      }
      Real x = 0, xu = 0, xv = 0;
      for (Integer i = 0; i < p; i++) { x += tv[i]*Lu[i][a]; xu += tv[i]*dLu[i][a]; xv += tdv[i]*Lu[i][a]; }
      X[k*N+a] = x; dXu[k*N+a] = xu; dXv[k*N+a] = xv;
    }
    if (fout && fnod) {
      Real s = 0;
      for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++) s += (*fnod)[i*p+j]*Lu[i][a]*Lv[j][a];
      (*fout)[a] = s;
    }
  }
}

// Metric G = [[Xu.Xu, Xu.Xv],[Xu.Xv, Xv.Xv]] at the target node.
template <class Real> void MetricAt(Real* G, const Integer p, const Vector<Real>& Fsh, const Real u0, const Real v0) {
  Vector<Real> uq(1), vq(1), X, dXu, dXv;
  uq[0] = u0; vq[0] = v0;
  EvalGeomDirect<Real>(X, dXu, dXv, nullptr, p, Fsh, nullptr, uq, vq);
  Real guu = 0, guv = 0, gvv = 0;
  for (Integer k = 0; k < DIM; k++) { guu += dXu[k]*dXu[k]; guv += dXu[k]*dXv[k]; gvv += dXv[k]*dXv[k]; }
  G[0] = guu; G[1] = guv; G[2] = guv; G[3] = gvv;
}

// -------------------------------------------- reference integrator (direct)

// I[c] = sum_q K_c(0, X_q) * f_q * |Xu x Xv|_q * wq  -- no tables, no factorization.
template <class Real, class Ker> void RefIntegrate(Vector<Real>& I, const Integer p, const Vector<Real>& Fsh, const Vector<Real>* fnod,
                                                   const Vector<Real>& uq, const Vector<Real>& vq, const Vector<Real>& wq, const Ker& ker) {
  static constexpr Integer KDIM0 = Ker::SrcDim(), KDIM1 = Ker::TrgDim();
  const Integer C = KDIM0*KDIM1;
  const Long N = uq.Dim();
  Vector<Real> X, dXu, dXv, fq;
  EvalGeomDirect<Real>(X, dXu, dXv, &fq, p, Fsh, fnod, uq, vq);
  Vector<Real> Xs(DIM*N), Xn(DIM*N), w(N);
  for (Long a = 0; a < N; a++) {
    const Real u0 = dXu[0*N+a], u1 = dXu[1*N+a], u2 = dXu[2*N+a];
    const Real v0 = dXv[0*N+a], v1 = dXv[1*N+a], v2 = dXv[2*N+a];
    const Real n0 = u1*v2 - u2*v1, n1 = u2*v0 - u0*v2, n2 = u0*v1 - u1*v0;
    const Real ar = sqrt<Real>(n0*n0 + n1*n1 + n2*n2), ia = (ar > 0 ? (Real)1/ar : (Real)0);
    for (Integer k = 0; k < DIM; k++) Xs[a*DIM+k] = X[k*N+a];
    Xn[a*DIM+0] = n0*ia; Xn[a*DIM+1] = n1*ia; Xn[a*DIM+2] = n2*ia;
    w[a] = ar*wq[a];
  }
  StaticArray<Real,DIM> Xt0{0,0,0};
  const Vector<Real> Xt0_v(DIM, Xt0, false);
  Matrix<Real> Mker;
  ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xs, Xn);
  I.ReInit(C); I.SetZero();
  for (Long a = 0; a < N; a++) {
    const Real ww = w[a] * (fnod ? fq[a] : (Real)1);
    for (Integer k0 = 0; k0 < KDIM0; k0++) for (Integer k1 = 0; k1 < KDIM1; k1++)
      I[k0*KDIM1+k1] += Mker[a*KDIM0+k0][k1]*ww;
  }
}

// ------------------------------------------- polar reference rule (indep.)

// Geometric panels grading toward BOTH ends of [A,B]; `lvl` panels per side.
template <class Real> void GradeBothEnds(Vector<Real>& x, Vector<Real>& w, const Real A, const Real B,
                                         const Integer lvl, const Vector<Real>& qn, const Vector<Real>& qw) {
  if (!(B > A)) return;
  const Real h = B - A;
  std::vector<Real> bnd;
  bnd.push_back(A);
  for (Integer k = lvl; k >= 1; k--) bnd.push_back(A + h*pow<Real>((Real)0.5, (Integer)k));
  for (Integer k = 2; k <= lvl; k++)  bnd.push_back(B - h*pow<Real>((Real)0.5, (Integer)k));
  bnd.push_back(B);
  for (size_t i = 0; i + 1 < bnd.size(); i++) {
    const Real len = bnd[i+1] - bnd[i];
    if (!(len > 0)) continue;
    for (Long m = 0; m < qn.Dim(); m++) { x.PushBack(bnd[i] + len*qn[m]); w.PushBack(len*qw[m]); }
  }
}

// True-polar rule about (u0,v0): four triangles, rho GL, sigma (angle) graded
// toward the metric foot AND toward both corner directions -- for a thin
// triangle rho_max(sigma) has O(d/L) features at the ends of the angular range,
// so grading only toward the foot leaves the reference wrong by ~1e-5 there.
// A different change of variables from Duffy, so a bug in the Duffy
// maps/tables cannot hide here.
template <class Real> void PolarRule(Vector<Real>& uq, Vector<Real>& vq, Vector<Real>& wq,
                                     const Real u0, const Real v0, const Real* G,
                                     const Integer nrho, const Integer siglvl, const Integer sigq) {
  uq.ReInit(0); vq.ReInit(0); wq.ReInit(0);
  Vector<Real> rn, rw, qn, qw;
  GLRule<Real>(rn, rw, nrho);
  GLRule<Real>(qn, qw, sigq);
  const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
  for (Integer kt = 0; kt < 4; kt++) {
    const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
    const Real b[2] = {cu[(kt+1)%4]-u0, cv[(kt+1)%4]-v0};
    const Real e[2] = {b[0]-a[0], b[1]-a[1]};
    const Real J0 = a[0]*b[1]-a[1]*b[0];
    SCTL_ASSERT(J0 > 0);
    const Real am = e[0]*(G[0]*e[0]+G[1]*e[1]) + e[1]*(G[2]*e[0]+G[3]*e[1]);
    const Real bm = a[0]*(G[0]*e[0]+G[1]*e[1]) + a[1]*(G[2]*e[0]+G[3]*e[1]);
    Real ts = -bm/am; ts = (ts < 0 ? (Real)0 : (ts > 1 ? (Real)1 : ts));
    const Real cs[2] = {a[0]+ts*e[0], a[1]+ts*e[1]};
    const Real Delta = atan2<Real>(a[0]*b[1]-a[1]*b[0], a[0]*b[0]+a[1]*b[1]);
    const Real th0 = atan2<Real>(a[1], a[0]);
    Real ss = atan2<Real>(a[0]*cs[1]-a[1]*cs[0], a[0]*cs[0]+a[1]*cs[1])/Delta;
    ss = (ss < 0 ? (Real)0 : (ss > 1 ? (Real)1 : ss));
    Vector<Real> sd, sw;
    GradeBothEnds<Real>(sd, sw, (Real)0, ss, siglvl, qn, qw);
    GradeBothEnds<Real>(sd, sw, ss, (Real)1, siglvl, qn, qw);
    for (Long m = 0; m < sd.Dim(); m++) {
      const Real sig = sd[m];
      const Real th = th0 + sig*Delta;
      const Real w0 = cos<Real>(th), w1 = sin<Real>(th);
      const Real rmax = J0/(w0*e[1]-w1*e[0]);
      for (Integer r = 0; r < nrho; r++) {
        const Real rho = rmax*rn[r];
        uq.PushBack(u0 + rho*w0);
        vq.PushBack(v0 + rho*w1);
        wq.PushBack(rmax*rmax*rn[r]*rw[r]*Delta*sw[m]);
      }
    }
  }
}

// ----------------------------------------------------------- Duffy tables

enum class TRule { Graded, Sinh, SinhComp };

template <class Real> struct TriTable {
  bool swap_ab = false;  // collapsed (s-only) coordinate is u  =>  local (alpha,beta) = (v,u)
  Real nsign = 1;        // sign that turns dX/dalpha x dX/dbeta into dX/du x dX/dv
  Real J0 = 0;           // |a x b| (2 * triangle area in parameter space)
  Long ns = 0, nt = 0;
  Vector<Real> sn, sw, tn, tw;
  Matrix<Real> Wb, WbD, WbT;              // (p x ns), (p x ns), (ns x p)
  Vector<Matrix<Real>> Wa, WaD, WaT;      // fused alpha table, ns entries: (p x nt), (p x nt), (nt x p)
  // Two-stage alpha form. alpha(s,t) is affine in t at fixed s, so L_r(alpha(s_i,.))
  // is degree p-1 in t and is reproduced exactly by its values at p reference nodes:
  //   Wa[i] = Mi[i] . Tt   with Mi[i][r][k] = L_r(alpha(s_i,theta_k)) (no t-rule, no metric)
  //                        and Tt[k][j] = l_k(t_j)   (the only t-dependent operator)
  Vector<Matrix<Real>> Mi, MiD, MiT;      // ns entries, each (p x p)
  Matrix<Real> Tt, TtT;                   // (p x nt), (nt x p)
  // Batched form: side-by-side copies so one GEMM per s-node replaces three, and the
  // Tt contraction leaves the s-loop entirely (one (ns*NR x p)(p x nt) GEMM).
  Matrix<Real> WbC;                       // (p x 2ns) = [Wb | WbD]
  Vector<Matrix<Real>> MiC;               // ns entries, each (p x 2p) = [Mi | MiD]
  Vector<Real> JW;                        // ns*nt : s_i * J0 * sw_i * tw_j
  Real tstar = 0, dOverL = 0, L = 0;
};

template <class Real> struct DuffyRule {
  Integer p = 0, ti = 0, tj = 0;
  std::vector<TriTable<Real>> tri;
  Long NPts() const { Long n = 0; for (const auto& t : tri) n += t.ns*t.nt; return n; }
};

// t-rule for one triangle. Graded: geometric panels toward t* (extra levels on
// top of ceil(log2(len*L/d))). Sinh: t = t* + (d/L) sinh(xi), single GL rule.
// SinhComp: the same substitution with tpar1 equal panels of order tpar2.
template <class Real> void BuildTRule(Vector<Real>& tn, Vector<Real>& tw, const Real tstar, const Real dOverL,
                                      const TRule tr, const Integer tpar1, const Integer tpar2) {
  if (tr == TRule::Graded) {
    const Real len = std::max(D_(tstar), D_(1-tstar));
    Integer lvl = (Integer)ceil<Real>(log2<Real>((Real)std::max(1.0, D_(len)/std::max(1e-300, D_(dOverL))))) + tpar1;
    if (lvl < 0) lvl = 0;
    if (lvl > 30) lvl = 30;
    Vector<Real> qn, qw;
    GLRule<Real>(qn, qw, tpar2);
    Vector<Real> d, w;
    QuadElemTestAccess<Real>::CenteredGraded1D(d, w, tstar, lvl, qn, qw);
    tn.ReInit(d.Dim()); tw.ReInit(d.Dim());
    for (Long i = 0; i < d.Dim(); i++) { tn[i] = tstar + d[i]; tw[i] = w[i]; }
    return;
  }
  const Real dd = dOverL;
  const Real x0 = -Asinh_<Real>(tstar/dd), x1 = Asinh_<Real>((1-tstar)/dd);
  auto emit = [&](const Real A, const Real B, const Vector<Real>& qn, const Vector<Real>& qw) {
    for (Long i = 0; i < qn.Dim(); i++) {
      const Real xi = A + (B-A)*qn[i];
      tn.PushBack(tstar + dd*Sinh_<Real>(xi));
      tw.PushBack(dd*Cosh_<Real>(xi)*(B-A)*qw[i]);
    }
  };
  tn.ReInit(0); tw.ReInit(0);
  Vector<Real> qn, qw;
  if (tr == TRule::Sinh) { GLRule<Real>(qn, qw, tpar1); emit(x0, x1, qn, qw); }
  else {
    GLRule<Real>(qn, qw, tpar2);
    for (Integer k = 0; k < tpar1; k++) emit(x0 + (x1-x0)*k/(Real)tpar1, x0 + (x1-x0)*(k+1)/(Real)tpar1, qn, qw);
  }
}

// Variant-C quantisation ranges. kappa = cot(theta) enters as a shift in units of
// the peak width; w = (d/L)_G/(d/L)_I is a pure scale, so it is binned in log.
static constexpr double KAP_MAX = 3.0, W_MIN = 0.15, W_MAX = 5.0;
template <class Real> Real BinKappa(const Real k, const Integer n) {
  const double h = 2*KAP_MAX/(double)n;
  long i = (long)std::floor((D_(k)+KAP_MAX)/h);
  i = std::max<long>(0, std::min<long>(n-1, i));
  return (Real)(-KAP_MAX + (i+0.5)*h);
}
template <class Real> Real BinWidth(const Real w, const Integer n) {
  const double lr = std::log(W_MAX/W_MIN);
  long i = (long)std::floor(std::log(std::max(W_MIN, std::min(W_MAX, D_(w)))/W_MIN)/lr*(double)n);
  i = std::max<long>(0, std::min<long>(n-1, i));
  return (Real)(W_MIN*std::exp(lr*(i+0.5)/(double)n));
}

// param_space: place the t-rule by parameter-space distance (variant B).
// nkap,nwid > 0: place it from the quantised (cot(theta), width-scale) pair (variant C).
// Neither: use the exact metric (variant A).
enum { TBL_FUSED = 1, TBL_PRE = 2, TBL_TONLY = 4 };
static Integer g_tbl = TBL_FUSED;
template <class Real> void BuildDuffyRule(DuffyRule<Real>& R, const Integer p, const Integer ti, const Integer tj,
                                          const Real* G, const Integer qs, const TRule tr, const Integer tpar1, const Integer tpar2,
                                          const bool param_space = false, const Integer nkap = 0, const Integer nwid = 0,
                                          const Integer tbl = -1) {
  if (tbl == -1) return BuildDuffyRule(R, p, ti, tj, G, qs, tr, tpar1, tpar2, param_space, nkap, nwid, g_tbl);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  const Real u0 = nds[ti], v0 = nds[tj];
  R.p = p; R.ti = ti; R.tj = tj;
  R.tri.assign(4, TriTable<Real>());
  Vector<Real> sn, sw;
  GLRule<Real>(sn, sw, qs);
  const Real cu[4] = {0,1,1,0}, cv[4] = {0,0,1,1};
  const Matrix<Real>& D = DMat<Real>(p);
  for (Integer kt = 0; kt < 4; kt++) {
    TriTable<Real>& T = R.tri[kt];
    const Real a[2] = {cu[kt]-u0, cv[kt]-v0};
    const Real b[2] = {cu[(kt+1)%4]-u0, cv[(kt+1)%4]-v0};
    const Real e[2] = {b[0]-a[0], b[1]-a[1]};
    T.J0 = a[0]*b[1]-a[1]*b[0];
    SCTL_ASSERT_MSG(T.J0 > 0, "triangle orientation");
    T.swap_ab = (fabs<Real>(e[0]) < fabs<Real>(e[1])); // e is axis aligned
    T.nsign = (T.swap_ab ? (Real)-1 : (Real)1);

    { // foot t* and peak width d/L, from the metric M (exact, param-space, or binned)
      auto place = [&](const Real* M, Real& ts_out, Real& dd_out, Real& L_out) {
        const Real Me[2] = {M[0]*e[0]+M[1]*e[1], M[2]*e[0]+M[3]*e[1]};
        const Real am = e[0]*Me[0] + e[1]*Me[1];
        const Real bm = a[0]*Me[0] + a[1]*Me[1];
        Real ts = -bm/am; ts = (ts < 0 ? (Real)0 : (ts > 1 ? (Real)1 : ts));
        const Real c[2] = {a[0]+ts*e[0], a[1]+ts*e[1]};
        const Real d2 = c[0]*(M[0]*c[0]+M[1]*c[1]) + c[1]*(M[2]*c[0]+M[3]*c[1]);
        ts_out = ts; L_out = sqrt<Real>(am); dd_out = sqrt<Real>(d2)/L_out;
      };
      const Real I2[4] = {1,0,0,1};
      Real tsG, ddG, LG, tsI, ddI, LI;
      place(G,  tsG, ddG, LG);
      place(I2, tsI, ddI, LI);
      if (param_space) { T.tstar = tsI; T.dOverL = ddI; T.L = LI; }
      else if (nkap > 0 && nwid > 0) {
        const Real kap = (tsI - tsG)/ddG;            // == cot(theta), signed
        const Real wsc = ddG/ddI;                    // == sin(theta)/r  or  sin(theta)*r
        T.dOverL = BinWidth<Real>(wsc, nwid)*ddI;
        Real ts = tsI - BinKappa<Real>(kap, nkap)*T.dOverL;
        T.tstar = (ts < 0 ? (Real)0 : (ts > 1 ? (Real)1 : ts));
        T.L = LG;
      } else { T.tstar = tsG; T.dOverL = ddG; T.L = LG; }
    }
    BuildTRule<Real>(T.tn, T.tw, T.tstar, T.dOverL, tr, tpar1, tpar2);
    T.sn = sn; T.sw = sw;
    T.ns = qs; T.nt = T.tn.Dim();

    const Real al0 = (T.swap_ab ? v0 : u0), be0 = (T.swap_ab ? u0 : v0);
    const Real aal = (T.swap_ab ? a[1] : a[0]), abe = (T.swap_ab ? a[0] : a[1]);
    const Real eal = (T.swap_ab ? e[1] : e[0]);

    { // collapsed direction: beta(s_i) = be0 + s_i*a_beta
      Vector<Real> bv(T.ns);
      for (Long i = 0; i < T.ns; i++) bv[i] = be0 + T.sn[i]*abe;
      LagMat(T.Wb, p, bv);
      T.WbD.ReInit(p, T.ns);
      Matrix<Real>::GEMM(T.WbD, D, T.Wb);
      TransposeTo(T.WbT, T.Wb);
      T.WbC.ReInit(p, 2*T.ns);
      for (Integer r = 0; r < p; r++) for (Long i = 0; i < T.ns; i++) { T.WbC[r][i] = T.Wb[r][i]; T.WbC[r][T.ns+i] = T.WbD[r][i]; }
    }
    if (tbl & TBL_FUSED) { // fused alpha table Wa[i] = L_r(alpha(s_i,t_j))
      T.Wa.ReInit(T.ns); T.WaD.ReInit(T.ns); T.WaT.ReInit(T.ns);
      Vector<Real> av(T.nt);
      for (Long i = 0; i < T.ns; i++) {
        for (Long j = 0; j < T.nt; j++) av[j] = al0 + T.sn[i]*(aal + T.tn[j]*eal);
        LagMat(T.Wa[i], p, av);
        T.WaD[i].ReInit(p, T.nt);
        Matrix<Real>::GEMM(T.WaD[i], D, T.Wa[i]);
        TransposeTo(T.WaT[i], T.Wa[i]);
      }
    }
    if (tbl & TBL_PRE) { // precomputable on (p,ti,tj,tri,q_s): no t-rule, no metric
      const Vector<Real>& th = QuadElemList<Real>::ParamNodes(p);
      T.Mi.ReInit(T.ns); T.MiD.ReInit(T.ns); T.MiT.ReInit(T.ns);
      Vector<Real> av(p);
      for (Long i = 0; i < T.ns; i++) {
        for (Integer k = 0; k < p; k++) av[k] = al0 + T.sn[i]*(aal + th[k]*eal);
        LagMat(T.Mi[i], p, av);
        T.MiD[i].ReInit(p, p);
        Matrix<Real>::GEMM(T.MiD[i], D, T.Mi[i]);
        TransposeTo(T.MiT[i], T.Mi[i]);
      }
      T.MiC.ReInit(T.ns);
      for (Long i = 0; i < T.ns; i++) {
        T.MiC[i].ReInit(p, 2*p);
        for (Integer r = 0; r < p; r++) for (Integer k = 0; k < p; k++) { T.MiC[i][r][k] = T.Mi[i][r][k]; T.MiC[i][r][p+k] = T.MiD[i][r][k]; }
      }
    }
    if (tbl & TBL_TONLY) { // the only per-target operator
      LagMat(T.Tt, p, T.tn);
      TransposeTo(T.TtT, T.Tt);
    }
    T.JW.ReInit(T.ns*T.nt);
    for (Long i = 0; i < T.ns; i++) for (Long j = 0; j < T.nt; j++) T.JW[i*T.nt+j] = T.sn[i]*T.J0*T.sw[i]*T.tw[j];
  }
}

// ------------------------------------------------------- Duffy evaluation

// Contraction of the alpha direction: Fused uses the (p x ns*nt) table Wa;
// TwoStage uses Mi[i] (precomputable) followed by the shared t-only operator Tt.
enum class Contract { Fused, TwoStage, TwoStageB };
static Contract g_contract = Contract::Fused;

// Batched two-stage contraction. Same arithmetic as Contract::TwoStage, reassociated so
// the shapes suit BLAS: the geometry, its two derivatives and the density share one GEMM
// per s-node against [Mi|MiD], and the Tt contraction and its adjoint each become a single
// large GEMM instead of ns medium ones.
template <class Real, class Ker> void DuffyEvalB(Matrix<Real>& Proj, Vector<Real>* Idir,
                                                 const DuffyRule<Real>& R, const Vector<Real>& Fsh,
                                                 const Vector<Real>* fnod, const Ker& ker) {
  static constexpr Integer KDIM0 = Ker::SrcDim(), KDIM1 = Ker::TrgDim();
  constexpr Integer NR = 3*DIM + 1;          // Hv, Hd, Hb, density -- rows per s-node
  constexpr Integer NA = 2*DIM + 1;          // Ai, Adi, hi        -- rows fed to [Mi|MiD]
  const Integer C = KDIM0*KDIM1;
  const Integer p = R.p;
  const Long nn = (Long)p*p;
  Proj.ReInit(C, nn); Proj.SetZero();
  if (Idir) { Idir->ReInit(C); Idir->SetZero(); }
  StaticArray<Real,DIM> Xt0{0,0,0};
  const Vector<Real> Xt0_v(DIM, Xt0, false);

  for (const TriTable<Real>& T : R.tri) {
    const Long ns = T.ns, nt = T.nt, nq = ns*nt;
    const Long sz = (DIM+1)*nn + 2*(DIM+1)*(Long)p*ns + (Long)NA*p + 2*(Long)NA*p
                  + (Long)ns*NR*p + (Long)ns*NR*nt + 2*(Long)DIM*nq + 2*nq
                  + nq*KDIM0*KDIM1 + (Long)C*nq + (Long)ns*C*p + (Long)C*p + (Long)C*p*ns + nn;
    ScratchBuf<Real> sb(sz);
    Long off = 0;
    auto take = [&](const Long n) { Iterator<Real> r = sb.begin() + off; off += n; return r; };

    Matrix<Real> FS((DIM+1)*p, p, take((DIM+1)*nn), false);
    Matrix<Real> G((DIM+1)*p, 2*ns, take(2*(DIM+1)*(Long)p*ns), false);
    Matrix<Real> As(NA, p, take((Long)NA*p), false), Tmp(NA, 2*p, take(2*(Long)NA*p), false);
    Matrix<Real> HGall(ns*NR, p, take((Long)ns*NR*p), false);
    Matrix<Real> XdX(ns*NR, nt, take((Long)ns*NR*nt), false);
    Vector<Real> Xs(DIM*nq, take((Long)DIM*nq), false), Xn(DIM*nq, take((Long)DIM*nq), false);
    Vector<Real> wq(nq, take(nq), false), fq(nq, take(nq), false);
    Matrix<Real> Mker(nq*KDIM0, KDIM1, take(nq*KDIM0*KDIM1), false);
    Matrix<Real> KW(ns*C, nt, take((Long)C*nq), false);
    Matrix<Real> Zall(ns*C, p, take((Long)ns*C*p), false);
    Matrix<Real> Yi(C, p, take((Long)C*p), false), Yall(C*p, ns, take((Long)C*p*ns), false);
    Matrix<Real> Pc(p, p, take(nn), false);

    FS.SetZero();
    for (Integer k = 0; k < DIM; k++)
      for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++)
        FS[k*p + (T.swap_ab ? j : i)][T.swap_ab ? i : j] = Fsh[k*nn + i*p + j];
    if (fnod) for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++)
      FS[DIM*p + (T.swap_ab ? j : i)][T.swap_ab ? i : j] = (*fnod)[i*p+j];

    Matrix<Real>::GEMM(G, FS, T.WbC);          // stage 1: collapsed index, value+derivative at once

    for (Long i = 0; i < ns; i++) {            // stage 2a: one GEMM per s-node
      for (Integer k = 0; k < DIM; k++) for (Integer m = 0; m < p; m++) {
        As[k][m] = G[k*p+m][i]; As[DIM+k][m] = G[k*p+m][ns+i];
      }
      for (Integer m = 0; m < p; m++) As[2*DIM][m] = G[DIM*p+m][i];
      Matrix<Real>::GEMM(Tmp, As, T.MiC[i]);
      for (Integer k = 0; k < DIM; k++) for (Integer m = 0; m < p; m++) {
        HGall[i*NR + k        ][m] = Tmp[k][m];          // value
        HGall[i*NR + DIM + k  ][m] = Tmp[k][p+m];        // d/d_alpha
        HGall[i*NR + 2*DIM + k][m] = Tmp[DIM+k][m];      // d/d_beta
      }
      for (Integer m = 0; m < p; m++) HGall[i*NR + 3*DIM][m] = Tmp[2*DIM][m];
    }

    Matrix<Real>::GEMM(XdX, HGall, T.Tt);      // stage 2b: one GEMM for every s-node at once

    for (Long i = 0; i < ns; i++) for (Long j = 0; j < nt; j++) {
      const Long q = i*nt + j;
      const Real a0 = XdX[i*NR+DIM+0][j], a1 = XdX[i*NR+DIM+1][j], a2 = XdX[i*NR+DIM+2][j];
      const Real b0 = XdX[i*NR+2*DIM+0][j], b1 = XdX[i*NR+2*DIM+1][j], b2 = XdX[i*NR+2*DIM+2][j];
      const Real n0 = T.nsign*(a1*b2-a2*b1), n1 = T.nsign*(a2*b0-a0*b2), n2 = T.nsign*(a0*b1-a1*b0);
      const Real ar = sqrt<Real>(n0*n0+n1*n1+n2*n2), ia = (ar > 0 ? (Real)1/ar : (Real)0);
      for (Integer k = 0; k < DIM; k++) Xs[q*DIM+k] = XdX[i*NR+k][j];
      Xn[q*DIM+0] = n0*ia; Xn[q*DIM+1] = n1*ia; Xn[q*DIM+2] = n2*ia;
      wq[q] = ar*T.JW[q];
      if (fnod) fq[q] = XdX[i*NR+3*DIM][j];
    }

    ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xs, Xn);
    for (Long i = 0; i < ns; i++) for (Long j = 0; j < nt; j++) {
      const Long q = i*nt + j;
      for (Integer k0 = 0; k0 < KDIM0; k0++) for (Integer k1 = 0; k1 < KDIM1; k1++)
        KW[i*C + k0*KDIM1+k1][j] = Mker[q*KDIM0+k0][k1]*wq[q];
    }
    if (Idir && fnod) for (Integer c = 0; c < C; c++) {
      Real u = 0;
      for (Long i = 0; i < ns; i++) for (Long j = 0; j < nt; j++) u += KW[i*C+c][j]*fq[i*nt+j];
      (*Idir)[c] += u;
    }

    Matrix<Real>::GEMM(Zall, KW, T.TtT);       // adjoint of stage 2b, also a single GEMM
    for (Long i = 0; i < ns; i++) {
      const Matrix<Real> Zi(C, p, (Iterator<Real>)Zall.begin() + i*(Long)C*p, false);
      Matrix<Real>::GEMM(Yi, Zi, T.MiT[i]);
      for (Integer c = 0; c < C; c++) for (Integer m = 0; m < p; m++) Yall[c*p+m][i] = Yi[c][m];
    }
    for (Integer c = 0; c < C; c++) {
      const Matrix<Real> Yc(p, ns, (Iterator<Real>)Yall.begin() + (Long)c*p*ns, false);
      Matrix<Real>::GEMM(Pc, Yc, T.WbT);
      for (Integer m = 0; m < p; m++) for (Integer n = 0; n < p; n++)
        Proj[c][T.swap_ab ? n*p+m : m*p+n] += Pc[m][n];
    }
  }
}

// Proj[c][node] = sum_q K_c(0,X_q) * L_node(u_q,v_q) * |Xu x Xv|_q * JW_q.
// When fnod is given, Idir[c] is the same quadrature contracted directly against
// the interpolated density; Proj.f and Idir must agree to roundoff (adjoint check).
// All temporaries come from the per-thread ScratchPool arena.
template <class Real, class Ker> void DuffyEval(Matrix<Real>& Proj, Vector<Real>* Idir,
                                                const DuffyRule<Real>& R, const Vector<Real>& Fsh,
                                                const Vector<Real>* fnod, const Ker& ker,
                                                const Contract mode_in = Contract::Fused, const bool use_global = true) {
  const Contract mode = (use_global ? g_contract : mode_in);
  if (mode == Contract::TwoStageB) { DuffyEvalB(Proj, Idir, R, Fsh, fnod, ker); return; }
  static constexpr Integer KDIM0 = Ker::SrcDim(), KDIM1 = Ker::TrgDim();
  const Integer C = KDIM0*KDIM1;
  const Integer p = R.p;
  const Long nn = (Long)p*p;
  Proj.ReInit(C, nn); Proj.SetZero();
  if (Idir) { Idir->ReInit(C); Idir->SetZero(); }
  StaticArray<Real,DIM> Xt0{0,0,0};
  const Vector<Real> Xt0_v(DIM, Xt0, false);

  for (const TriTable<Real>& T : R.tri) {
    const Long ns = T.ns, nt = T.nt, nq = ns*nt;
    const Long sz = DIM*nn + nn + 2*(Long)DIM*p*ns + (Long)p*ns + 2*(Long)DIM*nq + 2*nq
                  + 2*(Long)DIM*p + 3*(Long)DIM*p + 3*(Long)DIM*nt + nt + 2*(Long)p
                  + nq*KDIM0*KDIM1 + (Long)C*nq + (Long)C*p*ns + (Long)C*nt + 2*(Long)C*p + nn;
    ScratchBuf<Real> sb(sz);
    Long off = 0;
    auto take = [&](const Long n) { Iterator<Real> r = sb.begin() + off; off += n; return r; };

    Matrix<Real> Floc(DIM*p, p, take(DIM*nn), false), floc(p, p, take(nn), false);
    Matrix<Real> Gb(DIM*p, ns, take((Long)DIM*p*ns), false), Gd(DIM*p, ns, take((Long)DIM*p*ns), false);
    Matrix<Real> Hden(p, ns, take((Long)p*ns), false);
    Vector<Real> Xs(DIM*nq, take((Long)DIM*nq), false), Xn(DIM*nq, take((Long)DIM*nq), false);
    Vector<Real> wq(nq, take(nq), false), fq(nq, take(nq), false);
    Matrix<Real> Ai(DIM, p, take((Long)DIM*p), false), Adi(DIM, p, take((Long)DIM*p), false);
    Matrix<Real> HG(3*DIM, p, take(3*(Long)DIM*p), false);
    Matrix<Real> XdX(3*DIM, nt, take(3*(Long)DIM*nt), false);
    Matrix<Real> fi(1, nt, take(nt), false), hi(1, p, take(p), false), h2(1, p, take(p), false);
    Matrix<Real> Mker(nq*KDIM0, KDIM1, take(nq*KDIM0*KDIM1), false);
    Matrix<Real> KW(C, nq, take((Long)C*nq), false);
    Matrix<Real> Yall(C*p, ns, take((Long)C*p*ns), false);
    Matrix<Real> KWi(C, nt, take((Long)C*nt), false), Yi(C, p, take((Long)C*p), false), Zi(C, p, take((Long)C*p), false);
    Matrix<Real> Pc(p, p, take(nn), false);
    Matrix<Real> Hv(DIM, p, HG.begin(),                false);
    Matrix<Real> Hd(DIM, p, HG.begin() + (Long)DIM*p,  false);
    Matrix<Real> Hb(DIM, p, HG.begin() + 2*(Long)DIM*p,false);
    Matrix<Real> Xm (DIM, nt, XdX.begin(),                 false);
    Matrix<Real> dXa(DIM, nt, XdX.begin() + (Long)DIM*nt,  false);
    Matrix<Real> dXb(DIM, nt, XdX.begin() + 2*(Long)DIM*nt,false);

    for (Integer k = 0; k < DIM; k++)
      for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++)
        Floc[k*p + (T.swap_ab ? j : i)][T.swap_ab ? i : j] = Fsh[k*nn + i*p + j];
    if (fnod) for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++)
      floc[T.swap_ab ? j : i][T.swap_ab ? i : j] = (*fnod)[i*p+j];

    // Stage 1: contract the collapsed (beta) index.
    Matrix<Real>::GEMM(Gb, Floc, T.Wb);
    Matrix<Real>::GEMM(Gd, Floc, T.WbD);
    if (fnod) Matrix<Real>::GEMM(Hden, floc, T.Wb);

    // Stage 2: contract the alpha index, per s-node.
    for (Long i = 0; i < ns; i++) {
      for (Integer k = 0; k < DIM; k++) for (Integer m = 0; m < p; m++) { Ai[k][m] = Gb[k*p+m][i]; Adi[k][m] = Gd[k*p+m][i]; }
      if (fnod) for (Integer m = 0; m < p; m++) hi[0][m] = Hden[m][i];
      if (mode == Contract::Fused) {
        Matrix<Real>::GEMM(Xm,  Ai,  T.Wa[i]);
        Matrix<Real>::GEMM(dXa, Ai,  T.WaD[i]);
        Matrix<Real>::GEMM(dXb, Adi, T.Wa[i]);
        if (fnod) Matrix<Real>::GEMM(fi, hi, T.Wa[i]);
      } else {
        Matrix<Real>::GEMM(Hv, Ai,  T.Mi[i]);
        Matrix<Real>::GEMM(Hd, Ai,  T.MiD[i]);
        Matrix<Real>::GEMM(Hb, Adi, T.Mi[i]);
        Matrix<Real>::GEMM(XdX, HG, T.Tt);
        if (fnod) { Matrix<Real>::GEMM(h2, hi, T.Mi[i]); Matrix<Real>::GEMM(fi, h2, T.Tt); }
      }
      for (Long j = 0; j < nt; j++) {
        const Long q = i*nt + j;
        const Real a0 = dXa[0][j], a1 = dXa[1][j], a2 = dXa[2][j];
        const Real b0 = dXb[0][j], b1 = dXb[1][j], b2 = dXb[2][j];
        const Real n0 = T.nsign*(a1*b2-a2*b1), n1 = T.nsign*(a2*b0-a0*b2), n2 = T.nsign*(a0*b1-a1*b0);
        const Real ar = sqrt<Real>(n0*n0+n1*n1+n2*n2), ia = (ar > 0 ? (Real)1/ar : (Real)0);
        for (Integer k = 0; k < DIM; k++) Xs[q*DIM+k] = Xm[k][j];
        Xn[q*DIM+0] = n0*ia; Xn[q*DIM+1] = n1*ia; Xn[q*DIM+2] = n2*ia;
        wq[q] = ar*T.JW[q];
        if (fnod) fq[q] = fi[0][j];
      }
    }

    ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xs, Xn);
    for (Long q = 0; q < nq; q++)
      for (Integer k0 = 0; k0 < KDIM0; k0++) for (Integer k1 = 0; k1 < KDIM1; k1++)
        KW[k0*KDIM1+k1][q] = Mker[q*KDIM0+k0][k1]*wq[q];
    if (Idir && fnod) for (Integer c = 0; c < C; c++) { Real u = 0; for (Long q = 0; q < nq; q++) u += KW[c][q]*fq[q]; (*Idir)[c] += u; }

    // Projection: exact adjoint of stages 1-2 -- same operators, reversed order.
    for (Long i = 0; i < ns; i++) {
      for (Integer c = 0; c < C; c++) for (Long j = 0; j < nt; j++) KWi[c][j] = KW[c][i*nt+j];
      if (mode == Contract::Fused) Matrix<Real>::GEMM(Yi, KWi, T.WaT[i]);
      else { Matrix<Real>::GEMM(Zi, KWi, T.TtT); Matrix<Real>::GEMM(Yi, Zi, T.MiT[i]); }
      for (Integer c = 0; c < C; c++) for (Integer m = 0; m < p; m++) Yall[c*p+m][i] = Yi[c][m];
    }
    for (Integer c = 0; c < C; c++) {
      const Matrix<Real> Yc(p, ns, (Iterator<Real>)Yall.begin() + (Long)c*p*ns, false);
      Matrix<Real>::GEMM(Pc, Yc, T.WbT);
      for (Integer m = 0; m < p; m++) for (Integer n = 0; n < p; n++)
        Proj[c][T.swap_ab ? n*p+m : m*p+n] += Pc[m][n];
    }
  }
}

template <class Real> void ApplyProj(Vector<Real>& I, const Matrix<Real>& Proj, const Vector<Real>& f) {
  I.ReInit(Proj.Dim(0)); I.SetZero();
  for (Long c = 0; c < Proj.Dim(0); c++) { Real s = 0; for (Long q = 0; q < Proj.Dim(1); q++) s += Proj[c][q]*f[q]; I[c] = s; }
}

template <class Real> double RelErr(const Vector<Real>& a, const Vector<Real>& b) {
  Real num = 0, den = 0;
  for (Long i = 0; i < a.Dim(); i++) { const Real d = a[i]-b[i]; num = std::max(D_(num), D_(fabs<Real>(d))); den = std::max(D_(den), D_(fabs<Real>(b[i]))); }
  return D_(num)/std::max(1e-300, D_(den));
}

// ================================================================= mode: gate

// Small-r order of a kernel: evaluate along a parameter-space ray from the
// target node and fit the local power of r; also print the Duffy integrand
// K * s * |Xu x Xv|, which must stay bounded as s -> 0.
template <class Real, class Ker> void GateKernel(const ModelElem<Real>& el, const Integer ti, const Integer tj, const Ker& ker, const char* nm) {
  static constexpr Integer KDIM0 = Ker::SrcDim(), KDIM1 = Ker::TrgDim();
  const Integer p = el.p;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  const Real u0 = nds[ti], v0 = nds[tj];
  Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
  const Real dirs[4][2] = {{1,0},{0,1},{0.6,0.8},{-0.7,0.3}};
  std::printf("  %-14s", nm);
  double worst_slope = 0; double drift = 0;
  for (Integer dd = 0; dd < 4; dd++) {
    std::vector<double> lr, lk, lg;
    for (Integer k = 4; k <= 34; k += 2) {
      const Real s = pow<Real>((Real)0.5, (Integer)k);
      Vector<Real> uq(1), vq(1);
      uq[0] = u0 + s*(Real)dirs[dd][0]*(Real)0.3;
      vq[0] = v0 + s*(Real)dirs[dd][1]*(Real)0.3;
      Vector<Real> X, dXu, dXv;
      EvalGeomDirect<Real>(X, dXu, dXv, nullptr, p, Fsh, nullptr, uq, vq);
      const Real n0 = dXu[1]*dXv[2]-dXu[2]*dXv[1], n1 = dXu[2]*dXv[0]-dXu[0]*dXv[2], n2 = dXu[0]*dXv[1]-dXu[1]*dXv[0];
      const Real ar = sqrt<Real>(n0*n0+n1*n1+n2*n2), ia = 1/ar;
      Vector<Real> Xs(DIM), Xn(DIM);
      for (Integer m = 0; m < DIM; m++) Xs[m] = X[m];
      Xn[0] = n0*ia; Xn[1] = n1*ia; Xn[2] = n2*ia;
      StaticArray<Real,DIM> Xt0{0,0,0};
      const Vector<Real> Xt0_v(DIM, Xt0, false);
      Matrix<Real> Mker;
      ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xs, Xn);
      Real kmax = 0;
      for (Integer k0 = 0; k0 < KDIM0; k0++) for (Integer k1 = 0; k1 < KDIM1; k1++) kmax = std::max(D_(kmax), D_(fabs<Real>(Mker[k0][k1])));
      const Real r = sqrt<Real>(X[0]*X[0]+X[1]*X[1]+X[2]*X[2]);
      lr.push_back(std::log(D_(r))); lk.push_back(std::log(D_(kmax)));
      lg.push_back(D_(kmax)*D_(s)*D_(ar));
    }
    // slope over the last few decades
    const size_t n = lr.size();
    const double sl = (lk[n-1]-lk[n-4])/(lr[n-1]-lr[n-4]);
    worst_slope = std::min(worst_slope, sl);
    drift = std::max(drift, std::fabs(lg[n-1]/lg[n-5] - 1.0));
    std::printf("  p=%7.4f", -sl);
  }
  std::printf("   |K*s*dA drift|=%.1e   -> %s\n", drift, (-worst_slope < 1.5 ? "O(1/r): SAFE" : "NOT O(1/r)"));
}

template <class Real> void ModeGate(const Integer p) {
  std::printf("\n=== SECTION 7 GATE: small-r order of the kernels ===\n");
  std::printf("fitted p in |K| ~ r^-p, along 4 parameter-space rays from the target node\n");
  std::printf("K*s*dA drift = relative change of the Duffy integrand between s=2^-26 and 2^-34\n");
  for (const char* g : {"poly", "sphere", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    for (auto tt : {std::pair<Integer,Integer>{p/2,p/3}, std::pair<Integer,Integer>{0,0}}) {
      std::printf("\ngeom=%-7s target node (ti,tj)=(%d,%d)\n", g, (int)tt.first, (int)tt.second);
      GateKernel<Real>(el, tt.first, tt.second, Laplace3D_FxU(), "Laplace3D-FxU");
      GateKernel<Real>(el, tt.first, tt.second, Laplace3D_DxU(), "Laplace3D-DxU");
      GateKernel<Real>(el, tt.first, tt.second, Stokes3D_FxU(),  "Stokes3D-FxU");
      GateKernel<Real>(el, tt.first, tt.second, Stokes3D_DxU(),  "Stokes3D-DxU");
    }
  }
}

// ================================================================== mode: ref

// Analytic  \int\int du dv / (4 pi r)  over the unit square, flat element.
template <class Real> Real FlatSLExact(const Real u0, const Real v0) {
  auto rect = [](const Real A, const Real B) { return A*Asinh_<Real>(B/A) + B*Asinh_<Real>(A/B); };
  const Real I = rect(u0,v0) + rect(1-u0,v0) + rect(u0,1-v0) + rect(1-u0,1-v0);
  return I/(4*const_pi<Real>());
}

template <class Real> void ModeRef(const Integer p) {
  std::printf("\n=== REFERENCE CROSS-VALIDATION ===\n");
  const Integer ti = p/2, tj = p/3;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  for (const char* g : {"flat", "skew", "poly", "sphere", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
    Real G[4]; MetricAt<Real>(G, p, Fsh, nds[ti], nds[tj]);
    Vector<Real> ones((Long)p*p); for (Long i = 0; i < ones.Dim(); i++) ones[i] = 1;

    // Duffy reference (heavy) and a coarser one, to show it is converged.
    DuffyRule<Real> Rr, Rr2;
    BuildDuffyRule<Real>(Rr,  p, ti, tj, G, 48, TRule::Graded, 3, 26);
    BuildDuffyRule<Real>(Rr2, p, ti, tj, G, 36, TRule::Graded, 2, 20);
    Matrix<Real> P1, P2; Vector<Real> I1, I2;
    DuffyEval<Real>(P1, nullptr, Rr,  Fsh, nullptr, Laplace3D_FxU());
    DuffyEval<Real>(P2, nullptr, Rr2, Fsh, nullptr, Laplace3D_FxU());
    Vector<Real> A1, A2; ApplyProj(A1, P1, ones); ApplyProj(A2, P2, ones);

    // Independent polar reference.
    Vector<Real> uq, vq, wq, I3;
    PolarRule<Real>(uq, vq, wq, nds[ti], nds[tj], G, 36, 18, 20);
    RefIntegrate<Real>(I3, p, Fsh, nullptr, uq, vq, wq, Laplace3D_FxU());

    std::printf("  %-7s  SL: duffy_ref=%.20e  self-conv=%.2e  vs polar(%ld pts)=%.2e",
                g, D_(A1[0]), RelErr(A2, A1), (long)uq.Dim(), RelErr(I3, A1));
    if (!std::strcmp(g, "flat")) {
      Vector<Real> Ie(1); Ie[0] = FlatSLExact<Real>(nds[ti], nds[tj]);
      std::printf("  vs analytic=%.2e", RelErr(A1, Ie));
    }
    std::printf("\n");

    // Same for the double layer.
    Matrix<Real> Q1, Q2; DuffyEval<Real>(Q1, nullptr, Rr, Fsh, nullptr, Laplace3D_DxU());
    DuffyEval<Real>(Q2, nullptr, Rr2, Fsh, nullptr, Laplace3D_DxU());
    Vector<Real> B1, B2; ApplyProj(B1, Q1, ones); ApplyProj(B2, Q2, ones);
    Vector<Real> J3; RefIntegrate<Real>(J3, p, Fsh, nullptr, uq, vq, wq, Laplace3D_DxU());
    std::printf("  %-7s  DL: duffy_ref=%.20e  self-conv=%.2e  vs polar=%.2e\n",
                g, D_(B1[0]), RelErr(B2, B1), RelErr(J3, B1));
  }
}

// =================================================================== mode: v1

template <class Real, class Ker> void V1Kernel(const ModelElem<Real>& el, const Integer ti, const Integer tj, const Ker& ker, const char* nm,
                                               const Integer tlvl_extra, const Integer tq) {
  const Integer p = el.p;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
  Real G[4]; MetricAt<Real>(G, p, Fsh, nds[ti], nds[tj]);
  Vector<Real> ones((Long)p*p); for (Long i = 0; i < ones.Dim(); i++) ones[i] = 1;

  DuffyRule<Real> Rr; BuildDuffyRule<Real>(Rr, p, ti, tj, G, 48, TRule::Graded, 3, 26);
  Matrix<Real> Pr; DuffyEval<Real>(Pr, nullptr, Rr, Fsh, nullptr, ker);
  Vector<Real> Iref; ApplyProj(Iref, Pr, ones);

  DuffyRule<Real> Rt; BuildDuffyRule<Real>(Rt, p, ti, tj, G, 4, TRule::Graded, tlvl_extra, tq);
  std::printf("  %-14s t-rule graded(extra=%d,q=%d) nt=%ld :", nm, (int)tlvl_extra, (int)tq, (long)Rt.tri[0].nt);
  static const Integer qs_lst[] = {2,4,6,8,10,12,14,16,20,24,32,40};
  std::printf("\n     n_s :");
  for (Integer qs : qs_lst) std::printf(" %8d", (int)qs);
  std::printf("\n     err :");
  for (Integer qs : qs_lst) {
    DuffyRule<Real> R; BuildDuffyRule<Real>(R, p, ti, tj, G, qs, TRule::Graded, tlvl_extra, tq);
    Matrix<Real> P; DuffyEval<Real>(P, nullptr, R, Fsh, nullptr, ker);
    Vector<Real> I; ApplyProj(I, P, ones);
    std::printf(" %8.1e", RelErr(I, Iref));
  }
  std::printf("\n");
}

template <class Real> void ModeV1(const Integer p) {
  std::printf("\n=== V1: error vs n_s (s-direction), t-rule held fixed ===\n");
  std::printf("order p=%d; reference = Duffy with n_s=48, graded t (extra=3,q=26)\n", (int)p);
  for (const char* g : {"flat", "skew", "poly", "sphere", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    for (auto tt : {std::pair<Integer,Integer>{p/2,p/3}, std::pair<Integer,Integer>{0,0}}) {
      std::printf("\ngeom=%-7s (ti,tj)=(%d,%d)\n", g, (int)tt.first, (int)tt.second);
      for (auto tp : {std::pair<Integer,Integer>{0,8}, std::pair<Integer,Integer>{1,12}, std::pair<Integer,Integer>{2,20}}) {
        V1Kernel<Real>(el, tt.first, tt.second, Laplace3D_FxU(), "Laplace3D-FxU", tp.first, tp.second);
        V1Kernel<Real>(el, tt.first, tt.second, Laplace3D_DxU(), "Laplace3D-DxU", tp.first, tp.second);
      }
    }
  }
}

// ================================================================= mode: tune

template <class Real> struct NodeData { Integer ti = 0, tj = 0; Vector<Real> Fsh; Real G[4] = {0,0,0,0}; };

template <class Real> void BuildNodes(std::vector<NodeData<Real>>& nd, const ModelElem<Real>& el, const Integer stride) {
  const Integer p = el.p;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  nd.clear();
  for (Integer ti = 0; ti < p; ti += stride) for (Integer tj = 0; tj < p; tj += stride) {
    NodeData<Real> d; d.ti = ti; d.tj = tj;
    ShiftToTarget(d.Fsh, el, ti, tj);
    MetricAt<Real>(d.G, p, d.Fsh, nds[ti], nds[tj]);
    nd.push_back(std::move(d));
  }
}

// Reference operator blocks (one per target node), built once per (geometry, kernel).
template <class Real, class Ker> void BuildRefs(std::vector<Matrix<Real>>& Pr, const std::vector<NodeData<Real>>& nd,
                                                const Integer p, const Ker& ker, const Integer qs_ref, const Integer tx, const Integer tq) {
  Pr.resize(nd.size());
  for (size_t m = 0; m < nd.size(); m++) {
    DuffyRule<Real> R; BuildDuffyRule<Real>(R, p, nd[m].ti, nd[m].tj, nd[m].G, qs_ref, TRule::Graded, tx, tq);
    DuffyEval<Real>(Pr[m], nullptr, R, nd[m].Fsh, nullptr, ker);
  }
}

// Worst-case relative error over the target nodes for one (qs, t-rule) config,
// against the cached references, with both a constant and a varying density.
template <class Real, class Ker> void TuneOne(double& werr, double& avgpts, const std::vector<NodeData<Real>>& nd,
                                              const std::vector<Matrix<Real>>& Pr, const Integer p, const Ker& ker,
                                              const Integer qs, const TRule tr, const Integer tp1, const Integer tp2,
                                              const Vector<Real>& ones, const Vector<Real>& fvar,
                                              const bool param = false, const Integer nk = 0, const Integer nw = 0,
                                              double* maxpts = nullptr) {
  werr = 0; avgpts = 0;
  if (maxpts) *maxpts = 0;
  for (size_t m = 0; m < nd.size(); m++) {
    DuffyRule<Real> R; BuildDuffyRule<Real>(R, p, nd[m].ti, nd[m].tj, nd[m].G, qs, tr, tp1, tp2, param, nk, nw);
    if (maxpts) *maxpts = std::max(*maxpts, (double)R.NPts());
    Matrix<Real> P; DuffyEval<Real>(P, nullptr, R, nd[m].Fsh, nullptr, ker);
    Vector<Real> a, b;
    ApplyProj(a, P, ones); ApplyProj(b, Pr[m], ones);
    werr = std::max(werr, RelErr(a, b));
    ApplyProj(a, P, fvar); ApplyProj(b, Pr[m], fvar);
    werr = std::max(werr, RelErr(a, b));
    avgpts += (double)R.NPts();
  }
  avgpts /= (double)nd.size();
}

template <class Real> void ModeTune(const Integer p, const Integer stride) {
  std::printf("\n=== TUNE: q_s and t-rule ===\n");
  std::printf("worst relative error over target nodes (stride %d), constant AND varying density\n", (int)stride);
  std::printf("reference: Duffy with n_s=48, graded t (extra=3, q=26)\n");
  const Integer qs_ref = 48, txr = 3, tqr = 26;
  for (const char* g : {"poly", "twist4", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
    std::vector<NodeData<Real>> nd; BuildNodes(nd, el, stride);
    Vector<Real> ones((Long)p*p), fvar((Long)p*p);
    for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++) {
      ones[i*p+j] = 1;
      fvar[i*p+j] = sin<Real>((Real)2.3*nds[i] + (Real)1.1)*cos<Real>((Real)3.1*nds[j] - (Real)0.4);
    }
    std::vector<Matrix<Real>> R0, R1, R2, R3;
    BuildRefs<Real>(R0, nd, p, Laplace3D_FxU(), qs_ref, txr, tqr);
    BuildRefs<Real>(R1, nd, p, Laplace3D_DxU(), qs_ref, txr, tqr);
    BuildRefs<Real>(R2, nd, p, Stokes3D_FxU(),  qs_ref, txr, tqr);
    BuildRefs<Real>(R3, nd, p, Stokes3D_DxU(),  qs_ref, txr, tqr);
    const Integer qs_lst[] = {p+0, p+2, p+4, p+6, p+8, p+12, p+16};
    std::printf("\n--- geom=%s, order=%d, %d target nodes ---\n", g, (int)p, (int)nd.size());
    std::printf("max relative error over {Lap-SL,Lap-DL,Stk-SL,Stk-DL} x nodes x {const,varying density}\n");
    std::printf("%-22s %6s", "t-rule", "sum_nt");
    for (Integer qs : qs_lst) std::printf("   err@qs=%-2d (pts)", (int)qs);
    std::printf("\n");
    auto row = [&](const char* nm, const TRule tr, const Integer t1, const Integer t2) {
      double nt_tot = 0;
      std::printf("%-22s", nm);
      char buf[512]; int bl = 0;
      for (Integer qs : qs_lst) {
        double e[4], a[4];
        TuneOne<Real>(e[0], a[0], nd, R0, p, Laplace3D_FxU(), qs, tr, t1, t2, ones, fvar);
        TuneOne<Real>(e[1], a[1], nd, R1, p, Laplace3D_DxU(), qs, tr, t1, t2, ones, fvar);
        TuneOne<Real>(e[2], a[2], nd, R2, p, Stokes3D_FxU(),  qs, tr, t1, t2, ones, fvar);
        TuneOne<Real>(e[3], a[3], nd, R3, p, Stokes3D_DxU(),  qs, tr, t1, t2, ones, fvar);
        nt_tot = a[0]/(double)qs;
        bl += std::snprintf(buf+bl, sizeof(buf)-bl, " %7.1e(%5.0f)", std::max(std::max(e[0],e[1]),std::max(e[2],e[3])), a[0]);
      }
      std::printf(" %6.0f%s\n", nt_tot, buf);
    };
    char nm[64];
    for (Integer tx : {0, 1, 2}) for (Integer tq : {8, 12, 16}) {
      std::snprintf(nm, sizeof nm, "graded(x=%d,q=%d)", (int)tx, (int)tq);
      row(nm, TRule::Graded, tx, tq);
    }
    for (Integer nt : {16, 24, 32, 36, 40, 44, 48, 52, 56, 64}) {
      std::snprintf(nm, sizeof nm, "sinh(nt=%d)", (int)nt);
      row(nm, TRule::Sinh, nt, 0);
    }
    for (Integer np : {2, 3, 4}) for (Integer tq : {8, 12, 16}) {
      std::snprintf(nm, sizeof nm, "sinhcomp(%dx%d)", (int)np, (int)tq);
      row(nm, TRule::SinhComp, np, tq);
    }
  }
}

// ============================================================== mode: adjoint

template <class Real, class Ker> void AdjOne(const ModelElem<Real>& el, const Integer ti, const Integer tj, const Ker& ker, const char* nm) {
  const Integer p = el.p;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
  Real G[4]; MetricAt<Real>(G, p, Fsh, nds[ti], nds[tj]);
  Vector<Real> f((Long)p*p);
  for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++)
    f[i*p+j] = sin<Real>((Real)2.3*nds[i] + (Real)1.1)*cos<Real>((Real)3.1*nds[j] - (Real)0.4);
  DuffyRule<Real> R; BuildDuffyRule<Real>(R, p, ti, tj, G, p+4, TRule::Graded, 1, 12);
  Matrix<Real> P; Vector<Real> Idir;
  DuffyEval<Real>(P, &Idir, R, Fsh, &f, ker);
  Vector<Real> Iop; ApplyProj(Iop, P, f);
  // converged Duffy, and an independent polar reference, both with the same density
  DuffyRule<Real> Rr; BuildDuffyRule<Real>(Rr, p, ti, tj, G, 48, TRule::Graded, 3, 26);
  Matrix<Real> Pr; DuffyEval<Real>(Pr, nullptr, Rr, Fsh, nullptr, ker);
  Vector<Real> Idr; ApplyProj(Idr, Pr, f);
  Vector<Real> uq, vq, wq, Iref;
  PolarRule<Real>(uq, vq, wq, nds[ti], nds[tj], G, 40, 16, 20);
  RefIntegrate<Real>(Iref, p, Fsh, &f, uq, vq, wq, ker);
  std::printf("  %-14s (%2d,%2d)  adjoint=%.1e  duffy_ref vs polar=%.1e  tested(qs=p+4,graded(1,12),%ld pts) vs duffy_ref=%.1e\n",
              nm, (int)ti, (int)tj, RelErr(Iop, Idir), RelErr(Idr, Iref), (long)R.NPts(), RelErr(Iop, Idr));
}

template <class Real> void ModeAdjoint(const Integer p) {
  std::printf("\n=== ADJOINT / VARYING-DENSITY CHECK ===\n");
  for (const char* g : {"skew", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    std::printf("geom=%s\n", g);
    for (auto tt : {std::pair<Integer,Integer>{p/2,p/3}, std::pair<Integer,Integer>{0,p-1}}) {
      AdjOne<Real>(el, tt.first, tt.second, Laplace3D_FxU(), "Laplace3D-FxU");
      AdjOne<Real>(el, tt.first, tt.second, Laplace3D_DxU(), "Laplace3D-DxU");
      AdjOne<Real>(el, tt.first, tt.second, Stokes3D_FxU(),  "Stokes3D-FxU");
      AdjOne<Real>(el, tt.first, tt.second, Stokes3D_DxU(),  "Stokes3D-DxU");
    }
  }
}


// ============================================================== mode: compare

// One quadrature configuration. variant: 0 = A (metric sinh), 1 = B (parameter-space
// graded), 2 = C (sinh from binned cot(theta), width-scale).
struct Cfg {
  const char* var; Integer qs; TRule tr; Integer t1, t2; bool param; Integer nk, nw;
};
struct Row { double e[4]; double pts, maxpts; };

template <class Real> void EvalCfgs(std::vector<Row>& rows, const std::vector<Cfg>& cfg,
                                    const std::vector<NodeData<Real>>& nd, const Integer p,
                                    const std::vector<Matrix<Real>>& R0, const std::vector<Matrix<Real>>& R1,
                                    const std::vector<Matrix<Real>>& R2, const std::vector<Matrix<Real>>& R3,
                                    const Vector<Real>& ones, const Vector<Real>& fvar) {
  rows.resize(cfg.size());
  for (size_t i = 0; i < cfg.size(); i++) {
    const Cfg& c = cfg[i];
    double a[4], mx = 0;
    TuneOne<Real>(rows[i].e[0], a[0], nd, R0, p, Laplace3D_FxU(), c.qs, c.tr, c.t1, c.t2, ones, fvar, c.param, c.nk, c.nw, &mx);
    TuneOne<Real>(rows[i].e[1], a[1], nd, R1, p, Laplace3D_DxU(), c.qs, c.tr, c.t1, c.t2, ones, fvar, c.param, c.nk, c.nw);
    TuneOne<Real>(rows[i].e[2], a[2], nd, R2, p, Stokes3D_FxU(),  c.qs, c.tr, c.t1, c.t2, ones, fvar, c.param, c.nk, c.nw);
    TuneOne<Real>(rows[i].e[3], a[3], nd, R3, p, Stokes3D_DxU(),  c.qs, c.tr, c.t1, c.t2, ones, fvar, c.param, c.nk, c.nw);
    rows[i].pts = a[0]; rows[i].maxpts = mx;
  }
}

// Cheapest config of variant `var` whose worst error over the selected kernels is
// <= tol on EVERY geometry. kmask: bit k enables kernel k.
static Long BestPts(const std::vector<Cfg>& cfg, const std::vector<std::vector<Row>>& rows,
                    const char* var, const double tol, const Integer kmask, Integer* which = nullptr) {
  double best = 1e300; Integer bi = -1;
  for (size_t i = 0; i < cfg.size(); i++) {
    if (std::strcmp(cfg[i].var, var)) continue;
    bool ok = true; double pts = 0;
    for (size_t g = 0; g < rows.size() && ok; g++) {
      for (Integer k = 0; k < 4; k++) if ((kmask>>k)&1) if (!(rows[g][i].e[k] <= tol)) ok = false;
      pts = std::max(pts, rows[g][i].pts);
    }
    if (ok && pts < best) { best = pts; bi = (Integer)i; }
  }
  if (which) *which = bi;
  return (bi < 0 ? -1 : (Long)(best+0.5));
}

template <class Real> void ModeCompare(const Integer p, const Integer stride, const std::string& grid) {
  const bool tight = (grid == "tight");
  const char* gnames[] = {"flat", "skew", "poly", "sphere", "twist4", "twist"};
  const Integer NG = 6;
  std::printf("\n=== VARIANT COMPARISON AT MATCHED ACCURACY (order %d, %s grid) ===\n", (int)p, grid.c_str());
  std::printf("A = metric-aware sinh | B = parameter-space dyadic graded | C = sinh from binned (cot(theta), width)\n");
  std::printf("errors are worst case over target nodes (stride %d) x {constant, varying} density\n", (int)stride);

  // config grid
  std::vector<Cfg> cfg;
  {
    std::vector<Integer> qsl, ntl;
    if (tight) { qsl = {p+4, p+8, p+16}; ntl = {40,44,48,52,56,60,64,72,80}; }
    else       { qsl = {p, p+4, p+8, p+12, p+16}; ntl = {16,20,24,28,32,36,40,44,48,52,56,64,72}; }
    for (Integer qs : qsl) for (Integer nt : ntl) {
      cfg.push_back({"A", qs, TRule::Sinh, nt, 0, false, 0, 0});
      cfg.push_back({"C", qs, TRule::Sinh, nt, 0, false, 16, 12});
    }
    std::vector<Integer> xl = {0,1,2,3};
    std::vector<Integer> qtl = tight ? std::vector<Integer>{10,12,16,20} : std::vector<Integer>{6,8,10,12,16,20};
    for (Integer qs : (tight ? std::vector<Integer>{p+4,p+8,p+16} : std::vector<Integer>{p,p+4,p+8,p+16}))
      for (Integer x : xl) for (Integer qt : qtl)
        cfg.push_back({"B", qs, TRule::Graded, x, qt, true, 0, 0});
  }
  std::printf("%d configurations x %d geometries\n", (int)cfg.size(), (int)NG);

  std::vector<std::vector<Row>> rows(NG);
  for (Integer g = 0; g < NG; g++) {
    ModelElem<Real> el; BuildModelElem(el, p, gnames[g]);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
    std::vector<NodeData<Real>> nd; BuildNodes(nd, el, stride);
    Vector<Real> ones((Long)p*p), fvar((Long)p*p);
    for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++) {
      ones[i*p+j] = 1;
      fvar[i*p+j] = sin<Real>((Real)2.3*nds[i] + (Real)1.1)*cos<Real>((Real)3.1*nds[j] - (Real)0.4);
    }
    std::vector<Matrix<Real>> R0, R1, R2, R3;
    BuildRefs<Real>(R0, nd, p, Laplace3D_FxU(), 48, 3, 26);
    BuildRefs<Real>(R1, nd, p, Laplace3D_DxU(), 48, 3, 26);
    BuildRefs<Real>(R2, nd, p, Stokes3D_FxU(),  48, 3, 26);
    BuildRefs<Real>(R3, nd, p, Stokes3D_DxU(),  48, 3, 26);
    EvalCfgs<Real>(rows[g], cfg, nd, p, R0, R1, R2, R3, ones, fvar);
    std::printf("  [%s done, %d nodes]\n", gnames[g], (int)nd.size());
  }

  // ---- per-geometry cheapest config at each tolerance ----
  const double tols[] = {1e-6, 1e-8, 1e-10, 1e-12, 1e-13};
  const Long basep[] = {10528, 32376, 67392, 153600, 203840};
  for (Integer g = 0; g < NG; g++) {
    std::vector<std::vector<Row>> one(1, rows[g]);
    std::printf("\n--- geom=%s ---\n", gnames[g]);
    std::printf("%8s %8s | %-26s | %-26s | %-26s\n", "tol", "baseline", "A: pts (qs,nt) ratio", "B: pts (qs,x,qt) ratio", "C: pts (qs,nt) ratio");
    for (Integer it = 0; it < 5; it++) {
      if (tight && it < 3) continue;
      if (!tight && it > 3) continue;
      std::printf("%8.0e %8ld |", tols[it], (long)basep[it]);
      for (const char* v : {"A","B","C"}) {
        Integer wi = -1;
        const Long pts = BestPts(cfg, one, v, tols[it], 0xF, &wi);
        if (pts < 0) std::printf(" %-26s |", "  --");
        else if (cfg[wi].tr == TRule::Sinh)
          std::printf(" %6ld (%2d,%2d) %5.1fx    |", (long)pts, (int)cfg[wi].qs, (int)cfg[wi].t1, (double)basep[it]/(double)pts);
        else
          std::printf(" %6ld (%2d,%d,%2d) %5.1fx |", (long)pts, (int)cfg[wi].qs, (int)cfg[wi].t1, (int)cfg[wi].t2, (double)basep[it]/(double)pts);
      }
      std::printf("\n");
    }
  }

  // ---- single config that works on ALL geometries (the headline) ----
  std::printf("\n--- WORST CASE OVER ALL %d GEOMETRIES (one config must satisfy every geometry) ---\n", (int)NG);
  std::printf("%8s %9s | %-24s | %-26s | %-24s\n", "tol", "baseline", "A pts (qs,nt)", "B pts (qs,x,qt)", "C pts (qs,nt)");
  for (Integer it = 0; it < 5; it++) {
    if (tight && it < 3) continue;
    if (!tight && it > 3) continue;
    std::printf("%8.0e %9ld |", tols[it], (long)basep[it]);
    for (const char* v : {"A","B","C"}) {
      Integer wi = -1;
      const Long pts = BestPts(cfg, rows, v, tols[it], 0xF, &wi);
      if (pts < 0) std::printf(" %-24s |", "  -- (not reached)");
      else if (cfg[wi].tr == TRule::Sinh)
        std::printf(" %6ld (%2d,%2d) %5.1fx  |", (long)pts, (int)cfg[wi].qs, (int)cfg[wi].t1, (double)basep[it]/(double)pts);
      else
        std::printf(" %6ld (%2d,%d,%2d) %5.1fx|", (long)pts, (int)cfg[wi].qs, (int)cfg[wi].t1, (int)cfg[wi].t2, (double)basep[it]/(double)pts);
    }
    std::printf("\n");
  }
  // Laplace-only vs Stokes-only
  std::printf("\n--- same, Laplace only (SL,DL) / Stokes only (SL,DL) ---\n");
  for (Integer it = 0; it < 5; it++) {
    if (tight && it < 3) continue;
    if (!tight && it > 3) continue;
    std::printf("%8.0e |", tols[it]);
    for (Integer km = 0; km < 2; km++) {
      const Integer mask = (km == 0 ? 0x3 : 0xC);
      std::printf("  %s:", km == 0 ? "Lap" : "Stk");
      for (const char* v : {"A","B","C"}) {
        const Long pts = BestPts(cfg, rows, v, tols[it], mask);
        if (pts < 0) std::printf(" %s=--", v); else std::printf(" %s=%ld", v, (long)pts);
      }
      std::printf(" |");
    }
    std::printf("\n");
  }

  // ---- geometry sweep at one fixed config per variant ----
  std::printf("\n--- GEOMETRY SWEEP: error of the config each variant needs for 1e-10 over all geometries ---\n");
  for (const char* v : {"A","B","C"}) {
    Integer wi = -1; BestPts(cfg, rows, v, tight ? 1e-12 : 1e-10, 0xF, &wi);
    if (wi < 0) { std::printf("%s: no config reached the target\n", v); continue; }
    std::printf("%s  qs=%d %s(%d,%d)  pts=%.0f :", v, (int)cfg[wi].qs,
                cfg[wi].tr == TRule::Sinh ? "sinh" : "graded", (int)cfg[wi].t1, (int)cfg[wi].t2, rows[0][wi].pts);
    for (Integer g = 0; g < NG; g++) {
      double w = 0; for (Integer k = 0; k < 4; k++) w = std::max(w, rows[g][wi].e[k]);
      std::printf("  %s=%.1e", gnames[g], w);
    }
    std::printf("\n");
  }
}

// ============================================================= mode: binstudy

// How coarsely can (cot(theta), width-scale) be binned before variant C loses to
// variant A? Reports error vs bin count, and the table memory each implies.
template <class Real> void ModeBinStudy(const Integer p, const Integer stride) {
  const char* gnames[] = {"poly", "twist4", "twist"};
  const Integer qs = p+4, nt = 56;
  const double tbl_MB = 4.0*(double)p*p*2.0*(double)p*qs*nt*8.0/1048576.0; // 4 tri x p^2 nodes x {Wa,WaD}
  std::printf("\n=== VARIANT C BIN STUDY (order %d, qs=%d, nt=%d, %.0f pts/target) ===\n",
              (int)p, (int)qs, (int)nt, 4.0*qs*nt);
  std::printf("kappa = cot(theta) binned uniformly on [-%.1f,%.1f]; width scale binned in log on [%.2f,%.1f]\n",
              KAP_MAX, KAP_MAX, W_MIN, W_MAX);
  std::printf("one table set (all p^2 nodes x 4 triangles, values+derivatives) = %.0f MB in double\n", tbl_MB);
  std::printf("%5s %5s %9s %9s | %-34s\n", "Nkap", "Nwid", "d(cot)", "tables MB", "worst error over 4 kernels");
  std::printf("%5s %5s %9s %9s | %10s %10s %10s\n", "", "", "", "", "poly", "twist4", "twist");
  struct GD { std::vector<NodeData<Real>> nd; std::vector<Matrix<Real>> R[4]; Vector<Real> ones, fvar; };
  std::vector<GD> gd(3);
  for (Integer g = 0; g < 3; g++) {
    ModelElem<Real> el; BuildModelElem(el, p, gnames[g]);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
    BuildNodes(gd[g].nd, el, stride);
    gd[g].ones.ReInit((Long)p*p); gd[g].fvar.ReInit((Long)p*p);
    for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++) {
      gd[g].ones[i*p+j] = 1;
      gd[g].fvar[i*p+j] = sin<Real>((Real)2.3*nds[i] + (Real)1.1)*cos<Real>((Real)3.1*nds[j] - (Real)0.4);
    }
    BuildRefs<Real>(gd[g].R[0], gd[g].nd, p, Laplace3D_FxU(), 48, 3, 26);
    BuildRefs<Real>(gd[g].R[1], gd[g].nd, p, Laplace3D_DxU(), 48, 3, 26);
    BuildRefs<Real>(gd[g].R[2], gd[g].nd, p, Stokes3D_FxU(),  48, 3, 26);
    BuildRefs<Real>(gd[g].R[3], gd[g].nd, p, Stokes3D_DxU(),  48, 3, 26);
  }
  auto run = [&](const Integer nk, const Integer nw, const bool exact) {
    if (exact) std::printf("%5s %5s %9s %9s |", "exact", "(A)", "0", "on the fly");
    else std::printf("%5d %5d %9.3f %9.0f |", (int)nk, (int)nw, 2*KAP_MAX/(double)nk, tbl_MB*nk*nw);
    for (Integer g = 0; g < 3; g++) {
      double w = 0, a;
      double e;
      TuneOne<Real>(e, a, gd[g].nd, gd[g].R[0], p, Laplace3D_FxU(), qs, TRule::Sinh, nt, 0, gd[g].ones, gd[g].fvar, false, exact?0:nk, exact?0:nw); w = std::max(w,e);
      TuneOne<Real>(e, a, gd[g].nd, gd[g].R[1], p, Laplace3D_DxU(), qs, TRule::Sinh, nt, 0, gd[g].ones, gd[g].fvar, false, exact?0:nk, exact?0:nw); w = std::max(w,e);
      TuneOne<Real>(e, a, gd[g].nd, gd[g].R[2], p, Stokes3D_FxU(),  qs, TRule::Sinh, nt, 0, gd[g].ones, gd[g].fvar, false, exact?0:nk, exact?0:nw); w = std::max(w,e);
      TuneOne<Real>(e, a, gd[g].nd, gd[g].R[3], p, Stokes3D_DxU(),  qs, TRule::Sinh, nt, 0, gd[g].ones, gd[g].fvar, false, exact?0:nk, exact?0:nw); w = std::max(w,e);
      std::printf(" %10.2e", w);
    }
    std::printf("\n");
  };
  run(0, 0, true);
  for (auto b : {std::pair<Integer,Integer>{1,1}, {2,2}, {3,3}, {4,4}, {6,4}, {8,6}, {12,8}, {16,12}, {24,16}, {48,32}}) run(b.first, b.second, false);

  // Is C limited by the bins or by n_t?  Sweep n_t at a fixed bin count: if the
  // error is flat, the bins are the floor and no amount of n_t rescues it.
  std::printf("\n--- n_t sweep at fixed bins (Nkap=16, Nwid=12) vs exact metric (A) ---\n");
  std::printf("%6s | %-32s | %-32s\n", "nt", "C: binned (16,12)", "A: exact metric");
  std::printf("%6s | %10s %10s %10s | %10s %10s %10s\n", "", "poly", "twist4", "twist", "poly", "twist4", "twist");
  for (Integer ntv : {40, 56, 72, 88}) {
    std::printf("%6d |", (int)ntv);
    for (Integer pass = 0; pass < 2; pass++) {
      for (Integer g = 0; g < 3; g++) {
        double w = 0, a, e;
        const Integer nk = (pass ? 0 : 16), nw = (pass ? 0 : 12);
        TuneOne<Real>(e,a,gd[g].nd,gd[g].R[0],p,Laplace3D_FxU(),qs,TRule::Sinh,ntv,0,gd[g].ones,gd[g].fvar,false,nk,nw); w=std::max(w,e);
        TuneOne<Real>(e,a,gd[g].nd,gd[g].R[1],p,Laplace3D_DxU(),qs,TRule::Sinh,ntv,0,gd[g].ones,gd[g].fvar,false,nk,nw); w=std::max(w,e);
        TuneOne<Real>(e,a,gd[g].nd,gd[g].R[2],p,Stokes3D_FxU(), qs,TRule::Sinh,ntv,0,gd[g].ones,gd[g].fvar,false,nk,nw); w=std::max(w,e);
        TuneOne<Real>(e,a,gd[g].nd,gd[g].R[3],p,Stokes3D_DxU(), qs,TRule::Sinh,ntv,0,gd[g].ones,gd[g].fvar,false,nk,nw); w=std::max(w,e);
        std::printf(" %10.2e", w);
      }
      std::printf(pass ? "\n" : " |");
    }
  }
}


// ================================================================= mode: bench

// Table-build cost separated from quadrature cost. Mul-add counts are exact and
// load-independent; wall times are min-over-repetitions (robust to interference
// on a shared box) and are single-thread by construction -- run OMP_NUM_THREADS=1.
template <class Real, class Ker> double TimeQuad(const std::vector<DuffyRule<Real>>& R, const std::vector<NodeData<Real>>& nd,
                                                 const Ker& ker, const Integer reps) {
  double best = 1e300;
  Matrix<Real> P;
  for (Integer rep = 0; rep < reps; rep++) {
    const double t0 = omp_get_wtime();
    for (size_t m = 0; m < nd.size(); m++) DuffyEval<Real>(P, nullptr, R[m], nd[m].Fsh, nullptr, ker);
    best = std::min(best, (omp_get_wtime()-t0)/(double)nd.size());
  }
  return best;
}

template <class Real> void ModeBench(const Integer p, const Integer reps) {
  std::printf("\n=== COST: table build vs quadrature, per target (order %d, 1 thread) ===\n", (int)p);
  ModelElem<Real> el; BuildModelElem(el, p, "twist4");
  std::vector<NodeData<Real>> nd; BuildNodes(nd, el, 1);
  std::printf("geometry twist4, %d target nodes, min over %d repetitions\n", (int)nd.size(), (int)reps);
  std::printf("A1 = metric sinh, fused table rebuilt per target   A2 = two-stage, Mi cached   A3 = two-stage, Mi rebuilt per target\n");
  std::printf("A4 = two-stage BATCHED (one GEMM per s-node, single Tt GEMM), Mi cached\n");
  std::printf("B  = parameter-space graded, all precomputed        C  = binned sinh, all precomputed\n\n");
  std::printf("%-3s %3s %-7s %6s %5s | %8s %8s %8s | %8s %8s | %9s\n",
              "var", "qs", "trule", "pts", "nt", "t_table", "t_qd C1", "t_qd C9", "eff C1", "eff C9", "tbl MB");
  struct B { const char* nm; Integer var; Integer qs; TRule tr; Integer t1,t2; bool param; Integer nk,nw; };
  std::vector<B> lst;
  for (Integer nt : {36, 44, 56}) {
    lst.push_back({"A1", 1, p, TRule::Sinh, nt, 0, false, 0, 0});
    lst.push_back({"A2", 2, p, TRule::Sinh, nt, 0, false, 0, 0});
    lst.push_back({"A3", 5, p, TRule::Sinh, nt, 0, false, 0, 0});
    lst.push_back({"A4", 6, p, TRule::Sinh, nt, 0, false, 0, 0});
  }
  for (auto xq : {std::pair<Integer,Integer>{1,8},{1,10},{1,12},{1,16}})
    lst.push_back({"B ", 3, p, TRule::Graded, xq.first, xq.second, true, 0, 0});
  for (Integer nt : {44, 56, 64, 72})
    lst.push_back({"C ", 4, p, TRule::Sinh, nt, 0, false, 16, 12});

  for (const B& b : lst) {
    const Integer tbl_full = (b.var == 2 || b.var == 5 || b.var == 6 ? (TBL_PRE|TBL_TONLY) : TBL_FUSED);
    std::vector<DuffyRule<Real>> R(nd.size());
    for (size_t m = 0; m < nd.size(); m++)
      BuildDuffyRule<Real>(R[m], p, nd[m].ti, nd[m].tj, nd[m].G, b.qs, b.tr, b.t1, b.t2, b.param, b.nk, b.nw, tbl_full);
    // per-target build cost: A1 rebuilds the fused table; A2 only the t-only operator;
    // B and C read cached tables, so nothing is charged.
    double tb = 0;
    if (b.var != 3 && b.var != 4) {
      const Integer tbl_pt = (b.var == 2 || b.var == 6 ? TBL_TONLY : b.var == 5 ? (TBL_PRE|TBL_TONLY) : TBL_FUSED);
      tb = 1e300;
      DuffyRule<Real> tmp;
      for (Integer rep = 0; rep < reps; rep++) {
        const double t0 = omp_get_wtime();
        for (size_t m = 0; m < nd.size(); m++)
          BuildDuffyRule<Real>(tmp, p, nd[m].ti, nd[m].tj, nd[m].G, b.qs, b.tr, b.t1, b.t2, b.param, b.nk, b.nw, tbl_pt);
        tb = std::min(tb, (omp_get_wtime()-t0)/(double)nd.size());
      }
    }
    const Contract cm = (b.var == 6 ? Contract::TwoStageB : b.var == 2 || b.var == 5 ? Contract::TwoStage : Contract::Fused);
    double nt_mean = 0, pts = 0;
    for (size_t m = 0; m < nd.size(); m++) { pts += (double)R[m].NPts(); for (Integer k=0;k<4;k++) nt_mean += (double)R[m].tri[k].nt/4.0; }
    nt_mean /= (double)nd.size(); pts /= (double)nd.size();
    double tq1 = 1e300, tq9 = 1e300;
    { Matrix<Real> P;
      for (Integer rep = 0; rep < reps; rep++) {
        double t0 = omp_get_wtime();
        for (size_t m = 0; m < nd.size(); m++) DuffyEval<Real>(P, nullptr, R[m], nd[m].Fsh, nullptr, Laplace3D_FxU(), cm, false);
        tq1 = std::min(tq1, (omp_get_wtime()-t0)/(double)nd.size());
        t0 = omp_get_wtime();
        for (size_t m = 0; m < nd.size(); m++) DuffyEval<Real>(P, nullptr, R[m], nd[m].Fsh, nullptr, Stokes3D_FxU(), cm, false);
        tq9 = std::min(tq9, (omp_get_wtime()-t0)/(double)nd.size());
      }
    }
    const double P2 = (double)p*p;
    // stored per (target,tri): fused Wa+WaD (p x qs*nt) vs two-stage Mi+MiD (qs x p x p)
    const double mb = (b.var == 5 ? 0.0 : b.var == 2 || b.var == 6 ? 4.0*P2*3.0*b.qs*P2*8.0/1048576.0
                                  : 4.0*P2*2.0*p*b.qs*nt_mean*8.0/1048576.0)
                    * (b.nk ? (double)b.nk*b.nw : 1.0);
    std::printf("%-3s %3d %s%-5d %6.0f %5.1f | %8.1f %8.1f %8.1f | %8.1f %8.1f | %9.0f\n",
                b.nm, (int)b.qs, b.tr==TRule::Sinh?"nt":"x/q", (int)(b.tr==TRule::Sinh?b.t1:b.t1*100+b.t2),
                pts, nt_mean, tb*1e6, tq1*1e6, tq9*1e6, (tb+tq1)*1e6, (tb+tq9)*1e6, mb);
  }
  std::printf("\ntimes in microseconds per target, min over %d reps. eff = t_table + t_quad.\n", (int)reps);
}


// ================================================================ mode: factor

// alpha(s,t) is affine in t at fixed s, so L_r(alpha(s_i,.)) is degree p-1 in t and
// Wa[i] = Mi[i] . Tt exactly, with Mi[i] free of both the t-rule and the metric.
template <class Real> void ModeFactor(const Integer p) {
  std::printf("\n=== FACTORISATION CHECK: Wa[i] == Mi[i] . Tt ? ===\n");
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  std::printf("%-8s %7s %4s %6s | %11s %11s | %9s %9s %9s\n",
              "geom", "(ti,tj)", "tri", "nt", "max|Wa|", "max|Wa-M.T|", "rel", "max|Mi|", "max|Tt|");
  for (const char* g : {"poly", "twist4", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    for (auto tt : {std::pair<Integer,Integer>{p/2,p/3}, {0,0}, {0,p-1}, {p-1,p-1}}) {
      const Integer ti = tt.first, tj = tt.second;
      Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
      Real G[4]; MetricAt<Real>(G, p, Fsh, nds[ti], nds[tj]);
      DuffyRule<Real> R;
      BuildDuffyRule<Real>(R, p, ti, tj, G, p+4, TRule::Sinh, 56, 0, false, 0, 0, TBL_FUSED|TBL_PRE|TBL_TONLY);
      for (Integer kt = 0; kt < 4; kt++) {
        const TriTable<Real>& T4 = R.tri[kt];
        double wmax = 0, dmax = 0, mmax = 0, tmax = 0;
        for (Long j = 0; j < T4.nt; j++) for (Integer k = 0; k < p; k++) tmax = std::max(tmax, std::fabs(D_(T4.Tt[k][j])));
        Matrix<Real> prod(p, T4.nt);
        for (Long i = 0; i < T4.ns; i++) {
          Matrix<Real>::GEMM(prod, T4.Mi[i], T4.Tt);
          for (Integer r = 0; r < p; r++) for (Long j = 0; j < T4.nt; j++) {
            wmax = std::max(wmax, std::fabs(D_(T4.Wa[i][r][j])));
            dmax = std::max(dmax, std::fabs(D_(T4.Wa[i][r][j] - prod[r][j])));
          }
          for (Integer r = 0; r < p; r++) for (Integer k = 0; k < p; k++) mmax = std::max(mmax, std::fabs(D_(T4.Mi[i][r][k])));
        }
        std::printf("%-8s (%2d,%2d) %4d %6ld | %11.3e %11.3e | %9.1e %9.2f %9.2f\n",
                    g, (int)ti, (int)tj, (int)kt, (long)T4.nt, wmax, dmax, dmax/wmax, mmax, tmax);
      }
    }
  }
}

// ================================================================ mode: newval

template <class Real, class Ker> void NewValOne(const ModelElem<Real>& el, const Integer ti, const Integer tj,
                                                const Ker& ker, const char* nm, double* worst) {
  const Integer p = el.p;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
  Real G[4]; MetricAt<Real>(G, p, Fsh, nds[ti], nds[tj]);
  Vector<Real> f((Long)p*p);
  for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++)
    f[i*p+j] = sin<Real>((Real)2.3*nds[i] + (Real)1.1)*cos<Real>((Real)3.1*nds[j] - (Real)0.4);
  DuffyRule<Real> R;
  BuildDuffyRule<Real>(R, p, ti, tj, G, p+4, TRule::Sinh, 56, 0, false, 0, 0, TBL_FUSED|TBL_PRE|TBL_TONLY);
  Matrix<Real> P1, P2, P3; Vector<Real> D1, D2, D3;
  DuffyEval<Real>(P1, &D1, R, Fsh, &f, ker, Contract::Fused,     false);
  DuffyEval<Real>(P2, &D2, R, Fsh, &f, ker, Contract::TwoStage,  false);
  DuffyEval<Real>(P3, &D3, R, Fsh, &f, ker, Contract::TwoStageB, false);
  double dmax = 0, pmax = 0, bmax = 0;
  for (Long c = 0; c < P1.Dim(0); c++) for (Long q = 0; q < P1.Dim(1); q++) {
    pmax = std::max(pmax, std::fabs(D_(P1[c][q])));
    dmax = std::max(dmax, std::fabs(D_(P1[c][q]-P2[c][q])));
    bmax = std::max(bmax, std::fabs(D_(P1[c][q]-P3[c][q])));
  }
  const double drel = (pmax > 0 ? dmax/pmax : 0.0);
  const double brel = (pmax > 0 ? bmax/pmax : 0.0);
  Vector<Real> I1, I2, I3; ApplyProj(I1, P1, f); ApplyProj(I2, P2, f); ApplyProj(I3, P3, f);
  const double adj2 = RelErr(I2, D2), fus2 = RelErr(I1, D1), op = RelErr(I2, I1);
  const double adjb = RelErr(I3, D3);
  Vector<Real> uq, vq, wqv, Iref;
  PolarRule<Real>(uq, vq, wqv, nds[ti], nds[tj], G, 40, 16, 20);
  RefIntegrate<Real>(Iref, p, Fsh, &f, uq, vq, wqv, ker);
  std::printf("  %-14s (%2d,%2d) | 2stg-vs-fused %8.1e | BATCHED-vs-fused %8.1e | adj 2stg %8.1e | adj BATCHED %8.1e | op %8.1e | vs polar %8.1e\n",
              nm, (int)ti, (int)tj, drel, brel, adj2, adjb, op, RelErr(I2, Iref));
  *worst = std::max(*worst, std::max(std::max(brel, adjb), std::max(drel, std::max(adj2, op))));
}

template <class Real> void ModeNewVal(const Integer p) {
  std::printf("\n=== TWO-STAGE PATH VALIDATION (order %d) ===\n", (int)p);
  double worst = 0;
  for (const char* g : {"flat", "skew", "poly", "sphere", "twist4", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    std::printf("geom=%s\n", g);
    for (auto tt : {std::pair<Integer,Integer>{p/2,p/3}, {0,0}, {0,p-1}, {p-1,p-1}}) {
      NewValOne<Real>(el, tt.first, tt.second, Laplace3D_FxU(), "Laplace3D-FxU", &worst);
      NewValOne<Real>(el, tt.first, tt.second, Laplace3D_DxU(), "Laplace3D-DxU", &worst);
      NewValOne<Real>(el, tt.first, tt.second, Stokes3D_FxU(),  "Stokes3D-FxU",  &worst);
      NewValOne<Real>(el, tt.first, tt.second, Stokes3D_DxU(),  "Stokes3D-DxU",  &worst);
    }
  }
  std::printf("\nworst of {2stage-vs-fused, 2stage adjoint, operator agreement} = %.2e\n", worst);
}

// =============================================================== mode: dlscale

template <class Real, class Ker> double DLGap(const ModelElem<Real>& el, const Integer ti, const Integer tj, const Ker& ker) {
  const Integer p = el.p;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
  Real G[4]; MetricAt<Real>(G, p, Fsh, nds[ti], nds[tj]);
  DuffyRule<Real> R;
  BuildDuffyRule<Real>(R, p, ti, tj, G, p+4, TRule::Sinh, 56, 0, false, 0, 0, TBL_FUSED|TBL_PRE|TBL_TONLY);
  Matrix<Real> P1, P2;
  DuffyEval<Real>(P1, nullptr, R, Fsh, nullptr, ker, Contract::Fused,    false);
  DuffyEval<Real>(P2, nullptr, R, Fsh, nullptr, ker, Contract::TwoStage, false);
  double dmax = 0, pmax = 0;
  for (Long c = 0; c < P1.Dim(0); c++) for (Long q = 0; q < P1.Dim(1); q++) {
    pmax = std::max(pmax, std::fabs(D_(P1[c][q])));
    dmax = std::max(dmax, std::fabs(D_(P1[c][q]-P2[c][q])));
  }
  return (pmax > 0 ? dmax/pmax : 0.0);
}

template <class Real> void ModeDLScale(const Integer p) {
  std::printf("\n=== fused vs two-stage gap, per working precision (order %d) ===\n", (int)p);
  std::printf("machine eps for this Real = %.3e\n", D_(machine_eps<Real>()));
  std::printf("%-8s %7s | %10s %10s %10s %10s\n", "geom", "(ti,tj)", "Lap-SL", "Lap-DL", "Stk-SL", "Stk-DL");
  for (const char* g : {"poly", "twist4"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    for (auto tt : {std::pair<Integer,Integer>{p/2,p/3}, {0,0}, {p-1,p-1}}) {
      std::printf("%-8s (%2d,%2d) | %10.2e %10.2e %10.2e %10.2e\n", g, (int)tt.first, (int)tt.second,
                  DLGap<Real>(el, tt.first, tt.second, Laplace3D_FxU()),
                  DLGap<Real>(el, tt.first, tt.second, Laplace3D_DxU()),
                  DLGap<Real>(el, tt.first, tt.second, Stokes3D_FxU()),
                  DLGap<Real>(el, tt.first, tt.second, Stokes3D_DxU()));
    }
  }
}

// ================================================================== mode: diag

// How far parameter-space (t*, d/L) is from the metric values, in units of the
// metric peak width -- the quantity that decides whether the t-rule tables can
// be cached on (p,ti,tj,tri) alone (duffy.txt section 9).
template <class Real> void ModeDiag(const Integer p) {
  std::printf("\n=== DIAG: parameter-space vs metric t-rule placement, order=%d ===\n", (int)p);
  std::printf("measured max|t*_I - t*_G|/(d/L)_G  vs predicted |cot(theta)|\n");
  std::printf("measured max (d/L)_I/(d/L)_G       vs predicted max(r,1/r)/sin(theta),  r = |Xu|/|Xv|\n");
  std::printf("%-8s %10s | %8s %8s | %8s %8s | %8s\n", "geom", "min dd_G",
              "shift", "pred cot", "widthmax", "pred", "widthmin");
  const Real I2[4] = {1,0,0,1};
  for (const char* g : {"flat", "skew", "poly", "sphere", "twist4", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
    double mindd = 1e300, mxsh = 0, mxr = 0, mnr = 1e300, mxcot = 0, mxpr = 0;
    for (Integer ti = 0; ti < p; ti++) for (Integer tj = 0; tj < p; tj++) {
      Vector<Real> Fsh; ShiftToTarget(Fsh, el, ti, tj);
      Real G[4]; MetricAt<Real>(G, p, Fsh, nds[ti], nds[tj]);
      const double guu = D_(G[0]), guv = D_(G[1]), gvv = D_(G[3]);
      const double sdet = std::sqrt(guu*gvv - guv*guv);
      mxcot = std::max(mxcot, std::fabs(guv)/sdet);              // shift, in peak widths
      mxpr = std::max(mxpr, std::max(guu, gvv)/sdet);            // width ratio (e along u / along v)
      DuffyRule<Real> RG, RI;
      BuildDuffyRule<Real>(RG, p, ti, tj, G,  4, TRule::Sinh, 4, 0);
      BuildDuffyRule<Real>(RI, p, ti, tj, I2, 4, TRule::Sinh, 4, 0);
      for (Integer k = 0; k < 4; k++) {
        const double ddG = D_(RG.tri[k].dOverL), ddI = D_(RI.tri[k].dOverL);
        mindd = std::min(mindd, ddG);
        mxsh = std::max(mxsh, std::fabs(D_(RG.tri[k].tstar) - D_(RI.tri[k].tstar))/ddG);
        mxr = std::max(mxr, ddI/ddG); mnr = std::min(mnr, ddI/ddG);
      }
    }
    std::printf("%-8s %10.2e | %8.3f %8.3f | %8.3f %8.3f | %8.3f\n", g, mindd, mxsh, mxcot, mxr, mxpr, mnr);
  }
}

// ================================================================ mode: final

// Candidate configs over ALL p^2 target nodes, with the t-rule placed either by
// the surface metric at (u0,v0) or by parameter-space distance alone (G = I).
// The latter is what makes the tables a pure function of (p,ti,tj,tri).
template <class Real> void ModeFinal(const Integer p) {
  std::printf("\n=== FINAL: candidate t-rules, all %d target nodes, metric-aware vs parameter-space t* ===\n", (int)(p*p));
  const Integer qs_ref = 48, txr = 3, tqr = 26;
  for (const char* g : {"poly", "twist4", "twist"}) {
    ModelElem<Real> el; BuildModelElem(el, p, g);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
    std::vector<NodeData<Real>> nd; BuildNodes(nd, el, 1);
    Vector<Real> ones((Long)p*p), fvar((Long)p*p);
    for (Integer i = 0; i < p; i++) for (Integer j = 0; j < p; j++) {
      ones[i*p+j] = 1;
      fvar[i*p+j] = sin<Real>((Real)2.3*nds[i] + (Real)1.1)*cos<Real>((Real)3.1*nds[j] - (Real)0.4);
    }
    { // range of the peak-width parameter d/L over nodes x triangles
      double lo = 1e300, hi = 0, alo = 1e300, ahi = 0;
      const Real I2[4] = {1,0,0,1};
      for (const auto& d : nd) {
        DuffyRule<Real> R, Rp;
        BuildDuffyRule<Real>(R,  p, d.ti, d.tj, d.G, 4, TRule::Sinh, 4, 0);
        BuildDuffyRule<Real>(Rp, p, d.ti, d.tj, I2,  4, TRule::Sinh, 4, 0);
        for (Integer k = 0; k < 4; k++) {
          lo = std::min(lo, D_(R.tri[k].dOverL)); hi = std::max(hi, D_(R.tri[k].dOverL));
          alo = std::min(alo, D_(Rp.tri[k].dOverL)); ahi = std::max(ahi, D_(Rp.tri[k].dOverL));
        }
      }
      std::printf("\n--- geom=%s, order=%d --- d/L range: metric [%.4f,%.4f], param-space [%.4f,%.4f]\n",
                  g, (int)p, lo, hi, alo, ahi);
    }
    std::vector<Matrix<Real>> R0, R1, R2, R3;
    BuildRefs<Real>(R0, nd, p, Laplace3D_FxU(), qs_ref, txr, tqr);
    BuildRefs<Real>(R1, nd, p, Laplace3D_DxU(), qs_ref, txr, tqr);
    BuildRefs<Real>(R2, nd, p, Stokes3D_FxU(),  qs_ref, txr, tqr);
    BuildRefs<Real>(R3, nd, p, Stokes3D_DxU(),  qs_ref, txr, tqr);
    std::vector<NodeData<Real>> ndp = nd; // parameter-space placement: metric -> identity
    for (auto& d : ndp) { d.G[0] = 1; d.G[1] = 0; d.G[2] = 0; d.G[3] = 1; }
    std::printf("%-22s %5s %7s  %10s %10s %10s %10s | %7s %10s\n", "t-rule", "qs", "pts",
                "Lap-SL", "Lap-DL", "Stk-SL", "Stk-DL", "pts(par)", "param-t*");
    struct Cfg { const char* nm; Integer qs; TRule tr; Integer t1, t2; };
    const Cfg cfg[] = {
      {"sinh(24)",     p+4,  TRule::Sinh,   24, 0}, {"sinh(32)", p+4, TRule::Sinh, 32, 0},
      {"sinh(40)",     p+4,  TRule::Sinh,   40, 0}, {"sinh(48)", p+4, TRule::Sinh, 48, 0},
      {"sinh(56)",     p+4,  TRule::Sinh,   56, 0}, {"sinh(64)", p+4, TRule::Sinh, 64, 0},
      {"sinh(40)",     p+0,  TRule::Sinh,   40, 0}, {"sinh(40)", p+8, TRule::Sinh, 40, 0},
      {"sinh(56)",     p+8,  TRule::Sinh,   56, 0}, {"sinh(56)", p+16, TRule::Sinh, 56, 0},
      {"sinh(64)",     p+16, TRule::Sinh,   64, 0},
      {"graded(x=0,q=8)",  p+4, TRule::Graded, 0, 8},  {"graded(x=1,q=8)",  p+4, TRule::Graded, 1, 8},
      {"graded(x=1,q=12)", p+4, TRule::Graded, 1, 12}, {"graded(x=2,q=12)", p+4, TRule::Graded, 2, 12},
      {"graded(x=3,q=12)", p+4, TRule::Graded, 3, 12}, {"graded(x=3,q=16)", p+4, TRule::Graded, 3, 16},
      {"graded(x=3,q=16)", p+16, TRule::Graded, 3, 16},
    };
    for (const Cfg& c : cfg) {
      double e[4], a[4], ep[4], ap[4];
      TuneOne<Real>(e[0], a[0], nd, R0, p, Laplace3D_FxU(), c.qs, c.tr, c.t1, c.t2, ones, fvar);
      TuneOne<Real>(e[1], a[1], nd, R1, p, Laplace3D_DxU(), c.qs, c.tr, c.t1, c.t2, ones, fvar);
      TuneOne<Real>(e[2], a[2], nd, R2, p, Stokes3D_FxU(),  c.qs, c.tr, c.t1, c.t2, ones, fvar);
      TuneOne<Real>(e[3], a[3], nd, R3, p, Stokes3D_DxU(),  c.qs, c.tr, c.t1, c.t2, ones, fvar);
      TuneOne<Real>(ep[0], ap[0], ndp, R0, p, Laplace3D_FxU(), c.qs, c.tr, c.t1, c.t2, ones, fvar);
      TuneOne<Real>(ep[1], ap[1], ndp, R1, p, Laplace3D_DxU(), c.qs, c.tr, c.t1, c.t2, ones, fvar);
      TuneOne<Real>(ep[2], ap[2], ndp, R2, p, Stokes3D_FxU(),  c.qs, c.tr, c.t1, c.t2, ones, fvar);
      TuneOne<Real>(ep[3], ap[3], ndp, R3, p, Stokes3D_DxU(),  c.qs, c.tr, c.t1, c.t2, ones, fvar);
      std::printf("%-22s %5d %7.0f  %10.2e %10.2e %10.2e %10.2e | %7.0f %10.2e\n", c.nm, (int)c.qs, a[0],
                  e[0], e[1], e[2], e[3], ap[0], std::max(std::max(ep[0],ep[1]),std::max(ep[2],ep[3])));
    }
  }
}

// ================================================================= mode: base

// Shipped Adaptive self rule: Nu (graded u) x Nv (composite Alpert v) per target.
template <class Real> void ModeBase(const Integer p) {
  std::printf("\n=== BASELINE (shipped Adaptive self rule) point counts, order=%d ===\n", (int)p);
  std::printf("%8s %5s %5s %5s %6s %6s %6s %10s\n", "tol", "q_u", "L_u", "q_v", "L_v", "Nu", "Nv", "Nu*Nv");
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(p);
  Vector<Real> qn, qw, du, wu, dv, wv;
  for (Integer d = 4; d <= 14; d++) {
    const Real tol = pow<Real>((Real)0.1, (Integer)d);
    const Integer qu = QuadElemList<Real>::SelfQuadOrderProbe(tol);
    const Integer Lu = QuadElemList<Real>::SelfLevelsProbe(tol);
    const Integer qv = QuadElemTestAccess<Real>::BaseQuadOrder(tol);
    const Integer Lv = QuadElemTestAccess<Real>::VLevels(d);
    GLRule<Real>(qn, qw, qu);
    Long Nu = 0, Nv = 0;
    for (Integer ti = 0; ti < p; ti++) { QuadElemTestAccess<Real>::CenteredGraded1D(du, wu, nds[ti], Lu, qn, qw); Nu = std::max(Nu, du.Dim()); }
    for (Integer tj = 0; tj < p; tj++) { QuadElemTestAccess<Real>::LogSingV(dv, wv, nds[tj], Lv, qv); Nv = std::max(Nv, dv.Dim()); }
    std::printf("%8.0e %5d %5d %5d %6d %6ld %6ld %10ld\n", (double)pow<double>(0.1,(Integer)d), (int)qu, (int)Lu, (int)qv, (int)Lv, (long)Nu, (long)Nv, (long)(Nu*Nv));
  }
}

// ==================================================================== driver

template <class Real> void Run(const std::string& mode, const Integer p, const Integer stride, const std::string& grid) {
  if (mode == "gate")    ModeGate<Real>(p);
  else if (mode == "ref")     ModeRef<Real>(p);
  else if (mode == "v1")      ModeV1<Real>(p);
  else if (mode == "tune")    ModeTune<Real>(p, stride);
  else if (mode == "adjoint") ModeAdjoint<Real>(p);
  else if (mode == "base")    ModeBase<Real>(p);
  else if (mode == "final")   ModeFinal<Real>(p);
  else if (mode == "diag")    ModeDiag<Real>(p);
  else if (mode == "compare") ModeCompare<Real>(p, stride, grid);
  else if (mode == "binstudy") ModeBinStudy<Real>(p, stride);
  else if (mode == "bench")   ModeBench<Real>(p, std::max<Integer>(1, stride));
  else if (mode == "factor")  ModeFactor<Real>(p);
  else if (mode == "newval")  ModeNewVal<Real>(p);
  else if (mode == "dlscale") ModeDLScale<Real>(p);
  else std::printf("unknown mode %s\n", mode.c_str());
}

int main(int argc, char** argv) {
  const std::string mode = (argc > 1 ? argv[1] : "gate");
  const std::string prec = (argc > 2 ? argv[2] : "quad");
  const Integer p        = (argc > 3 ? (Integer)atoi(argv[3]) : 12);
  const Integer stride   = (argc > 4 ? (Integer)atoi(argv[4]) : 4);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::printf("mode=%s prec=%s order=%d stride=%d\n", mode.c_str(), prec.c_str(), (int)p, (int)stride);
  const std::string grid = (argc > 5 ? argv[5] : "wide");
  const std::string path = (argc > 6 ? argv[6] : "fused");
  if (path == "twostage") { g_contract = Contract::TwoStage; g_tbl = TBL_PRE | TBL_TONLY; }
  if (path == "twostageb") { g_contract = Contract::TwoStageB; g_tbl = TBL_PRE | TBL_TONLY; }
  std::printf("contraction path: %s\n", path.c_str());
  if (prec == "double") Run<double>(mode, p, stride, grid);
  else if (prec == "long") Run<long double>(mode, p, stride, grid);
#ifdef SCTL_QUAD_T
  else Run<QuadReal>(mode, p, stride, grid);
#endif
  return 0;
}
