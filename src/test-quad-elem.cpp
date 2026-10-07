/**
 * QuadElemList tests, from the building blocks to closed surfaces, so a failure points at the lowest
 * broken layer:
 *
 *   1. Building blocks: Alpert endpoint corrections, the centered log-singular rule, element
 *      geometry, the far-field rule and its cut-off distance, closest-point search, file and VTK
 *      output, and Copy.
 *   2. One element, for every quadrature scheme and element order: near- and self-interactions
 *      against closed forms on a flat element, and against an adaptive reference on a sphere patch
 *      small enough to resolve the sphere and the density to the requested tolerance.
 *   3. BoundaryIntegralOp on a twisted sphere resolved to the requested tolerance, for every scheme,
 *      at on- and off-surface targets against the adaptive reference.
 *
 * At a target on the surface, TensorProduct and Duffy return the principal value and Hedgehog the
 * limit from the side of the normal; the expected values differ by the jump of double-layer kernels.
 *   4. A convergence study on a finer sphere: surface area, the double-layer identity, and Green's
 *      identity at on- and off-surface targets.
 *
 * The default run, for CI, takes about a minute on 4 cores in a -O0 sanitizer build: element order 8,
 * one tolerance, a subset of the kernels, a coarse sphere, and no section 4. With any argument, the
 * full run covers orders 4-20, two tolerances, all kernels and section 4.
 *
 * make bin/test-quad-elem && OMP_NUM_THREADS=8 ./bin/test-quad-elem [full]
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <array>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace sctl;

// Extended precision for the references and the copy test: QuadReal where the build has it, else
// long double, which on some systems (macOS on arm64) is no wider than double
#ifdef SCTL_QUAD_T
using ExtReal = QuadReal;
#else
using ExtReal = long double;
#endif

namespace sctl {
template <class Real> struct QuadElemTestAccess {
  // The rule's nodes as absolute parameters v0 + delta
  static void LogSingularQuad1D(Vector<Real>& param, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder) {
    Vector<Real> delta;
    detail_tensorprod_singular::BuildCenteredLogSingular1D<Real>(delta, w, v0, Lvl, QuadOrder);
    param.ReInit(delta.Dim());
    for (Long i = 0; i < delta.Dim(); i++) param[i] = v0 + delta[i];
  }
  static Vector<Real> ElemCoord(const QuadElemList<Real>& qel, const Long elem_idx) {
    const Integer n = 3 * qel.Order() * qel.Order();
    return Vector<Real>(n, (Iterator<Real>)qel.coord.begin() + elem_idx * n, false);
  }
  static Real GetClosestNode(Real& ustar, Real& vstar, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg) {
    return detail_quadelem::GetClosestNode(ustar, vstar, ElemCoord(qel, elem_idx), qel.Order(), Xtrg);
  }
  static Real GetClosestPoint(Real& ustar, Real& vstar, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg) {
    const Integer n = 3 * qel.Order() * qel.Order();
    const Vector<Real> dcoord_du(n, (Iterator<Real>)qel.dcoord_du.begin() + elem_idx * n, false);
    const Vector<Real> dcoord_dv(n, (Iterator<Real>)qel.dcoord_dv.begin() + elem_idx * n, false);
    return detail_quadelem::GetClosestPoint(ustar, vstar, ElemCoord(qel, elem_idx), dcoord_du, dcoord_dv, qel.Order(), Xtrg);
  }
  static typename QuadElemList<Real>::QuadScheme Scheme(const QuadElemList<Real>& qel) {
    return qel.scheme_;
  }
};
}

namespace {

constexpr Integer COORD_DIM = 3;

template <class Real> using QuadScheme = typename QuadElemList<Real>::QuadScheme;

// The quadrature schemes, and the value each returns at a target on the surface: the principal value,
// or (one_sided) the limit from the side of the normal. For a double-layer kernel the two differ by
// the jump J*sigma, with J = +1/2 for the double layers and -1/2 for their adjoints.
template <class Real> struct SchemeInfo {
  std::string name;
  QuadScheme<Real> scheme;
  bool one_sided;
};

template <class Real> const std::vector<SchemeInfo<Real>>& Schemes() {
  static const std::vector<SchemeInfo<Real>> schemes{{"TensorProduct", QuadScheme<Real>::TensorProduct, false}, {"Duffy", QuadScheme<Real>::Duffy, false}, {"Hedgehog", QuadScheme<Real>::Hedgehog, true}};
  return schemes;
}

template <class T> T GlobalReduce(const T x, const Comm& comm, const CommOp op) {
  StaticArray<T,2> buf{x, 0};
  comm.Allreduce(buf + 0, buf + 1, 1, op);
  return buf[1];
}

// Largest deviation over the largest reference value
// The larger of a and b, or infinity if either is NaN: std::max(a, b) returns a when b is NaN, so a
// NaN result would pass. Infinity, unlike NaN, also survives a maximum taken across processes.
template <class Real> Real MaxErr(const Real a, const Real b) {
  return (a != a || b != b ? (Real)std::numeric_limits<double>::infinity() : std::max<Real>(a, b));
}

template <class Real> Real RelErr(const Vector<Real>& U, const Vector<Real>& U_ref) {
  SCTL_ASSERT(U.Dim() == U_ref.Dim());
  Real err = 0, ref = 0;
  for (Long i = 0; i < U.Dim(); i++) {
    err = MaxErr<Real>(err, fabs(U[i] - U_ref[i]));
    ref = MaxErr<Real>(ref, fabs(U_ref[i]));
  }
  return err / ref;
}

// u[c] = sum_r sigma[r] M[r][c]
template <class Real> Vector<Real> Apply(const Matrix<Real>& M, const Vector<Real>& sigma) {
  SCTL_ASSERT(M.Dim(0) == sigma.Dim());
  Vector<Real> u(M.Dim(1));
  u.SetZero();
  for (Long r = 0; r < M.Dim(0); r++) {
    for (Long c = 0; c < M.Dim(1); c++) u[c] += sigma[r] * M[r][c];
  }
  return u;
}

// One element over the unit parameter square: X(u,v) = (u, v, u*v) if curved, else (u, v, 0)
template <class Real> QuadElemList<Real> TestElem(const Integer order, const bool curved) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> coord(order * order * COORD_DIM);
  for (Integer i = 0; i < order; i++) {
    for (Integer j = 0; j < order; j++) {
      const Integer p = i * order + j;
      coord[p * COORD_DIM + 0] = nds[i];
      coord[p * COORD_DIM + 1] = nds[j];
      coord[p * COORD_DIM + 2] = (curved ? nds[i] * nds[j] : 0);
    }
  }
  return QuadElemList<Real>(order, coord);
}

// Smooth function of position with dof values per point (AoS)
template <class Real> Vector<Real> TestDensity(const Vector<Real>& X, const Integer dof) {
  const Long N = X.Dim() / COORD_DIM;
  Vector<Real> F(N * dof);
  for (Long i = 0; i < N; i++) {
    for (Integer k = 0; k < dof; k++) {
      F[i * dof + k] = cos<Real>(X[i * COORD_DIM + 0] + 2 * X[i * COORD_DIM + 1] - X[i * COORD_DIM + 2] + (Real)0.5 * k);
    }
  }
  return F;
}

// Point (u,v) of element elem of the cubed sphere of radius R with ppf x ppf elements per face,
// twisted about z (at height z, the point is rotated by twist*z)
template <class Real> std::array<Real,3> TwistedSpherePoint(const Long elem, const Real u, const Real v, const Integer ppf, const Real R, const Real twist) {
  static constexpr Integer face_map[6][3][3] = { // cube-face point (x,y,z) as coefficients of (1, a, b)
    {{ 1,  0, 0}, { 0, 1, 0}, { 0, 0,  1}},
    {{-1,  0, 0}, { 0, -1, 0}, { 0, 0,  1}},
    {{ 0,  1, 0}, { 1, 0, 0}, { 0, 0, -1}},
    {{ 0,  1, 0}, {-1, 0, 0}, { 0, 0,  1}},
    {{ 0,  1, 0}, { 0, 0, 1}, { 1, 0,  0}},
    {{ 0, -1, 0}, { 0, 0, 1}, {-1, 0,  0}}};
  const Integer face = (Integer)(elem / (ppf * ppf));
  const Real a = 2 * ((elem / ppf % ppf + u) / ppf) - 1, b = 2 * ((elem % ppf + v) / ppf) - 1;
  std::array<Real,3> p;
  for (Integer k = 0; k < COORD_DIM; k++) p[k] = face_map[face][k][0] + face_map[face][k][1] * a + face_map[face][k][2] * b;
  const Real s = R / sqrt<Real>(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  const Real x = p[0] * s, y = p[1] * s, z = p[2] * s;
  const Real c = cos<Real>(twist * z), sn = sin<Real>(twist * z);
  return {x * c + y * sn, -x * sn + y * c, z};
}

// That sphere, each process building its contiguous range of the elements
template <class Real> QuadElemList<Real> BuildTwistedSphere(const Integer order, const Integer ppf, const Real R, const Real twist, const Comm& comm) {
  const Long Nelem = 6 * ppf * ppf;
  const Long elem0 = Nelem * comm.Rank() / comm.Size();
  const Long elem1 = Nelem * (comm.Rank() + 1) / comm.Size();
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> X;
  for (Long elem = elem0; elem < elem1; elem++) {
    for (Integer i = 0; i < order; i++) {
      for (Integer j = 0; j < order; j++) {
        for (const Real x : TwistedSpherePoint(elem, nds[i], nds[j], ppf, R, twist)) X.PushBack(x);
      }
    }
  }
  return QuadElemList<Real>(order, X);
}

// Point (u,v) of the patch of the unit sphere of size h about +z (cube-face map)
template <class Real> std::array<Real,3> SpherePatchPoint(const Real h, const Real u, const Real v) {
  const Real a = h * (u - (Real)0.5), b = h * (v - (Real)0.5);
  const Real s = 1 / sqrt<Real>(a * a + b * b + 1);
  return {a * s, b * s, s};
}

// Largest discretization error of the element list against its exact map (element, u, v) -> X, over
// a grid on each element: of the geometry relative to elem_size, and of the interpolated TestDensity.
template <class Real, class Map> Real DiscretizationError(const QuadElemList<Real>& qel, const Map& map, const Real elem_size) {
  const Integer order = qel.Order(), Ns = 23;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> X, up(Ns), vp(Ns), Lu(order * Ns), Lv(order * Ns);
  qel.GetNodeCoord(&X, nullptr, nullptr);
  const Vector<Real> sigma = TestDensity(X, 1);
  for (Integer a = 0; a < Ns; a++) {
    up[a] = (a + (Real)0.5) / Ns;
    vp[a] = (a + (Real)0.3) / Ns;
  }
  LagrangeInterp<Real>::Interpolate(Lu, nds, up);
  LagrangeInterp<Real>::Interpolate(Lv, nds, vp);
  Real err = 0;
  for (Long e = 0; e < qel.Size(); e++) {
    Vector<Real> Xg;
    qel.GetGeom(&Xg, nullptr, nullptr, nullptr, nullptr, up, vp, e);
    for (Integer a = 0; a < Ns; a++) {
      for (Integer b = 0; b < Ns; b++) {
        const std::array<Real,3> x = map(e, up[a], vp[b]);
        const Integer p = a * Ns + b;
        Real r2 = 0, s = 0;
        for (Integer k = 0; k < COORD_DIM; k++) r2 += (Xg[p * COORD_DIM + k] - x[k]) * (Xg[p * COORD_DIM + k] - x[k]);
        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) s += sigma[e * order * order + i * order + j] * Lu[i * Ns + a] * Lv[j * Ns + b];
        }
        err = MaxErr<Real>(err, sqrt<Real>(r2) / elem_size);
        err = MaxErr<Real>(err, fabs(s - TestDensity(Vector<Real>{x[0], x[1], x[2]}, 1)[0]));
      }
    }
  }
  return err;
}

// One element on a patch of the unit sphere, the largest (halving the size from 1) whose geometry and
// density are resolved to tol; h is set to its size.
template <class Real> QuadElemList<Real> ResolvedSpherePatch(const Integer order, const Real tol, Real& h) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (h = 1; ; h /= 2) {
    Vector<Real> coord;
    for (Integer i = 0; i < order; i++) {
      for (Integer j = 0; j < order; j++) {
        for (const Real x : SpherePatchPoint(h, nds[i], nds[j])) coord.PushBack(x);
      }
    }
    QuadElemList<Real> qel(order, coord);
    const Real h_ = h;
    if (DiscretizationError(qel, [h_](Long, Real u, Real v) { return SpherePatchPoint(h_, u, v); }, h) <= tol || h < 1e-6) return qel;
  }
}

// Fewest elements per face of the twisted sphere whose geometry and density are resolved to tol
template <class Real> Integer ResolvedSpherePPF(const Integer order, const Real R, const Real twist, const Real tol) {
  Integer ppf = 1;
  while (DiscretizationError(BuildTwistedSphere<Real>(order, ppf, R, twist, Comm::Self()), [ppf, R, twist](Long e, Real u, Real v) { return TwistedSpherePoint(e, u, v, ppf, R, twist); }, 2 * R / ppf) > tol) ppf++;
  return ppf;
}

// Potential at the targets Xt from element elem_idx of qel carrying the nodal density sigma (AoS),
// independent of the library's schemes: adaptive quad-tree refinement of the parameter square with
// 12 x 12 Gauss-Legendre points on each leaf, a cell being split while its diameter exceeds its
// distance to the target, and dropped if still unresolved after 52 levels (a cell containing a
// target on the element, whose contribution vanishes with its size). Each target has an expansion
// point Pt (u,v) on the element; within 1/16 of it the source positions are taken relative to the
// element's value there from a Taylor expansion with coefficients computed in ExtReal, so that
// x_t - x_s stays accurate relative to its size as the cells shrink toward the target. With target
// normals, the kernel's target values are contracted with them in consecutive triples.
template <class Real, class Kernel> Vector<Real> ReferencePotential(const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& sigma, const Vector<Real>& Xt, const Vector<Real>& Pt, const Vector<Real>& normal_trg, const Kernel& ker) {
  using QR = ExtReal;
  constexpr Integer RefOrder = 12;
  constexpr Integer RefMaxDepth = 52;
  constexpr Integer KDIM0 = Kernel::SrcDim();
  constexpr Integer KDIM1 = Kernel::TrgDim();
  const Real TaylorRadius = (Real)1 / 16;
  const bool trg_dot_prod = (normal_trg.Dim() > 0);
  const Integer KDIM1_ = (trg_dot_prod ? KDIM1 / COORD_DIM : KDIM1);
  const Integer order = qel.Order();
  const Integer nnode = order * order;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Vector<Real>& gl_nds = LegQuadRule<Real>::nds(RefOrder);
  const Vector<Real>& gl_wts = LegQuadRule<Real>::wts(RefOrder);
  const Vector<Real> coord = QuadElemTestAccess<Real>::ElemCoord(qel, elem_idx); // component-major
  struct Cell {
    Real du0, du1, dv0, dv1; // relative to the expansion point
    Integer depth;
  };

  const Long Ntrg = Xt.Dim() / COORD_DIM;
  SCTL_ASSERT(Pt.Dim() == Ntrg * 2);
  Vector<Real> U(Ntrg * KDIM1_);
  #pragma omp parallel for schedule(dynamic)
  for (Long t = 0; t < Ntrg; t++) {
    const Real uf = Pt[t * 2 + 0], vf = Pt[t * 2 + 1];
    std::vector<Real> T(COORD_DIM * nnode); // T[(k*order+i)*order+j]: coefficient of du^i dv^j of X_k(uf+du, vf+dv) - X_k(uf, vf)
    { // Taylor coefficients of the element about (uf, vf)
      const auto basis_taylor = [&nds, order](const Real x0) { // L_m(x0 + d) = sum_i c[m*order+i] d^i
        std::vector<QR> c(order * order, 0);
        for (Integer m = 0; m < order; m++) {
          std::vector<QR> poly{1};
          QR den = 1;
          for (Integer l = 0; l < order; l++) {
            if (l == m) continue;
            std::vector<QR> next(poly.size() + 1, 0);
            for (size_t i = 0; i < poly.size(); i++) {
              next[i] += poly[i] * ((QR)x0 - (QR)nds[l]);
              next[i + 1] += poly[i];
            }
            poly.swap(next);
            den *= (QR)nds[m] - (QR)nds[l];
          }
          for (Integer i = 0; i < order; i++) c[m * order + i] = poly[i] / den;
        }
        return c;
      };
      const std::vector<QR> cu = basis_taylor(uf), cv = basis_taylor(vf);
      for (Integer k = 0; k < COORD_DIM; k++) {
        std::vector<QR> Cv(order * order, 0); // Cv[m*order+j] = sum_n coord[k][m][n] cv[n][j]
        for (Integer m = 0; m < order; m++) {
          for (Integer n = 0; n < order; n++) {
            for (Integer j = 0; j < order; j++) Cv[m * order + j] += (QR)coord[k * nnode + m * order + n] * cv[n * order + j];
          }
        }
        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) {
            QR s = 0;
            for (Integer m = 0; m < order; m++) s += cu[m * order + i] * Cv[m * order + j];
            T[(k * order + i) * order + j] = (Real)s;
          }
        }
        T[(k * order + 0) * order + 0] = 0;
      }
    }
    Vector<Real> Xf;
    qel.GetGeom(&Xf, nullptr, nullptr, nullptr, nullptr, Vector<Real>{uf}, Vector<Real>{vf}, elem_idx);
    const Vector<Real> offset{Xt[t * COORD_DIM + 0] - Xf[0], Xt[t * COORD_DIM + 1] - Xf[1], Xt[t * COORD_DIM + 2] - Xf[2]}; // target relative to X(uf, vf)

    const auto rel_pos = [&](Vector<Real>& dX, const Vector<Real>& du, const Vector<Real>& dv, const bool taylor) { // X(uf+du[a], vf+dv[b]) - X(uf, vf) at point a*Nv+b
      const Integer Nu = (Integer)du.Dim(), Nv = (Integer)dv.Dim();
      dX.ReInit(Nu * Nv * COORD_DIM);
      if (!taylor) {
        Vector<Real> up(Nu), vp(Nv), X;
        for (Integer a = 0; a < Nu; a++) up[a] = uf + du[a];
        for (Integer b = 0; b < Nv; b++) vp[b] = vf + dv[b];
        qel.GetGeom(&X, nullptr, nullptr, nullptr, nullptr, up, vp, elem_idx);
        for (Long i = 0; i < X.Dim(); i++) dX[i] = X[i] - Xf[i % COORD_DIM];
        return;
      }
      std::vector<Real> A(order * Nv);
      for (Integer k = 0; k < COORD_DIM; k++) {
        for (Integer i = 0; i < order; i++) { // Horner along v
          for (Integer b = 0; b < Nv; b++) {
            Real s = 0;
            for (Integer j = order - 1; j >= 0; j--) s = s * dv[b] + T[(k * order + i) * order + j];
            A[i * Nv + b] = s;
          }
        }
        for (Integer a = 0; a < Nu; a++) { // Horner along u
          for (Integer b = 0; b < Nv; b++) {
            Real s = 0;
            for (Integer i = order - 1; i >= 0; i--) s = s * du[a] + A[i * Nv + b];
            dX[(a * Nv + b) * COORD_DIM + k] = s;
          }
        }
      }
    };

    std::vector<Real> Xs, Ns, Fs; // leaf points relative to X(uf, vf), normals, and weighted densities
    std::vector<Cell> cells{{-uf, 1 - uf, -vf, 1 - vf, 0}};
    while (!cells.empty()) {
      const Cell c = cells.back();
      cells.pop_back();
      const bool taylor = std::max(std::max(fabs(c.du0), fabs(c.du1)), std::max(fabs(c.dv0), fabs(c.dv1))) <= TaylorRadius;
      const bool resolved = [&rel_pos, &offset, &c, taylor]() { // Diameter below the distance to the target
        Vector<Real> dX;
        rel_pos(dX, Vector<Real>{c.du0, (c.du0 + c.du1) / 2, c.du1}, Vector<Real>{c.dv0, (c.dv0 + c.dv1) / 2, c.dv1}, taylor);
        Real diam = 0, dist2 = 0;
        for (Integer p = 0; p < 9; p++) {
          Real r2 = 0;
          for (Integer k = 0; k < COORD_DIM; k++) r2 += (dX[p * COORD_DIM + k] - dX[4 * COORD_DIM + k]) * (dX[p * COORD_DIM + k] - dX[4 * COORD_DIM + k]);
          diam = std::max<Real>(diam, 2 * sqrt<Real>(r2));
        }
        for (Integer k = 0; k < COORD_DIM; k++) dist2 += (offset[k] - dX[4 * COORD_DIM + k]) * (offset[k] - dX[4 * COORD_DIM + k]);
        return diam < sqrt<Real>(dist2) - diam / 2;
      }();
      if (!resolved) {
        if (c.depth < RefMaxDepth) {
          const Real um = (c.du0 + c.du1) / 2, vm = (c.dv0 + c.dv1) / 2;
          cells.push_back({c.du0, um, c.dv0, vm, c.depth + 1});
          cells.push_back({um, c.du1, c.dv0, vm, c.depth + 1});
          cells.push_back({c.du0, um, vm, c.dv1, c.depth + 1});
          cells.push_back({um, c.du1, vm, c.dv1, c.depth + 1});
        }
        continue;
      }
      { // Gauss-Legendre points on the cell, and the density interpolated to them
        Vector<Real> du(RefOrder), dv(RefOrder), up(RefOrder), vp(RefOrder);
        for (Integer a = 0; a < RefOrder; a++) {
          du[a] = c.du0 + (c.du1 - c.du0) * gl_nds[a];
          dv[a] = c.dv0 + (c.dv1 - c.dv0) * gl_nds[a];
          up[a] = uf + du[a];
          vp[a] = vf + dv[a];
        }
        Vector<Real> dX, X, N, Xa, Lu(order * RefOrder), Lv(order * RefOrder);
        rel_pos(dX, du, dv, taylor);
        qel.GetGeom(&X, &N, &Xa, nullptr, nullptr, up, vp, elem_idx);
        LagrangeInterp<Real>::Interpolate(Lu, nds, up);
        LagrangeInterp<Real>::Interpolate(Lv, nds, vp);
        Vector<Real> sigma_v(order * RefOrder * KDIM0); // sigma interpolated along v
        sigma_v.SetZero();
        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) {
            for (Integer b = 0; b < RefOrder; b++) {
              for (Integer k = 0; k < KDIM0; k++) sigma_v[(i * RefOrder + b) * KDIM0 + k] += sigma[(i * order + j) * KDIM0 + k] * Lv[j * RefOrder + b];
            }
          }
        }
        for (Integer a = 0; a < RefOrder; a++) {
          for (Integer b = 0; b < RefOrder; b++) {
            const Integer q = a * RefOrder + b;
            const Real w = Xa[q] * (c.du1 - c.du0) * gl_wts[a] * (c.dv1 - c.dv0) * gl_wts[b];
            for (Integer k = 0; k < COORD_DIM; k++) {
              Xs.push_back(dX[q * COORD_DIM + k]);
              Ns.push_back(N[q * COORD_DIM + k]);
            }
            for (Integer k = 0; k < KDIM0; k++) {
              Real s = 0;
              for (Integer i = 0; i < order; i++) s += Lu[i * RefOrder + a] * sigma_v[(i * RefOrder + b) * KDIM0 + k];
              Fs.push_back(s * w);
            }
          }
        }
      }
    }
    Vector<Real> Ut;
    ker.Eval(Ut, offset, Vector<Real>((Long)Xs.size(), Ptr2Itr<Real>(Xs.data(), Xs.size()), false), Vector<Real>((Long)Ns.size(), Ptr2Itr<Real>(Ns.data(), Ns.size()), false), Vector<Real>((Long)Fs.size(), Ptr2Itr<Real>(Fs.data(), Fs.size()), false));
    for (Integer b = 0; b < KDIM1_; b++) {
      Real u = (trg_dot_prod ? 0 : Ut[b]);
      if (trg_dot_prod) {
        for (Integer l = 0; l < COORD_DIM; l++) u += Ut[b * COORD_DIM + l] * normal_trg[t * COORD_DIM + l];
      }
      U[t * KDIM1_ + b] = u;
    }
  }
  return U;
}

// Sum of ReferencePotential over the elements of qel, for the density sigma (AoS, all nodes), with the
// expansion point of target t at node trg_node[t] of element trg_elem[t] on that element, and at the
// center of the others.
template <class Real, class Kernel> Vector<Real> SurfaceReference(const QuadElemList<Real>& qel, const Vector<Real>& sigma, const Vector<Real>& Xt, const Vector<Long>& trg_elem, const Vector<Long>& trg_node, const Vector<Real>& normal_trg, const Kernel& ker) {
  constexpr Integer KDIM0 = Kernel::SrcDim();
  const Integer KDIM1 = (normal_trg.Dim() ? Kernel::TrgDim() / COORD_DIM : Kernel::TrgDim());
  const Integer order = qel.Order();
  const Long nnode = order * order;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> U(Xt.Dim() / COORD_DIM * KDIM1);
  U.SetZero();
  for (Long e = 0; e < qel.Size(); e++) {
    Vector<Real> Pt(trg_elem.Dim() * 2);
    for (Long t = 0; t < trg_elem.Dim(); t++) {
      Pt[t * 2 + 0] = (trg_elem[t] == e ? nds[trg_node[t] / order] : (Real)0.5);
      Pt[t * 2 + 1] = (trg_elem[t] == e ? nds[trg_node[t] % order] : (Real)0.5);
    }
    const Vector<Real> sigma_e(nnode * KDIM0, (Iterator<Real>)sigma.begin() + e * nnode * KDIM0, false);
    U += ReferencePotential(qel, e, sigma_e, Xt, Pt, normal_trg, ker);
  }
  return U;
}

// Kernels with a closed form for the flat element (0,1)^2 x {0}, normal +z, and constant density:
// the Laplace single and double layers, the Laplace adjoint double layer with target normal +z, and
// the Stokes single layer with density (0,0,1).
enum class FlatKernel { LaplaceSL, LaplaceDL, LaplaceAdjointDL, StokesSL };

// Their potentials at the targets Xt, as corner sums over the square of the antiderivatives of 1/R,
// z/R^3 and x/R^3 (with (x,y) relative to the target, z its height and R the distance), in ExtReal.
template <class Real> Vector<Real> FlatSquarePotential(const FlatKernel kernel, const Vector<Real>& Xt) {
  using QR = ExtReal;
  const Integer dof = (kernel == FlatKernel::StokesSL ? 3 : 1);
  const QR pi = const_pi<QR>();
  const Long Ntrg = Xt.Dim() / COORD_DIM;
  Vector<Real> U(Ntrg * dof);
  for (Long t = 0; t < Ntrg; t++) {
    const QR x0 = Xt[t * COORD_DIM + 0], y0 = Xt[t * COORD_DIM + 1], z = Xt[t * COORD_DIM + 2];
    QR sum_sl = 0, sum_dl = 0, sum_lnb = 0, sum_lna = 0;
    for (Integer ca = 0; ca < 2; ca++) {
      for (Integer cb = 0; cb < 2; cb++) {
        const QR a = (ca ? 1 - x0 : -x0), b = (cb ? 1 - y0 : -y0);
        const QR s = (ca == cb ? 1 : -1);
        const QR R = sqrt<QR>(a * a + b * b + z * z);
        const auto log_p_R = [R, z](const QR p, const QR q) { // log(p + R) without cancellation for p < 0; zero where its factor vanishes
          if (q == 0 && z == 0) return (QR)0;
          return (p >= 0 ? log<QR>(p + R) : log<QR>((q * q + z * z) / (R - p)));
        };
        const QR lnb = log_p_R(b, a), lna = log_p_R(a, b);
        const QR at = (z == 0 ? (QR)0 : atan<QR>(a * b / (z * R)));
        sum_sl += s * (a * lnb + b * lna - z * at);
        sum_dl += s * at;
        sum_lnb += s * lnb;
        sum_lna += s * lna;
      }
    }
    if (kernel == FlatKernel::LaplaceSL) {
      U[t] = (Real)(sum_sl / (4 * pi));
    } else if (kernel == FlatKernel::LaplaceDL) {
      U[t] = (Real)(sum_dl / (4 * pi));
    } else if (kernel == FlatKernel::LaplaceAdjointDL) {
      U[t] = (Real)(-sum_dl / (4 * pi));
    } else {
      U[t * 3 + 0] = (Real)(z * sum_lnb / (8 * pi));
      U[t * 3 + 1] = (Real)(z * sum_lna / (8 * pi));
      U[t * 3 + 2] = (Real)((sum_sl + z * sum_dl) / (8 * pi));
    }
  }
  return U;
}

// Near targets of the flat element and their normals: inside, near an edge, near a corner, beyond
// an edge and beyond a corner, on both sides at distances 1e-1, 1e-3 and 1e-7.
template <class Real> void FlatNearTargets(Vector<Real>& Xt, Vector<Real>& Nt) {
  const std::array<std::array<Real,2>,5> pos{{{0.4, 0.6}, {0.02, 0.5}, {0.02, 0.03}, {-0.03, 0.5}, {-0.03, -0.02}}};
  Xt.ReInit(0);
  Nt.ReInit(0);
  for (const auto& p : pos) {
    for (const Real d : {(Real)1e-1, (Real)1e-3, (Real)1e-7}) {
      for (const Real side : {(Real)1, (Real)-1}) {
        for (const Real x : {p[0], p[1], side * d}) Xt.PushBack(x);
        for (const Real n : {(Real)0, (Real)0, (Real)1}) Nt.PushBack(n);
      }
    }
  }
}

// The same target positions relative to a curved element of size h: offsets along the normal at
// points of the element, or at points displaced outward in its tangent plane beyond an edge or a
// corner, all scaled by h. Pt holds the parameters (u,v) of the element point each target is placed
// from.
template <class Real> void CurvedNearTargets(Vector<Real>& Xt, Vector<Real>& Nt, Vector<Real>& Pt, const QuadElemList<Real>& qel, const Real h) {
  Xt.ReInit(0);
  Nt.ReInit(0);
  Pt.ReInit(0);
  const std::array<std::array<Real,4>,5> pos{{{0.4, 0.6, 0, 0}, {0.02, 0.5, 0, 0}, {0.02, 0.03, 0, 0}, {0, 0.5, 0.03, 0}, {0, 0, 0.03, 0.03}}}; // (u, v, distances beyond the edges u = 0 and v = 0)
  for (const auto& p : pos) {
    const Vector<Real> up{p[0]}, vp{p[1]};
    Vector<Real> X, N, dXu, dXv;
    qel.GetGeom(&X, &N, nullptr, &dXu, &dXv, up, vp, 0);
    const Real lu = sqrt<Real>(dXu[0] * dXu[0] + dXu[1] * dXu[1] + dXu[2] * dXu[2]);
    const Real lv = sqrt<Real>(dXv[0] * dXv[0] + dXv[1] * dXv[1] + dXv[2] * dXv[2]);
    for (const Real d : {(Real)1e-1, (Real)1e-3, (Real)1e-7}) {
      for (const Real side : {(Real)1, (Real)-1}) {
        for (Integer k = 0; k < COORD_DIM; k++) Xt.PushBack(X[k] + h * (-p[2] * dXu[k] / lu - p[3] * dXv[k] / lv + side * d * N[k]));
        for (Integer k = 0; k < COORD_DIM; k++) Nt.PushBack(N[k]);
        Pt.PushBack(p[0]);
        Pt.PushBack(p[1]);
      }
    }
  }
}

// One row of errors, one column per element order; marked if any exceeds the limit
template <class Real> void PrintRow(const std::string& label, const std::vector<Real>& err, const Real limit) {
  const std::ios_base::fmtflags flags = std::cout.flags();
  const std::streamsize prec = std::cout.precision();
  std::cout << "    " << std::left << std::setw(32) << label << std::right << std::scientific << std::setprecision(1);
  bool over = false;
  for (const Real e : err) {
    std::cout << std::setw(10) << e;
    over = over || !(e < limit);
  }
  std::cout << (over ? "   <-- exceeds the limit" : "") << "\n";
  std::cout.flags(flags);
  std::cout.precision(prec);
}

template <class Real> void PrintHeader(const std::string& title, const Real tol, const Real limit, const std::vector<Integer>& orders) {
  std::cout << "  " << title << ", tol " << tol << ", limit " << limit << "\n    " << std::setw(32) << "" << "order:";
  for (size_t i = 0; i < orders.size(); i++) std::cout << std::setw(i ? 10 : 4) << orders[i];
  std::cout << "\n";
}

}

// ============================================================================================
// 1. Building blocks
// ============================================================================================

// Each endpoint correction integrates constants exactly (its weights sum to nskip - 1/2), and the
// corrected trapezoidal rule on [0,1] converges at the correction's order until round-off.
template <class Real> void test_AlpertQuadRule() {
  using Correction = typename AlpertQuadRule<Real>::EndpointCorrection;
  const auto integrate = [](const Correction& L, const Correction& R, const Integer N, const auto& f) {
    const Real h = (Real)1 / (N - 1);
    Real I = 0;
    for (Integer i = L.nskip; i <= N - 1 - R.nskip; i++) I += h * f(i * h);
    for (Long k = 0; k < L.nds.Dim(); k++) I += h * L.wts[k] * f(L.nds[k] * h);
    for (Long k = 0; k < R.nds.Dim(); k++) I += h * R.wts[k] * f(1 - R.nds[k] * h);
    return I;
  };
  const auto check = [&integrate](const Correction& L, const Correction& R, const Integer order, const auto& f, const Real I_exact) {
    for (const Correction* C : {&L, &R}) {
      Real wsum = 0;
      for (const Real w : C->wts) wsum += w;
      SCTL_ASSERT(fabs(wsum - (C->nskip - (Real)0.5)) < 1e-14);
    }
    const Real e0 = fabs(integrate(L, R, 64, f) - I_exact);
    const Real e1 = fabs(integrate(L, R, 128, f) - I_exact);
    SCTL_ASSERT(e1 < 1e-13 || log2(e0 / e1) > order - (Real)1.5);
  };

  const auto f_log = [](const Real x) { return log<Real>(x) / (1 + x) + cos<Real>(x); };
  const Real I_log = -const_pi<Real>() * const_pi<Real>() / 12 + sin<Real>(1);
  for (const Integer order : {2, 3, 4, 5, 6, 8, 10, 12, 14, 16}) {
    check(AlpertQuadRule<Real>::LogCorrection(order), AlpertQuadRule<Real>::SmoothCorrection(order), order, f_log, I_log);
  }

  const auto f_smooth = [](const Real x) { return exp<Real>(x) * cos<Real>(3 * x); };
  const Real I_smooth = (exp<Real>(1) * (cos<Real>(3) + 3 * sin<Real>(3)) - 1) / 10;
  for (const Integer order : {3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32}) {
    check(AlpertQuadRule<Real>::SmoothCorrection(order), AlpertQuadRule<Real>::SmoothCorrection(order), order, f_smooth, I_smooth);
  }
  for (Integer order = 0; order <= 32; order++) { // Untabulated orders get the next tabulated one
    const Integer next = (order <= 8 ? std::max<Integer>(order, 3) : (order + 3) / 4 * 4);
    SCTL_ASSERT(AlpertQuadRule<Real>::SmoothCorrection(order).nds.Dim() == AlpertQuadRule<Real>::SmoothCorrection(next).nds.Dim());
    SCTL_ASSERT(AlpertQuadRule<Real>::SmoothCorrection(order).nds[0] == AlpertQuadRule<Real>::SmoothCorrection(next).nds[0]);
  }
}

// The centered log-singular rule on [0,1] integrates log|v - v0| times smooth functions, and smooth
// functions, to near machine precision.
template <class Real> void test_LogSingularQuad1D() {
  const Real v0 = (Real)0.6;
  Vector<Real> param, w;
  QuadElemTestAccess<Real>::LogSingularQuad1D(param, w, v0, 5, 24);
  SCTL_ASSERT(param.Dim() == w.Dim() && param.Dim() > 0);
  for (const Real p : param) SCTL_ASSERT(p > 0 && p < 1);

  const auto quad = [&param, &w](const auto& f) {
    Real I = 0;
    for (Long i = 0; i < param.Dim(); i++) I += w[i] * f(param[i]);
    return I;
  };
  const Real a = v0, la = log<Real>(a), lb = log<Real>(1 - a);
  const Real I0 = a * la + (1 - a) * lb - 1; // int log|v - v0|
  const Real I1 = ((1 - a * a) / 2) * lb + (a * a / 2) * la - (Real)0.25 - a / 2; // int v log|v - v0|
  const Real I2 = ((1 - a * a * a) / 3) * lb - (Real)1 / 9 - a / 6 - a * a / 3 + (a * a * a / 3) * la; // int v^2 log|v - v0|
  const Real Icos = sin<Real>(3) / 3; // int cos(3v)
  SCTL_ASSERT(fabs(quad([](Real) { return (Real)1; }) - 1) < 1e-13);
  SCTL_ASSERT(fabs(quad([v0](Real v) { return log<Real>(fabs(v - v0)); }) - I0) < 1e-13);
  SCTL_ASSERT(fabs(quad([v0](Real v) { return v * log<Real>(fabs(v - v0)); }) - I1) < 1e-13);
  SCTL_ASSERT(fabs(quad([v0](Real v) { return (1 + v * v) * log<Real>(fabs(v - v0)) + cos<Real>(3 * v); }) - (I0 + I2 + Icos)) < 1e-13);
  SCTL_ASSERT(fabs(quad([](Real v) { return cos<Real>(3 * v); }) - Icos) < 1e-13);
}

// On z = u*v, which an element of order at least 3 represents exactly, GetGeom returns the analytic
// position, tangents, normal and area element, and GetNodeCoord their values at the nodes.
template <class Real> void test_GetGeom() {
  const Integer order = 8;
  const QuadElemList<Real> qel = TestElem<Real>(order, true);
  SCTL_ASSERT(qel.Size() == 1 && qel.Order() == order);

  const Vector<Real> up{0, (Real)0.13, (Real)0.5, (Real)0.77, 1}, vp{(Real)0.02, (Real)0.4, (Real)0.9};
  Vector<Real> X, Xn, Xa, dXu, dXv;
  qel.GetGeom(&X, &Xn, &Xa, &dXu, &dXv, up, vp, 0);
  const Real tol = 1e-13;
  for (Long a = 0; a < up.Dim(); a++) {
    for (Long b = 0; b < vp.Dim(); b++) {
      const Long p = a * vp.Dim() + b;
      const Real u = up[a], v = vp[b], s = sqrt<Real>(1 + u * u + v * v);
      const std::array<Real,3> X_{u, v, u * v}, Xu_{1, 0, v}, Xv_{0, 1, u}, Xn_{-v / s, -u / s, 1 / s};
      for (Integer k = 0; k < COORD_DIM; k++) {
        SCTL_ASSERT(fabs(X[p * COORD_DIM + k] - X_[k]) < tol);
        SCTL_ASSERT(fabs(dXu[p * COORD_DIM + k] - Xu_[k]) < tol);
        SCTL_ASSERT(fabs(dXv[p * COORD_DIM + k] - Xv_[k]) < tol);
        SCTL_ASSERT(fabs(Xn[p * COORD_DIM + k] - Xn_[k]) < tol);
      }
      SCTL_ASSERT(fabs(Xa[p] - s) < tol);
    }
  }

  Vector<Real> Xnode, Xnnode, Xg, Xng;
  Vector<Long> cnt;
  qel.GetNodeCoord(&Xnode, &Xnnode, &cnt);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  qel.GetGeom(&Xg, &Xng, nullptr, nullptr, nullptr, nds, nds, 0);
  SCTL_ASSERT(cnt.Dim() == 1 && cnt[0] == order * order);
  SCTL_ASSERT(Xnode.Dim() == Xg.Dim() && RelErr(Xnode, Xg) < tol && RelErr(Xnnode, Xng) < tol);
  for (Integer i = 0; i < order; i++) { // Node i*order+j at (u_i, v_j)
    for (Integer j = 0; j < order; j++) {
      SCTL_ASSERT(fabs(Xnode[(i * order + j) * COORD_DIM + 0] - nds[i]) < tol);
      SCTL_ASSERT(fabs(Xnode[(i * order + j) * COORD_DIM + 1] - nds[j]) < tol);
    }
  }
}

// The far-field rule is the element's Gauss-Legendre rule: its weights integrate the area of
// z = u*v, and at targets beyond the cut-off distance from every node (above the center, beyond an
// edge, beyond a corner) it evaluates the single-layer potential of a smooth density to within ten
// times the tolerance.
template <class Real> Real test_GetFarFieldNodes() {
  const Integer order = 12;
  const QuadElemList<Real> qel = TestElem<Real>(order, true);
  Vector<Real> Xnode;
  qel.GetNodeCoord(&Xnode, nullptr, nullptr);
  const Vector<Real> sigma = TestDensity(Xnode, 1);
  const Laplace3D_FxU ker;

  const Real area_ref = [](){ // int sqrt(1 + u^2 + v^2) over the unit square
    const Vector<Real>& x = LegQuadRule<Real>::nds(40);
    const Vector<Real>& w = LegQuadRule<Real>::wts(40);
    Real A = 0;
    for (Integer i = 0; i < 40; i++) {
      for (Integer j = 0; j < 40; j++) A += w[i] * w[j] * sqrt<Real>(1 + x[i] * x[i] + x[j] * x[j]);
    }
    return A;
  }();

  Real worst = 0;
  for (const Real tol : {(Real)1e-4, (Real)1e-8, (Real)1e-12}) {
    Vector<Real> X, Xn, wts, dist_far;
    Vector<Long> cnt;
    qel.GetFarFieldNodes(X, Xn, wts, dist_far, cnt, tol);
    SCTL_ASSERT(cnt.Dim() == 1 && cnt[0] == order * order && RelErr(X, Xnode) == 0);
    Real area = 0;
    for (const Real w : wts) area += w;
    SCTL_ASSERT(fabs(area - area_ref) < 1e-13);

    const std::array<std::array<Real,5>,3> dirs{{{0.5, 0.5, 0, 0, 1}, {0, 0.5, -1, 0, 0}, {0, 0, -1, -1, 0}}}; // (u, v, and the direction's coefficients of dX/du, dX/dv and the normal)
    for (const auto& d : dirs) {
      Vector<Real> Xb, Nb, dXu, dXv;
      qel.GetGeom(&Xb, &Nb, nullptr, &dXu, &dXv, Vector<Real>{d[0]}, Vector<Real>{d[1]}, 0);
      std::array<Real,3> dir;
      for (Integer k = 0; k < COORD_DIM; k++) dir[k] = d[2] * dXu[k] + d[3] * dXv[k] + d[4] * Nb[k];
      const Real ldir = sqrt<Real>(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
      const auto target = [&Xb, &dir, ldir](const Real s) {
        return Vector<Real>{Xb[0] + s * dir[0] / ldir, Xb[1] + s * dir[1] / ldir, Xb[2] + s * dir[2] / ldir};
      };
      const auto is_far = [&X, &dist_far, &target](const Real s) { // At least dist_far from every node
        const Vector<Real> Xt = target(s);
        for (Long i = 0; i < dist_far.Dim(); i++) {
          Real r2 = 0;
          for (Integer k = 0; k < COORD_DIM; k++) r2 += (Xt[k] - X[i * COORD_DIM + k]) * (Xt[k] - X[i * COORD_DIM + k]);
          if (r2 < dist_far[i] * dist_far[i]) return false;
        }
        return true;
      };
      Real s0 = 0, s1 = 1;
      while (!is_far(s1)) s1 *= 2;
      for (Integer iter = 0; iter < 50; iter++) {
        const Real s = (s0 + s1) / 2;
        (is_far(s) ? s1 : s0) = s;
      }
      const Vector<Real> Xt = target(s1);
      Vector<Real> F(wts.Dim()), U;
      for (Long i = 0; i < wts.Dim(); i++) F[i] = sigma[i] * wts[i];
      ker.Eval(U, Xt, X, Xn, F);
      const Real err = RelErr(U, ReferencePotential(qel, 0, sigma, Xt, Vector<Real>{(Real)0.5, (Real)0.5}, Vector<Real>(), ker)) / tol;
      worst = MaxErr(worst, err);
      SCTL_ASSERT(err < 10);
    }
  }
  return worst;
}

// A target lifted off the flat or curved element snaps back to its node, and the distance returned
// is the distance to that node.
template <class Real> void test_GetClosestNode(const bool curved) {
  const Integer order = 8;
  const QuadElemList<Real> qel = TestElem<Real>(order, curved);
  Vector<Real> X, Xn;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  const Integer trg_idx = 13;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Real utrg = nds[trg_idx / order], vtrg = nds[trg_idx % order];
  const Real d = (curved ? (Real)0.001 : (Real)0.1);
  const Real tol = (curved ? (Real)1e-8 : (Real)1e-9);

  Vector<Real> Xt(COORD_DIM);
  for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = X[trg_idx * COORD_DIM + k] + d * Xn[trg_idx * COORD_DIM + k];
  Real ustar, vstar;
  const Real dist = QuadElemTestAccess<Real>::GetClosestNode(ustar, vstar, qel, 0, Xt);
  SCTL_ASSERT(fabs(ustar - utrg) < tol && fabs(vstar - vtrg) < tol && fabs(dist - d) < tol);

  Xt[0] -= (Real)0.0013;
  Xt[1] += (Real)0.0005;
  Real r2 = 0;
  for (Integer k = 0; k < COORD_DIM; k++) r2 += (Xt[k] - X[trg_idx * COORD_DIM + k]) * (Xt[k] - X[trg_idx * COORD_DIM + k]);
  const Real dist2 = QuadElemTestAccess<Real>::GetClosestNode(ustar, vstar, qel, 0, Xt);
  SCTL_ASSERT(fabs(ustar - utrg) < tol && fabs(vstar - vtrg) < tol && fabs(dist2 - sqrt<Real>(r2)) < tol);
}

// GetClosestPoint finds the foot of the perpendicular at an off-node point of the flat or curved
// element, and for a generic target a point where the residual is orthogonal to both tangents;
// on the curved element to 1% of the distance.
template <class Real> void test_GetClosestPoint(const bool curved) {
  const QuadElemList<Real> qel = TestElem<Real>(8, curved);
  const Real u0 = (Real)0.37, v0 = (Real)0.62;
  const Real d = (curved ? (Real)0.01 : (Real)0.1);
  const Real tol = (curved ? (Real)1e-2 * d : (Real)1e-9);
  Vector<Real> Xs, Ns;
  qel.GetGeom(&Xs, &Ns, nullptr, nullptr, nullptr, Vector<Real>{u0}, Vector<Real>{v0}, 0);
  Vector<Real> Xt(COORD_DIM);
  for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = Xs[k] + d * Ns[k];

  Real ustar, vstar;
  const Real dist = QuadElemTestAccess<Real>::GetClosestPoint(ustar, vstar, qel, 0, Xt);
  SCTL_ASSERT(fabs(ustar - u0) < tol && fabs(vstar - v0) < tol && fabs(dist - d) < tol);

  if (!curved) { // A tangential shift moves the foot with it
    Xt[0] -= (Real)0.0013;
    Xt[1] += (Real)0.0005;
    const Real dist2 = QuadElemTestAccess<Real>::GetClosestPoint(ustar, vstar, qel, 0, Xt);
    SCTL_ASSERT(fabs(ustar - (u0 - (Real)0.0013)) < tol && fabs(vstar - (v0 + (Real)0.0005)) < tol && fabs(dist2 - d) < tol);
    return;
  }

  const Vector<Real> Xt2{Xs[0] + (Real)0.05, Xs[1] - (Real)0.03, Xs[2] + (Real)0.08};
  QuadElemTestAccess<Real>::GetClosestPoint(ustar, vstar, qel, 0, Xt2);
  SCTL_ASSERT(ustar > tol && ustar < 1 - tol && vstar > tol && vstar < 1 - tol);
  Vector<Real> Xc, dXu, dXv;
  qel.GetGeom(&Xc, nullptr, nullptr, &dXu, &dXv, Vector<Real>{ustar}, Vector<Real>{vstar}, 0);
  Real ru = 0, rv = 0, tu = 0, tv = 0, rr = 0;
  for (Integer k = 0; k < COORD_DIM; k++) {
    const Real r = Xc[k] - Xt2[k];
    ru += r * dXu[k];
    rv += r * dXv[k];
    tu += dXu[k] * dXu[k];
    tv += dXv[k] * dXv[k];
    rr += r * r;
  }
  SCTL_ASSERT(fabs(ru) < (Real)1e-2 * sqrt<Real>(tu * rr) && fabs(rv) < (Real)1e-2 * sqrt<Real>(tv * rr));
}

// Write then Read reproduces the element list exactly, with one file per process.
template <class Real> void test_WriteRead(const Comm& comm) {
  const QuadElemList<Real> qel = BuildTwistedSphere<Real>(8, 2, 1, (Real)0.3, comm);
  const std::string fname = (std::filesystem::temp_directory_path() / "sctl-test-quad-elem-").string();
  qel.Write(fname, comm);
  QuadElemList<Real> qel2;
  qel2.template Read<Real>(fname, comm);
  std::remove((fname + detail_quadelem::RankFileName("", comm)).c_str());

  Vector<Real> X, Xn, X2, Xn2;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  qel2.GetNodeCoord(&X2, &Xn2, nullptr);
  SCTL_ASSERT(qel2.Size() == qel.Size() && qel2.Order() == qel.Order());
  SCTL_ASSERT(X2.Dim() == X.Dim() && (X.Dim() == 0 || (RelErr(X2, X) == 0 && RelErr(Xn2, Xn) == 0)));
}

// Copy to ExtReal and back keeps the nodes and the scheme, and the normals of the copy are those of a
// list built in ExtReal from the same coordinates.
template <class Real> void test_Copy() {
  QuadElemList<Real> qel = TestElem<Real>(8, true);
  qel.SetQuadScheme(QuadScheme<Real>::Hedgehog);
  QuadElemList<ExtReal> qel_q;
  qel.Copy(qel_q);
  QuadElemList<Real> qel2;
  qel_q.Copy(qel2);

  Vector<Real> X, X2;
  Vector<ExtReal> Xq;
  qel.GetNodeCoord(&X, nullptr, nullptr);
  qel_q.GetNodeCoord(&Xq, nullptr, nullptr);
  qel2.GetNodeCoord(&X2, nullptr, nullptr);
  SCTL_ASSERT(qel_q.Size() == qel.Size() && qel_q.Order() == qel.Order() && Xq.Dim() == X.Dim());
  for (Long i = 0; i < X.Dim(); i++) SCTL_ASSERT(Xq[i] == (ExtReal)X[i] && X2[i] == X[i]);
  SCTL_ASSERT(QuadElemTestAccess<ExtReal>::Scheme(qel_q) == QuadScheme<ExtReal>::Hedgehog);
  Vector<ExtReal> Xn_q, Xn_ext;
  qel_q.GetNodeCoord(nullptr, &Xn_q, nullptr);
  QuadElemList<ExtReal>(qel.Order(), X).GetNodeCoord(nullptr, &Xn_ext, nullptr);
  SCTL_ASSERT(Xn_q.Dim() == Xn_ext.Dim());
  for (Long i = 0; i < Xn_q.Dim(); i++) SCTL_ASSERT(Xn_q[i] == Xn_ext[i]);
  SCTL_ASSERT(QuadElemTestAccess<Real>::Scheme(qel2) == QuadScheme<Real>::Hedgehog);
}

// GetVTUData appends each element as an (order+1) x (order+1) grid of quadrilaterals whose inner
// vertices are the nodes, carrying the nodal values; WriteVTK writes a non-empty file per process,
// and the index file on the first.
template <class Real> void test_VTU(const Comm& comm) {
  const Integer order = 8, dof = 2, Ng = order + 2;
  const QuadElemList<Real> qel = BuildTwistedSphere<Real>(order, 1, 1, (Real)0.3, comm);
  const Long Nelem = qel.Size();
  Vector<Real> X;
  qel.GetNodeCoord(&X, nullptr, nullptr);
  const Vector<Real> F = TestDensity(X, dof);

  VTUData vtu;
  qel.GetVTUData(vtu, F);
  SCTL_ASSERT(vtu.coord.Dim() == Nelem * Ng * Ng * COORD_DIM && vtu.value.Dim() == Nelem * Ng * Ng * dof);
  const Long Ncell = Nelem * (Ng - 1) * (Ng - 1);
  SCTL_ASSERT(vtu.connect.Dim() == Ncell * 4 && vtu.offset.Dim() == Ncell && vtu.types.Dim() == Ncell);
  for (Long e = 0; e < Nelem; e++) { // cell (i,j) of element e: grid points (i,j), (i,j+1), (i+1,j+1), (i+1,j)
    for (Integer i = 0; i + 1 < Ng; i++) {
      for (Integer j = 0; j + 1 < Ng; j++) {
        const Long c = (e * (Ng - 1) + i) * (Ng - 1) + j, g = e * Ng * Ng + i * Ng + j;
        SCTL_ASSERT(vtu.connect[4 * c] == g && vtu.connect[4 * c + 1] == g + 1 && vtu.connect[4 * c + 2] == g + Ng + 1 && vtu.connect[4 * c + 3] == g + Ng);
        SCTL_ASSERT(vtu.offset[c] == 4 * (c + 1) && vtu.types[c] == 9); // 9: VTK_QUAD
      }
    }
  }
  const Real tol = 1e-6; // VTUData stores float
  for (Long e = 0; e < Nelem; e++) {
    for (Integer i = 0; i < order; i++) {
      for (Integer j = 0; j < order; j++) {
        const Long g = e * Ng * Ng + (i + 1) * Ng + (j + 1), p = e * order * order + i * order + j;
        for (Integer k = 0; k < COORD_DIM; k++) SCTL_ASSERT(fabs(vtu.coord[g * COORD_DIM + k] - X[p * COORD_DIM + k]) < tol);
        for (Integer k = 0; k < dof; k++) SCTL_ASSERT(fabs(vtu.value[g * dof + k] - F[p * dof + k]) < tol);
      }
    }
  }

  const std::string fname = (std::filesystem::temp_directory_path() / "sctl-test-quad-elem-vtk").string();
  qel.WriteVTK(fname, F, comm);
  const std::string fname_rank = fname + detail_quadelem::RankFileName("", comm) + ".vtu";
  SCTL_ASSERT(std::filesystem::exists(fname_rank) && std::filesystem::file_size(fname_rank) > 0);
  std::remove(fname_rank.c_str());
  if (!comm.Rank()) {
    SCTL_ASSERT(std::filesystem::exists(fname + ".pvtu"));
    std::remove((fname + ".pvtu").c_str());
  }
}

// An empty, default-constructed list (order 0) next to a sphere, with the targets on the nodes, leaves
// the potential unchanged
template <class Real> void test_EmptyList(const Comm& comm) {
  const QuadElemList<Real> qel = BuildTwistedSphere<Real>(4, 2, 1, (Real)0.3, comm);
  Vector<Real> X;
  qel.GetNodeCoord(&X, nullptr, nullptr);
  const Vector<Real> sigma = TestDensity(X, 1);
  const auto potential = [&qel, &sigma, &comm](const bool with_empty) {
    BoundaryIntegralOp<Real,Laplace3D_FxU> op(Laplace3D_FxU(), false, comm);
    op.SetAccuracy((Real)1e-6);
    op.AddElemList(qel, "sphere");
    if (with_empty) op.AddElemList(QuadElemList<Real>(), "empty");
    Vector<Real> U;
    op.ComputePotential(U, sigma);
    return U;
  };
  SCTL_ASSERT(RelErr(potential(true), potential(false)) == 0);
}

// SurfaceSingularDegree gives the growth exponent along the surface of every SCTL kernel, with and without
// the target-normal contraction: 2 or more where the Duffy and TensorProduct self-interactions are wrong
// (checked against the average of the one-sided Hedgehog limits on twisted-sphere elements), 1 or less elsewhere
template <class Real> void test_SurfaceSingularDegree() {
  Real mu = (Real)1.7;
  Helmholtz3D_FxU h_fxu;
  Helmholtz3D_DxU h_dxu;
  Helmholtz3D_FxdU h_fxdu;
  HelmholtzDiff3D_FxdU hd_fxdu;
  h_fxu.SetCtxPtr(&mu);
  h_dxu.SetCtxPtr(&mu);
  h_fxdu.SetCtxPtr(&mu);
  hd_fxdu.SetCtxPtr(&mu);
  const auto check = [](const auto& ker, const Integer d, const Integer d_dot) { // d_dot < 0: no target-normal contraction for this kernel
    SCTL_ASSERT(fabs(detail_dispatch::SurfaceSingularDegree<Real>(ker, false, (Real)1e-4) - d) < (Real)0.1);
    if (d_dot >= 0) SCTL_ASSERT(fabs(detail_dispatch::SurfaceSingularDegree<Real>(ker, true, (Real)1e-4) - d_dot) < (Real)0.1);
  };
  check(Laplace3D_FxU(), 1, -1);
  check(Laplace3D_DxU(), 1, -1);
  check(Laplace3D_FxdU(), 2, 1);
  check(Laplace3D_Fxd2U(), 3, 3);
  check(Laplace3D_DxdU(), 3, 3);
  check(Stokes3D_FxU(), 1, 1);
  check(Stokes3D_DxU(), 1, 0);
  check(Stokes3D_FxT(), 2, 1);
  check(Stokes3D_FSxU(), 2, 1);
  check(Stokes3D_FxUP(), 2, -1);
  check(BiotSavart3D_FxU(), 2, 2);
  check(BiotSavart3D_FxdU(), 3, 3);
  check(h_fxu, 1, -1);
  check(h_dxu, 1, -1);
  check(h_fxdu, 2, 2);
  check(hd_fxdu, 0, 0);
}

#ifdef SCTL_QUAD_T
// Duffy, Stokes single layer, at a tolerance of 1e-30, for which its angular rule would need more than
// its largest (128 points): that rule is used, without an error
void test_ManyDigits() {
  QuadElemList<QuadReal> qel = TestElem<QuadReal>(4, true);
  qel.SetQuadScheme(QuadScheme<QuadReal>::Duffy);
  Vector<Matrix<QuadReal>> M(1);
  QuadElemList<QuadReal>::SelfInterac(M, Stokes3D_FxU(), (QuadReal)1e-30, false, &qel);
  for (Long i = 0; i < M[0].Dim(0) * M[0].Dim(1); i++) SCTL_ASSERT(fabs(M[0][0][i]) < (QuadReal)1e300); // fails on NaN too
}
#endif

// ============================================================================================
// 2. One element: near- and self-interactions for every scheme and element order
// ============================================================================================

// The adaptive reference against the closed forms on the flat element with constant density, at the
// targets of FlatNearTargets and at nnodes nodes spread over the element (every node if 0), for the
// Laplace single and double layers, and with all_kernels also the adjoint double layer and the Stokes
// single layer. Returns the largest error at each: {near, self}.
template <class Real> std::array<Real,2> test_ReferenceFlat(const std::vector<Integer>& orders, const Integer nnodes, const bool all_kernels) {
  std::array<Real,2> err{0, 0};
  const auto check = [&err, &orders, nnodes](const auto& ker, const FlatKernel flat_kernel) {
    constexpr Integer KDIM0 = std::decay_t<decltype(ker)>::SrcDim();
    const bool trg_dot_prod = (flat_kernel == FlatKernel::LaplaceAdjointDL);
    const bool self = (flat_kernel == FlatKernel::LaplaceSL || flat_kernel == FlatKernel::StokesSL); // the double layers vanish at the nodes
    for (const Integer order : orders) {
      const QuadElemList<Real> qel = TestElem<Real>(order, false);
      Vector<Real> sigma(order * order * KDIM0);
      for (Long i = 0; i < sigma.Dim(); i++) sigma[i] = (KDIM0 == 1 || i % KDIM0 == 2 ? 1 : 0);
      Vector<Real> Xt, Nt, Pt;
      FlatNearTargets(Xt, Nt);
      for (Long t = 0; t < Xt.Dim() / COORD_DIM; t++) { // expansion at the target's foot, moved onto the element
        Pt.PushBack(std::min<Real>(1, std::max<Real>(0, Xt[t * COORD_DIM + 0])));
        Pt.PushBack(std::min<Real>(1, std::max<Real>(0, Xt[t * COORD_DIM + 1])));
      }
      err[0] = MaxErr(err[0], RelErr(ReferencePotential(qel, 0, sigma, Xt, Pt, (trg_dot_prod ? Nt : Vector<Real>()), ker), FlatSquarePotential(flat_kernel, Xt)));
      if (self) {
        Vector<Real> Xnode, X, Pn;
        qel.GetNodeCoord(&Xnode, nullptr, nullptr);
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Integer nnode = order * order;
        for (Integer i = 0; i < (nnodes ? nnodes : nnode); i++) {
          const Integer p = (nnodes ? (i * nnode) / nnodes + order / 2 : i) % nnode;
          for (Integer k = 0; k < COORD_DIM; k++) X.PushBack(Xnode[p * COORD_DIM + k]);
          Pn.PushBack(nds[p / order]);
          Pn.PushBack(nds[p % order]);
        }
        err[1] = MaxErr(err[1], RelErr(ReferencePotential(qel, 0, sigma, X, Pn, Vector<Real>(), ker), FlatSquarePotential(flat_kernel, X)));
      }
    }
  };
  check(Laplace3D_FxU(), FlatKernel::LaplaceSL);
  check(Laplace3D_DxU(), FlatKernel::LaplaceDL);
  if (all_kernels) {
    check(Laplace3D_FxdU(), FlatKernel::LaplaceAdjointDL);
    check(Stokes3D_FxU(), FlatKernel::StokesSL);
  }
  return err;
}

// Near-interactions of the flat element with constant density against the closed forms, at the
// targets of FlatNearTargets in one call. Returns the error for each scheme and element order.
template <class Real, class Kernel> std::vector<std::vector<Real>> test_NearFlat(const Kernel& ker, const FlatKernel flat_kernel, const Real tol, const std::vector<Integer>& orders) {
  constexpr Integer KDIM0 = Kernel::SrcDim();
  const bool trg_dot_prod = (flat_kernel == FlatKernel::LaplaceAdjointDL);
  Vector<Real> Xt, Nt;
  FlatNearTargets(Xt, Nt);
  const Vector<Real> U_ref = FlatSquarePotential(flat_kernel, Xt);
  std::vector<std::vector<Real>> err(Schemes<Real>().size(), std::vector<Real>(orders.size()));
  for (Integer o = 0; o < (Integer)orders.size(); o++) {
    QuadElemList<Real> qel = TestElem<Real>(orders[o], false);
    Vector<Real> sigma(orders[o] * orders[o] * KDIM0);
    for (Long i = 0; i < sigma.Dim(); i++) sigma[i] = (KDIM0 == 1 || i % KDIM0 == 2 ? 1 : 0);
    for (Integer s = 0; s < (Integer)Schemes<Real>().size(); s++) {
      qel.SetQuadScheme(Schemes<Real>()[s].scheme);
      Matrix<Real> M;
      QuadElemList<Real>::NearInterac(M, Xt, (trg_dot_prod ? Nt : Vector<Real>()), ker, tol, 0, &qel);
      err[s][o] = RelErr(Apply(M, sigma), U_ref);
    }
  }
  return err;
}

// Near-interactions of a sphere patch resolved to tol, with a smooth density, against the adaptive
// reference, at the targets of CurvedNearTargets in one call.
template <class Real, class Kernel> std::vector<std::vector<Real>> test_NearCurved(const Kernel& ker, const bool trg_dot_prod, const Real tol, const std::vector<Integer>& orders) {
  constexpr Integer KDIM0 = Kernel::SrcDim();
  std::vector<std::vector<Real>> err(Schemes<Real>().size(), std::vector<Real>(orders.size()));
  for (Integer o = 0; o < (Integer)orders.size(); o++) {
    Real h;
    QuadElemList<Real> qel = ResolvedSpherePatch<Real>(orders[o], tol, h);
    Vector<Real> X, Xt, Nt, Pt;
    qel.GetNodeCoord(&X, nullptr, nullptr);
    CurvedNearTargets(Xt, Nt, Pt, qel, h);
    const Vector<Real> Nt_ = (trg_dot_prod ? Nt : Vector<Real>());
    const Vector<Real> sigma = TestDensity(X, KDIM0);
    const Vector<Real> U_ref = ReferencePotential(qel, 0, sigma, Xt, Pt, Nt_, ker);
    for (Integer s = 0; s < (Integer)Schemes<Real>().size(); s++) {
      qel.SetQuadScheme(Schemes<Real>()[s].scheme);
      Matrix<Real> M;
      QuadElemList<Real>::NearInterac(M, Xt, Nt_, ker, tol, 0, &qel);
      err[s][o] = RelErr(Apply(M, sigma), U_ref);
    }
  }
  return err;
}

// Self-interactions at every node of the flat element with constant density against the closed
// forms (single layers only; the double layers vanish on a plane).
template <class Real, class Kernel> std::vector<std::vector<Real>> test_SelfFlat(const Kernel& ker, const FlatKernel flat_kernel, const Real tol, const std::vector<Integer>& orders) {
  constexpr Integer KDIM0 = Kernel::SrcDim();
  std::vector<std::vector<Real>> err(Schemes<Real>().size(), std::vector<Real>(orders.size()));
  for (Integer o = 0; o < (Integer)orders.size(); o++) {
    QuadElemList<Real> qel = TestElem<Real>(orders[o], false);
    Vector<Real> X;
    qel.GetNodeCoord(&X, nullptr, nullptr);
    const Vector<Real> U_ref = FlatSquarePotential(flat_kernel, X);
    Vector<Real> sigma(orders[o] * orders[o] * KDIM0);
    for (Long i = 0; i < sigma.Dim(); i++) sigma[i] = (KDIM0 == 1 || i % KDIM0 == 2 ? 1 : 0);
    for (Integer s = 0; s < (Integer)Schemes<Real>().size(); s++) {
      qel.SetQuadScheme(Schemes<Real>()[s].scheme);
      Vector<Matrix<Real>> M(1);
      QuadElemList<Real>::SelfInterac(M, ker, tol, false, &qel);
      err[s][o] = RelErr(Apply(M[0], sigma), U_ref);
    }
  }
  return err;
}

// Self-interactions of a sphere patch resolved to tol, with a smooth density, against the adaptive
// reference (plus jump * sigma for the one-sided schemes), at the first nnodes of: the corner node,
// the middle node, and nodes next to two opposite edges.
template <class Real, class Kernel> std::vector<std::vector<Real>> test_SelfCurved(const Kernel& ker, const bool trg_dot_prod, const Real jump, const Real tol, const std::vector<Integer>& orders, const Integer nnodes) {
  constexpr Integer KDIM0 = Kernel::SrcDim();
  const Integer KDIM1 = (trg_dot_prod ? Kernel::TrgDim() / COORD_DIM : Kernel::TrgDim());
  SCTL_ASSERT(jump == 0 || KDIM0 == KDIM1);
  std::vector<std::vector<Real>> err(Schemes<Real>().size(), std::vector<Real>(orders.size()));
  for (Integer o = 0; o < (Integer)orders.size(); o++) {
    const Integer order = orders[o];
    Real h;
    QuadElemList<Real> qel = ResolvedSpherePatch<Real>(order, tol, h);
    Vector<Real> X, Xn;
    qel.GetNodeCoord(&X, &Xn, nullptr);
    const Vector<Real> sigma = TestDensity(X, KDIM0);
    const std::array<Integer,4> all_nodes{0, (order / 2) * order + order / 2, order / 2, (order - 1) * order + order / 3};
    const std::vector<Integer> nodes(all_nodes.begin(), all_nodes.begin() + nnodes);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    Vector<Real> Xt, Nt, Pt;
    for (const Integer t : nodes) {
      for (Integer k = 0; k < COORD_DIM; k++) {
        Xt.PushBack(X[t * COORD_DIM + k]);
        Nt.PushBack(Xn[t * COORD_DIM + k]);
      }
      Pt.PushBack(nds[t / order]);
      Pt.PushBack(nds[t % order]);
    }
    const Vector<Real> U_ref = ReferencePotential(qel, 0, sigma, Xt, Pt, (trg_dot_prod ? Nt : Vector<Real>()), ker);
    for (Integer s = 0; s < (Integer)Schemes<Real>().size(); s++) {
      qel.SetQuadScheme(Schemes<Real>()[s].scheme);
      Vector<Matrix<Real>> M(1);
      QuadElemList<Real>::SelfInterac(M, ker, tol, trg_dot_prod, &qel);
      const Vector<Real> U_all = Apply(M[0], sigma);
      Vector<Real> U, U_expect;
      for (Integer n = 0; n < (Integer)nodes.size(); n++) {
        for (Integer k = 0; k < KDIM1; k++) {
          U.PushBack(U_all[nodes[n] * KDIM1 + k]);
          U_expect.PushBack(U_ref[n * KDIM1 + k] + (Schemes<Real>()[s].one_sided ? jump * sigma[nodes[n] * KDIM0 + k] : 0));
        }
      }
      err[s][o] = RelErr(U, U_expect);
    }
  }
  return err;
}

// ============================================================================================
// 3. BoundaryIntegralOp on a small twisted sphere
// ============================================================================================

// Potential on a twisted sphere of elements of the given order, as many per face as resolve its
// geometry and the density to tol, with the density a smooth function of position, at targets on
// about nnodes nodes per process (self) and at 1e-3 and 0.1 element sizes off those nodes on both
// sides (near and far), in one target set, against the adaptive reference summed over all elements
// (plus jump * sigma at the nodes for the one-sided schemes). Returns the error for each tolerance
// and scheme.
template <class Real, class Kernel> std::vector<std::vector<Real>> test_BIO(const Kernel& ker, const bool trg_dot_prod, const Real jump, const Integer order, const std::vector<Real>& tols, const Long nnodes, const Comm& comm) {
  constexpr Integer KDIM0 = Kernel::SrcDim();
  const Integer KDIM1 = (trg_dot_prod ? Kernel::TrgDim() / COORD_DIM : Kernel::TrgDim());
  SCTL_ASSERT(jump == 0 || KDIM0 == KDIM1);
  const Real R = 1, twist = (Real)0.3;
  const Long nnode = order * order;
  std::vector<std::vector<Real>> err(tols.size(), std::vector<Real>(Schemes<Real>().size()));
  for (size_t i = 0; i < tols.size(); i++) {
    const Integer ppf = ResolvedSpherePPF<Real>(order, R, twist, tols[i]);
    const Real elem_size = 2 * R / ppf;
    QuadElemList<Real> qel = BuildTwistedSphere<Real>(order, ppf, R, twist, comm);
    const QuadElemList<Real> qel_all = BuildTwistedSphere<Real>(order, ppf, R, twist, Comm::Self());
    const Long elem0 = [&qel, &comm]() { // global index of the first local element
      StaticArray<Long,2> n{qel.Size(), 0};
      comm.Scan(n + 0, n + 1, 1, CommOp::SUM);
      return n[1] - qel.Size();
    }();

    Vector<Real> X, Xn, Xt, Nt, sigma_t;
    Vector<Long> trg_elem, trg_node; // global element and node of each target, for the targets on a node
    qel.GetNodeCoord(&X, &Xn, nullptr);
    const Vector<Real> sigma = TestDensity(X, KDIM0);
    const Long Nnode = X.Dim() / COORD_DIM;
    for (Long n = 0; n < Nnode; n += std::max<Long>(1, Nnode / nnodes)) {
      for (const Real d : {(Real)0, (Real)1e-3, (Real)-1e-3, (Real)0.1, (Real)-0.1}) {
        for (Integer k = 0; k < COORD_DIM; k++) {
          Xt.PushBack(X[n * COORD_DIM + k] + d * elem_size * Xn[n * COORD_DIM + k]);
          Nt.PushBack(Xn[n * COORD_DIM + k]);
        }
        for (Integer k = 0; k < KDIM1; k++) sigma_t.PushBack(d == 0 ? sigma[n * KDIM0 + k] : 0); // density at the targets on a node
        trg_elem.PushBack(elem0 + n / nnode);
        trg_node.PushBack(n % nnode);
      }
    }

    Vector<Real> X_all;
    qel_all.GetNodeCoord(&X_all, nullptr, nullptr);
    const Vector<Real> U_ref = SurfaceReference(qel_all, TestDensity(X_all, KDIM0), Xt, trg_elem, trg_node, (trg_dot_prod ? Nt : Vector<Real>()), ker);

    for (size_t s = 0; s < Schemes<Real>().size(); s++) {
      qel.SetQuadScheme(Schemes<Real>()[s].scheme);
      Vector<Real> U;
      BoundaryIntegralOp<Real,Kernel> op(ker, trg_dot_prod, comm);
      op.SetAccuracy(tols[i]);
      op.AddElemList(qel);
      op.SetTargetCoord(Xt);
      if (trg_dot_prod) op.SetTargetNormal(Nt);
      op.ComputePotential(U, sigma);
      const Vector<Real> U_expect = U_ref + (Schemes<Real>()[s].one_sided ? jump : 0) * sigma_t;
      Real e = 0, ref = 0;
      for (Long j = 0; j < U.Dim(); j++) {
        e = MaxErr<Real>(e, fabs(U[j] - U_expect[j]));
        ref = MaxErr<Real>(ref, fabs(U_expect[j]));
      }
      err[i][s] = GlobalReduce(e, comm, CommOp::MAX) / GlobalReduce(ref, comm, CommOp::MAX);
    }
  }
  return err;
}

// The normal derivative of the single layer of sigma = xyz (a spherical harmonic of degree 3) on the unit
// twisted sphere, resolved to tol, at targets 1e-4 element sizes inside and outside the sphere directly
// above nodes, against its closed form (3/7 r^2 Y inside, -4/7 r^-5 Y outside, Y = xyz on the unit sphere).
// The far-field rule's term of the node under a target, about 1e8 times the result here, is added by the
// far-field evaluation and subtracted by the near correction, which must compute it identically. Returns the
// error for each scheme.
template <class Real> std::vector<Real> test_NearNodeTargets(const Real tol, const Comm& comm) {
  const Integer order = 8;
  const Real R = 1, twist = (Real)0.3;
  const Integer ppf = ResolvedSpherePPF<Real>(order, R, twist, tol);
  const Real d = (Real)1e-4 * 2 * R / ppf;
  QuadElemList<Real> qel = BuildTwistedSphere<Real>(order, ppf, R, twist, comm);

  Vector<Real> X, sigma, Xt, Nt, U_ref;
  qel.GetNodeCoord(&X, nullptr, nullptr);
  const Long Nnode = X.Dim() / COORD_DIM;
  for (Long n = 0; n < Nnode; n++) sigma.PushBack(X[n * COORD_DIM + 0] * X[n * COORD_DIM + 1] * X[n * COORD_DIM + 2]);
  for (Long n = 0; n < Nnode; n += 7) {
    const Real r = sqrt<Real>(X[n * COORD_DIM + 0] * X[n * COORD_DIM + 0] + X[n * COORD_DIM + 1] * X[n * COORD_DIM + 1] + X[n * COORD_DIM + 2] * X[n * COORD_DIM + 2]);
    const Real Y = (X[n * COORD_DIM + 0] / r) * (X[n * COORD_DIM + 1] / r) * (X[n * COORD_DIM + 2] / r);
    for (const Real rad : {R - d, R + d}) {
      for (Integer k = 0; k < COORD_DIM; k++) {
        Xt.PushBack(X[n * COORD_DIM + k] / r * rad);
        Nt.PushBack(X[n * COORD_DIM + k] / r);
      }
      U_ref.PushBack(rad < R ? (Real)3 / 7 * rad * rad * Y : -(Real)4 / 7 * Y / pow<Real>(rad, 5));
    }
  }

  std::vector<Real> err(Schemes<Real>().size());
  for (size_t s = 0; s < Schemes<Real>().size(); s++) {
    qel.SetQuadScheme(Schemes<Real>()[s].scheme);
    Vector<Real> U;
    BoundaryIntegralOp<Real,Laplace3D_FxdU> op(Laplace3D_FxdU(), true, comm);
    op.SetAccuracy(tol);
    op.AddElemList(qel);
    op.SetTargetCoord(Xt);
    op.SetTargetNormal(Nt);
    op.ComputePotential(U, sigma);
    Real e = 0, ref = 0;
    for (Long j = 0; j < U.Dim(); j++) {
      e = MaxErr<Real>(e, fabs(U[j] - U_ref[j]));
      ref = MaxErr<Real>(ref, fabs(U_ref[j]));
    }
    err[s] = GlobalReduce(e, comm, CommOp::MAX) / GlobalReduce(ref, comm, CommOp::MAX);
  }
  return err;
}

// The reference summed over a closed twisted sphere (order 16, resolved to 1e-12) against exact
// results, at 8 nodes and at points 0.1 inside them: the double-layer identity for a constant density
// q (D[q] = -q/2 on the surface, -q inside) and Green's identity for the field u of a point source
// outside (S[du/dn] - D[u] = u/2 on the surface, u inside). Returns the largest error, relative to
// q/2 and to max|u|.
template <class Real> Real test_ReferenceSphere() {
  const Integer order = 16;
  const Real R = 1, twist = (Real)0.3;
  const QuadElemList<Real> qel = BuildTwistedSphere<Real>(order, ResolvedSpherePPF<Real>(order, R, twist, (Real)1e-12), R, twist, Comm::Self());
  Vector<Real> X, Xn, Xt;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  const Long nnode = order * order, Nnode = X.Dim() / COORD_DIM;
  Vector<Long> trg_elem, trg_node;
  for (Long n = Nnode / 16; n < Nnode; n += Nnode / 8) {
    for (const Real d : {(Real)0, (Real)0.1}) { // even targets on the surface, odd ones inside
      for (Integer k = 0; k < COORD_DIM; k++) Xt.PushBack(X[n * COORD_DIM + k] - d * Xn[n * COORD_DIM + k]);
      trg_elem.PushBack(n / nnode);
      trg_node.PushBack(n % nnode);
    }
  }
  const Long Ntrg = Xt.Dim() / COORD_DIM;

  Real err = 0;
  const auto dl_identity = [&](const auto& ker) {
    constexpr Integer KDIM0 = std::decay_t<decltype(ker)>::SrcDim();
    Vector<Real> q(Nnode * KDIM0);
    for (Long i = 0; i < Nnode; i++) {
      for (Integer k = 0; k < KDIM0; k++) q[i * KDIM0 + k] = k + 1;
    }
    const Vector<Real> U = SurfaceReference(qel, q, Xt, trg_elem, trg_node, Vector<Real>(), ker);
    for (Long t = 0; t < Ntrg; t++) {
      for (Integer k = 0; k < KDIM0; k++) err = MaxErr<Real>(err, fabs(U[t * KDIM0 + k] + (t % 2 ? 1 : (Real)0.5) * (k + 1)) / ((Real)0.5 * (k + 1)));
    }
  };
  const auto greens_identity = [&](const auto& ker_sl, const auto& ker_dl, const auto& ker_grad) {
    constexpr Integer KDIM0 = std::decay_t<decltype(ker_sl)>::SrcDim();
    const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2}, Xn0{0, 0, 0};
    Vector<Real> F0(KDIM0), u_surf, du, u_trg;
    for (Integer k = 0; k < KDIM0; k++) F0[k] = (Real)0.7 - (Real)0.5 * k;
    ker_sl.Eval(u_surf, X, X0, Xn0, F0);
    ker_grad.Eval(du, X, X0, Xn0, F0);
    ker_sl.Eval(u_trg, Xt, X0, Xn0, F0);
    Vector<Real> du_dn(Nnode * KDIM0);
    for (Long i = 0; i < Nnode; i++) {
      for (Integer j = 0; j < KDIM0; j++) {
        Real s = 0;
        for (Integer k = 0; k < COORD_DIM; k++) s += du[(i * KDIM0 + j) * COORD_DIM + k] * Xn[i * COORD_DIM + k];
        du_dn[i * KDIM0 + j] = s;
      }
    }
    const Vector<Real> U = SurfaceReference(qel, du_dn, Xt, trg_elem, trg_node, Vector<Real>(), ker_sl) - SurfaceReference(qel, u_surf, Xt, trg_elem, trg_node, Vector<Real>(), ker_dl);
    Real e = 0, ref = 0;
    for (Long t = 0; t < Ntrg; t++) {
      for (Integer k = 0; k < KDIM0; k++) {
        e = MaxErr<Real>(e, fabs(U[t * KDIM0 + k] - (t % 2 ? 1 : (Real)0.5) * u_trg[t * KDIM0 + k]));
        ref = MaxErr<Real>(ref, fabs(u_trg[t * KDIM0 + k]));
      }
    }
    err = MaxErr(err, e / ref);
  };
  dl_identity(Laplace3D_DxU());
  dl_identity(Stokes3D_DxU());
  greens_identity(Laplace3D_FxU(), Laplace3D_DxU(), Laplace3D_FxdU());
  greens_identity(Stokes3D_FxU(), Stokes3D_DxU(), Stokes3D_FxT());
  return err;
}

// ============================================================================================
// 4. Convergence study on a finer sphere (opt-in)
// ============================================================================================

// The far-field weights sum to the sphere's area 4 pi R^2. Returns the relative error.
template <class Real> Real test_SurfaceArea(const QuadElemList<Real>& qel, const Real R, const Comm& comm) {
  Vector<Real> X, Xn, wts, dist_far;
  Vector<Long> cnt;
  qel.GetFarFieldNodes(X, Xn, wts, dist_far, cnt, 1);
  Real area = 0;
  for (const Real w : wts) area += w;
  return fabs(GlobalReduce(area, comm, CommOp::SUM) - 4 * const_pi<Real>() * R * R) / (4 * const_pi<Real>() * R * R);
}

// Double-layer identity on a closed surface with outward normals, for a constant density q: on the
// surface D[q] = -q/2 (principal value), and 0 as the limit from outside (one_sided). Returns the
// largest deviation relative to q/2.
template <class Real, class KerDL> Real test_DLIdentity(const QuadElemList<Real>& qel, const Comm& comm, const Real tol, const bool one_sided) {
  constexpr Integer KDIM0 = KerDL::SrcDim();
  BoundaryIntegralOp<Real,KerDL> op(KerDL(), false, comm);
  op.SetAccuracy(tol);
  op.AddElemList(qel);
  Vector<Real> X;
  qel.GetNodeCoord(&X, nullptr, nullptr);
  const Long Nnode = X.Dim() / COORD_DIM;
  Vector<Real> q(Nnode * KDIM0), U;
  for (Long i = 0; i < Nnode; i++) {
    for (Integer k = 0; k < KDIM0; k++) q[i * KDIM0 + k] = k + 1;
  }
  op.ComputePotential(U, q);
  Real err = 0;
  const Real expect = (one_sided ? 0 : (Real)-0.5);
  for (Long i = 0; i < q.Dim(); i++) err = MaxErr<Real>(err, fabs(U[i] / q[i] - expect) / (Real)0.5);
  return GlobalReduce(err, comm, CommOp::MAX);
}

// Green's identity on a closed surface for the field u of a point source X0 outside it:
// S[du/dn] - D[u] = u at interior targets (trg_dist > 0 inward along the normal), and on the surface
// (trg_dist = 0, targets at the nodes) u/2 with the principal value of D, or 0 as the limit from
// outside (one_sided). Returns max|error| / max|u| over the targets.
template <class Real, class KerSL, class KerDL, class KerGrad> Real test_GreensIdentity(const QuadElemList<Real>& qel, const Comm& comm, const Real tol, const Vector<Real>& X0, const Real trg_dist, const bool one_sided) {
  constexpr Integer KDIM0 = KerSL::SrcDim();
  const KerSL ker_sl;
  const KerDL ker_dl;
  const KerGrad ker_grad;
  Vector<Real> X, Xn;
  qel.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim() / COORD_DIM;
  const Vector<Real> Xtrg = X - trg_dist * Xn;

  const Vector<Real> Xn0{0, 0, 0};
  Vector<Real> F0(KDIM0), u_surf, du, u_trg;
  for (Integer k = 0; k < KDIM0; k++) F0[k] = (Real)0.7 - (Real)0.5 * k;
  ker_sl.Eval(u_surf, X, X0, Xn0, F0);
  ker_grad.Eval(du, X, X0, Xn0, F0);
  ker_sl.Eval(u_trg, Xtrg, X0, Xn0, F0);
  Vector<Real> du_dn(N * KDIM0);
  for (Long i = 0; i < N; i++) {
    for (Integer j = 0; j < KDIM0; j++) {
      Real s = 0;
      for (Integer k = 0; k < COORD_DIM; k++) s += du[(i * KDIM0 + j) * COORD_DIM + k] * Xn[i * COORD_DIM + k];
      du_dn[i * KDIM0 + j] = s;
    }
  }

  BoundaryIntegralOp<Real,KerSL> op_sl(ker_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> op_dl(ker_dl, false, comm);
  op_sl.AddElemList(qel);
  op_dl.AddElemList(qel);
  op_sl.SetAccuracy(tol);
  op_dl.SetAccuracy(tol);
  if (trg_dist > 0) {
    op_sl.SetTargetCoord(Xtrg);
    op_dl.SetTargetCoord(Xtrg);
  }
  Vector<Real> Us, Ud;
  op_sl.ComputePotential(Us, du_dn);
  op_dl.ComputePotential(Ud, u_surf);
  const Vector<Real> U = Us - Ud + (trg_dist > 0 ? (Real)0 : (one_sided ? (Real)1 : (Real)0.5)) * u_surf; // u at every target
  Real err = 0, ref = 0;
  for (Long i = 0; i < U.Dim(); i++) {
    err = MaxErr<Real>(err, fabs(U[i] - u_trg[i]));
    ref = MaxErr<Real>(ref, fabs(u_trg[i]));
  }
  return GlobalReduce(err, comm, CommOp::MAX) / GlobalReduce(ref, comm, CommOp::MAX);
}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    using Real = double;
    const Comm comm = Comm::World();
    const bool root = !comm.Rank();
    const auto passed = [root](const std::string& name) {
      if (root) std::cout << name << ": PASSED\n";
    };
    // Pass limits, relative to the requested tolerance: correct code reaches up to 3.3 on one element
    // (order 4, curved, at 1e-10) and 0.31 on the sphere
    constexpr Real ErrFactor = 10, ErrFactorSphere = 1;
    constexpr Real RefLimitNear = 1e-12, RefLimitSelf = 1e-11, RefLimitSphere = 1e-11; // limits for the adaptive reference itself
    const bool full = (argc > 1);
    const std::vector<Integer> orders = (full ? std::vector<Integer>{4, 8, 12, 16, 20} : std::vector<Integer>{8});
    const std::vector<Real> tols = (full ? std::vector<Real>{1e-5, 1e-10} : std::vector<Real>{1e-6});
    const Integer self_nodes = (full ? 4 : 2);
    Integer nfail = 0; // rows with an error over the limit, reported at the end
    if (root) std::cout << (full ? "Full run\n" : "Default run (pass any argument for the full run)\n");

    if (root) std::cout << "==================== 1. Building blocks ====================\n";
    test_AlpertQuadRule<Real>();
    passed("test_AlpertQuadRule");
    test_LogSingularQuad1D<Real>();
    passed("test_LogSingularQuad1D");
    test_GetGeom<Real>();
    passed("test_GetGeom");
    const Real far_err = test_GetFarFieldNodes<Real>();
    if (root) std::cout << "  largest far-field error beyond the cut-off distance: " << far_err << " x tol\n";
    passed("test_GetFarFieldNodes");
    for (const bool curved : {false, true}) {
      test_GetClosestNode<Real>(curved);
      test_GetClosestPoint<Real>(curved);
    }
    passed("test_GetClosestNode, test_GetClosestPoint");
    test_WriteRead<Real>(comm);
    passed("test_WriteRead");
    test_Copy<Real>();
    passed("test_Copy");
    test_VTU<Real>(comm);
    passed("test_VTU");
    test_EmptyList<Real>(comm);
    passed("test_EmptyList");
    test_SurfaceSingularDegree<Real>();
    passed("test_SurfaceSingularDegree");
#ifdef SCTL_QUAD_T
    test_ManyDigits();
    passed("test_ManyDigits");
#endif

    if (root) std::cout << "\n==================== 2. One element, every scheme ====================\n";
    {
      const std::array<Real,2> e = (full ? test_ReferenceFlat<Real>({4, 12, 20}, 0, true) : test_ReferenceFlat<Real>({8}, 4, false));
      if (root) std::cout << "  adaptive reference vs closed forms on the flat element: near " << e[0] << ", self " << e[1] << "\n";
      SCTL_ASSERT(e[0] < RefLimitNear && e[1] < RefLimitSelf);
      passed("test_ReferenceFlat");
    }
    const auto check = [root, &nfail](const std::string& label, const std::vector<std::vector<Real>>& err, const Real limit) {
      for (size_t s = 0; s < Schemes<Real>().size(); s++) {
        if (root) PrintRow(label + " " + Schemes<Real>()[s].name, err[s], limit);
        for (const Real e : err[s]) {
          if (!(e < limit)) {
            nfail++;
            break;
          }
        }
      }
    };
    for (const Real tol : tols) {
      if (root) PrintHeader("near, flat element, closed form", tol, ErrFactor * tol, orders);
      check("Laplace3D-FxU", test_NearFlat(Laplace3D_FxU(), FlatKernel::LaplaceSL, tol, orders), ErrFactor * tol);
      check("Laplace3D-DxU", test_NearFlat(Laplace3D_DxU(), FlatKernel::LaplaceDL, tol, orders), ErrFactor * tol);
      check("Laplace3D-FxdU.n", test_NearFlat(Laplace3D_FxdU(), FlatKernel::LaplaceAdjointDL, tol, orders), ErrFactor * tol);
      check("Stokes3D-FxU", test_NearFlat(Stokes3D_FxU(), FlatKernel::StokesSL, tol, orders), ErrFactor * tol);
    }
    for (const Real tol : tols) {
      if (root) PrintHeader("near, curved element, adaptive reference", tol, ErrFactor * tol, orders);
      check("Laplace3D-DxU", test_NearCurved(Laplace3D_DxU(), false, tol, orders), ErrFactor * tol);
      check("Laplace3D-FxdU.n", test_NearCurved(Laplace3D_FxdU(), true, tol, orders), ErrFactor * tol);
      check("Stokes3D-FxU", test_NearCurved(Stokes3D_FxU(), false, tol, orders), ErrFactor * tol);
      if (full) {
        check("Laplace3D-FxU", test_NearCurved(Laplace3D_FxU(), false, tol, orders), ErrFactor * tol);
        check("Stokes3D-DxU", test_NearCurved(Stokes3D_DxU(), false, tol, orders), ErrFactor * tol);
        check("Stokes3D-FxT.n", test_NearCurved(Stokes3D_FxT(), true, tol, orders), ErrFactor * tol);
      }
    }
    for (const Real tol : tols) {
      if (root) PrintHeader("self, flat element, closed form", tol, ErrFactor * tol, orders);
      check("Laplace3D-FxU", test_SelfFlat(Laplace3D_FxU(), FlatKernel::LaplaceSL, tol, orders), ErrFactor * tol);
      check("Stokes3D-FxU", test_SelfFlat(Stokes3D_FxU(), FlatKernel::StokesSL, tol, orders), ErrFactor * tol);
    }
    for (const Real tol : tols) {
      if (root) PrintHeader("self, curved element, adaptive reference", tol, ErrFactor * tol, orders);
      check("Laplace3D-DxU", test_SelfCurved(Laplace3D_DxU(), false, (Real)0.5, tol, orders, self_nodes), ErrFactor * tol);
      check("Laplace3D-FxdU.n", test_SelfCurved(Laplace3D_FxdU(), true, (Real)-0.5, tol, orders, self_nodes), ErrFactor * tol);
      check("Stokes3D-FxU", test_SelfCurved(Stokes3D_FxU(), false, (Real)0, tol, orders, self_nodes), ErrFactor * tol);
      if (full) {
        check("Laplace3D-FxU", test_SelfCurved(Laplace3D_FxU(), false, (Real)0, tol, orders, self_nodes), ErrFactor * tol);
        check("Stokes3D-DxU", test_SelfCurved(Stokes3D_DxU(), false, (Real)0.5, tol, orders, self_nodes), ErrFactor * tol);
        check("Stokes3D-FxT.n", test_SelfCurved(Stokes3D_FxT(), true, (Real)-0.5, tol, orders, self_nodes), ErrFactor * tol);
      }
    }

    if (root) std::cout << "\n==================== 3. BoundaryIntegralOp, resolved twisted sphere ====================\n";
    if (full) {
      const Real e = test_ReferenceSphere<Real>();
      if (root) std::cout << "  adaptive reference summed over a closed sphere vs the double-layer and Green's identities: " << e << "\n";
      SCTL_ASSERT(e < RefLimitSphere);
      passed("test_ReferenceSphere");
    }
    {
      const Integer bio_order = (full ? 12 : 8);
      const std::vector<Real> bio_tols = (full ? tols : std::vector<Real>{1e-4});
      const Long bio_nodes = (full ? 16 : 2);
      std::vector<std::pair<std::string, std::vector<std::vector<Real>>>> err{
        {"Laplace3D-FxdU.n", test_BIO(Laplace3D_FxdU(), true, (Real)-0.5, bio_order, bio_tols, bio_nodes, comm)}};
      if (full) {
        err.push_back({"Laplace3D-DxU", test_BIO(Laplace3D_DxU(), false, (Real)0.5, bio_order, bio_tols, bio_nodes, comm)});
        err.push_back({"Stokes3D-FxU", test_BIO(Stokes3D_FxU(), false, (Real)0, bio_order, bio_tols, bio_nodes, comm)});
      }
      for (size_t i = 0; i < bio_tols.size(); i++) {
        const Real limit = ErrFactorSphere * bio_tols[i];
        if (root) std::cout << "  order " << bio_order << ", tol " << bio_tols[i] << ", limit " << limit << "\n";
        for (size_t s = 0; s < Schemes<Real>().size(); s++) {
          bool over = false;
          if (root) std::cout << "    " << std::left << std::setw(14) << Schemes<Real>()[s].name << std::right;
          for (const auto& e : err) {
            if (root) std::cout << "  " << e.first << " " << std::scientific << std::setprecision(1) << e.second[i][s] << std::defaultfloat << std::setprecision(6);
            over = over || !(e.second[i][s] < limit);
          }
          if (root) std::cout << (over ? "   <-- exceeds the limit" : "") << "\n";
          nfail += (over ? 1 : 0);
        }
      }
    }

    if (root && nfail) std::cout << "\n" << nfail << " rows exceed the limit\n";
    SCTL_ASSERT(nfail == 0);
    passed("single-element and sphere tests");
    {
      const Real tol = (Real)1e-6;
      const std::vector<Real> e = test_NearNodeTargets<Real>(tol, comm);
      if (root) std::cout << "  targets 1e-4 element sizes off nodes, Laplace3D-FxdU.n, tol " << tol << ", limit " << ErrFactorSphere * tol << ":";
      for (size_t s = 0; s < e.size(); s++) {
        if (root) std::cout << "  " << Schemes<Real>()[s].name << " " << std::scientific << std::setprecision(1) << e[s] << std::defaultfloat << std::setprecision(6);
        SCTL_ASSERT(e[s] < ErrFactorSphere * tol);
      }
      if (root) std::cout << "\n";
      passed("test_NearNodeTargets");
    }

    if (full) {
      const Integer order = 12, ppf = 12;
      const Real R = 1, tol = 1e-9, rel_tol = 1e-9;
      const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2}; // outside the sphere
      if (root) std::cout << "\n==================== 4. Sphere, order " << order << ", " << ppf << " x " << ppf << " elements per face, tol " << tol << " ====================\n";
      for (const auto& sc : Schemes<Real>()) {
        QuadElemList<Real> qel = BuildTwistedSphere<Real>(order, ppf, R, 0, comm);
        qel.SetQuadScheme(sc.scheme);
        const std::array<std::pair<std::string,Real>,7> errs{{
          {"area", test_SurfaceArea(qel, R, comm)},
          {"DL Laplace", test_DLIdentity<Real, Laplace3D_DxU>(qel, comm, tol, sc.one_sided)},
          {"DL Stokes", test_DLIdentity<Real, Stokes3D_DxU>(qel, comm, tol, sc.one_sided)},
          {"Green Laplace", test_GreensIdentity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(qel, comm, tol, X0, 0, sc.one_sided)},
          {"Green Stokes", test_GreensIdentity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(qel, comm, tol, X0, 0, sc.one_sided)},
          {"Green Laplace, interior", test_GreensIdentity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(qel, comm, tol, X0, (Real)0.02, sc.one_sided)},
          {"Green Stokes, interior", test_GreensIdentity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(qel, comm, tol, X0, (Real)0.02, sc.one_sided)}}};
        if (root) {
          std::cout << "  " << std::setw(14) << sc.name << ":";
          for (const auto& e : errs) std::cout << " " << e.first << " " << e.second << ",";
          std::cout << "\n";
        }
        for (const auto& e : errs) SCTL_ASSERT(e.second < rel_tol);
      }
      passed("sphere study");
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
