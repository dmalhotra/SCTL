/**
 * QuadElemList test suite, ordered from the simplest building blocks to full-geometry
 * boundary-integral identities so a failure points at the lowest broken layer:
 *
 *   1. Unit tests (single element / kernel building blocks): parametric grid, closest-point
 *      projection, and the singular/near quadrature schemes (adaptive log-singular,
 *      rectangular-polar, hybrid) against closed-form or upsampled references.
 *   2. Sphere tests (whole closed surface, progressively harder): surface area from the far-field
 *      weights, the double-layer constant-density identity, and the Green's representation identity,
 *      run for every quadrature scheme on a regular cubed sphere.
 *
 * make bin/test-quad-elem && export OMP_NUM_THREADS=4 && ./bin/test-quad-elem
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

using namespace sctl;

// ============================================================================================
// 1. UNIT TESTS  (single-element / building-block level)
// ============================================================================================
template <class Real> Vector<Real> get_testsurf(const Integer order, const Integer nelem_perside) {
    // First define surface
    const auto fsurf = [](const Real x, const Real y) {
        return x*y;
    };

    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, nelem_perside); // Get x-y grid on [0,1]x[0,1]
    // Get z value on x-y grid for surface.
    for (int i=0; i<coord0.Dim()/3; i++) {
        coord0[i*3 + 2] = fsurf(coord0[i*3+0], coord0[i*3+1]);
    }
    return coord0;
}

template <class Real> void test_ParamGrid() {
    // Tensor grid generation directly on ParamGrid.
    const Long order = 4;
    const Long nelem_perside = 2;
    const Long N_per_side = order * nelem_perside; // 8 nodes per side
    const Long N_total = N_per_side * N_per_side;  // 64 tensor-grid points

    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, nelem_perside);
    SCTL_ASSERT(coord0.Dim() == N_total * 3);

    // Expected order-4 GL nodes mapped to [0,1], split into 2 panels.
    const Real x_param_exp[8] = {
        0.034715922101486804, // panel 0
        0.165004739103786020,
        0.334995260896213980,
        0.465284077898513196,
        0.534715922101486804, // panel 1
        0.665004739103786020,
        0.834995260896213980,
        0.965284077898513196
    };

    // Tensor product x_param_exp (x) x_param_exp, AoS (u slow, v fast), z = 0.
    const Real tol = 1e-12;
    for (Long xind = 0; xind < N_per_side; xind++) {
        for (Long yind = 0; yind < N_per_side; yind++) {
            const Long idx = (xind * N_per_side + yind) * 3;
            SCTL_ASSERT(fabs(coord0[idx + 0] - x_param_exp[xind]) < tol);
            SCTL_ASSERT(fabs(coord0[idx + 1] - x_param_exp[yind]) < tol);
            SCTL_ASSERT(fabs(coord0[idx + 2] - (Real)0) < tol);
        }
    }
}

template <class Real> void test_GetClosestNode_plane() {
    // Flat patch z = 0; lifted target must snap back to the surface node.
    const Long COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);

    Vector<Real> X, Xn;
    qel.GetNodeCoord(&X, &Xn, nullptr);
    const int trg_idx = 13; // arbitrary point on surface
    const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>) X.begin() + trg_idx * COORD_DIM, false);
    const Vector<Real> Xntrg(COORD_DIM, (Iterator<Real>) Xn.begin() + trg_idx * COORD_DIM, false);
    const Real utrg = coord0[trg_idx*COORD_DIM + 0];
    const Real vtrg = coord0[trg_idx*COORD_DIM + 1];

    Vector<Real> Xtrg_shifted = Xtrg;
    Xtrg_shifted[2] = 0.1;

    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestNode(ustar, vstar, 0, Xtrg_shifted);

    const Real tol = 1e-9;
    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist  - 0.1) < tol);

    // Now shift the target away from x and y as well.
    Xtrg_shifted[0] -= 0.0013;
    Xtrg_shifted[1] += 0.0005;
    const Real exp_dist = sqrt<Real>(0.1*0.1 + 0.0013*0.0013 + 0.0005*0.0005);

    const Real dist2 = qel.GetClosestNode(ustar, vstar, 0, Xtrg_shifted);

    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist2  - exp_dist) < tol);
}

template <class Real> void test_GetClosestNode_curved() {
    // Curved patch z = u*v; target lifted along the normal must snap to its node.
    const Integer COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = get_testsurf<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);

    Vector<Real> X, Xn;
    qel.GetNodeCoord(&X, &Xn, nullptr);
    const int trg_idx = 13; // arbitrary point on surface
    const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>) X.begin() + trg_idx * COORD_DIM, false);
    const Vector<Real> Xntrg(COORD_DIM, (Iterator<Real>) Xn.begin() + trg_idx * COORD_DIM, false);
    const Real utrg = coord0[trg_idx*COORD_DIM + 0];
    const Real vtrg = coord0[trg_idx*COORD_DIM + 1];
    const Real d = 0.001;
    Vector<Real> Xtrg_shifted = Xtrg + d * Xntrg;

    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestNode(ustar, vstar, 0, Xtrg_shifted);

    const Real tol = 1e-8;
    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist  - d ) < tol);

    // Now shift the target away from x and y as well.
    Xtrg_shifted[0] -= 0.0013;
    Xtrg_shifted[1] += 0.0005;
    const Real exp_dist = sqrt<Real>((d*Xntrg[0]-0.0013)*(d*Xntrg[0]-0.0013) + (d*Xntrg[1]+0.0005)*(d*Xntrg[1]+0.0005) + (d*Xntrg[2])*(d*Xntrg[2]));

    const Real dist2 = qel.GetClosestNode(ustar, vstar, 0, Xtrg_shifted);

    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist2  - exp_dist) < tol);
}

template <class Real> void test_GetClosestPoint_plane() {
    // Flat patch z = 0: GetClosestPoint must recover the exact projection at an off-node (u,v).
    const Integer COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);

    // Off-node surface point and its normal (= +z for the plane).
    const Real u0 = 0.37, v0 = 0.62;
    Vector<Real> up{u0}, vp{v0}, Xsurf, Nsurf;
    qel.GetGeom(&Xsurf, &Nsurf, nullptr, nullptr, nullptr, up, vp, 0);

    // Target lifted a distance d along the normal.
    const Real d = 0.1;
    Vector<Real> Xtrg(COORD_DIM);
    for (Integer k = 0; k < COORD_DIM; k++) Xtrg[k] = Xsurf[k] + d * Nsurf[k];

    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestPoint(ustar, vstar, 0, Xtrg);

    const Real tol = 1e-9;
    SCTL_ASSERT(fabs(ustar - u0) < tol);
    SCTL_ASSERT(fabs(vstar - v0) < tol);
    SCTL_ASSERT(fabs(dist  - d) < tol);

    // Tangential shift: projection follows it, dist stays = d.
    Xtrg[0] -= 0.0013;
    Xtrg[1] += 0.0005;
    const Real dist2 = qel.GetClosestPoint(ustar, vstar, 0, Xtrg);
    SCTL_ASSERT(fabs(ustar - (u0 - (Real)0.0013)) < tol);
    SCTL_ASSERT(fabs(vstar - (v0 + (Real)0.0005)) < tol);
    SCTL_ASSERT(fabs(dist2 - d) < tol);
}

template <class Real> void test_GetClosestPoint_curved() {
    // Curved patch z = u*v: GetClosestPoint must find the foot of the perpendicular at an off-node (u,v).
    const Integer COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = get_testsurf<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);

    // Off-node surface point + normal; small offset so (u0,v0) is the unique foot.
    const Real u0 = 0.37, v0 = 0.62;
    Vector<Real> up{u0}, vp{v0}, Xsurf, Nsurf;
    qel.GetGeom(&Xsurf, &Nsurf, nullptr, nullptr, nullptr, up, vp, 0);
    const Real d = 0.01;
    Vector<Real> Xtrg(COORD_DIM);
    for (Integer k = 0; k < COORD_DIM; k++) Xtrg[k] = Xsurf[k] + d * Nsurf[k];

    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestPoint(ustar, vstar, 0, Xtrg);

    const Real tol = 1e-7;
    SCTL_ASSERT(fabs(ustar - u0) < tol);
    SCTL_ASSERT(fabs(vstar - v0) < tol);
    SCTL_ASSERT(fabs(dist  - d) < tol);

    // Generic target: residual (closest point - target) must be orthogonal to both tangents.
    Vector<Real> Xt2(COORD_DIM);
    Xt2[0] = Xsurf[0] + (Real)0.05;
    Xt2[1] = Xsurf[1] - (Real)0.03;
    Xt2[2] = Xsurf[2] + (Real)0.08;
    qel.GetClosestPoint(ustar, vstar, 0, Xt2);
    SCTL_ASSERT(ustar > tol && ustar < 1 - tol && vstar > tol && vstar < 1 - tol); // interior min

    Vector<Real> u1{ustar}, v1{vstar}, Xc, dXu, dXv;
    qel.GetGeom(&Xc, nullptr, nullptr, &dXu, &dXv, u1, v1, 0);
    Real ru = 0, rv = 0, tu = 0, tv = 0, rr = 0;
    for (Integer k = 0; k < COORD_DIM; k++) {
        const Real r = Xc[k] - Xt2[k];
        ru += r * dXu[k]; rv += r * dXv[k];
        tu += dXu[k]*dXu[k]; tv += dXv[k]*dXv[k]; rr += r*r;
    }
    const Real rn = sqrt<Real>(rr);
    // std::cout << "tu = " << sqrt<Real>(tu) << ", tv = " << sqrt<Real>(tv) << ", rn = " << rn << ", lhs = " << fabs(ru) <<", rhs = " << (Real)1e-8 * sqrt<Real>(tu) * rn << std::endl;
    SCTL_ASSERT(fabs(ru) < tol * sqrt<Real>(tu) * rn);
    SCTL_ASSERT(fabs(rv) < tol * sqrt<Real>(tv) * rn);
}


// Reference near-singular evaluation: integrate the BIO on element `elem_idx`
// against target `Xt` via uniform nsub x nsub refinement with order-`order` GL on
// each panel, using the same Lagrange-interpolant density as NearInterac. As nsub
// grows this converges to the exact integral NearInterac computes to tolerance.
// Returns the target potential (Kernel::TrgDim() reals).
template <class Real, class Kernel> Vector<Real> direct_upsampled_potential(
    const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& sigma,
    const Vector<Real>& Xt, const Kernel& ker, const Long nsub) {

    const Integer order = qel.Order();
    const Integer KDIM0 = Kernel::SrcDim();
    const Integer KDIM1 = Kernel::TrgDim();
    const Long nq = (Long)order * order;
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    const Vector<Real>& wts = LegQuadRule<Real>::wts(order);

    Vector<Real> u(KDIM1);
    u.SetZero();

    Vector<Real> u_param(order), v_param(order);
    for (Long pi = 0; pi < nsub; pi++) {
        for (Long pj = 0; pj < nsub; pj++) {
            for (Integer a = 0; a < order; a++) u_param[a] = (nds[a] + pi) / (Real)nsub;
            for (Integer b = 0; b < order; b++) v_param[b] = (nds[b] + pj) / (Real)nsub;

            // Geometry on this panel's order x order GL grid.
            Vector<Real> X, Xn, Xa;
            qel.GetGeom(&X, &Xn, &Xa, nullptr, nullptr, u_param, v_param, elem_idx);

            // Lagrange weights from patch nodes to panel quad nodes.
            Vector<Real> Lu(order * order), Lv(order * order);
            LagrangeInterp<Real>::Interpolate(Lu, nds, u_param);
            LagrangeInterp<Real>::Interpolate(Lv, nds, v_param);

            // Interpolate the nodal density onto the panel quad nodes.
            Vector<Real> sigma_q(nq * KDIM0);
            sigma_q.SetZero();
            for (Integer a = 0; a < order; a++) {
                for (Integer b = 0; b < order; b++) {
                    const Long q = a * order + b;
                    for (Integer i = 0; i < order; i++) {
                        for (Integer j = 0; j < order; j++) {
                            const Real L = Lu[i * order + a] * Lv[j * order + b];
                            for (Integer k0 = 0; k0 < KDIM0; k0++) {
                                sigma_q[q * KDIM0 + k0] += sigma[(i * order + j) * KDIM0 + k0] * L;
                            }
                        }
                    }
                }
            }

            // Kernel matrix from this panel's sources to the target (scaled, matches NearInterac).
            Matrix<Real> Mker; // (nq*KDIM0 x KDIM1)
            ker.template KernelMatrix<Real, false>(Mker, Xt, X, Xn);

            for (Integer a = 0; a < order; a++) {
                for (Integer b = 0; b < order; b++) {
                    const Long q = a * order + b;
                    // Surface quad weight with the 1/nsub^2 panel Jacobian.
                    const Real wq = Xa[q] * wts[a] * wts[b] / ((Real)nsub * (Real)nsub);
                    for (Integer k0 = 0; k0 < KDIM0; k0++) {
                        for (Integer k1 = 0; k1 < KDIM1; k1++) {
                            u[k1] += Mker[q * KDIM0 + k0][k1] * sigma_q[q * KDIM0 + k0] * wq;
                        }
                    }
                }
            }
        }
    }
    return u;
}

// Forward declaration of the friend shim (defined below) that exposes QuadElemList's private
// static quadrature helpers (log-singular 1D rule, rectangular-polar 1D COV) to the tests; the
// shim's full definition appears later in namespace sctl.
namespace sctl { template <class Real> struct QuadElemTestAccess; }

template <class Real, class Kernel> void test_NearInterac(const Kernel& ker, const bool curved, const char* label, const typename QuadElemList<Real>::QuadScheme scheme = QuadElemList<Real>::QuadScheme::Adaptive, const Real rel_tol = 1e-6, const Integer cov_order = 0, const Integer max_depth = 30) {
    const Integer COORD_DIM = 3;
    const Integer order = 24;
    const Integer KDIM0 = Kernel::SrcDim();
    const Integer KDIM1 = Kernel::TrgDim();
    const Long nnode = (Long)order * order;
    const Long elem_idx = 0;

    // Single element: flat plane z = 0 or curved testsurf z = u*v.
    Vector<Real> coord0 = curved ? get_testsurf<Real>(order, 1)
                                 : QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);
    const Integer q = 10;
    qel.SetQuadScheme(scheme, q, cov_order, max_depth);

    // Near-singular target: offset d along the normal at an interior point.
    const Real u0 = 0.4, v0 = 0.6, d = 0.01;
    Vector<Real> up{u0}, vp{v0}, Xsurf, Nsurf;
    qel.GetGeom(&Xsurf, &Nsurf, nullptr, nullptr, nullptr, up, vp, elem_idx);
    Vector<Real> Xt(COORD_DIM);
    for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = Xsurf[k] + d * Nsurf[k];

    // Smooth nodal density (AoS); both schemes integrate the same interpolant.
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    Vector<Real> sigma(nnode * KDIM0);
    for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
                sigma[(i * order + j) * KDIM0 + k0] = cos<Real>(nds[i] + 2 * nds[j] + (Real)0.5 * k0);
            }
        }
    }

    // Near-interaction matrix and potential M^T * sigma.
    Matrix<Real> M;
    Vector<Real> normal_trg; // empty: no target-normal contraction
    const Real tol = 1e-08;
    QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, normal_trg, ker, tol, elem_idx, &qel);
    SCTL_ASSERT(M.Dim(0) == nnode * KDIM0 && M.Dim(1) == KDIM1); // single target

    Vector<Real> u_near(KDIM1);
    u_near.SetZero();
    for (Long r = 0; r < nnode * KDIM0; r++) {
        for (Integer k1 = 0; k1 < KDIM1; k1++) u_near[k1] += sigma[r] * M[r][k1];
    }

    // Reference potential: uniform upsampled direct quadrature (nsub=100), accurate at the moderate
    // near distance d=0.01 used here. (Deep near needs a RectPolar gold instead.)
    const Vector<Real> u_ref = direct_upsampled_potential<Real, Kernel>(qel, elem_idx, sigma, Xt, ker, 100);

    // Relative error in the target potential.
    Real err2 = 0, ref2 = 0;
    for (Integer k1 = 0; k1 < KDIM1; k1++) {
        const Real e = u_near[k1] - u_ref[k1];
        err2 += e * e;
        ref2 += u_ref[k1] * u_ref[k1];
    }
    const Real rel_err = sqrt<Real>(err2) / sqrt<Real>(ref2);

    std::cout << "  test_NearInterac (" << label << "): rel_err = " << rel_err << "\n";
    SCTL_ASSERT(rel_err < rel_tol);
}

// Singular self-interaction vs. closed-form references on the flat unit square
// (z = 0), where r_3 = 0 and n = (0,0,1) give analytic answers for constant density:
//   Laplace3D-FxU, sigma=1       :  u = (1/4pi) I0
//   Stokes3D-FxU,  q=(0,0,1)     :  u = (0,0,(1/8pi) I0)
//   Stokes3D-DxU,  q arbitrary   :  u = 0
// I0 is the in-plane Newtonian potential of the unit square (1/r antiderivative
// F(X,Y) = X ln(Y+R) + Y ln(X+R)). Applied as u = sigma^T M.
template <class Real, class Kernel> void test_SelfInterac(const Kernel& ker, const typename QuadElemList<Real>::QuadScheme scheme = QuadElemList<Real>::QuadScheme::Adaptive, const Real rel_tol = 1e-6, const Integer q = 10, const Real tol = 1e-10, const Integer cov_order = 0, const Integer max_depth = 30) {
    const Integer order = 12;
    const Long nnode = (Long)order * order;
    const Integer KDIM0 = Kernel::SrcDim();
    const Integer KDIM1 = Kernel::TrgDim();
    SCTL_ASSERT(KDIM1 <= 3);

    // Flat unit square z = 0.
    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);
    qel.SetQuadScheme(scheme, q, cov_order, max_depth);

    // Self-interaction matrix (no target-normal contraction).
    Vector<Matrix<Real>> M_lst(1);
    QuadElemList<Real>::template SelfInterac<Kernel>(M_lst, ker, tol, /*trg_dot_prod=*/false, &qel);

    // Shape + finiteness.
    SCTL_ASSERT(M_lst.Dim() == 1);
    const Matrix<Real>& M = M_lst[0];
    SCTL_ASSERT(M.Dim(0) == nnode * KDIM0 && M.Dim(1) == nnode * KDIM1);
    for (Long r = 0; r < M.Dim(0); r++) {
        for (Long c = 0; c < M.Dim(1); c++) SCTL_ASSERT(std::isfinite(M[r][c]));
    }

    // I0: corner sum of the 1/r antiderivative.
    auto I0 = [](Real x0, Real y0) {
        auto F = [](Real X, Real Y) { const Real R = sqrt<Real>(X*X + Y*Y); return X*log<Real>(Y + R) + Y*log<Real>(X + R); };
        return F(1 - x0, 1 - y0) - F(1 - x0, -y0) - F(-x0, 1 - y0) + F(-x0, -y0);
    };

    // Per-kernel constant density q and the closed-form reference u_exact(x0,y0).
    const std::string& kname = Kernel::Name();
    Vector<Real> qden(KDIM0); qden.SetZero();
    auto u_exact = [&](Real x0, Real y0, Real* ue) {
        for (Integer k = 0; k < KDIM1; k++) ue[k] = 0;
        if (kname == "Laplace3D-FxU")      ue[0] = I0(x0, y0) / (4 * const_pi<Real>());
        else if (kname == "Stokes3D-FxU")  ue[2] = I0(x0, y0) / (8 * const_pi<Real>());
        else if (kname == "Stokes3D-DxU")  { /* u == 0 */ }
        else SCTL_ASSERT_MSG(false, "test_SelfInterac: unsupported kernel");
    };
    if (kname == "Laplace3D-FxU")      qden[0] = 1;            // sigma = 1
    else if (kname == "Stokes3D-FxU")  qden[2] = 1;            // q = (0,0,1) (normal)
    else if (kname == "Stokes3D-DxU")  qden[0] = 1;            // q arbitrary
    else SCTL_ASSERT_MSG(false, "test_SelfInterac: unsupported kernel");

    // Apply to the constant density and compare at every node (relative error
    // for the single layers, absolute for the zero double layer).
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    Real max_abs = 0, ref_scale = 0;
    for (Integer ti = 0; ti < order; ti++) {
        for (Integer tj = 0; tj < order; tj++) {
            const Long t = ti * order + tj;
            Real u[3] = {0, 0, 0};
            for (Long p = 0; p < nnode; p++)
                for (Integer k0 = 0; k0 < KDIM0; k0++)
                    for (Integer k1 = 0; k1 < KDIM1; k1++)
                        u[k1] += qden[k0] * M[p*KDIM0 + k0][t*KDIM1 + k1];
            Real ue[3];
            u_exact(nds[ti], nds[tj], ue);
            for (Integer k1 = 0; k1 < KDIM1; k1++) {
                max_abs   = std::max<Real>(max_abs, fabs(u[k1] - ue[k1]));
                ref_scale = std::max<Real>(ref_scale, fabs(ue[k1]));
            }
        }
    }
    const Real err = (ref_scale > 0 ? max_abs / ref_scale : max_abs);
    std::cout << "  test_SelfInterac (" << kname << "): err = " << err << "\n";
    SCTL_ASSERT(err < rel_tol);
}


// Friend shim forwarding to QuadElemList's private static helpers (must be in namespace sctl).
namespace sctl {
template <class Real> struct QuadElemTestAccess {
    // The library now emits the log-singular 1D Alpert rule as offsets from v0
    // (LogSingularQuad1DCentered); reconstruct the absolute nodes param = v0 + delta so the
    // tests below keep the original absolute-node semantics.
    static void LogSingularQuad1D(Vector<Real>& param, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder) {
        Vector<Real> delta;
        QuadElemList<Real>::LogSingularQuad1DCentered(delta, w, v0, Lvl, QuadOrder);
        param.ReInit(delta.Dim());
        for (Long i = 0; i < delta.Dim(); i++) param[i] = v0 + delta[i];
    }
    static void RectPolarNodes1D(Vector<Real>& nodes, Vector<Real>& wts, const Real alpha, const Integer q, const Vector<Real>& gl_nds, const Vector<Real>& gl_wts) {
        QuadElemList<Real>::RectPolarNodes1D(nodes, wts, alpha, q, gl_nds, gl_wts);
    }
};
}

// Sanity-check the rectangular-polar 1D COV: nodes stay in [0,1], weights sum to 1,
// and the COV weight vanishes at the singularity u* = (alpha+1)/2.
template <class Real> void test_RectPolarNodes1D() {
    const Integer order = 256, q = 10;
    // const Vector<Real>& gl_nds = QuadElemList<Real>::ParamNodes(order);
    // const Vector<Real>& gl_wts = sctl::LegQuadRule<Real>::wts(order);
    Vector<Real> gl_nds, gl_wts;
    sctl::LegQuadRule<Real>::ComputeNdsWts(&gl_nds, &gl_wts, order);
    for (const Real ustar : {(Real)0.2, (Real)0.5, (Real)0.77}) {
        const Real alpha = 2*ustar - 1;
        Vector<Real> nds, wts;
        sctl::QuadElemTestAccess<Real>::RectPolarNodes1D(nds, wts, alpha, q, gl_nds, gl_wts);
        Real wsum = 0;
        for (Long i = 0; i < nds.Dim(); i++) {
            SCTL_ASSERT(nds[i] > -1e-12 && nds[i] < 1 + 1e-12);
            SCTL_ASSERT(wts[i] > -1e-12); // monotone COV => nonnegative weights
            wsum += wts[i];
        }
        // sum(w) -> 1 to GL accuracy on eta' (structural check, not machine eps).
        std::cout << "  test_RectPolarNodes1D (u*=" << (double)ustar << "): sum(w)=" << (double)wsum
                  << "  err=" << (double)fabs(wsum - 1) << "\n";
        SCTL_ASSERT(fabs(wsum - 1) < 1e-8);

        // Node nearest u* should have tiny weight relative to the largest.
        Long isng = 0; Real dmin = -1, wmax = 0;
        for (Long i = 0; i < nds.Dim(); i++) {
            const Real d = fabs(nds[i] - ustar);
            if (dmin < 0 || d < dmin) { dmin = d; isng = i; }
            wmax = std::max<Real>(wmax, wts[i]);
        }
        std::cout << "      node nearest u*: out=" << (double)nds[isng]
                  << " (in=" << (double)gl_nds[isng] << ")  w=" << (double)wts[isng]
                  << "  w/wmax=" << (double)(wts[isng]/wmax) << "\n";
    }
}

// Verify the Alpert 1D log-singular rule (LogSingularQuad1D) for I[f] = int_0^1 f
// with a log singularity at interior v0, against closed-form integrals of:
//   (a) log|v-v0|  (b) v log|v-v0|  (c) (1+v^2) log|v-v0|+cos(3v)  (d) cos(3v)
// Closed forms from int_0^1 v^k log|v-a| dv.
template <class Real> void test_LogSingularQuad1D() {
    const Real v0 = (Real)0.6;
    const Integer Lvl = 5, QuadOrder = 24; // grading levels per side + GL order on smooth panels

    Vector<Real> param, w;
    QuadElemTestAccess<Real>::LogSingularQuad1D(param, w, v0, Lvl, QuadOrder);

    // Structural sanity: sizes match, nodes in (0,1), weights sum to 1.
    SCTL_ASSERT(param.Dim() == w.Dim());
    SCTL_ASSERT(param.Dim() > 0);
    Real wsum = 0;
    for (Long i = 0; i < param.Dim(); i++) {
        SCTL_ASSERT(param[i] > (Real)0 && param[i] < (Real)1);
        wsum += w[i];
    }
    SCTL_ASSERT(fabs(wsum - (Real)1) < (Real)1e-12);

    auto quad = [&](auto f) {
        Real I = 0;
        for (Long i = 0; i < param.Dim(); i++) I += w[i] * f(param[i]);
        return I;
    };

    const Real a = v0;
    const Real la = log<Real>(a), lb = log<Real>(1 - a);

    // (a) f = log|v - v0|
    {
        const Real I = quad([&](Real v) { return log<Real>(fabs(v - v0)); });
        const Real I_exact = a * la + (1 - a) * lb - 1;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=log|v-v0|        I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-10);
    }

    // (b) f = v * log|v - v0|
    {
        const Real I = quad([&](Real v) { return v * log<Real>(fabs(v - v0)); });
        const Real I_exact = ((1 - a * a) / 2) * lb + (a * a / 2) * la - (Real)0.25 - a / 2;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=v*log|v-v0|      I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-10);
    }

    // (c) f = (1 + v^2) * log|v - v0| + cos(3 v); int v^2 log via F(x) = ((x^3-a^3)/3)log|x-a| - x^3/9 - a x^2/6 - a^2 x/3.
    {
        const Real I = quad([&](Real v) {
            return (1 + v * v) * log<Real>(fabs(v - v0)) + cos<Real>(3 * v);
        });
        const Real I0 = a * la + (1 - a) * lb - 1;                                   // \int v^0 log
        const Real F1 = ((1 - a * a * a) / 3) * lb - (Real)1 / 9 - a / 6 - a * a / 3; // F(1)
        const Real F0 = (-a * a * a / 3) * la;                                        // F(0)
        const Real I2 = F1 - F0;                                                      // \int v^2 log
        const Real Icos = sin<Real>((Real)3) / 3;                                     // \int_0^1 cos(3v)
        const Real I_exact = I0 + I2 + Icos;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=(1+v^2)log+cos   I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-9);
    }

    // (d) purely smooth integrand; rule must still be high order.
    {
        const Real I = quad([&](Real v) { return cos<Real>(3 * v); });
        const Real I_exact = sin<Real>((Real)3) / 3;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=cos(3v)          I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-10);
    }
}

// Check the interpolation floor of the self-interaction quadrature: IntegrateBlock
// samples the order-`order` tensor-product Lagrange interpolant (not the true field)
// at the Alpert nodes. For a non-polynomial field on the curved testsurf (z = u*v)
// this is inexact; confirm the error sits at the expected spectral level.
template <class Real> void test_QuadNodeInterp() {
    const Integer order = 12;
    const Long elem_idx = 0;

    // Non-flat patch z = u*v; its order-12 interpolant is what the quadrature integrates.
    Vector<Real> coord0 = get_testsurf<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);

    // Non-polynomial scalar field in physical space (non-polynomial in (u,v)).
    auto g = [](const Real* X) {
        return exp<Real>((Real)0.5 * X[0]) * cos<Real>(X[1]) + sin<Real>(X[2]);
    };

    // Field values at the patch nodes (interpolation data).
    Vector<Real> Xpatch;
    qel.GetGeom(&Xpatch, nullptr, nullptr, nullptr, nullptr, nds, nds, elem_idx);
    Vector<Real> f_patch(order * order);
    for (Long p = 0; p < order * order; p++) f_patch[p] = g(&Xpatch[p * 3]);

    // Alpert nodes in u and v forming the tensor-product target grid (node (a,b) at a*Nv + b).
    Vector<Real> u_param, v_param, wu, wv;
    QuadElemTestAccess<Real>::LogSingularQuad1D(u_param, wu, (Real)0.3, /*Lvl*/ 4, /*QuadOrder*/ order);
    QuadElemTestAccess<Real>::LogSingularQuad1D(v_param, wv, (Real)0.6, /*Lvl*/ 4, /*QuadOrder*/ order);
    const Long Nu = u_param.Dim(), Nv = v_param.Dim();

    // Lagrange weights from patch nodes to the Alpert nodes (as in IntegrateBlock).
    Vector<Real> Mu(order * Nu), Mv(order * Nv);
    LagrangeInterp<Real>::Interpolate(Mu, nds, u_param);
    LagrangeInterp<Real>::Interpolate(Mv, nds, v_param);

    // Exact field at the Alpert nodes, via the surface geometry there.
    Vector<Real> Xquad;
    qel.GetGeom(&Xquad, nullptr, nullptr, nullptr, nullptr, u_param, v_param, elem_idx);

    // Compare the tensor-product Lagrange interpolant against the exact field.
    Real max_err = 0, max_f = 0;
    for (Long a = 0; a < Nu; a++) {
        for (Long b = 0; b < Nv; b++) {
            Real f_interp = 0;
            for (Integer i = 0; i < order; i++) {
                for (Integer j = 0; j < order; j++) {
                    f_interp += f_patch[i * order + j] * Mu[i * Nu + a] * Mv[j * Nv + b];
                }
            }
            const Real f_exact = g(&Xquad[(a * Nv + b) * 3]);
            max_err = std::max<Real>(max_err, fabs(f_interp - f_exact));
            max_f   = std::max<Real>(max_f, fabs(f_exact));
        }
    }
    const Real rel_err = max_err / max_f;
    std::cout << "  test_QuadNodeInterp: order=" << order << " Nu=" << Nu << " Nv=" << Nv
              << " max_abs_err=" << max_err << " rel_err=" << rel_err << "\n";
    const Real rel_tol = 1e-6;
    SCTL_ASSERT(rel_err < rel_tol);
}

// Visualize the Hybrid scheme (adaptive near + rectangular-polar self) on a single
// flat order-12 panel: dump the near quadtree leaf GL grid and the RP self grid to VTK
// for inspection in ParaView. Reuses QuadElemList's WriteNearInteracVTK (adaptive) and
// WriteSelfInteracRPVTK (RectPolar) writers.
template <class Real> void hybrid_scheme_vis() {
    using QS = typename QuadElemList<Real>::QuadScheme;
    const Integer order = 12;
    const Long evis = 0;

    // Flat panel z = 0, order 12, single element.
    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);
    // Hybrid = adaptive near + RectPolar self; max_depth=12 caps the adaptive near
    // quadtree; q/cov_order feed the RP self grid rendering.
    qel.SetQuadScheme(QS::Hybrid, /*q=*/6, /*cov_order=*/200, /*max_depth=*/12);

    // --- Near (adaptive): target at (u*,v*)=(0.314,0.157), 0.02 off surface along normal.
    Vector<Real> up{(Real)0.314}, vp{(Real)0.157}, Xc, Xn;
    qel.GetGeom(&Xc, &Xn, nullptr, nullptr, nullptr, up, vp, evis);
    Vector<Real> Xtrg(3);
    for (Integer k = 0; k < 3; k++) Xtrg[k] = Xc[k] + (Real)0.02 * Xn[k];
    qel.WriteNearInteracVTK("hybrid-near-elem0", evis, Xtrg, /*tol=*/1e-9, Comm::Self());

    // --- Self (RectPolar): singular point at the 8th x-node and 10th y-node (1-based).
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    const Real u0 = nds[7], v0 = nds[9];
    qel.WriteSelfInteracRPVTK("hybrid-self-elem0", evis, u0, v0, /*Nbeta=*/200, Comm::Self());

    // --- Original order-12 tensor GL grid of the panel itself (quad mesh).
    qel.WriteVTK("hybrid-panel-grid", Vector<Real>(), Comm::Self());

    std::cout << "  hybrid_scheme_vis: wrote hybrid-near-elem0-*, hybrid-self-elem0-*, hybrid-panel-grid-* VTK files\n";
}

// ============================================================================================
// 2. SPHERE TESTS  (whole closed surface; progressively harder than the unit tests above)
// ============================================================================================

namespace {

// --- Distributed-memory helpers -------------------------------------------------
// Under MPI each rank owns only a slice of the geometry (see BuildTwistedSphere), so scalar
// norms/areas accumulated over local nodes must be reduced across ranks before comparison, and
// result prints are emitted on rank 0 only.
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
// twisted about z: at height z, {x,y} rotated by theta_twist*z. Regular sphere: theta_twist = 0.
// Every rank builds the full node array X, then the QuadElemList constructor keeps only this rank's
// contiguous element slice (replicate-then-slice partitioning).
template <class Real>
QuadElemList<Real> BuildTwistedSphere(Long ElemOrder, Long PatchPerFace, Real Radius, Real theta_twist = 0., const Comm& comm = Comm::Self()) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(ElemOrder);
  for (Integer face = 0; face < 6; face++) {
    for (Long iu = 0; iu < PatchPerFace; iu++) {
      for (Long iv = 0; iv < PatchPerFace; iv++) {
        for (Long i = 0; i < ElemOrder; i++) {
          const Real a = 2 * ((iu + nds[i]) / (Real)PatchPerFace) - 1;
          for (Long j = 0; j < ElemOrder; j++) {
            const Real b = 2 * ((iv + nds[j]) / (Real)PatchPerFace) - 1;
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

// Far-field quadrature weights must sum to the analytic sphere area 4 pi R^2.
// Returns the relative area error |A - 4 pi R^2| / (4 pi R^2).
double test_SurfaceArea(const QuadElemList<double>& elem_lst, double Radius, const Comm& comm) {
  Vector<double> wts, Xtemp, Xntemp, dist_far;
  Vector<Long> elem_wise_temp;
  elem_lst.GetFarFieldNodes(Xtemp, Xntemp, wts, dist_far, elem_wise_temp, 1);
  double Area = 0.;
  for (int i = 0; i < wts.Dim(); i++) Area += wts[i];
  Area = GlobalReduce(Area, comm, CommOp::SUM); // weights are distributed across ranks
  const double Area_exact = 4. * const_pi<double>() * Radius * Radius;
  const double rel_err = std::fabs(Area - Area_exact) / Area_exact;
  if (!comm.Rank()) std::cout << "  surface area: Jacobian=" << Area << ", exact=" << Area_exact
                              << ", rel err=" << rel_err << std::endl;
  return rel_err;
}

} // end anonymous namespace (TU-local helpers: GlobalReduce, FacePoint, BuildTwistedSphere, test_SurfaceArea)


// Double-layer constant-density identity on a closed surface (Laplace or Stokes):
// D[q] = c*q for constant q, with c = -1/2 for the outward-normal convention used here.
// Sign convention: this kernel (r = x_trg-x_src, source normal) gives c = -1/2 for an outward
// normal, +1/2 for inward. Returns the max relative error over the node components.
template <class Real, class KerDL> Real test_DLIdentity(const QuadElemList<Real>& elem_lst, const Comm& comm, const Real quad_tol = 1e-8) {
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
  if (!comm.Rank()) std::cout << std::setprecision(8) << "  DL constant-density identity: max relative error = " << cx_relerr_avg << std::endl;
  return cx_relerr_avg;
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
// Returns the relative error max|Uerr|/max|Uref| (reduced across ranks) so callers can tabulate it.
template <class Real, class KerSL, class KerDL, class KerGrad> Real test_greens_identity(const QuadElemList<Real>& elem_lst, const Comm& comm,
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
  Real rel_err = 0;
  { // Print error
    StaticArray<Real,2> max_err{0,0};
    StaticArray<Real,2> max_val{0,0};
    for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
    for (auto x : Uref) max_val[0] = std::max<Real>(max_val[0], fabs(x));
    comm.Allreduce(max_err+0, max_err+1, 1, CommOp::MAX);
    comm.Allreduce(max_val+0, max_val+1, 1, CommOp::MAX);
    rel_err = max_err[1]/max_val[1];
    if (!pid) std::cout<<"  Green's identity error = "<<rel_err<<'\n';
  }
  sctl::Profile::print(&comm, {"t_avg", "f/s_avg"});
  sctl::Profile::reset();
  sctl::Profile::Enable(false);
  return rel_err;
}


int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const bool root = !comm.Rank();

    // ======================================================================================
    // 1. Unit tests -- single element / kernel building blocks (each uses Comm::Self()).
    // ======================================================================================
    if (root) std::cout << "==================== Unit tests ====================\n";
    test_ParamGrid<Real>();
    std::cout << "test_ParamGrid: PASSED\n";
    test_GetClosestNode_plane<Real>();
    std::cout << "test_GetClosestNode_plane: PASSED\n";
    test_GetClosestNode_curved<Real>();
    std::cout << "test_GetClosestNode_curved: PASSED\n";
    test_GetClosestPoint_plane<Real>();
    std::cout << "test_GetClosestPoint_plane: PASSED\n";
    test_GetClosestPoint_curved<Real>();
    std::cout << "test_GetClosestPoint_curved: PASSED\n";

    std::cout << "--- Scheme 1: adaptive and/or log singular special quadrature ---\n";
    test_LogSingularQuad1D<Real>();
    std::cout << "test_LogSingularQuad1D: PASSED\n";
    test_QuadNodeInterp<Real>();
    std::cout << "test_QuadNodeInterp: PASSED\n";
    // NearInterac: adaptive scheme vs. upsampled direct quadrature.
    const Stokes3D_FxU ker_FxU;
    const Stokes3D_DxU ker_DxU;
    const Laplace3D_FxU ker_lapFxU;
    test_NearInterac<Real>(ker_FxU, false, "Stokes3D_FxU / plane");
    std::cout << "test_NearInterac (Stokes3D_FxU / plane): PASSED\n";
    test_NearInterac<Real>(ker_FxU, true,  "Stokes3D_FxU / testsurf");
    std::cout << "test_NearInterac (Stokes3D_FxU / testsurf): PASSED\n";
    test_NearInterac<Real>(ker_DxU, false, "Stokes3D_DxU / plane");
    std::cout << "test_NearInterac (Stokes3D_DxU / plane): PASSED\n";
    test_NearInterac<Real>(ker_DxU, true,  "Stokes3D_DxU / testsurf");
    std::cout << "test_NearInterac (Stokes3D_DxU / testsurf): PASSED\n";
    // SelfInterac vs. closed-form references on the flat unit square (all three kernels).
    test_SelfInterac<Real>(ker_lapFxU);
    std::cout << "test_SelfInterac (Laplace3D_FxU / plane): PASSED\n";
    test_SelfInterac<Real>(ker_FxU);
    std::cout << "test_SelfInterac (Stokes3D_FxU / plane): PASSED\n";
    test_SelfInterac<Real>(ker_DxU);
    std::cout << "test_SelfInterac (Stokes3D_DxU / plane): PASSED\n";

    // Convergence in the adaptive dyadic-refinement depth cap (max_depth knob, {4,8,12,30}).
    // rel_tol loosened to 1e0 so the assert never trips and only the printed err reveals the trend.
    {
      using QSA = QuadElemList<Real>::QuadScheme;
      std::cout << "  Adaptive self-interac convergence, Sto_FxU (tol=1e-12; max_depth -> rel_err):\n";
      for (const Integer depth : {4, 8, 12, 30}) {
        std::cout << "    max_depth=" << depth << ": ";
        test_SelfInterac<Real>(ker_FxU, QSA::Adaptive, /*rel_tol=*/1e0, /*q=*/10, /*tol=*/1e-12, /*cov_order=*/0, /*max_depth=*/depth);
      }
      std::cout << "  Adaptive near-interac convergence, Sto_FxU / testsurf (max_depth -> rel_err):\n";
      for (const Integer depth : {4, 8, 12, 30}) {
        std::cout << "    max_depth=" << depth << ": ";
        test_NearInterac<Real>(ker_FxU, true, "adaptive depth sweep", QSA::Adaptive, /*rel_tol=*/1e0, /*cov_order=*/0, /*max_depth=*/depth);
      }
    }

    // Scheme 2: rectangular-polar COV (Bruno 2018); accuracy driven by Nbeta, not field order.
    using QS = QuadElemList<Real>::QuadScheme;
    std::cout << "--- Scheme 2: rectangular-polar change of variable ---\n";
    test_RectPolarNodes1D<Real>();
    std::cout << "test_RectPolarNodes1D: PASSED\n";

    const Integer Nbeta = 200;
    test_NearInterac<Real>(ker_FxU, false, "RP Stokes3D_FxU / plane",    QS::RectPolar, 1e-7, Nbeta);
    test_NearInterac<Real>(ker_FxU, true,  "RP Stokes3D_FxU / testsurf", QS::RectPolar, 1e-7, Nbeta);
    test_NearInterac<Real>(ker_DxU, false, "RP Stokes3D_DxU / plane",    QS::RectPolar, 1e-7, Nbeta);
    test_NearInterac<Real>(ker_DxU, true,  "RP Stokes3D_DxU / testsurf", QS::RectPolar, 1e-7, Nbeta);
    std::cout << "test_NearInterac (RectPolar, Nbeta=" << Nbeta << "): PASSED\n";
    test_SelfInterac<Real>(ker_lapFxU, QS::RectPolar, 1e-7, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/200);
    std::cout << "test_SelfInterac Lap_FxU (RectPolar, Nbeta=200): PASSED\n";
    test_SelfInterac<Real>(ker_FxU, QS::RectPolar, 1e-7, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/200);
    std::cout << "test_SelfInterac Sto_FxU (RectPolar, Nbeta=200): PASSED\n";
    test_SelfInterac<Real>(ker_DxU, QS::RectPolar, 1e-7, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/200);
    std::cout << "test_SelfInterac Sto_DxU (RectPolar, Nbeta=200): PASSED\n";
    // Convergence in Nbeta (Nbeta, not q, drives accuracy).
    std::cout << "  RP self-interac convergence, Sto_FxU (q=10; Nbeta -> max_rel):\n";
    for (const Integer nb : {48, 100, 200, 512}) {
      std::cout << "    Nbeta=" << nb << ": ";
      test_SelfInterac<Real>(ker_FxU, QS::RectPolar, 1e0, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/nb);
    }

    // Scheme 3: Hybrid = adaptive near + rectangular-polar self.
    std::cout << "--- Scheme 3: Hybrid (adaptive near + RectPolar self) ---\n";
    test_NearInterac<Real>(ker_FxU, false, "Hybrid Stokes3D_FxU / plane",    QS::Hybrid, 1e-7, /*cov_order=*/0);
    test_NearInterac<Real>(ker_FxU, true,  "Hybrid Stokes3D_FxU / testsurf", QS::Hybrid, 1e-7, /*cov_order=*/0);
    std::cout << "test_NearInterac (Hybrid, adaptive near): PASSED\n";
    test_SelfInterac<Real>(ker_lapFxU, QS::Hybrid, 1e-7, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/200);
    std::cout << "test_SelfInterac Lap_FxU (Hybrid, RP self, Nbeta=200): PASSED\n";
    test_SelfInterac<Real>(ker_FxU, QS::Hybrid, 1e-7, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/200);
    std::cout << "test_SelfInterac Sto_FxU (Hybrid, RP self, Nbeta=200): PASSED\n";

    // Hybrid scheme visualization (flat order-12 panel): VTK dump for ParaView inspection only,
    // no assertions. Disabled in the test build (exercises the VTK writers, not the quadrature).
    // std::cout << "--- Hybrid scheme visualization (flat order-12 panel) ---\n";
    // hybrid_scheme_vis<Real>();

    // ======================================================================================
    // 2. Sphere tests (OPT-IN) -- order 12, 12 patches/face, REGULAR sphere, per scheme, tol = 1e-9.
    //    Each returns a max relative error, gated below rel_tol. RP/Duffy reach ~1e-8 or better, but
    //    the Adaptive near/self path floors much higher on the Stokes DL constant-density identity
    //    (~3e-6 at order 12, even untwisted), which sets the achievable floor -- so the gate is 1e-5.
    //
    //    This is a heavy convergence STUDY (864-element BIE solves x 4 schemes) -- too slow and too
    //    large for the sanitizer CI matrix: an order-12, 864-element solve overflows the runner's
    //    8 MB stack under ASan's redzone-inflated frames (raw SIGSEGV). It is therefore OPT-IN: a
    //    bare invocation (`make test` / CI) runs only the unit + single-element scheme tests above;
    //    pass ANY argument to run the full study:  ./bin/test-quad-elem full
    // ======================================================================================
    if (argc > 1) {
    const Long ElemOrder = 12, PatchPerFace = 12;
    const Real Radius = 1;
    const Real tol = 1e-9;
    const Real rel_tol = 1e-5;                          // required accuracy: err < 1e-5 (see note above)
    const Vector<Real> X0{(Real)1.3, (Real)1.2, (Real)0.2}; // exterior source for Green's identity

    struct SchemeCfg { const char* name; QS scheme; };
    const std::vector<SchemeCfg> schemes = {
      {"RP",       QS::RectPolar},
      {"Adaptive", QS::Adaptive},
      {"Hybrid",   QS::Hybrid},
      {"Duffy",    QS::Duffy},
    };

    if (root) std::cout << "\n==================== Sphere tests (order " << ElemOrder << ", "
                        << PatchPerFace << " patches/face, regular sphere, tol " << tol << ") ====================\n";
    Real overall_worst = 0;
    for (const auto& sc : schemes) {
      if (root) std::cout << "\n---------- scheme = " << sc.name << " ----------\n";
      QuadElemList<Real> qel = BuildTwistedSphere<Real>(ElemOrder, PatchPerFace, Radius, /*theta_twist=*/0., comm);
      // max_depth = 12 is the tol=1e-9 ladder value (u-grading depth). The self-accuracy cap is the
      // v-direction composite-Alpert levels (VLevelsForDigits, deepened to digits-2), not u.
      qel.SetQuadScheme(sc.scheme, /*q=*/6, /*cov_order=*/200, /*max_depth=*/12);

      // Collect every error first (so one run prints the full per-scheme matrix), then gate on the
      // scheme's worst. The Stokes DL constant-density identity is the hardest probe for the Adaptive
      // near/self path (~3e-6 at order 12, even untwisted), so it sets the achievable floor.
      const Real e_area   = test_SurfaceArea(qel, Radius, comm);
      const Real e_dl_lap = test_DLIdentity<Real, Laplace3D_DxU>(qel, comm, tol);
      const Real e_dl_stk = test_DLIdentity<Real, Stokes3D_DxU >(qel, comm, tol);
      const Real e_gr_lap = test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(qel, comm, tol, X0, /*trg_dist=*/0.);
      const Real e_gr_stk = test_greens_identity<Real, Stokes3D_FxU,  Stokes3D_DxU,  Stokes3D_FxT  >(qel, comm, tol, X0, /*trg_dist=*/0.);
      const Real scheme_worst = std::max(std::max(std::max(e_area, e_dl_lap), std::max(e_dl_stk, e_gr_lap)), e_gr_stk);
      overall_worst = std::max(overall_worst, scheme_worst);

      if (root) std::cout << "  scheme " << sc.name << " worst rel error = " << scheme_worst
                          << "  (area=" << e_area << " DL_lap=" << e_dl_lap << " DL_stk=" << e_dl_stk
                          << " greens_lap=" << e_gr_lap << " greens_stk=" << e_gr_stk << ")\n";
      SCTL_ASSERT(scheme_worst < rel_tol);
    }
    if (root) std::cout << "\nAll tests PASSED (overall worst rel error " << overall_worst << " < " << rel_tol << ")\n";
    } else if (root) {
      std::cout << "\nUnit + single-element scheme tests PASSED."
                   " (Sphere convergence study skipped -- pass any argument to run it.)\n";
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
