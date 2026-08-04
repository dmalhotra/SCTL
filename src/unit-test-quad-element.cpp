#include <sctl.hpp>
#include "sctl/experimental/quad_element.cpp" // template definitions
#include "sctl/experimental/gmsh_reader.cpp"  // gmsh -> QuadElemList importer

using namespace sctl;

// Friend shim forwarding to QuadElemList's private helpers (must be in namespace sctl).
namespace sctl {
template <class Real> struct QuadElemTestAccess {
    static Real GetClosestNode(const QuadElemList<Real>& qel, Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) {
        return qel.GetClosestNode(ustar, vstar, elem_idx, Xtrg);
    }
    static Real GetClosestPoint(const QuadElemList<Real>& qel, Real& ustar, Real& vstar, const Long elem_idx, const Vector<Real>& Xtrg) {
        return qel.GetClosestPoint(ustar, vstar, elem_idx, Xtrg);
    }
};
}


// Equidistant tensor grid of nelem_perside^2 panels of GL nodes on [0,1]^2, z = 0. One
// order x order block per panel, u-slow/v-fast, so QuadElemList's constructor slices it into one
// element per panel; a single global row-major grid would instead hand each element a strip
// spanning several panels, leaving the positions plausible but the tangents meaningless.
template <class Real> Vector<Real> param_grid(const Integer order, const Integer nelem_perside) {
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    Vector<Real> coord((Long)nelem_perside*nelem_perside*order*order*3);
    Long idx = 0;
    for (Integer pu = 0; pu < nelem_perside; pu++) {
        for (Integer pv = 0; pv < nelem_perside; pv++) {
            for (Integer i = 0; i < order; i++) {
                for (Integer j = 0; j < order; j++) {
                    coord[idx + 0] = (nds[i] + pu) / nelem_perside;
                    coord[idx + 1] = (nds[j] + pv) / nelem_perside;
                    coord[idx + 2] = 0;
                    idx += 3;
                }
            }
        }
    }
    SCTL_ASSERT(idx == coord.Dim());
    return coord;
}

template <class Real> Vector<Real> get_testsurf(const Integer order, const Integer nelem_perside) {
    // First define surface
    const auto fsurf = [](const Real x, const Real y) {
        return x*y;
    };

    Vector<Real> coord0 = param_grid<Real>(order, nelem_perside); // Get x-y grid on [0,1]x[0,1]
    // Get z value on x-y grid for surface.
    for (int i=0; i<coord0.Dim()/3; i++) {
        coord0[i*3 + 2] = fsurf(coord0[i*3+0], coord0[i*3+1]);
    }
    return coord0;
}

template <class Real> void test_ParamGrid() {
    // The driver-local grid helper, and that each block it emits becomes one valid element.
    const Long order = 4;
    const Long nelem_perside = 2;
    const Long N_per_side = order * nelem_perside; // 8 nodes per side
    const Long N_total = N_per_side * N_per_side;  // 64 tensor-grid points

    Vector<Real> coord0 = param_grid<Real>(order, nelem_perside);
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

    // One order x order block per panel, AoS (u slow, v fast), z = 0, panels in (pu,pv) order.
    // NOT a single global row-major grid: Init slices this into elements of order^2 consecutive
    // nodes, so a global grid would hand each element a strip spanning several panels.
    const Real tol = 1e-12;
    for (Long pu = 0; pu < nelem_perside; pu++) {
        for (Long pv = 0; pv < nelem_perside; pv++) {
            const Long ebase = (pu * nelem_perside + pv) * order * order;
            for (Long i = 0; i < order; i++) {
                for (Long j = 0; j < order; j++) {
                    const Long idx = (ebase + i * order + j) * 3;
                    SCTL_ASSERT(fabs(coord0[idx + 0] - x_param_exp[pu * order + i]) < tol);
                    SCTL_ASSERT(fabs(coord0[idx + 1] - x_param_exp[pv * order + j]) < tol);
                    SCTL_ASSERT(fabs(coord0[idx + 2] - (Real)0) < tol);
                }
            }
        }
    }

    // Each element must be a panel in its own right: on this grid every tangent is constant and
    // axis-aligned with length 1/nelem_perside. Getting the node ordering wrong leaves the
    // positions looking plausible while the tangents -- hence normals and area -- are meaningless.
    {
        QuadElemList<Real> qel(order, coord0);
        SCTL_ASSERT(qel.Size() == nelem_perside * nelem_perside);
        const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
        const Real h = (Real)1 / nelem_perside;
        for (Long e = 0; e < qel.Size(); e++) {
            Vector<Real> X, Xn, dXu, dXv;
            qel.GetGeom(&X, &Xn, nullptr, &dXu, &dXv, nds, nds, e);
            for (Long t = 0; t < order * order; t++) {
                SCTL_ASSERT(fabs(dXu[t*3+0] - h) < tol && fabs(dXu[t*3+1]) < tol && fabs(dXu[t*3+2]) < tol);
                SCTL_ASSERT(fabs(dXv[t*3+0]) < tol && fabs(dXv[t*3+1] - h) < tol && fabs(dXv[t*3+2]) < tol);
                SCTL_ASSERT(fabs(fabs(Xn[t*3+2]) - (Real)1) < tol);
            }
        }
    }
}

template <class Real> void test_GetClosestNode_plane() {
    // Flat patch z = 0; lifted target must snap back to the surface node.
    const Long COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = param_grid<Real>(order, 1);
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
    const Real dist = QuadElemTestAccess<Real>::GetClosestNode(qel, ustar, vstar, 0, Xtrg_shifted);

    const Real tol = 1e-9;
    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist  - 0.1) < tol);

    // Now shift the target away from x and y as well.
    Xtrg_shifted[0] -= 0.0013;
    Xtrg_shifted[1] += 0.0005;
    const Real exp_dist = sqrt<Real>(0.1*0.1 + 0.0013*0.0013 + 0.0005*0.0005);

    const Real dist2 = QuadElemTestAccess<Real>::GetClosestNode(qel, ustar, vstar, 0, Xtrg_shifted);

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
    const Real dist = QuadElemTestAccess<Real>::GetClosestNode(qel, ustar, vstar, 0, Xtrg_shifted);

    const Real tol = 1e-8;
    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist  - d ) < tol);

    // Now shift the target away from x and y as well.
    Xtrg_shifted[0] -= 0.0013;
    Xtrg_shifted[1] += 0.0005;
    const Real exp_dist = sqrt<Real>((d*Xntrg[0]-0.0013)*(d*Xntrg[0]-0.0013) + (d*Xntrg[1]+0.0005)*(d*Xntrg[1]+0.0005) + (d*Xntrg[2])*(d*Xntrg[2]));

    const Real dist2 = QuadElemTestAccess<Real>::GetClosestNode(qel, ustar, vstar, 0, Xtrg_shifted);

    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist2  - exp_dist) < tol);
}

template <class Real> void test_GetClosestPoint_plane() {
    // Flat patch z = 0: GetClosestPoint must recover the exact projection at an off-node (u,v).
    const Integer COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = param_grid<Real>(order, 1);
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
    const Real dist = QuadElemTestAccess<Real>::GetClosestPoint(qel, ustar, vstar, 0, Xtrg);

    const Real tol = 1e-9;
    SCTL_ASSERT(fabs(ustar - u0) < tol);
    SCTL_ASSERT(fabs(vstar - v0) < tol);
    SCTL_ASSERT(fabs(dist  - d) < tol);

    // Tangential shift: projection follows it, dist stays = d.
    Xtrg[0] -= 0.0013;
    Xtrg[1] += 0.0005;
    const Real dist2 = QuadElemTestAccess<Real>::GetClosestPoint(qel, ustar, vstar, 0, Xtrg);
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
    const Real dist = QuadElemTestAccess<Real>::GetClosestPoint(qel, ustar, vstar, 0, Xtrg);

    const Real tol = 1e-7;
    SCTL_ASSERT(fabs(ustar - u0) < tol);
    SCTL_ASSERT(fabs(vstar - v0) < tol);
    SCTL_ASSERT(fabs(dist  - d) < tol);

    // Generic target: residual (closest point - target) must be orthogonal to both tangents.
    Vector<Real> Xt2(COORD_DIM);
    Xt2[0] = Xsurf[0] + (Real)0.05;
    Xt2[1] = Xsurf[1] - (Real)0.03;
    Xt2[2] = Xsurf[2] + (Real)0.08;
    QuadElemTestAccess<Real>::GetClosestPoint(qel, ustar, vstar, 0, Xt2);
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

template <class Real, class Kernel> void test_NearInterac(const Kernel& ker, const bool curved, const char* label, const Real rel_tol = 1e-6) {
    const Integer COORD_DIM = 3;
    const Integer order = 16; // dispatch instantiates {8,12,16}
    const Integer KDIM0 = Kernel::SrcDim();
    const Integer KDIM1 = Kernel::TrgDim();
    const Long nnode = (Long)order * order;
    const Long elem_idx = 0;

    // Single element: flat plane z = 0 or curved testsurf z = u*v.
    Vector<Real> coord0 = curved ? get_testsurf<Real>(order, 1)
                                 : param_grid<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);

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
    // near distance d=0.01 used here.
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

// Cross-check the near potential at a flat-panel target d above (0.4,0.6) from several
// independent high-accuracy methods to establish a gold reference and measure hedgehog against it.
template <class Real, class Kernel> void test_SelfInterac(const Kernel& ker, const Real rel_tol = 1e-6, const Real tol = 1e-10) {
    const Integer order = 12;
    const Long nnode = (Long)order * order;
    const Integer KDIM0 = Kernel::SrcDim();
    const Integer KDIM1 = Kernel::TrgDim();
    SCTL_ASSERT(KDIM1 <= 3);

    // Flat unit square z = 0.
    Vector<Real> coord0 = param_grid<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);

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

    // Off-node target grid in u and v (node (a,b) at a*Nv + b).
    Vector<Real> u_param, v_param, wu, wv;
    LegQuadRule<Real>::ComputeNdsWts(&u_param, &wu, 40);
    LegQuadRule<Real>::ComputeNdsWts(&v_param, &wv, 40);
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

// gmsh import pipeline: read a quad-meshed sphere, resample onto a target_order
// GL grid, and check (1) nodes lie approximately on the sphere, (2) they are
// ordered as a tensor GL grid (u-slow), (3) node count matches the order.
template <class Real> void test_GmshReader(const char* fname = "./sphere", const Integer order = 6, const Real Radius = 1) {
    { std::ifstream is(fname); if (!is.good()) { std::cout << "test_GmshReader: SKIPPED (mesh file '" << fname << "' not found)\n"; return; } }

    // Raw quad patches (tensor (i,j) order, u-slow) + the resampled GL coords.
    Integer nside = 0;
    Vector<Real> src;
    const Long nquad = GmshReader<Real>::ReadQuadPatches(fname, nside, src);
    const Vector<Real> coord = GmshReader<Real>::ReadQuadCoord(fname, order);

    QuadElemList<Real> qel(order, coord);
    SCTL_ASSERT(qel.Order() == order);
    SCTL_ASSERT(qel.Size() == nquad);

    // (3) tensor GL grid of the correct order: order*order nodes per element.
    const Long N = order, nn_trg = N * N;
    SCTL_ASSERT(coord.Dim() == nquad * nn_trg * 3);

    // (1) every node lies approximately on the sphere of radius `Radius`.
    // Linear (Q1) patches are flat chords, so GL nodes sit inside the sphere by O(h^2).
    Real max_rad_err = 0;
    for (Long i = 0; i < coord.Dim() / 3; i++) {
        const Real r = sqrt<Real>(coord[i*3+0]*coord[i*3+0] + coord[i*3+1]*coord[i*3+1] + coord[i*3+2]*coord[i*3+2]);
        max_rad_err = std::max<Real>(max_rad_err, fabs(r - Radius));
    }
    const Real sphere_tol = (nside == 2 ? (Real)3e-2 : (Real)1e-6); // resolution-dependent for flat Q1
    std::cout << "  test_GmshReader: nquad=" << nquad << " gmsh_nside=" << nside << " order=" << order
              << " max|r-R|=" << max_rad_err << "\n";
    SCTL_ASSERT(max_rad_err < sphere_tol);

    // (2) ordered correctly: independently re-evaluate each patch's gmsh shape
    // functions at the GL grid using explicit per-point Lagrange basis values
    // (a different contraction than the importer's GEMM), and require node-by-node
    // agreement with the importer output -- any (u,v)/transpose ordering bug fails.
    Vector<Real> eq(nside);
    for (Integer k = 0; k < nside; k++) eq[k] = (nside == 1) ? (Real)0.5 : (Real)k / (Real)(nside - 1);
    const Vector<Real>& gl = QuadElemList<Real>::ParamNodes(order);
    // Lagrange basis weights at each GL node: Lu(a,i) row-major nside x N from Interpolate(eq, gl).
    Vector<Real> Lw(nside * N); LagrangeInterp<Real>::Interpolate(Lw, eq, gl); // Lw[i*N + a]
    const Long nn_src = (Long)nside * nside;
    Real max_ord_err = 0;
    for (Long e = 0; e < nquad; e++) {
        for (Long a = 0; a < N; a++) {        // u-target index (slow)
            for (Long b = 0; b < N; b++) {    // v-target index (fast)
                Real X[3] = {0,0,0};
                for (Integer i = 0; i < nside; i++) {
                    for (Integer j = 0; j < nside; j++) {
                        const Real w = Lw[i*N + a] * Lw[j*N + b];
                        const Long s = (e * nn_src + (Long)i*nside + j) * 3;
                        X[0] += w * src[s+0]; X[1] += w * src[s+1]; X[2] += w * src[s+2];
                    }
                }
                const Long p = (e * nn_trg + a*N + b) * 3; // u-slow flat node index
                for (Integer k = 0; k < 3; k++) max_ord_err = std::max<Real>(max_ord_err, fabs(coord[p+k] - X[k]));
            }
        }
    }
    std::cout << "  test_GmshReader: ordering cross-check max_abs_err=" << max_ord_err << "\n";
    SCTL_ASSERT(max_ord_err < (Real)1e-11);
}

int main(int argc, char** argv) {

    using Real = double;

    test_GmshReader<Real>();
    std::cout << "test_GmshReader: PASSED\n";
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

    test_QuadNodeInterp<Real>();
    std::cout << "test_QuadNodeInterp: PASSED\n";

    // NearInterac (split-at-foot) vs. upsampled direct quadrature.
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

    // SelfInterac (Duffy) vs. closed-form references on the flat unit square.
    test_SelfInterac<Real>(ker_lapFxU);
    std::cout << "test_SelfInterac (Laplace3D_FxU / plane): PASSED\n";
    test_SelfInterac<Real>(ker_FxU);
    std::cout << "test_SelfInterac (Stokes3D_FxU / plane): PASSED\n";
    test_SelfInterac<Real>(ker_DxU);
    std::cout << "test_SelfInterac (Stokes3D_DxU / plane): PASSED\n";

    return 0;
}
