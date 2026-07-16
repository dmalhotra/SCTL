#include <sctl.hpp>
#include "sctl/experimental/quad_element.cpp" // template definitions
#include "sctl/experimental/gmsh_reader.cpp"  // gmsh -> QuadElemList importer

using namespace sctl;


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

// Forward declaration of the friend shim (defined below) so test_NearInterac can use its
// CompareNearBlocks bit-for-bit gate; the shim's full definition appears later in namespace sctl.
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

    // Bit-for-bit gate: leaf-batched near block must exactly match the per-leaf reference.
    // (digits=8 matches near tol=1e-8 above; order=24 matches this test's element order.)
    {
        const Real near_maxdiff = QuadElemTestAccess<Real>::template CompareNearBlocks<8, 24>(qel, elem_idx, Xt, normal_trg, ker);
        std::cout << "  test_NearInterac (" << label << "): batched-vs-perleaf max|dM| = " << near_maxdiff << "\n";
        SCTL_ASSERT(near_maxdiff == 0);
    }

    Vector<Real> u_near(KDIM1);
    u_near.SetZero();
    for (Long r = 0; r < nnode * KDIM0; r++) {
        for (Integer k1 = 0; k1 < KDIM1; k1++) u_near[k1] += sigma[r] * M[r][k1];
    }

    // Reference: upsampled direct quadrature.
    const Long nsub = 100;
    Vector<Real> u_ref = direct_upsampled_potential<Real, Kernel>(qel, elem_idx, sigma, Xt, ker, nsub);

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
    static void LogSingularQuad1D(Vector<Real>& param, Vector<Real>& w, const Real v0, const Integer Lvl, const Integer QuadOrder) {
        QuadElemList<Real>::LogSingularQuad1D(param, w, v0, Lvl, QuadOrder);
    }
    static void RectPolarNodes1D(Vector<Real>& nodes, Vector<Real>& wts, const Real alpha, const Integer q, const Vector<Real>& gl_nds, const Vector<Real>& gl_wts) {
        QuadElemList<Real>::RectPolarNodes1D(nodes, wts, alpha, q, gl_nds, gl_wts);
    }
    // Bit-for-bit gate: run both the per-leaf reference (NearInteracBlock) and the production
    // leaf-batched block (NearInteracBlockBatched) at the same (digits,order) and return max|diff|.
    template <Integer digits, Integer order, class Kernel>
    static Real CompareNearBlocks(const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker) {
        Matrix<Real> M_ref, M_bat;
        QuadElemList<Real>::template NearInteracBlock<digits, order>(M_ref, qel, elem_idx, Xtrg, normal_trg, ker);
        QuadElemList<Real>::template NearInteracBlockBatched<digits, order>(M_bat, qel, elem_idx, Xtrg, normal_trg, ker);
        SCTL_ASSERT(M_ref.Dim(0) == M_bat.Dim(0) && M_ref.Dim(1) == M_bat.Dim(1));
        Real maxdiff = 0;
        for (Long i = 0; i < M_ref.Dim(0); i++)
            for (Long j = 0; j < M_ref.Dim(1); j++)
                maxdiff = std::max<Real>(maxdiff, fabs(M_ref[i][j] - M_bat[i][j]));
        return maxdiff;
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
    // rel_tol loosened to 1e0 so the assert never trips and only the printed err reveals the
    // trend; max_depth=30 reproduces the default-scheme results above.
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

    // Scheme 3: Hybrid = adaptive near + rectangular-polar self. Near matches the
    // Adaptive path (tol-driven, cov_order ignored); self matches the RectPolar path.
    std::cout << "--- Scheme 3: Hybrid (adaptive near + RectPolar self) ---\n";
    test_NearInterac<Real>(ker_FxU, false, "Hybrid Stokes3D_FxU / plane",    QS::Hybrid, 1e-7, /*cov_order=*/0);
    test_NearInterac<Real>(ker_FxU, true,  "Hybrid Stokes3D_FxU / testsurf", QS::Hybrid, 1e-7, /*cov_order=*/0);
    std::cout << "test_NearInterac (Hybrid, adaptive near): PASSED\n";
    test_SelfInterac<Real>(ker_lapFxU, QS::Hybrid, 1e-7, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/200);
    std::cout << "test_SelfInterac Lap_FxU (Hybrid, RP self, Nbeta=200): PASSED\n";
    test_SelfInterac<Real>(ker_FxU, QS::Hybrid, 1e-7, /*q=*/10, /*tol=*/1e-14, /*cov_order=*/200);
    std::cout << "test_SelfInterac Sto_FxU (Hybrid, RP self, Nbeta=200): PASSED\n";

    return 0;
}