/**
 * Accuracy test for the gmsh -> QuadElemList eval pipeline on an OPEN CYLINDER
 * (axis = x, radius 0.5, length 1, centered at the origin), using CSBQ's
 * SlenderElemList as the reference "truth".
 *
 * The same continuous surface is discretized two ways:
 *   - gmsh mesh ./cylinder_ord9 (2112 order-9 quad patches) resampled onto an
 *     order-16 QuadElemList;
 *   - a CSBQ SlenderElemList: straight centerline along x, Nelem=30 panels,
 *     ElemOrder=10, FourierOrder=64 (30*64=1920 patches ~ 2112).
 *
 * The SAME analytic surface density (periodic in the circumferential angle,
 * smooth in x, band-limited so CSBQ resolves it) is sampled at each list's own
 * nodes. We evaluate the single-layer potential (Laplace and Stokes) at common
 * off-surface targets in four regions (near/far x interior/exterior) and report
 * the relative-L2 difference gmsh-vs-CSBQ per region -- i.e. the gmsh geometry's
 * eval error against the convergent CSBQ reference.
 *
 * Build (from the SCTL_quad_element root, after `. ./sctl_source`):
 *   make MPI=1 bin/test-cylinder-gmsh-vs-csbq
 * Run (hybrid MPI + OpenMP):
 *   OMP_NUM_THREADS=4 mpirun -n <N> --map-by :PE=4 ./bin/test-cylinder-gmsh-vs-csbq
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <sctl/experimental/gmsh_reader.cpp>
#include <csbq/slender_element.hpp>
#include <csbq/slender_element.cpp>

#include <iomanip>
#include <random>
#include <string>
#include <vector>

using namespace sctl;

namespace {

// Geometry of the open cylinder.
constexpr double CYL_R = 0.5;   // radius (in the y-z plane)
constexpr double CYL_L = 1.0;   // axial length, x in [-L/2, L/2]

// --- Distributed-memory helper --------------------------------------------------
// Surface-area sums are accumulated over each rank's local nodes and reduced.
inline double GlobalReduce(double x, const Comm& comm, CommOp op) {
  StaticArray<double,2> buf; buf[0] = x; buf[1] = 0;
  comm.Allreduce(buf+0, buf+1, 1, op);
  return buf[1];
}

// --- Shared analytic surface density --------------------------------------------
// A band-limited random field that depends only on the 3D surface point, so it is
// sampled identically on both discretizations:
//   sigma_c(x,theta) = sum_{m,k} [A cos(m*theta) + B sin(m*theta)] * cos(k*pi*(x+L/2)/L)
// periodic in theta (required on the tube), smooth in x. Angular modes m <= Mmax are
// kept well below FourierOrder/2 so the CSBQ reference resolves the density exactly.
class PeriodicDensity {
  static constexpr int Mmax = 6;  // max angular (Fourier) mode
  static constexpr int Kmax = 4;  // max axial mode
  static constexpr int NC   = 3;  // components (Laplace uses 0; Stokes uses 0,1,2)
  double A[NC][Mmax+1][Kmax+1];
  double B[NC][Mmax+1][Kmax+1];

 public:
  explicit PeriodicDensity(unsigned seed = 12345u) {
    std::mt19937 rng(seed); // fixed seed => identical coefficients on every rank
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int c = 0; c < NC; c++)
      for (int m = 0; m <= Mmax; m++)
        for (int k = 0; k <= Kmax; k++) {
          const double decay = std::exp(-0.35 * (m + k)); // keep it smooth
          A[c][m][k] = U(rng) * decay;
          B[c][m][k] = (m == 0 ? 0.0 : U(rng) * decay);   // sin(0)=0
        }
  }

  // Evaluate component `comp` of the density at surface point P (AoS xyz).
  double operator()(const double* P, int comp) const {
    const double x  = P[0];
    const double th = std::atan2(P[2], P[1]);
    double s = 0;
    for (int m = 0; m <= Mmax; m++) {
      const double cm = std::cos(m * th), sm = std::sin(m * th);
      for (int k = 0; k <= Kmax; k++) {
        const double ax = std::cos(k * const_pi<double>() * (x + CYL_L / 2) / CYL_L);
        s += (A[comp][m][k] * cm + B[comp][m][k] * sm) * ax;
      }
    }
    return s;
  }
};

// Build a density vector (AoS, size Nnode*KDIM0) on a list's own surface nodes.
template <Integer KDIM0>
Vector<double> BuildDensity(const Vector<double>& Xnodes, const PeriodicDensity& dens) {
  const Long Nnode = Xnodes.Dim() / 3;
  Vector<double> F(Nnode * KDIM0);
  for (Long i = 0; i < Nnode; i++)
    for (Integer c = 0; c < KDIM0; c++)
      F[i * KDIM0 + c] = dens(&Xnodes[i * 3], c);
  return F;
}

// --- Off-surface targets --------------------------------------------------------
struct TargetSet {
  std::string name;
  Vector<double> X; // AoS xyz, replicated identically on all ranks
};

// Place `n` points on a coaxial cylinder of radius `r`, with x in [-xr,xr] and
// random circumferential angle. Deterministic (fixed seed) => identical on all ranks.
TargetSet MakeTargets(const std::string& name, double r, double xr, Long n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> Uth(0.0, 2 * const_pi<double>());
  std::uniform_real_distribution<double> Ux(-xr, xr);
  TargetSet t; t.name = name; t.X.ReInit(n * 3);
  for (Long i = 0; i < n; i++) {
    const double th = Uth(rng), x = Ux(rng);
    t.X[i*3+0] = x;
    t.X[i*3+1] = r * std::cos(th);
    t.X[i*3+2] = r * std::sin(th);
  }
  return t;
}

// Surface area from far-field quadrature weights (sanity: both surfaces ~ 2*pi*R*L).
double SurfaceArea(const ElementListBase<double>& elem_lst, const Comm& comm) {
  Vector<double> X, Xn, wts, dist_far; Vector<Long> cnt;
  elem_lst.GetFarFieldNodes(X, Xn, wts, dist_far, cnt, 1e-10);
  double a = 0; for (Long i = 0; i < wts.Dim(); i++) a += wts[i];
  return GlobalReduce(a, comm, CommOp::SUM);
}

// --- Per-kernel comparison ------------------------------------------------------
// Both element lists get the SAME (replicated) targets, so ComputePotential returns
// the full potential (sources reduced across ranks) identically on every rank.
template <class Kernel>
void RunKernel(const QuadElemList<double>& qel, const SlenderElemList<double>& sel,
               const PeriodicDensity& dens, const std::vector<TargetSet>& cats,
               const char* kname, double tol, const Comm& comm) {
  static constexpr Integer KDIM0 = Kernel::SrcDim();
  const Kernel ker;

  // Density on each list's own surface nodes (same analytic field).
  Vector<double> Xg, Xs;
  qel.GetNodeCoord(&Xg, nullptr, nullptr);
  sel.GetNodeCoord(&Xs, nullptr, nullptr);
  const Vector<double> Fg = BuildDensity<KDIM0>(Xg, dens);
  const Vector<double> Fs = BuildDensity<KDIM0>(Xs, dens);

  static constexpr Integer TDIM = Kernel::TrgDim();

  // Concatenate all regions into ONE target array so each operator sets up its
  // near-field quadrature once (not once per region), then slice the result.
  Vector<double> Xall;
  std::vector<std::pair<Long,Long>> span; // (start_point, num_points) per region
  for (const auto& c : cats) {
    const Long n = c.X.Dim() / 3;
    span.emplace_back(Xall.Dim() / 3, n);
    for (Long i = 0; i < c.X.Dim(); i++) Xall.PushBack(c.X[i]);
  }

  BoundaryIntegralOp<double, Kernel> Bg(ker, /*trg_normal_dot_prod=*/false, comm);
  Bg.SetAccuracy(tol); Bg.AddElemList(qel); Bg.SetTargetCoord(Xall);
  BoundaryIntegralOp<double, Kernel> Bc(ker, /*trg_normal_dot_prod=*/false, comm);
  Bc.SetAccuracy(tol); Bc.AddElemList(sel); Bc.SetTargetCoord(Xall);

  Vector<double> Ug, Uc;
  Bg.ComputePotential(Ug, Fg);
  Bc.ComputePotential(Uc, Fs); // reference
  SCTL_ASSERT(Ug.Dim() == Uc.Dim());

  if (!comm.Rank()) {
    std::cout << "\n  " << kname << " single-layer (SrcDim=" << KDIM0 << ", TrgDim=" << TDIM << "), tol=" << tol << "\n";
    std::cout << "    " << std::left << std::setw(16) << "region"
              << std::setw(16) << "rel-L2" << std::setw(16) << "max-abs-err" << "\n";
  }

  for (size_t c = 0; c < cats.size(); c++) {
    const Long p0 = span[c].first, np = span[c].second;
    double err2 = 0, ref2 = 0, maxe = 0;
    for (Long p = p0; p < p0 + np; p++)
      for (Integer d = 0; d < TDIM; d++) {
        const Long i = p * TDIM + d;
        const double e = Ug[i] - Uc[i];
        err2 += e * e; ref2 += Uc[i] * Uc[i];
        maxe = std::max(maxe, std::fabs(e));
      }
    const double rel = (ref2 > 0 ? std::sqrt(err2 / ref2) : 0.0);
    if (!comm.Rank())
      std::cout << "    " << std::left << std::setw(16) << cats[c].name
                << std::setw(16) << std::scientific << std::setprecision(3) << rel
                << std::setw(16) << maxe << "\n";
  }
}

// Build the CSBQ reference cylinder: straight centerline along x, constant radius.
// Replicate-then-slice under MPI: each rank builds ONLY its contiguous panel slice
// [Nelem*rank/size, Nelem*(rank+1)/size), matching CSBQ's GenericGeom convention and
// the QuadElemList/GmshReader partitioning (so BoundaryIntegralOp does not double count).
SlenderElemList<double> BuildCsbqCylinder(Long Nelem, Long ElemOrder, Long FourierOrder, const Comm& comm) {
  const Long k0 = Nelem * (comm.Rank() + 0) / comm.Size();
  const Long k1 = Nelem * (comm.Rank() + 1) / comm.Size();
  Vector<Long> eo, fo;
  Vector<double> Xc, rad;
  const Vector<double>& nds = SlenderElemList<double>::CenterlineNodes(ElemOrder); // [0,1]
  for (Long i = k0; i < k1; i++) {
    eo.PushBack(ElemOrder); fo.PushBack(FourierOrder);
    for (Long j = 0; j < ElemOrder; j++) {
      const double s = (i + nds[j]) / Nelem;            // global s in [0,1]
      Xc.PushBack(-CYL_L / 2 + CYL_L * s);              // x
      Xc.PushBack(0.0);                                 // y (centerline on axis)
      Xc.PushBack(0.0);                                 // z
      rad.PushBack(CYL_R);
    }
  }
  // Default orientation => circular cross-section in the y-z plane (matches the mesh).
  return SlenderElemList<double>(eo, fo, Xc, rad);
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();

    const Long GmshOrder   = 16;
    const Long Nelem       = 30;
    const Long ElemOrder   = 10;
    const Long FourierOrder = 64;
    const double tol       = 1e-11;

    if (!comm.Rank()) {
      std::cout << "Open-cylinder eval accuracy: gmsh QuadElemList vs CSBQ SlenderElemList\n";
      std::cout << "  cylinder: axis=x, R=" << CYL_R << ", L=" << CYL_L << ", centered at origin\n";
      std::cout << "  gmsh: ./cylinder_ord9 resampled to order " << GmshOrder << "\n";
      std::cout << "  csbq: Nelem=" << Nelem << " ElemOrder=" << ElemOrder
                << " FourierOrder=" << FourierOrder << "\n";
      std::cout << "  MPI ranks: " << comm.Size() << "\n";
    }

    // Geometry.
    QuadElemList<double> qel = GmshReader<double>::LoadQuadElemList("./cylinder_ord9", GmshOrder, comm);
    SlenderElemList<double> sel = BuildCsbqCylinder(Nelem, ElemOrder, FourierOrder, comm);

    // Surface-area sanity check (both should be ~ 2*pi*R*L).
    const double area_exact = 2 * const_pi<double>() * CYL_R * CYL_L;
    const double area_g = SurfaceArea(qel, comm);
    const double area_s = SurfaceArea(sel, comm);
    if (!comm.Rank())
      std::cout << "  surface area: gmsh=" << area_g << " csbq=" << area_s
                << " exact=" << area_exact << "\n";

    // Off-surface targets: near/far x interior/exterior.
    std::vector<TargetSet> cats;
    cats.push_back(MakeTargets("near-exterior", 0.55, 0.4, 24, 101)); // just outside
    cats.push_back(MakeTargets("near-interior", 0.45, 0.4, 24, 202)); // just inside
    cats.push_back(MakeTargets("far-exterior",  2.00, 0.4, 24, 303)); // far outside
    cats.push_back(MakeTargets("far-interior",  0.05, 0.3, 24, 404)); // deep interior, near axis

    PeriodicDensity dens(/*seed=*/777u);

    RunKernel<Laplace3D_FxU>(qel, sel, dens, cats, "Laplace3D_FxU", tol, comm);
    RunKernel<Stokes3D_FxU >(qel, sel, dens, cats, "Stokes3D_FxU",  tol, comm);

    if (!comm.Rank()) std::cout << "\nDone.\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
