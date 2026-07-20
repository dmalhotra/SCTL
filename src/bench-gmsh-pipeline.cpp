/**
 * Microbenchmark for the gmsh -> quad-element -> boundary-integral pipeline.
 *
 * Mirrors the end-to-end usage of test_GmshVsTwistSphere (src/test-quad-elem.cpp)
 * but measures WALL TIME instead of accuracy. For each mesh size it times the
 * three pipeline stages -- gmsh load, BIO Setup, BIO Eval -- and prints the
 * hierarchical sctl::Profile breakdown (SetupFarField / SetupSingular / SetupNear,
 * EvalFar / EvalNear) so the bottleneck stage is directly readable. Under
 * -DBENCH_QUAD it also prints the intra-kernel phase breakdown (KernelEval etc.),
 * which is the phase where the scalar Laplace and matrix Stokes kernels diverge.
 *
 * Two problem sizes are compared (natural per-mesh resample order, matching
 * test_GmshVsTwistSphere):
 *     ./sphere       small, coarse mesh, target_order 4
 *     ./sphere_ord9  large,  Q9 mesh,     target_order 16
 * and two kernels: Laplace3D_DxU (scalar) vs Stokes3D_DxU (matrix), Adaptive scheme.
 *
 * Build (coarse: wall time + stage tree only):
 *     make bench-gmsh
 * Build (adds intra-kernel phase breakdown):
 *     make BENCH=1 bench-gmsh
 * Run single-threaded first for clean attribution, then scale threads:
 *     OMP_NUM_THREADS=1 ./bin/bench-gmsh-pipeline
 *     OMP_NUM_THREADS=8 ./bin/bench-gmsh-pipeline
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <sctl/experimental/gmsh_reader.hpp>
#include <sctl/experimental/gmsh_reader.cpp>
#include <sctl/experimental/bench_quad.hpp>
#include <fstream>
#include <string>
#include <vector>

using namespace sctl;

namespace {

// Sum/reduce a scalar across MPI ranks (per-rank Size()/node counts -> global).
inline Long GlobalReduce(Long x, const Comm& comm, CommOp op) {
  StaticArray<Long, 2> buf; buf[0] = x; buf[1] = 0;
  comm.Allreduce(buf + 0, buf + 1, 1, op);
  return buf[1];
}

// One measured (size x kernel) run, collected for the final comparison table.
struct Row {
  std::string mesh;
  std::string kernel;
  Long        nelem  = 0;
  Long        nnode  = 0;
  double      t_load = 0;
  double      t_setup = 0;
  double      t_eval  = 0;
};

// Run the full gmsh -> QuadElemList -> BoundaryIntegralOp pipeline once for one
// kernel and record the per-stage wall times. Returns false if the mesh file is
// absent (matching test_GmshVsTwistSphere's clean skip).
template <class Kernel>
bool bench_gmsh_pipeline(const char* fname, Integer target_order, double tol,
                         const Comm& comm, Row& out) {
  using Real = double;
  static constexpr Integer KDIM0 = Kernel::SrcDim(); // 1 (Laplace) or 3 (Stokes)
  const Kernel ker;

  { std::ifstream is(fname); if (!is.good()) {
      if (!comm.Rank()) std::printf("  SKIPPED (mesh '%s' not found)\n", fname);
      return false;
  } }

  if (!comm.Rank())
    std::printf("\n---- %s  |  %s  order=%d  tol=%.0e ----\n",
                fname, Kernel::Name().c_str(), (int)target_order, tol);

  // --- Stage 1: gmsh load (parse MSH 4.1 + resample each patch to a GL grid). ---
  Profile::reset();
  const double tl0 = bench::Wtime();
  Profile::Tic("gmsh load", &comm, true);
  QuadElemList<Real> qel = GmshReader<Real>::LoadQuadElemList(fname, target_order, comm);
  Profile::Toc();
  const double t_load = bench::Wtime() - tl0;

  // Density at surface nodes (AoS), sampled from a smooth non-polynomial field.
  Vector<Real> Xnodes;
  qel.GetNodeCoord(&Xnodes, nullptr, nullptr);
  const Long Nnode = Xnodes.Dim() / 3;
  Vector<Real> F(Nnode * KDIM0);
  for (Long i = 0; i < Nnode; i++)
    for (Integer k = 0; k < KDIM0; k++)
      F[i * KDIM0 + k] = std::exp(Xnodes[i * 3 + (k % 3)]);

  BoundaryIntegralOp<Real, Kernel> BIOp(ker, /*trg_normal_dot_prod=*/false, comm);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(qel);
  // Targets default to the surface nodes (on-surface): exercises SelfInterac
  // (singular), the near list, and the far field in one pass.

  // --- Stage 2: Setup (near list + singular/near/far operator assembly). ---
  // The intra-kernel phase timers (KernelEval etc.) live inside IntegrateBlock,
  // which runs here (SelfInterac/NearInterac), not in Eval -- so bracket Setup.
  bench::Reset();
  const double ts0 = bench::Wtime();
  Profile::Tic("Setup", &comm, true);
  BIOp.Setup();
  Profile::Toc();
  const double t_setup = bench::Wtime() - ts0;

  // --- Stage 3: Eval (apply). Setup is cached, so this times the apply only. ---
  Vector<Real> U;
  const double te0 = bench::Wtime();
  Profile::Tic("Eval", &comm, true);
  BIOp.ComputePotential(U, F);
  Profile::Toc();
  const double t_eval = bench::Wtime() - te0;

  const Long nelem = GlobalReduce(qel.Size(), comm, CommOp::SUM);
  const Long Nnode_g = GlobalReduce(Nnode, comm, CommOp::SUM);

  if (!comm.Rank()) {
    std::printf("  nelem=%ld  nnode=%ld\n", nelem, Nnode_g);
    std::printf("  wall:  load=%.4g s   setup=%.4g s   eval=%.4g s\n", t_load, t_setup, t_eval);
    std::printf("  setup: %.3g ms/elem   eval: %.3g us/node\n",
                nelem ? 1e3 * t_setup / nelem : 0.0,
                Nnode_g ? 1e6 * t_eval / Nnode_g : 0.0);
  }
  // Hierarchical stage tree: SetupFarField / SetupSingular / SetupNear, EvalFar / EvalNear.
  Profile::print(&comm, {"t_max", "f_max", "f/s_avg"});
  // Intra-kernel phase breakdown of Setup's operator assembly (KernelEval is the
  // phase where the scalar Laplace and matrix Stokes kernels diverge).
  bench::Report("setup intra-kernel phases", t_setup);

  out = Row{fname, Kernel::Name(), nelem, Nnode_g, t_load, t_setup, t_eval};
  return true;
}

void print_summary(const Comm& comm, const std::vector<Row>& rows) {
  if (comm.Rank() || rows.empty()) return;
  std::printf("\n==================== SUMMARY (wall time, s) ====================\n");
  std::printf("%-16s %-14s %8s %9s %10s %10s %10s\n",
              "mesh", "kernel", "nelem", "nnode", "load", "setup", "eval");
  for (const Row& r : rows)
    std::printf("%-16s %-14s %8ld %9ld %10.4g %10.4g %10.4g\n",
                r.mesh.c_str(), r.kernel.c_str(), r.nelem, r.nnode,
                r.t_load, r.t_setup, r.t_eval);
  std::printf("================================================================\n");
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    Profile::Enable(true);

    // (mesh file, natural resample order) -- matches test_GmshVsTwistSphere.
    struct Size { const char* fname; Integer order; double tol; };
    const Size sizes[] = {
      {"./sphere",      4,  1e-6},
      {"./sphere_ord9", 16, 1e-9},
    };

    std::vector<Row> rows;
    for (const Size& s : sizes) {
      Row r;
      if (bench_gmsh_pipeline<Laplace3D_DxU>(s.fname, s.order, s.tol, comm, r)) rows.push_back(r);
      if (bench_gmsh_pipeline<Stokes3D_DxU >(s.fname, s.order, s.tol, comm, r)) rows.push_back(r);
    }
    print_summary(comm, rows);
  }
  Comm::MPI_Finalize();
  return 0;
}
