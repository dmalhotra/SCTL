#include <sctl.hpp>

#include <iostream>
#include <string>

#include "test-utils.hpp"

namespace {

using sctl::Comm;
using sctl::CommOp;
using sctl::Integer;
using sctl::Long;
using sctl::Periodicity;
using sctl::Vector;

constexpr Integer DIM = 3;
using Real = double;
using PtTree = sctl::PtTree<Real, DIM>;

/**
 * Particle data round-trips through the tree unchanged, for data added before and after a
 * `Broadcast` has filled the group's ghost slots, and across the refinements that move it.
 *
 * A `Broadcast` leaves the group's per-node counts covering the ghost nodes as well, so a data set
 * added after one is laid out against those counts with its own items in the owned window, while
 * one added before keeps the owned-only layout. Both must read back as they went in. Only above one
 * rank are there ghost nodes at all, so this says nothing on a single rank.
 */
void TestParticleDataLayout(const Comm& comm) {
  const Integer np = comm.Size(), rank = comm.Rank();
  const Long N = (np > 2 && rank == np - 1 ? 0 : 5000 + 100 * rank);  // one empty rank when there are enough
  const Long dof = 2;

  Vector<Real> X(N * DIM), f(N * dof);
  for (Long i = 0; i < N; i++) {
    for (Integer k = 0; k < DIM; k++) X[i * DIM + k] = (Real)(((i * 37 + k * 11 + rank * 101) % 1000)) / 1000;
    for (Long k = 0; k < dof; k++) f[i * dof + k] = (Real)(rank * 1000000 + i * dof + k);
  }

  const auto check = [&comm, &f](const PtTree& tree, const std::string& name) {
    Vector<Real> out;
    tree.GetParticleData(out, name);
    Long bad = (out.Dim() != f.Dim());
    if (!bad) for (Long i = 0; i < out.Dim(); i++) bad += (out[i] != f[i]);
    Long tot = 0;
    comm.Allreduce(sctl::Ptr2ConstItr<Long>(&bad, 1), sctl::Ptr2Itr<Long>(&tot, 1), 1, CommOp::SUM);
    SCTL_ASSERT_MSG(tot == 0, ("test-pt-tree: " + name + " does not round-trip").c_str());
  };

  PtTree tree(comm);
  tree.AddParticles("pt", X);
  tree.AddParticleData("v1", "pt", f);
  tree.UpdateRefinement(X, 100, true, Periodicity::NONE, 1);
  check(tree, "v1");

  tree.Broadcast<Real>("pt");        // fills the group's ghost slots
  tree.AddParticleData("v2", "pt", f);  // laid out against counts that now cover them
  check(tree, "v1");
  check(tree, "v2");

  tree.Broadcast<Real>("v2");  // the ghost slots are already sized, so this moves nothing
  check(tree, "v2");

  tree.UpdateRefinement(X, 60, true, Periodicity::NONE, 1);
  check(tree, "v1");
  check(tree, "v2");
}

/**
 * A tree whose walk produces far more nodes than `Tree::UpdateRefinement` reserves room for, so the
 * overflow leaves the pool scratch: clumps of identical coordinates, more than `M` to a clump, split
 * at every level down to `MAX_DEPTH` because duplicates never separate.
 *
 * Checks the node list is the complete linear tree it claims to be -- sorted, and its leaves tiling
 * the domain end to end with no gap or overlap. Nodes dropped, duplicated or reordered on the way
 * out of the overflow break that. Built on the self communicator so the list covers the whole
 * domain whatever the run's rank count.
 *
 * Which way the overflow goes depends on how much room the thread's pool chunk happens to have:
 * grown into the chunk where it fits, into a heap buffer where it does not. The tree must come out
 * the same either way, which is what this asks.
 */
void TestWalkOverflow() {
  using Morton = sctl::Morton<DIM>;
  const Comm& self = Comm::Self();
  const Long M = 512, per = M + 1, clumps = 400;
  const Long N = clumps * per;
  Vector<Real> X(N * DIM);
  for (Long g = 0; g < clumps; g++) {
    Real c[DIM];
    for (Integer d = 0; d < DIM; d++) c[d] = (Real)((g * 7919 + d * 104729) % 100003) / (Real)100003;
    for (Long j = 0; j < per; j++)
      for (Integer d = 0; d < DIM; d++) X[((g * per + j) * DIM) + d] = c[d];
  }

  sctl::Tree<DIM> tree(self);
  tree.UpdateRefinement(X, M, false, Periodicity::NONE, 0);
  const Vector<Morton>& mid = tree.GetNodeMID();
  const auto& attr = tree.GetNodeAttr();
  SCTL_ASSERT(mid.Dim() > 0);

  // Morton equality takes the depth in too; here only where a box starts matters.
  const auto same_box = [](const Morton& a, const Morton& b) { return !(a.mid < b.mid) && !(b.mid < a.mid); };

  Long leaves = 0;
  Morton reach = Morton();  // how far the cover has got; leaves tile in order, each starting where the last ended
  for (Long i = 0; i < mid.Dim(); i++) {
    if (i) SCTL_ASSERT_MSG(mid[i - 1] < mid[i], "walk overflow: the node list is not sorted");
    if (!attr[i].Leaf) continue;
    SCTL_ASSERT_MSG(same_box(mid[i], reach), "walk overflow: the leaves leave a gap or overlap");
    reach = mid[i].Next();
    leaves++;
  }
  SCTL_ASSERT_MSG(same_box(reach, Morton().Next()), "walk overflow: the leaves stop short of the domain end");
  SCTL_ASSERT(leaves > clumps);  // every clump had to be resolved into a box of its own
  std::cout << "  walk overflow: " << mid.Dim() << " nodes, " << leaves << " leaves, cover complete\n";
}

}  // namespace

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const Comm& comm = Comm::World();
    const auto run = [&comm]() {
      sctl::PtTree<double, 2>::test();  // the usage example
      TestParticleDataLayout(comm);
      if (!comm.Rank()) TestWalkOverflow();
      comm.Barrier();
    };
    run();
    { // and again where the OpenMP team is smaller than the tree build asks for
      const test_utils::TrimmedOmpTeam team;
      if (!comm.Rank()) team.Report("  again");
      if (team.Trimmed()) run();
    }
    if (!comm.Rank()) std::cout << "test-pt-tree passed on " << comm.Size() << " ranks\n";
  }
  sctl::Comm::MPI_Finalize();

  return 0;
}
