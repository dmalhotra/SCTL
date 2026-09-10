#include <sctl.hpp>

#include <iostream>
#include <string>

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

}  // namespace

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  {
    const Comm& comm = Comm::World();
    sctl::PtTree<double, 2>::test();  // the usage example
    TestParticleDataLayout(comm);
    comm.Barrier();
    if (!comm.Rank()) std::cout << "test-pt-tree passed on " << comm.Size() << " ranks\n";
  }
  sctl::Comm::MPI_Finalize();

  return 0;
}
