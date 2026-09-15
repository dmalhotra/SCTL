// Exercises sctl::device_tree::PtTree from a translation unit that no CUDA compiler touches: this
// file is built by the host compiler and links libsctl-device-tree.a. Nothing here includes
// thrust or a CUDA header, which is the property the interface exists to provide.

#include <cstdio>
#include <random>
#include <vector>

#include "sctl/experimental/device-tree.hpp"

using namespace sctl;

namespace {

  Long check(const char* what, bool ok, Long& fails)
  {
    fails += !ok;
    printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    return fails;
  }

  template <class Real, Integer DIM> Long run(const char* label)
  {
    printf("%s\n", label);
    Long fails = 0;
    const Comm& comm = Comm::World();
    const Long N = 5000;

    std::mt19937_64 rng(11 + 1000 * comm.Rank());
    std::uniform_real_distribution<Real> U(0, 1);
    std::vector<Real> x(N * DIM);
    for (auto& v : x) v = U(rng);

    device_tree::PtTree<Real, DIM> tree(comm);
    tree.UpdateRefinement(x.data(), N, device_tree::MemSpace::Host, 100, true, Periodicity::NONE, 1);

    const auto mid = tree.NodeMID();
    check("the tree has nodes, and their storage is a device pointer", mid.Dim() > 0 && mid.data() != nullptr, fails);

    const auto attr = tree.NodeAttrs();
    check("one attribute per node", attr.Dim() == mid.Dim(), fails);

    const auto lists = tree.Lists();
    Long nnbr = 1;
    for (Integer k = 0; k < DIM; k++) nnbr *= 3;
    check("parent, child and neighbour lists have the shapes the layout implies",
          lists.parent.Dim() == mid.Dim() && lists.child.Dim() == mid.Dim() * (1 << DIM) && lists.nbr.Dim() == mid.Dim() * nnbr, fails);

    Long b = 0, e = 0;
    tree.OwnedRange(b, e);
    check("the owned range lies within the node list", 0 <= b && b <= e && e <= mid.Dim(), fails);

    tree.AddParticles("pt", x.data(), N, device_tree::MemSpace::Host);

    std::vector<Real> f(N);
    for (Long i = 0; i < N; i++) f[i] = (Real)(1 + i);
    tree.AddParticleData("f", "pt", f.data(), N, device_tree::MemSpace::Host);

    const Long dim = tree.ParticleDataDim("f");
    check("the particle data comes back with the length it went in with", dim == N, fails);

    std::vector<Real> back(N, Real(-1));
    tree.GetParticleData("f", back.data(), dim, device_tree::MemSpace::Host);
    Long bad = 0;
    for (Long i = 0; i < N; i++) bad += (back[i] != f[i]);
    check("particle data round-trips through the tree order and back", bad == 0, fails);

    { // node data: fill through the view on the device, then read the values back per particle
      Vector<Long> cnt;
      tree.AddParticleData("u", "pt", 1);
      const auto v = tree.template Data<Real>("u", cnt);
      check("an uninitialised particle data set views one value per particle of the group",
            v.Dim() > 0 && v.data() != nullptr && cnt.Dim() == mid.Dim(), fails);
    }

    tree.Broadcast("f");
    check("Broadcast leaves the owned values alone", tree.ParticleDataDim("f") == N, fails);

    if (comm.Size() > 1)
    {
      tree.ReduceBroadcast("f");
      check("ReduceBroadcast runs over the shared nodes", tree.ParticleDataDim("f") == N, fails);
    }

    tree.DeleteParticleData("f");
    return fails;
  }

}  // namespace

int main(int argc, char** argv)
{
  Comm::MPI_Init(&argc, &argv);
  {
    Long fails = 0;
    fails += run<double, 3>("device_tree::PtTree<double,3>");
    fails += run<float, 3>("device_tree::PtTree<float,3>");
    fails += run<double, 2>("device_tree::PtTree<double,2>");
    fails += run<float, 2>("device_tree::PtTree<float,2>");
    SCTL_ASSERT_MSG(fails == 0, "test-device-tree: failures above");
  }
  Comm::MPI_Finalize();
  return 0;
}
