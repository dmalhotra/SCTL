// Verifies the backend dispatch of GPUTree and PtTree:
//   thrust::device_vector  -> GPU path (thrust)
//   std::vector            -> CPU path (omp_par, chunked walk)
// Runs each path on N random particles and checks that the node sequence is sorted in (code, depth)
// lex order and that AddParticles + GetParticleData round-trips the particle order, which is the
// sort's permutation.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 20
#endif

#include "sctl/experimental/gpu-tree.hpp"

using Real    = double;
constexpr int kDim = 3;
using NodeMID = sctl::Morton<kDim>;
using GPUTree = gpu_tree::GPUTree<Real, kDim>;
using Long    = gpu_tree::Long;

template <class V> double ms(V t0, V t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Leaf nodes are sorted lex by (code, depth).
static bool leaves_sorted(const std::vector<NodeMID>& v) {
  for (size_t i = 1; i < v.size(); ++i) {
    const NodeMID& a = v[i - 1];
    const NodeMID& b = v[i];
    if (b.mid < a.mid) return false;
    const bool codes_equal = !(a.mid < b.mid) && !(b.mid < a.mid);
    if (codes_equal && a.depth > b.depth) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  Long N = (argc > 1) ? std::stoll(argv[1]) : 1'000'000;
  const Long M = 4;

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<Real> U(0.0, 1.0);
  std::vector<Real> coord_h(N * kDim);
  for (Real& x : coord_h) x = U(rng);

  // --- CPU path ---------------------------------------------------------
  {
    std::vector<Real> coord = coord_h;
    GPUTree tr(sctl::Comm::Self());
    gpu_tree::PtTree<Real, kDim, std::vector> pt(sctl::Comm::Self());

    auto t0 = std::chrono::steady_clock::now();
    tr.UpdateRefinement(coord, M);
    auto t1 = std::chrono::steady_clock::now();
    pt.UpdateRefinement(coord, M);
    pt.AddParticles("pt", coord);
    std::vector<Real> back;
    pt.GetParticleData(back, "pt");
    auto t2 = std::chrono::steady_clock::now();

    const bool sorted_ok = leaves_sorted(tr.GetNodeMID());
    const bool perm_ok = (back == coord);
    std::printf("CPU (std::vector):  UpdateRefinement          %.2f ms  sorted=%s\n", ms(t0, t1), sorted_ok ? "ok" : "FAIL");
    std::printf("CPU (std::vector):  AddParticles + round trip %.2f ms  permutation=%s\n", ms(t1, t2), perm_ok ? "ok" : "FAIL");
  }

  // --- GPU path ---------------------------------------------------------
  {
    thrust::device_vector<Real> coord(coord_h.begin(), coord_h.end());
    gpu_tree::GPUTree<Real, kDim, thrust::device_vector> tr(sctl::Comm::Self());
    gpu_tree::PtTree<Real, kDim, thrust::device_vector> pt(sctl::Comm::Self());

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    tr.UpdateRefinement(coord, M);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    pt.UpdateRefinement(coord, M);
    pt.AddParticles("pt", coord);
    thrust::device_vector<Real> back;
    pt.GetParticleData(back, "pt");
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    std::vector<NodeMID> tree_h(tr.GetNodeMID().size());
    thrust::copy(tr.GetNodeMID().begin(), tr.GetNodeMID().end(), tree_h.begin());
    const bool sorted_ok = leaves_sorted(tree_h);
    const bool perm_ok = (back.size() == coord.size()) && thrust::equal(back.begin(), back.end(), coord.begin());
    std::printf("GPU (device_vector): UpdateRefinement          %.2f ms  sorted=%s\n", ms(t0, t1), sorted_ok ? "ok" : "FAIL");
    std::printf("GPU (device_vector): AddParticles + round trip %.2f ms  permutation=%s\n", ms(t1, t2), perm_ok ? "ok" : "FAIL");
  }

  sctl::Comm::MPI_Finalize();
  return 0;
}
