#ifndef _SCTL_GMSH_READER_CPP_
#define _SCTL_GMSH_READER_CPP_

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sctl.hpp>
#include "sctl/experimental/gmsh_reader.hpp"
#include "sctl/experimental/quad_element.cpp"

namespace sctl {

  template <class Real> Integer GmshReader<Real>::QuadTypeToNside(Integer gmsh_type) {
    // Full-tensor Lagrange quadrangles: order p -> (p+1) nodes per side.
    switch (gmsh_type) {
      case 3:  return 2;  // 4-node   (Q1)
      case 10: return 3;  // 9-node   (Q2)
      case 36: return 4;  // 16-node  (Q3)
      case 37: return 5;  // 25-node  (Q4)
      case 38: return 6;  // 36-node  (Q5)
      case 47: return 7;  // 49-node  (Q6)
      case 48: return 8;  // 64-node  (Q7)
      case 49: return 9;  // 81-node  (Q8)
      case 50: return 10; // 100-node (Q9)
      default: return 0;  // not a full-tensor quad (incl. 8-node serendipity type 16)
    }
  }

  template <class Real> Vector<Long> GmshReader<Real>::GmshQuadTensorPerm(Integer nside) {
    // gmsh high-order quad node order: 4 corners (CCW), then the 4 edges
    // (each from lower- to higher-numbered principal vertex), then the
    // interior face nodes ordered the same way, recursively. Reference quad
    // corners map to (i,j) tensor indices v1=(0,0) v2=(hi,0) v3=(hi,hi) v4=(0,hi).
    SCTL_ASSERT(nside >= 2);
    Vector<Long> perm; // perm[gmsh_local_idx] = i*nside + j
    std::function<void(Integer, Integer)> fill = [&](Integer lo, Integer hi) {
      if (lo > hi) return;
      const auto push = [&](Integer i, Integer j) { perm.PushBack((Long)i * nside + j); };
      if (lo == hi) { push(lo, lo); return; }             // single center node
      push(lo, lo); push(hi, lo); push(hi, hi); push(lo, hi); // corners CCW
      for (Integer i = lo + 1; i < hi; i++) push(i, lo);   // bottom edge j=lo
      for (Integer j = lo + 1; j < hi; j++) push(hi, j);   // right edge  i=hi
      for (Integer i = hi - 1; i > lo; i--) push(i, hi);   // top edge    j=hi (reversed)
      for (Integer j = hi - 1; j > lo; j--) push(lo, j);   // left edge   i=lo (reversed)
      fill(lo + 1, hi - 1);                                // interior
    };
    fill(0, nside - 1);
    SCTL_ASSERT(perm.Dim() == (Long)nside * nside);
    return perm;
  }

  template <class Real> Long GmshReader<Real>::ReadQuadPatches(const std::string& fname, Integer& nside, Vector<Real>& src) {
    std::ifstream is(fname);
    SCTL_ASSERT_MSG(is.good(), (std::string("GmshReader: cannot open file ") + fname).c_str());

    // Read the next non-empty line and split into whitespace-separated tokens.
    std::string line;
    const auto next = [&](std::vector<std::string>& tok) -> bool {
      while (std::getline(is, line)) {
        std::istringstream ss(line);
        tok.clear();
        std::string t;
        while (ss >> t) tok.push_back(t);
        if (!tok.empty()) return true;
      }
      return false;
    };

    std::unordered_map<Long, std::array<Real, 3>> node_pos; // tag -> (x,y,z)
    std::vector<std::pair<Integer, std::vector<Long>>> quads; // (nside, gmsh-ordered node tags)
    std::unordered_map<Integer, Long> skipped_surf; // non-quad surface type -> count

    std::vector<std::string> tok;
    while (next(tok)) {
      if (tok[0] == "$MeshFormat") {
        std::vector<std::string> t;
        SCTL_ASSERT(next(t) && t.size() >= 3);
        const double ver = std::stod(t[0]);
        SCTL_ASSERT_MSG((int)std::stol(t[1]) == 0, "GmshReader: only ASCII MSH files are supported");
        SCTL_ASSERT_MSG(ver >= 4.0 && ver < 5.0, "GmshReader: only MSH 4.x format is supported");
      } else if (tok[0] == "$Nodes") {
        std::vector<std::string> h;
        SCTL_ASSERT(next(h) && h.size() >= 4); // numEntityBlocks numNodes minTag maxTag
        const Long nblk = std::stol(h[0]);
        for (Long b = 0; b < nblk; b++) {
          std::vector<std::string> bh;
          SCTL_ASSERT(next(bh) && bh.size() >= 4); // entityDim entityTag parametric numNodesInBlock
          const Long nb = std::stol(bh[3]);
          std::vector<Long> tags(nb);
          for (Long i = 0; i < nb; i++) { std::vector<std::string> t; SCTL_ASSERT(next(t)); tags[i] = std::stol(t[0]); }
          for (Long i = 0; i < nb; i++) {
            std::vector<std::string> t;
            SCTL_ASSERT(next(t) && t.size() >= 3);
            node_pos[tags[i]] = {{(Real)std::stod(t[0]), (Real)std::stod(t[1]), (Real)std::stod(t[2])}};
          }
        }
      } else if (tok[0] == "$Elements") {
        std::vector<std::string> h;
        SCTL_ASSERT(next(h) && h.size() >= 4); // numEntityBlocks numElements minTag maxTag
        const Long nblk = std::stol(h[0]);
        for (Long b = 0; b < nblk; b++) {
          std::vector<std::string> bh;
          SCTL_ASSERT(next(bh) && bh.size() >= 4); // entityDim entityTag elementType numElementsInBlock
          const Integer dim = (Integer)std::stol(bh[0]);
          const Integer etype = (Integer)std::stol(bh[2]);
          const Long nb = std::stol(bh[3]);
          const Integer ns = QuadTypeToNside(etype);
          std::cout << "GmshReader CHECK: Number of Lagrange nodes on each side ( = order + 1): " << ns << std::endl;
          for (Long e = 0; e < nb; e++) {
            std::vector<std::string> t;
            SCTL_ASSERT(next(t)); // elemTag node1 node2 ...
            if (ns > 0) {
              const Long nnode = (Long)ns * ns;
              SCTL_ASSERT_MSG((Long)t.size() == nnode + 1, "GmshReader: quad element node count mismatch");
              std::vector<Long> nodes(nnode);
              for (Long k = 0; k < nnode; k++) nodes[k] = std::stol(t[1 + k]);
              quads.emplace_back(ns, std::move(nodes));
            } else if (dim == 2) {
              skipped_surf[etype]++;
            }
          }
        }
      }
    }

    for (const auto& kv : skipped_surf) {
      std::cout << "GmshReader: warning: skipped " << kv.second << " non-quad surface element(s) of gmsh type " << kv.first << "\n";
    }
    SCTL_ASSERT_MSG(!quads.empty(), "GmshReader: no quadrilateral elements found");

    // All quads must share the same order (QuadElemList requires uniform order).
    nside = quads[0].first;
    for (const auto& q : quads) SCTL_ASSERT_MSG(q.first == nside, "GmshReader: mesh mixes quad orders; not supported");

    const Vector<Long> perm = GmshQuadTensorPerm(nside); // gmsh-local -> tensor flat i*nside+j
    const Long nn = (Long)nside * nside;
    const Long nquad = (Long)quads.size();
    src.ReInit(nquad * nn * 3);
    for (Long e = 0; e < nquad; e++) {
      const std::vector<Long>& nodes = quads[e].second;
      for (Long g = 0; g < nn; g++) { // gmsh-local index g -> tensor slot perm[g]
        const auto it = node_pos.find(nodes[g]);
        SCTL_ASSERT_MSG(it != node_pos.end(), "GmshReader: element references unknown node tag");
        const std::array<Real, 3>& X = it->second;
        const Long slot = (e * nn + perm[g]) * 3;
        src[slot + 0] = X[0]; src[slot + 1] = X[1]; src[slot + 2] = X[2];
      }
    }
    return nquad;
  }

  template <class Real> Vector<Real> GmshReader<Real>::ReadQuadCoord(const std::string& fname, Integer target_order) {
    SCTL_ASSERT(target_order > 0);
    Integer nside = 0;
    Vector<Real> src;
    const Long nquad = ReadQuadPatches(fname, nside, src);
    const Long nn_src = (Long)nside * nside;
    const Long N = target_order;
    const Long nn_trg = N * N;

    // 1D interpolation: equispaced source params (nside pts on [0,1]) -> GL target (ParamNodes).
    Vector<Real> eq(nside);
    for (Integer k = 0; k < nside; k++) eq[k] = (nside == 1) ? (Real)0.5 : (Real)k / (Real)(nside - 1);
    const Vector<Real>& gl = QuadElemList<Real>::ParamNodes(target_order);
    Matrix<Real> Mu(nside, N); // Mu(i,a): row-major nside x N
    { Vector<Real> v(nside * N, Mu.begin(), false); LagrangeInterp<Real>::Interpolate(v, eq, gl); }
    const Matrix<Real> MuT = Mu.Transpose(); // (N x nside)
    const Matrix<Real>& Mv = Mu;             // same rule in v

    Vector<Real> coord(nquad * nn_trg * 3);
    Matrix<Real> srcM(nside, nside), tmp(nside, N), outM(N, N);
    for (Long e = 0; e < nquad; e++) {
      for (Integer k = 0; k < 3; k++) {
        for (Long t = 0; t < nn_src; t++) srcM.begin()[t] = src[(e * nn_src + t) * 3 + k]; // (i,j) -> i*nside+j (contiguous row-major)
        Matrix<Real>::GEMM(tmp, srcM, Mv);   // (nside x nside).(nside x N) = (nside x N)
        Matrix<Real>::GEMM(outM, MuT, tmp);  // (N x nside).(nside x N) = (N x N)
        for (Long p = 0; p < nn_trg; p++) coord[(e * nn_trg + p) * 3 + k] = outM.begin()[p]; // p = a*N + b, u-slow
      }
    }
    return coord;
  }

  template <class Real> QuadElemList<Real> GmshReader<Real>::LoadQuadElemList(const std::string& fname, Integer target_order, const Comm& comm) {
    return QuadElemList<Real>(target_order, ReadQuadCoord(fname, target_order), comm);
  }

}

#endif // _SCTL_GMSH_READER_CPP_
