// Per-function tests for sctl/morton.{hpp,txx}.
//
// sctl::Morton<DIM> is currently exercised indirectly via test-pt-tree (high-level Tree
// integration), test-tree-edge-periodic (Tree under periodic + balance21), and
// test-nodemid-vs-morton (compares against the gpu_tree::NodeMID port). This file
// adds a dedicated standalone unit test: every public member exercised on randomized
// inputs + boundary edge cases, verified via self-consistent invariants.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <sstream>

#include "sctl/common.hpp"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/vector.hpp"
#include "sctl/vector.txx"
#include "sctl/morton.hpp"
#include "sctl/morton.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::Integer;
using Real = double;
constexpr Integer DIM = 3;
using Morton = sctl::Morton<DIM>;

int main() {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<Real> U(0.0, 1.0);
  std::uniform_int_distribution<int>   Ud(0, Morton::MAX_DEPTH);

  // --- Default ctor: depth=0, coord components all zero ---
  std::printf("default ctor :\n");
  {
    Morton m;
    CHECK(m.Depth() == 0);
    std::array<Real, DIM> c;
    m.Coord(c);
    for (Integer d = 0; d < DIM; ++d) CHECK(c[d] == 0.0);
  }

  // --- MaxDepth() ---
  std::printf("MaxDepth :\n");
  {
    CHECK(Morton::MaxDepth() == Morton::MAX_DEPTH);
  }

  // --- INVALID_DEPTH sentinel ---
  std::printf("INVALID_DEPTH :\n");
  {
    CHECK(Morton::INVALID_DEPTH == (uint8_t)0xFF);
  }

  // --- Coord round-trip: build from coord+depth, read coord back ---
  // For valid depths and coords, ceiling(c*2^depth)/2^depth rounds to the truncated
  // representation; we check the rounded-integer-grid equality.
  std::printf("Coord round-trip :\n");
  {
    for (int t = 0; t < 200; ++t) {
      Real coord[DIM];
      for (Integer d = 0; d < DIM; ++d) coord[d] = U(rng);
      const uint8_t depth = (uint8_t)Ud(rng);
      Morton m(sctl::Ptr2ConstItr<Real>(coord, DIM), depth);

      CHECK(m.Depth() == depth);
      std::array<Real, DIM> c;
      m.Coord(c);
      // Both `coord` and `c` round to the same integer at the given depth.
      const std::uint64_t maxCoord = std::uint64_t(1) << Morton::MAX_DEPTH;
      const std::uint64_t mask     = ~(((std::uint64_t)1 << (Morton::MAX_DEPTH - depth)) - 1);
      for (Integer d = 0; d < DIM; ++d) {
        const std::uint64_t a_ix = (std::uint64_t)std::floor(coord[d] * maxCoord) & mask;
        const std::uint64_t b_ix = (std::uint64_t)std::floor(c[d]     * maxCoord) & mask;
        CHECK(a_ix == b_ix);
      }
    }
  }

  // --- Coord<Real>() std::array overload ---
  std::printf("Coord<Real>() :\n");
  {
    Real coord[DIM] = {0.5, 0.25, 0.75};
    Morton m(sctl::Ptr2ConstItr<Real>(coord, DIM), Morton::MAX_DEPTH);
    std::array<Real, DIM> c = m.Coord<Real>();
    for (Integer d = 0; d < DIM; ++d) CHECK(c[d] == coord[d]);
  }

  // --- Ancestor(level) :
  //   level == depth: returns self
  //   level <  depth: returns a coarser node containing self
  //   Ancestor∘Ancestor at coarser depth = Ancestor of the coarser depth.
  std::printf("Ancestor :\n");
  {
    Real coord[DIM] = {0.3, 0.6, 0.9};
    Morton m(sctl::Ptr2ConstItr<Real>(coord, DIM), Morton::MAX_DEPTH);
    Morton ma = m.Ancestor(Morton::MAX_DEPTH);
    CHECK(ma == m);
    Morton mc = m.Ancestor(0);  // root
    CHECK(mc.Depth() == 0);
    // Idempotency at the same coarser depth
    Morton mc1 = m.Ancestor(3);
    Morton mc2 = mc1.Ancestor(3);
    CHECK(mc1 == mc2);
  }

  // --- DFD : same code, depth set to `level`. No bit manipulation. ---
  std::printf("DFD :\n");
  {
    Real coord[DIM] = {0.1, 0.7, 0.3};
    Morton m(sctl::Ptr2ConstItr<Real>(coord, DIM), 5);
    Morton d = m.DFD(Morton::MAX_DEPTH);
    CHECK(d.Depth() == Morton::MAX_DEPTH);
    // After DFD at MAX_DEPTH the code is unchanged (no masking).
    std::array<Real, DIM> cm, cd;
    m.Coord(cm);
    d.Coord(cd);
    for (Integer i = 0; i < DIM; ++i) CHECK(cm[i] == cd[i]);
  }

  // --- Next : depth-aware successor in Morton order. Strictly greater except at
  //   the root where there is no successor (and at the top of the code space,
  //   where sctl saturates by setting a coord to maxCoord, putting it outside
  //   [0,1) -- a sentinel that we don't test directly). For depth > 0 and
  //   non-saturated inputs, Next > self.
  std::printf("Next :\n");
  {
    // Pick a coord well below the top so Next stays in-range.
    Real coord[DIM] = {0.1, 0.2, 0.3};
    Morton m(sctl::Ptr2ConstItr<Real>(coord, DIM), 5);
    Morton n = m.Next();
    CHECK(m < n);
    // Next at depth=0 may saturate (no representable successor); we only
    // check it doesn't crash.
    Morton root;
    Morton root_next = root.Next();
    (void)root_next;
    CHECK(true);
  }

  // --- Children : 2^DIM children at depth+1, each containing distinct sub-regions ---
  std::printf("Children :\n");
  {
    Morton root;
    sctl::Vector<Morton> children;
    root.Children(children);
    CHECK(children.Dim() == (1 << DIM));
    // All children are strict descendants of root, all at depth 1.
    for (Long i = 0; i < children.Dim(); ++i) {
      CHECK(children[i].Depth() == 1);
      CHECK(root.isAncestor(children[i]));
    }
    // All 2^DIM children are distinct
    std::set<Morton> uniq;
    for (Long i = 0; i < children.Dim(); ++i) uniq.insert(children[i]);
    CHECK((Long)uniq.size() == children.Dim());
  }

  // --- isAncestor : strict; descendant.depth > self.depth + descendant.Ancestor(self.depth) == self ---
  std::printf("isAncestor :\n");
  {
    Real coord[DIM] = {0.4, 0.4, 0.4};
    Morton root;
    Morton mid (sctl::Ptr2ConstItr<Real>(coord, DIM), 5);
    Morton leaf(sctl::Ptr2ConstItr<Real>(coord, DIM), Morton::MAX_DEPTH);
    CHECK(root.isAncestor(mid));
    CHECK(root.isAncestor(leaf));
    CHECK(mid.isAncestor(leaf));
    // Self is NOT its own (strict) ancestor.
    CHECK(!mid.isAncestor(mid));
    CHECK(!root.isAncestor(root));
    // A leaf isn't ancestor of mid.
    CHECK(!leaf.isAncestor(mid));
  }

  // --- Comparison operators total-order: matches reflexivity, antisymmetry, transitivity ---
  std::printf("comparison ops :\n");
  {
    Real ca[DIM] = {0.1, 0.2, 0.3};
    Real cb[DIM] = {0.5, 0.5, 0.5};
    Morton a(sctl::Ptr2ConstItr<Real>(ca, DIM), 8);
    Morton b(sctl::Ptr2ConstItr<Real>(cb, DIM), 8);
    CHECK(a < b);
    CHECK(b > a);
    CHECK(a <= a);
    CHECK(b >= b);
    CHECK(a != b);
    CHECK(a == a);
    CHECK(!(a > b));
    CHECK(!(b < a));
  }

  // --- operator- : -1 intersecting, 0 touching, >0 separated ---
  std::printf("operator- :\n");
  {
    // Self with self: intersecting.
    Real ca[DIM] = {0.3, 0.3, 0.3};
    Morton a(sctl::Ptr2ConstItr<Real>(ca, DIM), Morton::MAX_DEPTH);
    CHECK((a - a) == -1);
    // Two leaf cells at the opposite corners of the cube: separated.
    Real cb[DIM] = {0.99999, 0.99999, 0.99999};
    Morton b(sctl::Ptr2ConstItr<Real>(cb, DIM), Morton::MAX_DEPTH);
    CHECK((a - b) > 0);
  }

  // --- NbrList: 3^DIM entries. Periodic at level 0 wraps every neighbour to root;
  //   non-periodic at root has all "off-domain" neighbours flagged with depth=INVALID_DEPTH.
  std::printf("NbrList :\n");
  {
    Morton root;
    sctl::Vector<Morton> nbrs;
    root.NbrList(nbrs, /*level=*/0, /*periodic=*/sctl::all_periodic(DIM));
    Long pow3 = 1;
    for (Integer d = 0; d < DIM; ++d) pow3 *= 3;
    CHECK(nbrs.Dim() == pow3);
    // Periodic at level 0: every neighbour wraps to the root.
    for (Long i = 0; i < nbrs.Dim(); ++i) CHECK(nbrs[i] == root);

    sctl::Vector<Morton> nbrs2;
    root.NbrList(nbrs2, 0, /*periodic=*/sctl::Periodicity::NONE);
    CHECK(nbrs2.Dim() == pow3);
    // Non-periodic at root level 0: all off-axis offsets are out-of-bounds.
    // Only the self entry (at the centre index (3^DIM-1)/2) is valid.
    Long n_valid = 0, n_invalid = 0;
    for (Long i = 0; i < nbrs2.Dim(); ++i) {
      if (nbrs2[i].Depth() == Morton::INVALID_DEPTH) ++n_invalid;
      else ++n_valid;
    }
    CHECK(n_valid   == 1);
    CHECK(n_invalid == pow3 - 1);
  }

  // --- operator<< (smoke) ---
  std::printf("operator<< :\n");
  {
    Morton m;
    std::ostringstream os;
    os << m;
    CHECK(!os.str().empty());
  }

  TEST_SUMMARY_RETURN();
}
