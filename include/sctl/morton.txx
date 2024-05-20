#ifndef _SCTL_MORTON_TXX_
#define _SCTL_MORTON_TXX_

#include <ostream>              // for ostream
#include <stdlib.h>             // for abs
#include <algorithm>            // for max
#include <array>                // for array
#include <cstdint>              // for uint8_t, int8_t
#include <ostream>              // for basic_ostream, char_traits, operator<<
#include <type_traits>          // for remove_reference

#include "sctl/common.hpp"      // for Integer, Long, SCTL_ASSERT, SCTL_NAME...
#include SCTL_INCLUDE(morton.hpp)      // for Morton
#include SCTL_INCLUDE(iterator.hpp)    // for ConstIterator
#include SCTL_INCLUDE(math_utils.hpp)  // for floor
#include SCTL_INCLUDE(math_utils.txx)  // for pow
#include SCTL_INCLUDE(tree.hpp)        // for Morton
#include SCTL_INCLUDE(vector.hpp)      // for Vector

namespace SCTL_NAMESPACE {

  template <Integer DIM> constexpr Integer Morton<DIM>::MaxDepth() {
    return MAX_DEPTH;
  }

  template <Integer DIM> Morton<DIM>::Morton() {
    depth = 0;
    for (Integer i = 0; i < DIM; i++) x[i] = 0;
  }

  template <Integer DIM> template <class T> Morton<DIM>::Morton(ConstIterator<T> coord, uint8_t depth_) {
    depth = depth_;
    SCTL_ASSERT(depth <= MAX_DEPTH);
    UINT_T mask = ~((((UINT_T)1) << (MAX_DEPTH - depth)) - 1);
    for (Integer i = 0; i < DIM; i++) x[i] = mask & (UINT_T)floor((double)coord[i] * maxCoord);
  }

  template <Integer DIM> int8_t Morton<DIM>::Depth() const {
    return depth;
  }

  template <Integer DIM> template <class ArrayType> void Morton<DIM>::Coord(ArrayType&& coord) const {
    using Real = typename std::remove_reference<decltype(coord[0])>::type;
    static const Real factor = 1.0 / (Real)maxCoord;
    for (Integer i = 0; i < DIM; i++) coord[i] = (Real)x[i] * factor;
  }
  template <Integer DIM> template <class Real> std::array<Real,DIM> Morton<DIM>::Coord() const {
    std::array<Real,DIM> x_real;
    Coord(x_real);
    return x_real;
  }

  template <Integer DIM> Morton<DIM> Morton<DIM>::Next() const {
    UINT_T mask = ((UINT_T)1) << (MAX_DEPTH - depth);
    Integer d, i;

    Morton m = *this;
    for (d = depth; d >= 0; d--) {
      for (i = 0; i < DIM; i++) {
        m.x[i] = (m.x[i] ^ mask);
        if ((m.x[i] & mask)) break;
      }
      if (i < DIM) break;
      mask = (mask << 1);
    }

    if (d < 0) d = 0;
    m.depth = (uint8_t)d;

    return m;
  }

  template <Integer DIM> Morton<DIM> Morton<DIM>::Ancestor(uint8_t ancestor_level) const {
    UINT_T mask = ~((((UINT_T)1) << (MAX_DEPTH - ancestor_level)) - 1);

    Morton m;
    for (Integer i = 0; i < DIM; i++) m.x[i] = x[i] & mask;
    m.depth = ancestor_level;
    return m;
  }

  template <Integer DIM> Morton<DIM> Morton<DIM>::DFD(uint8_t level) const {
    Morton m = *this;
    m.depth = level;
    return m;
  }

  template <Integer DIM> void Morton<DIM>::NbrList(Vector<Morton>& nbrs, uint8_t level, bool periodic) const {
    static constexpr Integer MAX_NBRS = sctl::pow<DIM,Integer>(3);
    if (nbrs.Dim() != MAX_NBRS) nbrs.ReInit(MAX_NBRS);

    const UINT_T box_size = (((UINT_T)1) << (MAX_DEPTH - level));
    const UINT_T mask = ~(box_size - 1);

    for (Integer i = 0; i < DIM; i++) nbrs[0].x[i] = x[i] & mask;
    nbrs[0].depth = level;
    Integer Nnbrs = 1;

    if (periodic) {
      constexpr UINT_T mask0 = (maxCoord - 1);
      for (Integer i = 0; i < DIM; i++) {
        for (Integer j = 0; j < Nnbrs; j++) {
          const auto m0 = nbrs[j];
          auto& m1 = nbrs[0*Nnbrs+j];
          auto& m2 = nbrs[1*Nnbrs+j];
          auto& m3 = nbrs[2*Nnbrs+j];
          m1 = m0;
          m2 = m0;
          m3 = m0;
          m1.x[i] = (m0.x[i] - box_size) & mask0;
          m2.x[i] = (m0.x[i]           ) & mask0;
          m3.x[i] = (m0.x[i] + box_size) & mask0;
        }
        Nnbrs *= 3;
      }
    } else {
      constexpr UINT_T mask0 = (maxCoord - 1);
      for (Integer i = 0; i < DIM; i++) {
        for (Integer j = 0; j < Nnbrs; j++) {
          const auto m0 = nbrs[j];
          auto& m1 = nbrs[0*Nnbrs+j];
          auto& m2 = nbrs[1*Nnbrs+j];
          auto& m3 = nbrs[2*Nnbrs+j];
          m1 = m0;
          m2 = m0;
          m3 = m0;
          m1.x[i] = (m0.x[i] - box_size) & mask0;
          m2.x[i] = (m0.x[i]           ) & mask0;
          m3.x[i] = (m0.x[i] + box_size) & mask0;
          if (m0.x[i] < box_size) m1.depth = -1;
          if (m0.x[i] + box_size >= maxCoord) m3.depth = -1;
        }
        Nnbrs *= 3;
      }
    }
  }

  template <Integer DIM> void Morton<DIM>::Children(Vector<Morton> &nlst) const {
    static const Integer cnt = (1UL << DIM);
    if (nlst.Dim() != cnt) nlst.ReInit(cnt);

    for (Integer i = 0; i < DIM; i++) nlst[0].x[i] = x[i];
    nlst[0].depth = (uint8_t)(depth + 1);

    Integer k = 1;
    UINT_T mask = (((UINT_T)1) << (MAX_DEPTH - (depth + 1)));
    for (Integer i = 0; i < DIM; i++) {
      for (Integer j = 0; j < k; j++) {
        nlst[j + k] = nlst[j];
        nlst[j + k].x[i] += mask;
      }
      k = (k << 1);
    }
  }

  template <Integer DIM> bool Morton<DIM>::operator<(const Morton &m) const {
    UINT_T diff = 0;
    for (Integer i = 0; i < DIM; i++) diff = diff | (x[i] ^ m.x[i]);
    if (!diff) return depth < m.depth;

    UINT_T mask = 1;
    for (Integer i = 4 * sizeof(UINT_T); i > 0; i = (i >> 1)) {
      UINT_T mask_ = (mask << i);
      if (mask_ <= diff) mask = mask_;
    }

    for (Integer i = DIM - 1; i >= 0; i--) {
      if (mask & (x[i] ^ m.x[i])) return x[i] < m.x[i];
    }
    return false; // TODO: check
  }

  template <Integer DIM> bool Morton<DIM>::operator>(const Morton &m) const {
    return m < (*this);
  }

  template <Integer DIM> bool Morton<DIM>::operator!=(const Morton &m) const {
    for (Integer i = 0; i < DIM; i++)
      if (x[i] != m.x[i]) return true;
    return (depth != m.depth);
  }

  template <Integer DIM> bool Morton<DIM>::operator==(const Morton &m) const {
    return !(*this != m);
  }

  template <Integer DIM> bool Morton<DIM>::operator<=(const Morton &m) const {
    return !(*this > m);
  }

  template <Integer DIM> bool Morton<DIM>::operator>=(const Morton &m) const {
    return !(*this < m);
  }

  template <Integer DIM> bool Morton<DIM>::isAncestor(Morton const &descendant) const {
    return descendant.depth > depth && descendant.Ancestor(depth) == *this;
  }

  template <Integer DIM> Long Morton<DIM>::operator-(const Morton<DIM> &I) const {
    // Intersecting -1
    // Touching 0

    Long offset0 = 1 << (MAX_DEPTH - depth - 1);
    Long offset1 = 1 << (MAX_DEPTH - I.depth - 1);

    Long diff = 0;
    for (Integer i = 0; i < DIM; i++) {
      diff = std::max<Long>(diff, abs(((Long)x[i] + offset0) - ((Long)I.x[i] + offset1)));
    }
    if (diff < offset0 + offset1) return -1;
    Integer max_depth = std::max(depth, I.depth);
    diff = (diff - offset0 - offset1) >> (MAX_DEPTH - max_depth);
    return diff;
  }

  template <Integer DIM> std::ostream& operator<<(std::ostream &out, const Morton<DIM> &mid) {
    double a = 0;
    double s = 1u << DIM;
    for (Integer j = Morton<DIM>::MAX_DEPTH; j >= 0; j--) {
      for (Integer i = DIM - 1; i >= 0; i--) {
        s = s * 0.5;
        if (mid.x[i] & (((typename Morton<DIM>::UINT_T)1) << j)) a += s;
      }
    }
    out << "(";
    for (Integer i = 0; i < DIM; i++) {
      out << mid.x[i] * 1.0 / Morton<DIM>::maxCoord << ",";
    }
    out << (int)mid.depth << "," << a << ")";
    return out;
  }

}

#endif // _SCTL_MORTON_TXX_
