/**
 * @file morton.txx
 * Template implementations of MortonCode and Morton from morton-code.hpp.
 */

#ifndef _SCTL_MORTON_TXX_
#define _SCTL_MORTON_TXX_

#include <cstdint>  // for uint8_t, uint64_t

// libstdc++ marks std::count{l,r}_zero as __host__-only constexpr, so the C++20 path is host-only;
// nvcc/hipcc keep using the GCC builtins, which work in both host and device code.
#if __cplusplus >= 202002L && !defined(__CUDACC__) && !defined(__HIPCC__)
#define SCTL_MORTON_CODE_USE_STD_BITOPS 1
#include <bit>  // for std::countl_zero, std::countr_zero
#endif

#include "sctl/morton.hpp"
#include "sctl/vector.hpp"  // for Vector (legacy NbrList/Children overloads)

namespace sctl {

namespace detail {

// __builtin_clzll(0) / __builtin_ctzll(0) are UB; callers must guard against zero.
SCTL_GPU_HD inline int clzll(std::uint64_t x) {
#ifdef SCTL_MORTON_CODE_USE_STD_BITOPS
  return std::countl_zero(x);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __clzll(static_cast<long long>(x));
#else
  return __builtin_clzll(x);
#endif
}

SCTL_GPU_HD inline int ctzll(std::uint64_t x) {
#ifdef SCTL_MORTON_CODE_USE_STD_BITOPS
  return std::countr_zero(x);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __ffsll(static_cast<long long>(x)) - 1;
#else
  return __builtin_ctzll(x);
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// MortonCode::MortonBig<NWORDS_>
// ---------------------------------------------------------------------------

template <Integer DIM> template <int NWORDS_> constexpr bool MortonCode<DIM>::MortonBig<NWORDS_>::operator<(const MortonBig& o) const {
  for (int i = NWORDS_ - 1; i >= 0; --i)
    if (w[i] != o.w[i]) return w[i] < o.w[i];
  return false;
}

template <Integer DIM> template <int NWORDS_> constexpr bool MortonCode<DIM>::MortonBig<NWORDS_>::operator==(const MortonBig& o) const {
  for (int i = 0; i < NWORDS_; ++i)
    if (w[i] != o.w[i]) return false;
  return true;
}

template <Integer DIM> template <int NWORDS_> constexpr typename MortonCode<DIM>::template MortonBig<NWORDS_> MortonCode<DIM>::MortonBig<NWORDS_>::operator|(const MortonBig& o) const {
  MortonBig r{};
  for (int i = 0; i < NWORDS_; ++i) r.w[i] = w[i] | o.w[i];
  return r;
}

template <Integer DIM> template <int NWORDS_> constexpr typename MortonCode<DIM>::template MortonBig<NWORDS_> MortonCode<DIM>::MortonBig<NWORDS_>::operator&(const MortonBig& o) const {
  MortonBig r{};
  for (int i = 0; i < NWORDS_; ++i) r.w[i] = w[i] & o.w[i];
  return r;
}

template <Integer DIM> template <int NWORDS_> constexpr typename MortonCode<DIM>::template MortonBig<NWORDS_> MortonCode<DIM>::MortonBig<NWORDS_>::operator^(const MortonBig& o) const {
  MortonBig r{};
  for (int i = 0; i < NWORDS_; ++i) r.w[i] = w[i] ^ o.w[i];
  return r;
}

template <Integer DIM> template <int NWORDS_> constexpr typename MortonCode<DIM>::template MortonBig<NWORDS_> MortonCode<DIM>::MortonBig<NWORDS_>::operator+(const MortonBig& o) const {
  MortonBig r{};
  std::uint64_t carry = 0;
  for (int i = 0; i < NWORDS_; ++i) {
    const std::uint64_t s1 = w[i] + o.w[i];
    const std::uint64_t c1 = (s1 < w[i]) ? 1u : 0u;
    const std::uint64_t s2 = s1 + carry;
    const std::uint64_t c2 = (s2 < s1) ? 1u : 0u;
    r.w[i] = s2;
    carry = c1 + c2;
  }
  return r;
}

template <Integer DIM> template <int NWORDS_> constexpr typename MortonCode<DIM>::template MortonBig<NWORDS_>& MortonCode<DIM>::MortonBig<NWORDS_>::operator|=(const MortonBig& o) {
  for (int i = 0; i < NWORDS_; ++i) w[i] |= o.w[i];
  return *this;
}

template <Integer DIM> template <int NWORDS_> constexpr typename MortonCode<DIM>::template MortonBig<NWORDS_> MortonCode<DIM>::MortonBig<NWORDS_>::operator<<(int s) const {
  MortonBig r{};
  const int ws = s / 64, bs = s % 64;
  for (int i = NWORDS_ - 1; i >= 0; --i) {
    std::uint64_t v = 0;
    if (i - ws >= 0) v |= w[i - ws] << bs;
    if (bs != 0 && i - ws - 1 >= 0) v |= w[i - ws - 1] >> (64 - bs);
    r.w[i] = v;
  }
  return r;
}

template <Integer DIM> template <int NWORDS_> constexpr typename MortonCode<DIM>::template MortonBig<NWORDS_> MortonCode<DIM>::MortonBig<NWORDS_>::operator>>(int s) const {
  MortonBig r{};
  const int ws = s / 64, bs = s % 64;
  for (int i = 0; i < NWORDS_; ++i) {
    std::uint64_t v = 0;
    if (i + ws < NWORDS_) v |= w[i + ws] >> bs;
    if (bs != 0 && i + ws + 1 < NWORDS_) v |= w[i + ws + 1] << (64 - bs);
    r.w[i] = v;
  }
  return r;
}

// ---------------------------------------------------------------------------
// MortonCode
// ---------------------------------------------------------------------------

template <Integer DIM> SCTL_GPU_HD int MortonCode<DIM>::highest_bit_pos(const MortonInteger& x) {
  if constexpr (STORAGE_BITS <= 64) {
    const std::uint64_t v = static_cast<std::uint64_t>(x);
    if (v == 0) return -1;
    return 63 - detail::clzll(v);
  } else {
    for (int i = NWORDS - 1; i >= 0; --i)
      if (x.w[i] != 0) return i * 64 + (63 - detail::clzll(x.w[i]));
    return -1;
  }
}

template <Integer DIM> SCTL_GPU_HD uint8_t MortonCode<DIM>::coarsest_depth(const MortonInteger& c) {
  int tzs;
  if constexpr (STORAGE_BITS <= 64) {
    const std::uint64_t v = static_cast<std::uint64_t>(c);
    if (v == 0) return 0;
    tzs = detail::ctzll(v);
  } else {
    tzs = -1;
    for (int i = 0; i < NWORDS; ++i)
      if (c.w[i] != 0) {
        tzs = i * 64 + detail::ctzll(c.w[i]);
        break;
      }
    if (tzs < 0) return 0;
  }
  return static_cast<uint8_t>(MAX_DEPTH - tzs / DIM);
}

// log-step doubling: at each Step, r <- (r | (r << (DIM-1)*2^Step)) & mask_Step, where mask_Step keeps
// bit p iff (p mod (DIM*2^Step)) < 2^Step. Bottoms out at Step == -1.
template <Integer DIM> template <Integer Step> SCTL_GPU_HD typename MortonCode<DIM>::MortonInteger MortonCode<DIM>::spread_step(MortonInteger r) {
  if constexpr (Step >= 0) {
    constexpr int shift = static_cast<int>((DIM - 1) * (Integer(1) << Step));
    constexpr MortonInteger mask = [] {
      MortonInteger m{};
      constexpr Integer total_bits = DIM * MAX_DEPTH;
      constexpr Integer stride     = DIM * (Integer(1) << Step);
      constexpr Integer keep       =       (Integer(1) << Step);
      for (Integer p = 0; p < total_bits; ++p)
        if ((p % stride) < keep) m |= MortonInteger(1) << static_cast<int>(p);
      return m;
    }();
    r = (r | (r << shift)) & mask;
    return spread_step<Step - 1>(r);
  } else {
    return r;
  }
}

template <Integer DIM> SCTL_GPU_HD typename MortonCode<DIM>::MortonInteger MortonCode<DIM>::spread_bits(std::uint64_t xi) {
  constexpr Integer NumSteps = [] {
    Integer l = 0;
    while ((Integer(1) << l) < MAX_DEPTH) ++l;
    return l;
  }();
  return spread_step<NumSteps - 1>(static_cast<MortonInteger>(xi));
}

// Inverse of `spread_step`. At each Step (running from 0 up to NumSteps-1), pulls bits paired by the
// corresponding spread step back together: r <- (r | (r >> (DIM-1)*2^Step)) & mask_Step, where
// mask_Step keeps bit p iff (p mod (DIM*2^(Step+1))) < 2^(Step+1). Bottoms out when Step == NumSteps.
template <Integer DIM> template <Integer Step> SCTL_GPU_HD typename MortonCode<DIM>::MortonInteger MortonCode<DIM>::compact_step(MortonInteger r) {
  constexpr Integer NumSteps = [] {
    Integer l = 0;
    while ((Integer(1) << l) < MAX_DEPTH) ++l;
    return l;
  }();
  if constexpr (Step < NumSteps) {
    constexpr int shift = static_cast<int>((DIM - 1) * (Integer(1) << Step));
    constexpr MortonInteger mask = [] {
      MortonInteger m{};
      constexpr Integer total_bits = DIM * MAX_DEPTH;
      constexpr Integer stride     = DIM * (Integer(1) << (Step + 1));
      constexpr Integer keep       =       (Integer(1) << (Step + 1));
      for (Integer p = 0; p < total_bits; ++p)
        if ((p % stride) < keep) m |= MortonInteger(1) << static_cast<int>(p);
      return m;
    }();
    r = (r | (r >> shift)) & mask;
    return compact_step<Step + 1>(r);
  } else {
    return r;
  }
}

template <Integer DIM> SCTL_GPU_HD std::uint64_t MortonCode<DIM>::compact_bits(MortonInteger code, Integer d) {
  // Shift coord d's bits into the "stride DIM, keep 1" positions, mask, then compact.
  constexpr MortonInteger initial_mask = [] {
    MortonInteger m{};
    for (Integer p = 0; p < MAX_DEPTH * DIM; p += DIM) m |= MortonInteger(1) << static_cast<int>(p);
    return m;
  }();
  MortonInteger r = (code >> static_cast<int>(d)) & initial_mask;
  r = compact_step<0>(r);
  // The compacted xi occupies the low MAX_DEPTH bits; safe to cast to uint64_t (MAX_DEPTH < 64).
  if constexpr (STORAGE_BITS <= 64) {
    return static_cast<std::uint64_t>(r);
  } else {
    return r.w[0];
  }
}

// Interleave DIM per-axis ints into a code, unrolled over `d` (force-inlined for nbr_emit_).
template <Integer DIM> template <Integer d>
[[gnu::always_inline]] inline SCTL_GPU_HD typename MortonCode<DIM>::MortonInteger MortonCode<DIM>::interleave(const std::uint64_t* xi) {
  if constexpr (d < DIM) {
    return static_cast<MortonInteger>(spread_bits(xi[d]) << static_cast<int>(d)) | interleave<d + 1>(xi);
  } else {
    return MortonInteger(0);
  }
}

// Ctor body: clamp coords to [0,1), scale to ints, interleave. Plain inline, not the force-inlined
// integer overload above (forcing it regressed gcc's DIM=4 ctor ~3x).
template <Integer DIM> template <class Real, class>
SCTL_GPU_HD typename MortonCode<DIM>::MortonInteger MortonCode<DIM>::interleave(const Real* coord) {
  // 2^MAX_DEPTH in u64 to avoid overflow when MortonInteger is exactly MAX_DEPTH bits wide.
  constexpr std::uint64_t max_coord_u64 = std::uint64_t(1) << MAX_DEPTH;
  constexpr std::uint64_t max_xi = max_coord_u64 - 1;
  const Real scale = static_cast<Real>(max_coord_u64);

  MortonInteger code{};
  for (Integer d = 0; d < DIM; ++d) {
    Real c = coord[d];
    if (!(c > Real(0))) c = Real(0);  // also handles NaN
    if (c >= Real(1))   c = Real(1);

    std::uint64_t xi = static_cast<std::uint64_t>(c * scale);
    if (xi > max_xi) xi = max_xi;  // clamp on round-up at the upper edge

    code |= spread_bits(xi) << static_cast<int>(d);
  }
  return code;
}

template <Integer DIM> template <class Real> SCTL_GPU_HD MortonCode<DIM>::MortonCode(const Real* coord) : code(interleave(coord)) {}

template <Integer DIM> SCTL_GPU_HD bool MortonCode<DIM>::operator<(const MortonCode& other) const {
  return code < other.code;
}

template <Integer DIM> SCTL_GPU_HD Morton<DIM> MortonCode<DIM>::CommonAncestor(const MortonCode& other) const {
  const MortonInteger diff = code ^ other.code;
  const int p = highest_bit_pos(diff);
  if (p < 0) return Morton<DIM>{*this, static_cast<uint8_t>(MAX_DEPTH)};  // codes identical
  const uint8_t d = static_cast<uint8_t>((TOTAL_BITS - 1 - p) / DIM);
  const int k = TOTAL_BITS - static_cast<int>(d) * static_cast<int>(DIM);
  // shift-by-TOTAL_BITS on built-in MortonInteger is UB; guard with k < TOTAL_BITS (only triggers at d == 0).
  const MortonInteger anc_code = (k < TOTAL_BITS) ? ((code >> k) << k) : MortonInteger{};
  return Morton<DIM>{MortonCode(anc_code), d};
}

template <Integer DIM> SCTL_GPU_HD Morton<DIM> MortonCode<DIM>::Ancestor(uint8_t depth) const {
  const int k = TOTAL_BITS - static_cast<int>(depth) * static_cast<int>(DIM);
  const MortonInteger anc_code = (k < TOTAL_BITS) ? ((code >> k) << k) : MortonInteger{};
  return Morton<DIM>{MortonCode(anc_code), depth};
}

// ---------------------------------------------------------------------------
// Morton
// ---------------------------------------------------------------------------

template <Integer DIM> template <class T> Morton<DIM>::Morton(ConstIterator<T> coord, uint8_t depth_) {
  T c[DIM];
  for (Integer i = 0; i < DIM; ++i) c[i] = coord[i];
  *this = MortonCode<DIM>(static_cast<const T*>(c)).Ancestor(depth_);
}

template <Integer DIM> SCTL_GPU_HD uint8_t Morton<DIM>::Depth() const {
  return depth;
}

// De-interleaves the code and rescales to [0,1) in the element type of `coord`.
template <Integer DIM> template <class ArrayType> SCTL_GPU_HD void Morton<DIM>::Coord(ArrayType&& coord) const {
  constexpr std::uint64_t maxCoord = std::uint64_t(1) << MAX_DEPTH;
  using ElemT = typename std::remove_reference<decltype(coord[0])>::type;
  const ElemT factor = ElemT(1) / static_cast<ElemT>(maxCoord);
  for (Integer d = 0; d < DIM; ++d) {
    const std::uint64_t xi = MortonCode<DIM>::compact_bits(mid.code, d);
    coord[d] = static_cast<ElemT>(xi) * factor;
  }
}

template <Integer DIM> SCTL_GPU_HD Morton<DIM> Morton<DIM>::Next() const {
  // At the root, `k = DIM * MAX_DEPTH` so the increment sets bit `TOTAL_BITS` — the
  // first bit of the extra-level storage (`STORAGE_BITS = TOTAL_BITS + DIM`). This
  // produces a "past-end" sentinel that sorts strictly greater than every valid Morton,
  // matching the way `sctl::Tree::UpdateRefinement` (and friends) use
  // `Morton<DIM>().Next()` as a `+infinity` upper bound in `std::lower_bound` partitioning.
  using MortonInteger = typename MortonCode<DIM>::MortonInteger;
  const int k = static_cast<int>(DIM) * (static_cast<int>(MAX_DEPTH) - static_cast<int>(depth));
  const MortonInteger new_code = mid.code + (MortonInteger(1) << k);
  const uint8_t new_depth = MortonCode<DIM>::coarsest_depth(new_code);
  return Morton{MortonCode<DIM>(new_code), new_depth};
}

template <Integer DIM> SCTL_GPU_HD Morton<DIM> Morton<DIM>::Ancestor(uint8_t level) const {
  return mid.Ancestor(level);
}

template <Integer DIM> SCTL_GPU_HD Morton<DIM> Morton<DIM>::DFD(uint8_t level) const {
  return Morton{mid, level};
}

template <Integer DIM> SCTL_GPU_HD std::array<Morton<DIM>, (1 << DIM)> Morton<DIM>::Children() const {
  using MI = typename MortonCode<DIM>::MortonInteger;
  std::array<Morton, (1 << DIM)> out{};
  // Child k's code: parent's code with bit i of k setting coord i's bit at level (MAX_DEPTH-depth-1).
  // In interleaved space that's `(k << shift)` with shift = DIM*(MAX_DEPTH-depth-1).
  const int shift = static_cast<int>(DIM) * (static_cast<int>(MAX_DEPTH) - static_cast<int>(depth) - 1);
  for (Integer k = 0; k < (1 << DIM); ++k) {
    const MI child_code = mid.code | (MI(static_cast<std::uint64_t>(k)) << shift);
    out[k] = Morton{MortonCode<DIM>(child_code), static_cast<uint8_t>(depth + 1)};
  }
  return out;
}

// Per-axis offsets for a fixed neighbor `idx`, unrolled over `d`: j = (idx/3^d) mod 3 - 1 is constant,
// so j==0 axes drop their bounds checks; with DYN==false `is_periodic(PER,d)` folds too (else runtime).
template <Integer DIM> template <Periodicity PER, bool DYN, Integer idx, Integer d>
[[gnu::always_inline]] inline SCTL_GPU_HD void Morton<DIM>::nbr_fill_(const std::uint64_t* xi_self, std::uint64_t box_size, std::uint64_t maxCoord,
                                        Periodicity periodicity, std::uint64_t* xi_nbr, bool& out_of_bounds) {
  if constexpr (d < DIM) {
    constexpr int j = static_cast<int>(idx / pow<d, Integer>(3) % 3) - 1;  // -1, 0, or +1
    const std::uint64_t self = xi_self[d];
    std::uint64_t v = self;
    if constexpr (j < 0) {
      if (self < box_size) {
        if constexpr (DYN) {
          if (is_periodic(periodicity, d)) v = self + maxCoord - box_size;
          else out_of_bounds = true;
        } else if constexpr (is_periodic(PER, d)) {
          v = self + maxCoord - box_size;
        } else {
          out_of_bounds = true;
        }
      } else {
        v = self - box_size;
      }
    } else if constexpr (j > 0) {
      v = self + box_size;
      if (v >= maxCoord) {
        if constexpr (DYN) {
          if (is_periodic(periodicity, d)) v -= maxCoord;
          else out_of_bounds = true;
        } else if constexpr (is_periodic(PER, d)) {
          v -= maxCoord;
        } else {
          out_of_bounds = true;
        }
      }
    }
    xi_nbr[d] = v;
    nbr_fill_<PER, DYN, idx, d + 1>(xi_self, box_size, maxCoord, periodicity, xi_nbr, out_of_bounds);
  }
}

// Outer loop over neighbor index `idx` (compile-time), recursing over all 3^DIM offset combos:
// idx = j_0 + 3*j_1 + 9*j_2 + ..., j_d ∈ {0,1,2} → offset (-1,0,+1)*box_size on axis d.
template <Integer DIM> template <Periodicity PER, bool DYN, Integer idx>
[[gnu::always_inline]] inline SCTL_GPU_HD void Morton<DIM>::nbr_emit_(const std::uint64_t* xi_self, std::uint64_t box_size, std::uint64_t maxCoord,
                                        Periodicity periodicity, uint8_t level,
                                        std::array<Morton, pow<DIM, std::size_t>(3)>& out) {
  if constexpr (idx < pow<DIM, Integer>(3)) {
    using MI = typename MortonCode<DIM>::MortonInteger;
    std::uint64_t xi_nbr[DIM];
    bool out_of_bounds = false;
    nbr_fill_<PER, DYN, idx, 0>(xi_self, box_size, maxCoord, periodicity, xi_nbr, out_of_bounds);
    if (out_of_bounds) {
      out[idx] = Morton{MortonCode<DIM>(MI(0)), Morton::INVALID_DEPTH};
    } else {
      out[idx] = Morton{MortonCode<DIM>(MortonCode<DIM>::interleave(xi_nbr)), level};
    }
    nbr_emit_<PER, DYN, idx + 1>(xi_self, box_size, maxCoord, periodicity, level, out);
  }
}

// Compact, non-unrolled emitter (the readable reference form of `nbr_emit_`). Used on-device for
// DIM>=4, where the unrolled emitters spill and tank occupancy; keeps register pressure low.
template <Integer DIM>
SCTL_GPU_HD void Morton<DIM>::nbr_loop_(const std::uint64_t* xi_self, std::uint64_t box_size, std::uint64_t maxCoord,
                                        Periodicity periodicity, uint8_t level,
                                        std::array<Morton, pow<DIM, std::size_t>(3)>& out) {
  using MI = typename MortonCode<DIM>::MortonInteger;
  // 3^DIM offset combos: idx = j_0 + 3*j_1 + 9*j_2 + ..., j_d in {0,1,2} -> offset (-1,0,+1)*box_size.
  for (Integer idx = 0; idx < pow<DIM, Integer>(3); ++idx) {
    std::uint64_t xi_nbr[DIM];
    bool out_of_bounds = false;
    Integer tmp = idx;
    for (Integer d = 0; d < DIM; ++d) {
      const int j = static_cast<int>(tmp % 3) - 1;  // -1, 0, or +1
      tmp /= 3;
      const std::uint64_t self = xi_self[d];
      std::uint64_t v = self;
      if (j < 0) {
        if (self < box_size) { if (is_periodic(periodicity, d)) v = self + maxCoord - box_size; else out_of_bounds = true; }
        else v = self - box_size;
      } else if (j > 0) {
        v = self + box_size;
        if (v >= maxCoord) { if (is_periodic(periodicity, d)) v -= maxCoord; else out_of_bounds = true; }
      }
      xi_nbr[d] = v;
    }
    if (out_of_bounds) out[idx] = Morton{MortonCode<DIM>(MI(0)), Morton::INVALID_DEPTH};
    else out[idx] = Morton{MortonCode<DIM>(MortonCode<DIM>::interleave(xi_nbr)), level};
  }
}

template <Integer DIM> SCTL_GPU_HD std::array<Morton<DIM>, pow<DIM, std::size_t>(3)> Morton<DIM>::NbrList(uint8_t level, Periodicity periodicity) const {
  static_assert(DIM <= PERIODICITY_MAX_DIM, "NbrList: DIM exceeds the Periodicity bitmask width");
  std::array<Morton, pow<DIM, std::size_t>(3)> out{};

  // Step 1: truncate to `level` and extract per-coord ints.
  const Morton base = Ancestor(level);
  std::uint64_t xi_self[DIM];
  for (Integer d = 0; d < DIM; ++d) {
    xi_self[d] = MortonCode<DIM>::compact_bits(base.mid.code, d);
  }
  const std::uint64_t box_size = std::uint64_t(1) << (MAX_DEPTH - level);
  const std::uint64_t maxCoord = std::uint64_t(1) << MAX_DEPTH;

  // Step 2: emit the 3^DIM neighbors.
#if defined(__CUDA_ARCH__)
  // On device the fully-unrolled emitters exhaust the register budget at high DIM (DIM>=4 hits the
  // 255-reg cap + spills -> low occupancy); the compact `nbr_loop_` keeps occupancy high and is
  // ~1.6x faster there. Host/lower-DIM keep the unrolled switch (faster on CPU and device DIM<=3).
  // The `else` (not a bare `return`) keeps the unrolled switch from being instantiated on this path.
  if constexpr (DIM >= 4) {
    nbr_loop_(xi_self, box_size, maxCoord, periodicity, level, out);
  } else
#endif
  // Dispatch runtime periodicity to a PER-specialized, unrolled, force-inlined emitter (DYN==false);
  // other masks runtime. ~2-3x faster than `nbr_loop_` on host/DIM<=3 (which is the readable form).
  switch (periodicity) {
    case Periodicity::NONE: nbr_emit_<Periodicity::NONE, false, 0>(xi_self, box_size, maxCoord, periodicity, level, out); break;
    case Periodicity::X:    nbr_emit_<Periodicity::X,    false, 0>(xi_self, box_size, maxCoord, periodicity, level, out); break;
    case Periodicity::Y:    nbr_emit_<Periodicity::Y,    false, 0>(xi_self, box_size, maxCoord, periodicity, level, out); break;
    case Periodicity::Z:    nbr_emit_<Periodicity::Z,    false, 0>(xi_self, box_size, maxCoord, periodicity, level, out); break;
    case Periodicity::XY:   nbr_emit_<Periodicity::XY,   false, 0>(xi_self, box_size, maxCoord, periodicity, level, out); break;
    case Periodicity::XYZ:  nbr_emit_<Periodicity::XYZ,  false, 0>(xi_self, box_size, maxCoord, periodicity, level, out); break;
    default:                nbr_emit_<Periodicity::NONE, true,  0>(xi_self, box_size, maxCoord, periodicity, level, out); break;
  }
  return out;
}

template <Integer DIM> SCTL_GPU_HD bool Morton<DIM>::operator<(const Morton& o) const {
  // Lex Morton order: code first, depth as tiebreaker (matches sctl::Morton::operator<).
  if (mid.code == o.mid.code) return depth < o.depth;
  return mid.code < o.mid.code;
}

template <Integer DIM> SCTL_GPU_HD bool Morton<DIM>::operator>(const Morton& o) const { return o < *this; }
template <Integer DIM> SCTL_GPU_HD bool Morton<DIM>::operator<=(const Morton& o) const { return !(*this > o); }
template <Integer DIM> SCTL_GPU_HD bool Morton<DIM>::operator>=(const Morton& o) const { return !(*this < o); }

template <Integer DIM> SCTL_GPU_HD bool Morton<DIM>::operator==(const Morton& o) const { return mid.code == o.mid.code && depth == o.depth; }

template <Integer DIM> SCTL_GPU_HD bool Morton<DIM>::operator!=(const Morton& o) const { return !(*this == o); }

template <Integer DIM> SCTL_GPU_HD bool Morton<DIM>::isAncestor(const Morton& descendant) const { return descendant.depth > depth && descendant.Ancestor(depth) == *this; }

template <Integer DIM> SCTL_GPU_HD Long Morton<DIM>::operator-(const Morton& o) const {
  // Direct port of sctl::Morton<DIM>::operator-: -1 intersecting, 0 touching, >0 separated.
  const std::uint64_t offset0 = std::uint64_t(1) << (MAX_DEPTH - depth);
  const std::uint64_t offset1 = std::uint64_t(1) << (MAX_DEPTH - o.depth);
  std::uint64_t diff = 0;
  for (Integer d = 0; d < DIM; ++d) {
    const std::uint64_t x0 = MortonCode<DIM>::compact_bits(mid.code, d);
    const std::uint64_t x1 = MortonCode<DIM>::compact_bits(o.mid.code, d);
    const std::uint64_t Xc0 = x0 * 2 + offset0;  // center * 2, no /2 rounding loss
    const std::uint64_t Xc1 = x1 * 2 + offset1;
    const std::uint64_t d_val = (Xc0 > Xc1) ? (Xc0 - Xc1) : (Xc1 - Xc0);
    if (d_val > diff) diff = d_val;
  }
  if (diff < offset0 + offset1) return -1;
  const Integer max_d = (depth > o.depth) ? depth : o.depth;
  return static_cast<Long>((diff - offset0 - offset1) >> (MAX_DEPTH + 1 - max_d));
}

// sctl::Tree-compat overloads: write std::array result into a Vector outparam.
template <Integer DIM> void Morton<DIM>::NbrList(Vector<Morton>& nlst, uint8_t level, Periodicity periodicity) const {
  const auto arr = NbrList(level, periodicity);
  nlst.ReInit(static_cast<Long>(arr.size()));
  for (Long i = 0; i < static_cast<Long>(arr.size()); ++i) nlst[i] = arr[i];
}

template <Integer DIM> void Morton<DIM>::Children(Vector<Morton>& nlst) const {
  const auto arr = Children();
  nlst.ReInit(static_cast<Long>(arr.size()));
  for (Long i = 0; i < static_cast<Long>(arr.size()); ++i) nlst[i] = arr[i];
}

}  // namespace sctl

#endif  // _SCTL_MORTON_TXX_
