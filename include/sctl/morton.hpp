// Morton (Z-order) code and tree node (`MortonCode`, `Morton`); host + device.

#ifndef _SCTL_MORTON_HPP_
#define _SCTL_MORTON_HPP_

#include <array>        // for std::array (Children, NbrList return types)
#include <cstdint>      // for uint8_t, uint16_t, uint32_t, uint64_t, int64_t
#include <ostream>      // for operator<< (host-only)
#include <type_traits>  // for conditional

#include "sctl/common.hpp"      // for Integer, Long
#include "sctl/iterator.hpp"    // for ConstIterator (used by Morton ctor for sctl tree compat)
#include "sctl/math_utils.hpp"  // for pow (declaration)
#include "sctl/math_utils.txx"  // for pow's constexpr definition (needed at class-template instantiation)

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 15
#endif

// Mark host-callable functions also callable from device code under nvcc/hipcc.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SCTL_GPU_HD __host__ __device__
#else
#define SCTL_GPU_HD
#endif

namespace sctl {

template <class T> class Vector;  // fwd-decl for legacy NbrList/Children outparam overloads

/** Maximum tree depth (number of refinement levels from the root). */
constexpr Integer MAX_DEPTH = SCTL_MAX_DEPTH;

template <Integer DIM> class MortonCode;
template <Integer DIM> class Morton;

/**
 * Morton (Z-order) code via interleaved bit encoding: bit `level*DIM + d` of the underlying code
 * carries bit `level` of `coord[d]` (level 0 = LSB / finest). `operator<` is a single integer
 * compare, so a radix sort over codes is a Morton-order sort.
 *
 * @tparam DIM Number of spatial dimensions.
 */
template <Integer DIM> class MortonCode {
  static_assert(DIM > 0, "MortonCode: DIM must be positive");
  static_assert(MAX_DEPTH > 0 && MAX_DEPTH < 64, "MortonCode: MAX_DEPTH must be in (0, 64)");
  static_assert(MAX_DEPTH * DIM > 0, "MortonCode: MAX_DEPTH*DIM must be positive");

 public:
  /** Trivial default ctor: code is uninitialised. Use `MortonCode{}` for a zero code. */
  MortonCode() = default;

  /** @param[in] coord `DIM` coordinates; inputs outside [0,1) are clamped. */
  template <class Real> SCTL_GPU_HD explicit MortonCode(const Real* coord);

  /** Morton-order less-than. */
  SCTL_GPU_HD bool operator<(const MortonCode& other) const;

  /**
   * Deepest tree box containing both `*this` and `other`. For identical inputs the result is
   * the leaf itself at depth `MAX_DEPTH`.
   *
   * @return `(mid, depth)` with `depth` measured from the root (0 = root, MAX_DEPTH = leaf).
   */
  SCTL_GPU_HD Morton<DIM> CommonAncestor(const MortonCode& other) const;

  /**
   * Ancestor box at the given tree depth (low `(MAX_DEPTH-depth)*DIM` code bits zeroed).
   *
   * @param[in] depth Tree depth, `0 <= depth <= MAX_DEPTH`.
   */
  SCTL_GPU_HD Morton<DIM> Ancestor(uint8_t depth) const;

 private:
  /** Active code range: `MAX_DEPTH` levels of `DIM` bits each, MSB-first per coord. */
  static constexpr Integer TOTAL_BITS = MAX_DEPTH * DIM;

  /**
   * Storage width of `MortonInteger`. One full extra level above the active range so the
   * "past-end" sentinel produced by `Morton::Next()` at the root (bit `TOTAL_BITS = DIM*MAX_DEPTH`
   * set) has somewhere to live. Mirrors how `sctl::Morton<DIM>::Next()` saturates by setting
   * `x[0] = 1 << MAX_DEPTH`, the bit one level above the active per-coord range.
   */
  static constexpr Integer STORAGE_BITS = TOTAL_BITS + DIM;

  /** Multi-word unsigned integer used as `MortonInteger` when `STORAGE_BITS > 64`. `w[0]` is the LSW. */
  template <int NWORDS_> struct MortonBig {
    std::uint64_t w[NWORDS_];

    MortonBig() = default;
    constexpr explicit MortonBig(std::uint64_t x) : w{} { w[0] = x; }

    constexpr bool operator<(const MortonBig& o) const;
    constexpr bool operator==(const MortonBig& o) const;
    constexpr MortonBig operator|(const MortonBig& o) const;
    constexpr MortonBig operator&(const MortonBig& o) const;
    constexpr MortonBig operator^(const MortonBig& o) const;
    constexpr MortonBig operator+(const MortonBig& o) const;  // wraps mod 2^(64*NWORDS_)
    constexpr MortonBig& operator|=(const MortonBig& o);
    constexpr MortonBig operator<<(int s) const;
    constexpr MortonBig operator>>(int s) const;
  };

  static constexpr int NWORDS = static_cast<int>((STORAGE_BITS + 63) / 64);

  /** Smallest unsigned holding STORAGE_BITS bits: built-in uintN_t for <=64, else MortonBig<NWORDS>. */
  using MortonInteger = typename std::conditional<(STORAGE_BITS <= 16), std::uint16_t,
                        typename std::conditional<(STORAGE_BITS <= 32), std::uint32_t,
                        typename std::conditional<(STORAGE_BITS <= 64), std::uint64_t,
                        MortonBig<NWORDS>>::type>::type>::type;

  SCTL_GPU_HD explicit MortonCode(MortonInteger code_) : code(code_) {}

  /** Highest set bit position (0-indexed from LSB), or -1 when `x == 0`. */
  static SCTL_GPU_HD int highest_bit_pos(const MortonInteger& x);

  /** Coarsest depth at which `c` is a valid box ID (low `(MAX_DEPTH-d)*DIM` bits zero). 0 for `c == 0`. */
  static SCTL_GPU_HD uint8_t coarsest_depth(const MortonInteger& c);

  /** Spread MAX_DEPTH bits of `xi` to DIM-spaced positions in `O(log MAX_DEPTH)` mask/shift steps. */
  template <Integer Step> static SCTL_GPU_HD MortonInteger spread_step(MortonInteger r);

  static SCTL_GPU_HD MortonInteger spread_bits(std::uint64_t xi);

  /** Inverse of `spread_step`: pulls DIM-spaced bits back to contiguous positions. */
  template <Integer Step> static SCTL_GPU_HD MortonInteger compact_step(MortonInteger r);

  /** Extract coordinate `d` from an interleaved code (de-interleave). Inverse of `spread_bits`. */
  static SCTL_GPU_HD std::uint64_t compact_bits(MortonInteger code, Integer d);

  /** Interleave DIM per-axis ints into a code (force-inlined; `nbr_emit_`'s hot path). */
  template <Integer d = 0> static SCTL_GPU_HD MortonInteger interleave(const std::uint64_t* xi);

  /** Real-coord ctor body: clamp to `[0,1)`, scale to ints, interleave. Separate plain-inline overload
   *  (force-inlining regressed gcc's DIM=4 ctor ~3x); `enable_if` avoids clashing with the `uint64_t*` one. */
  template <class Real, class = std::enable_if_t<!std::is_integral<Real>::value>>
  static SCTL_GPU_HD MortonInteger interleave(const Real* coord);

  MortonInteger code;

  friend class Morton<DIM>;
};

/**
 * Tree node: a Morton code with its depth from the root (0 = root, `MAX_DEPTH` = leaf). For any
 * non-leaf node the low `(MAX_DEPTH-depth)*DIM` bits of `mid.code` are zero.
 */
template <Integer DIM> class Morton {
 public:
  /**
   * Sentinel `depth` value for "invalid" / "missing" nodes (see `NbrList`). Picked as
   * `0xFF` (= `(uint8_t)-1`); legal depths are in `[0, MAX_DEPTH]` and `MAX_DEPTH < 64`,
   * so this never collides with a real depth.
   */
  static constexpr uint8_t INVALID_DEPTH = 0xFF;

  /** Maximum tree depth (alias for the namespace-scope `sctl::MAX_DEPTH`). */
  static constexpr Integer MAX_DEPTH = sctl::MAX_DEPTH;
  static constexpr Integer MaxDepth() { return MAX_DEPTH; }

  MortonCode<DIM> mid;  ///< Morton code of the node (low bits zeroed for non-leaves).
  uint8_t depth;        ///< Tree depth (0 = root, MAX_DEPTH = leaf). `INVALID_DEPTH` flags an invalid node.

  /** Default ctor: ROOT (zero code, depth 0). */
  SCTL_GPU_HD Morton() : mid(MortonCode<DIM>{}), depth(0) {}

  /** Aggregate-style ctor (preserves `Morton{code, depth}` initialiser-list syntax). */
  SCTL_GPU_HD Morton(MortonCode<DIM> mid_, uint8_t depth_) : mid(mid_), depth(depth_) {}

  /** Construct the depth-`depth_` box containing `coord` (truncates low bits). Accepts a raw
   *  `const T*` (release) or `ConstIterator<T>` (under SCTL_MEMDEBUG). */
  template <class T> explicit Morton(ConstIterator<T> coord, uint8_t depth_ = MAX_DEPTH);

  /** Returns this node's depth (alias for the `depth` field). */
  SCTL_GPU_HD uint8_t Depth() const;

  /** Write per-dim coordinates `[0,1)^DIM` into `coord`. Inverse of `MortonCode(const Real*)`. */
  template <class ArrayType> SCTL_GPU_HD void Coord(ArrayType&& coord) const;

  /** Return per-dim coordinates as `std::array<Real, DIM>`. */
  template <class Real> std::array<Real, DIM> Coord() const {
    std::array<Real, DIM> arr{};
    Coord(arr);
    return arr;
  }

  /**
   * Next consecutive node in Morton order at this depth (code += `1 << (DIM*(MAX_DEPTH-depth))`).
   * Carry propagation may zero additional low bits, so the result's depth is `<= depth`.
   *
   * At `depth == 0` the increment sets the storage bit one level above the active code range,
   * producing a "past-end" sentinel that sorts strictly greater than every valid `Morton` —
   * matching the way `Morton<DIM>().Next()` is used as a `+infinity` upper bound in
   * `std::lower_bound` partitioning across `sctl::Tree` and the boundary kernels.
   */
  SCTL_GPU_HD Morton Next() const;

  /**
   * Ancestor at the given tree level (low `(MAX_DEPTH-level)*DIM` bits of the code zeroed,
   * `depth = level`). Independent of `this->depth`.
   */
  SCTL_GPU_HD Morton Ancestor(uint8_t level) const;

  /** Deepest first descendant at the given level: same code, `depth = level`. No bit work. */
  SCTL_GPU_HD Morton DFD(uint8_t level = MAX_DEPTH) const;

  /**
   * `2^DIM` children at depth `this->depth + 1`. Child `k` (`k` in `[0, 1<<DIM)`) has bit `i` of `k`
   * setting the `i`-th coordinate's bit at level `MAX_DEPTH-(depth+1)`. Caller must ensure
   * `depth < MAX_DEPTH`.
   */
  SCTL_GPU_HD std::array<Morton, (1 << DIM)> Children() const;

  /**
   * `3^DIM` same-level neighbors (ancestor truncated to `level`). The neighbor for offset
   * `(δ_0,…,δ_{DIM-1})`, `δ_d ∈ {-1,0,+1}`, is at index `Σ (δ_d + 1) 3^d`; self is the centre.
   * Periodic axes wrap; non-periodic out-of-domain neighbors get `depth = INVALID_DEPTH`.
   */
  SCTL_GPU_HD std::array<Morton, pow<DIM, std::size_t>(3)> NbrList(uint8_t level, Periodicity periodicity) const;

  /** sctl::Tree-compat overloads: write into a Vector outparam (host-only). */
  void NbrList(Vector<Morton>& nlst, uint8_t level, Periodicity periodicity) const;
  void Children(Vector<Morton>& nlst) const;

  /** Lexicographic Morton order: by code first, with `depth` as tiebreaker. */
  SCTL_GPU_HD bool operator<(const Morton& o) const;

  /** Equivalent to `o < *this`. */
  SCTL_GPU_HD bool operator>(const Morton& o) const;

  /** Negation of `operator>`. */
  SCTL_GPU_HD bool operator<=(const Morton& o) const;

  /** Negation of `operator<`. */
  SCTL_GPU_HD bool operator>=(const Morton& o) const;

  SCTL_GPU_HD bool operator==(const Morton& o) const;
  SCTL_GPU_HD bool operator!=(const Morton& o) const;

  /** Strict ancestor test: `desc.depth > this->depth && desc.Ancestor(this->depth) == *this`. */
  SCTL_GPU_HD bool isAncestor(const Morton& descendant) const;

  /**
   * Spatial separation in code units, per `sctl::Morton::operator-`:
   *   `-1` if the two boxes intersect,
   *   `0`  if they touch (share a face/edge/corner only),
   *   `>0` quantized gap otherwise.
   */
  SCTL_GPU_HD Long operator-(const Morton& o) const;

  /** Stream insertion (host-only): prints `(x_0, x_1, ..., depth)` with coords as double. */
  template <class CharT, class Traits>
  friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const Morton& n) {
    os << "(";
    const std::array<double, DIM> c = n.template Coord<double>();
    for (Integer d = 0; d < DIM; ++d) os << c[d] << ",";
    os << static_cast<int>(n.depth) << ")";
    return os;
  }

 private:
  /**
   * `NbrList` helpers, unrolled at compile time so the neighbor index `idx` and axis `d`
   * (and hence the per-axis offset `j ∈ {-1,0,+1}`) are constants. With `j` constant, the
   * `j<0`/`j>0` bounds-check branches collapse via `if constexpr`: the `j==0` axes emit no
   * bounds work at all. Mirrors the recursive `if constexpr` style of `MortonCode::spread_step`.
   *
   * Periodicity is also a template parameter: `NbrList` dispatches the runtime `Periodicity` via a
   * `switch` to a `PER`-specialized instantiation (`DYN == false`) so `is_periodic(PER, d)` is a
   * compile-time constant and the wrap-vs-out-of-bounds branch folds away. Unenumerated masks fall
   * through to the `DYN == true` instantiation, which reads the runtime `periodicity` argument.
   */
  template <Periodicity PER, bool DYN, Integer idx, Integer d>
  static SCTL_GPU_HD void nbr_fill_(const std::uint64_t* xi_self, std::uint64_t box_size, std::uint64_t maxCoord,
                                    Periodicity periodicity, std::uint64_t* xi_nbr, bool& out_of_bounds);

  template <Periodicity PER, bool DYN, Integer idx>
  static SCTL_GPU_HD void nbr_emit_(const std::uint64_t* xi_self, std::uint64_t box_size, std::uint64_t maxCoord,
                                    Periodicity periodicity, uint8_t level,
                                    std::array<Morton, pow<DIM, std::size_t>(3)>& out);

  /** Compact (non-unrolled) emitter; the readable reference form of `nbr_emit_`. Used on-device for
   *  DIM>=4 where the unrolled emitters spill and tank occupancy (host/DIM<=3 use the switch). */
  static SCTL_GPU_HD void nbr_loop_(const std::uint64_t* xi_self, std::uint64_t box_size, std::uint64_t maxCoord,
                                    Periodicity periodicity, uint8_t level,
                                    std::array<Morton, pow<DIM, std::size_t>(3)>& out);
};

}  // namespace sctl

#include "sctl/morton.txx"

#endif  // _SCTL_MORTON_HPP_
