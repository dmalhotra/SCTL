#ifndef _SCTL_FUNCTION_TABLE_TXX_
#define _SCTL_FUNCTION_TABLE_TXX_

#include <algorithm>               // for max, max_element
#include <cmath>                   // for cos, floor, pow, fabs
#include <type_traits>             // for conditional, integral_constant, is_same
#include <utility>                 // for integer_sequence, make_integer_sequence
#include <vector>                  // for vector

#include "sctl/common.hpp"         // for Integer, Long, SCTL_ASSERT_MSG, sctl
#include "sctl/function-table.hpp" // for FunctionTable
#include "sctl/intrin-wrapper.hpp" // for eval_poly_rows_intrin
#include "sctl/math_utils.hpp"     // for const_pi, machine_eps
#include "sctl/vector.hpp"         // for Vector
#include "sctl/vec.hpp"            // for Vec, max, min, floor, FMA, Convert, permute, blend

namespace sctl {

  namespace detail_function_table {
    // sum c[i] t^i, i < R, by Estrin's scheme; c is overwritten
    template <Integer R, class VecR> inline VecR estrin(VecR (&c)[R], const VecR& t) {
      VecR p = t;
      Integer len = R;
      while (len > 1) {
        for (Integer i = 0; i < len / 2; i++) c[i] = FMA(c[2 * i + 1], p, c[2 * i]);
        if (len % 2) c[len / 2] = c[len - 1];
        len = (len + 1) / 2;
        p = p * p;
      }
      return c[0];
    }
    // The indices of rows_poly for the rows of N coefficients of the N lanes: the even and the odd part at each stage,
    // and the lane and the coefficient group of each element after it.
    template <class Real, Integer N> struct RowsPlan {
      static constexpr Integer E = ((Integer)(16 / sizeof(Real)) < N ? (Integer)(16 / sizeof(Real)) : N); // elements of a 128-bit lane, at most N
      // the element of the pair (a, b) = (v[2j], v[2j+1]), N and above in b, of the even (parity 0) or odd (1) part of
      // element k at stage s >= 1: the first half from a and the second from b, of the even scalars of each 128-bit
      // lane in the first log2(E) stages, then of the even 128-bit lanes
      static constexpr Integer source(const Integer s, const Integer k, const Integer parity) {
        const bool in_lane = ((((Integer)1) << s) <= E);
        const Integer gs = (in_lane ? E : N); // size of the group of elements split in two halves
        const Integer z = (in_lane ? 1 : E); // size of the parts moved
        const Integer g = k / gs;
        const Integer r = k % gs;
        const Integer sub = r / z;
        const Integer off = r % z;
        const Integer half = gs / z / 2;
        return (sub < half ? g * gs + (2 * sub + parity) * z + off : N + g * gs + (2 * (sub - half) + parity) * z + off);
      }
      static constexpr Integer lane(const Integer s, const Integer j, const Integer k) { // the lane of element k of vector j after stage s
        if (s == 0) return j;
        const Integer i = source(s, k, 0);
        return (i < N ? lane(s - 1, 2 * j, i) : lane(s - 1, 2 * j + 1, i - N));
      }
      static constexpr Integer group(const Integer s, const Integer j, const Integer k) { // the coefficient group of element k of vector j after stage s
        if (s == 0) return k;
        const Integer i = source(s, k, 0);
        return (i < N ? group(s - 1, 2 * j, i) : group(s - 1, 2 * j + 1, i - N)) / 2;
      }
      static constexpr Integer stages() {
        Integer s = 0;
        while ((((Integer)1) << s) < N) s++;
        return s;
      }
      static constexpr bool natural() { // the lanes in order after the last stage, in one coefficient group
        for (Integer k = 0; k < N; k++) {
          if (lane(stages(), 0, k) != k || group(stages(), 0, k) != 0) return false;
        }
        return true;
      }
    };
    // v[j] = even + odd t^(2^(s-1)) of the pair (v[2j], v[2j+1]); K: the elements
    template <class Plan, Integer s, Integer j, class VecR, Integer V, Integer... K> inline void rows_pair(VecR (&v)[V], const VecR& tp, std::integer_sequence<Integer, K...>) {
      v[j] = FMA(blend<Plan::source(s, K, 1)...>(v[2 * j], v[2 * j + 1]), permute<Plan::lane(s, j, K)...>(tp), blend<Plan::source(s, K, 0)...>(v[2 * j], v[2 * j + 1]));
    }
    // stage s for the pairs J
    template <class Plan, Integer s, class VecR, Integer V, Integer... J> inline void rows_stage(VecR (&v)[V], const VecR& tp, std::integer_sequence<Integer, J...>) {
      (rows_pair<Plan, s, J>(v, tp, std::make_integer_sequence<Integer, VecR::Size()>()), ...);
    }
    // sum_{i<B} p[l][i] t_l^i in each lane l: for B = N in vector registers, the rows transposed by blend and evaluated by
    // Estrin's scheme together, one step of the scheme after each stage of the transpose; else eval_poly_rows_intrin,
    // also for 8 floats, where its AVX kernel builds the arrangements of t in fewer instructions
    template <Integer B, class Real, Integer N> inline Vec<Real,N> rows_poly(const Real* const (&p)[N], const Vec<Real,N>& t) {
      using VecR = Vec<Real,N>;
      if constexpr (B != N || array_lanes<typename VecR::VData> || (std::is_same<Real,float>::value && N == 8)) {
        return VecR(eval_poly_rows_intrin<B>(p, t.get()));
      } else {
        using Plan = RowsPlan<Real, N>;
        static_assert(Plan::natural(), "rows_poly: the lanes out of order.");
        static_assert(Plan::stages() <= 4, "rows_poly: at most 16 coefficients.");
        VecR v[N];
        for (Integer j = 0; j < N; j++) v[j] = VecR::Load(p[j]);
        VecR tp = t;
        const auto stage = [&v, &tp](auto sc) {
          static constexpr Integer s = decltype(sc)::value;
          if constexpr (s <= Plan::stages()) {
            rows_stage<Plan, s>(v, tp, std::make_integer_sequence<Integer, (N >> s)>());
            tp = tp * tp;
          }
        };
        stage(std::integral_constant<Integer, 1>());
        stage(std::integral_constant<Integer, 2>());
        stage(std::integral_constant<Integer, 3>());
        stage(std::integral_constant<Integer, 4>());
        return v[0];
      }
    }
    // sum c_i t^i of the rows idx[l] of R coefficients of the lanes l, by blocks of B coefficients
    template <Integer R, class Real, Integer N, class Idx> inline Vec<Real,N> eval_rows(const Real* coef, const Idx (&idx)[N], const Vec<Real,N>& t) {
      using VecR = Vec<Real,N>;
      static constexpr Integer B = (N < 8 ? N : 8); // as the specializations of eval_poly_rows_intrin
      VecR q[R / B]; // the polynomial of each block
      for (Integer b = 0; b < R / B; b++) {
        const Real* p[N];
        for (Integer l = 0; l < N; l++) p[l] = coef + (Long)idx[l] * R + b * B;
        q[b] = rows_poly<B>(p, t);
      }
      VecR tb = t; // t^B
      for (Integer i = 1; i < B; i *= 2) tb = tb * tb;
      return estrin(q, tb);
    }
  }

  template <class Real> template <class Fn> FunctionTable<Real>::FunctionTable(const Fn& f, Real lb, Real ub, Integer digits, Integer degree) {
    SCTL_ASSERT_MSG(lb < ub, "FunctionTable: the interval must have lb < ub.");
    SCTL_ASSERT_MSG(degree < 32, "FunctionTable: the degree must be below 32.");
    const double tol = std::max(std::pow(10.0, -(double)digits), 8 * (double)machine_eps<Real>());
    if (degree >= 0) {
      SCTL_ASSERT_MSG(Build(*this, f, lb, ub, tol, degree, -1, false), "FunctionTable: the function is not resolved to the requested digits.");
      return;
    }
    for (const bool uniform : {true, false}) { // the lowest degree whose table fits in max_bytes, uniform tables first
      for (const Integer p : {7, 15, 23}) {
        if (Build(*this, f, lb, ub, tol, p, max_bytes, uniform)) return;
      }
    }
    SCTL_ASSERT_MSG(Build(*this, f, lb, ub, tol, 23, -1, false), "FunctionTable: the function is not resolved to the requested digits.");
  }

  template <class Real> template <class Fn> bool FunctionTable<Real>::Build(FunctionTable& T, const Fn& f, const double lb, const double ub, const double tol, const Integer p, const Long bytes, const bool uniform) {
    static constexpr Long max_depth = 30;
    static constexpr Long max_leaves = ((Long)1) << 22;
    static constexpr double max_cond = 32; // largest sum |m_i| of the monomial coefficients over the largest |f|
    const Integer n = p + 1; // number of coefficients
    const Integer row = (n + 7) / 8 * 8;
    const Long max_pieces = (bytes < 0 ? -1 : bytes / (row * (Long)sizeof(Real)));
    Long depth_lim = max_depth; // largest depth of a leaf
    if (uniform && max_pieces >= 0) {
      depth_lim = 0;
      while ((((Long)1) << (depth_lim + 1)) <= max_pieces) depth_lim++;
    }
    std::vector<double> cos_tab(n * n); // cos(pi k (2j+1)/(2n))
    for (Integer k = 0; k < n; k++) {
      for (Integer j = 0; j < n; j++) cos_tab[k * n + j] = std::cos(const_pi<double>() * k * (2 * j + 1) / (2 * n));
    }

    const auto fit = [&f, &cos_tab, n, tol](double* m, const double a, const double b) { // m[i], i < n: the monomial coefficients in t of the interpolant at the first-kind Chebyshev nodes; true where it meets tol
      double fx[32];
      double c[32] = {};
      double scale = 0;
      double mean = 0;
      for (Integer j = 0; j < n; j++) {
        fx[j] = (double)f((Real)((a + b) / 2 + (b - a) / 2 * cos_tab[n + j]));
        scale = std::max(scale, std::fabs(fx[j]));
        mean += fx[j] / n;
      }
      for (Integer k = 0; k < n; k++) { // of fx - mean, for a rounding error relative to the variation of f
        double s = 0;
        for (Integer j = 0; j < n; j++) s += (fx[j] - mean) * cos_tab[k * n + j];
        c[k] = s * (k == 0 ? 1 : 2) / n;
      }
      c[0] += mean;
      { // m = sum_k c_k T_k, by T_(k+1) = 2t T_k - T_(k-1)
        double T0[32] = {1};
        double T1[32] = {0, 1};
        for (Integer i = 0; i < n; i++) m[i] = c[0] * T0[i] + (n > 1 ? c[1] * T1[i] : 0);
        for (Integer k = 2; k < n; k++) {
          double T2[32];
          T2[0] = -T0[0];
          for (Integer i = 1; i < n; i++) T2[i] = 2 * T1[i - 1] - T0[i];
          for (Integer i = 0; i < n; i++) {
            m[i] += c[k] * T2[i];
            T0[i] = T1[i];
            T1[i] = T2[i];
          }
        }
      }
      const auto horner = [m, n](const double t) {
        double r = (double)(Real)m[n - 1];
        for (Integer i = n - 2; i >= 0; i--) r = r * t + (double)(Real)m[i];
        return r;
      };
      const Integer n_test = 2 * n + 1; // test points, equispaced with both ends
      double err = 0;
      for (Integer i = 0; i < n_test; i++) {
        const double t = -1 + 2 * (double)i / (n_test - 1);
        const double fi = (double)f((Real)((a + b) / 2 + (b - a) / 2 * t));
        scale = std::max(scale, std::fabs(fi));
        err = std::max(err, std::fabs(horner(t) - fi));
      }
      const double tail = std::fabs(c[n - 1]); // above tol where the series has not converged; its rounding is about n eps scale
      double cond = 0;
      double noise = 0; // bound of 4 eps |x f'(x)|: the change of f from rounding x to Real, in a sample and through the interpolation
      for (Integer i = 0; i < n; i++) {
        cond += std::fabs(m[i]);
        noise += i * std::fabs(m[i]);
      }
      noise *= 4 * (double)machine_eps<Real>() * std::max(std::fabs(a), std::fabs(b)) * 2 / (b - a);
      return !(err > tol * scale + noise) && !(tail > std::max(tol, 4 * n * (double)machine_eps<double>()) * scale + noise) && !(cond > max_cond * scale); // also for f = 0
    };

    std::vector<Long> leaf_depth, leaf_index;
    { // adaptive bisection of [lb, ub] for the depth of each leaf
      std::vector<Long> stack_depth(1, 0), stack_index(1, 0);
      double m[32];
      while (stack_depth.size()) {
        const Long d = stack_depth.back();
        const Long i = stack_index.back();
        stack_depth.pop_back();
        stack_index.pop_back();
        const double w = (ub - lb) / (double)(((Long)1) << d);
        if (fit(m, lb + i * w, lb + (i + 1) * w)) {
          leaf_depth.push_back(d);
          leaf_index.push_back(i);
          if (max_pieces >= 0 && (Long)leaf_depth.size() > max_pieces) return false;
        } else {
          if (d >= depth_lim || (Long)(leaf_depth.size() + stack_depth.size()) > max_leaves) return false;
          stack_depth.push_back(d + 1);
          stack_index.push_back(2 * i + 1);
          stack_depth.push_back(d + 1);
          stack_index.push_back(2 * i);
        }
      }
    }

    const auto cell_depths = [&leaf_depth, &leaf_index](std::vector<Long>& e, const Long d0) { // for the cells at depth d0, the depth of their deepest leaf below d0
      e.assign(((Long)1) << d0, 0);
      for (size_t l = 0; l < leaf_depth.size(); l++) {
        if (leaf_depth[l] >= d0) {
          Long& ec = e[leaf_index[l] >> (leaf_depth[l] - d0)];
          ec = std::max(ec, leaf_depth[l] - d0);
        }
      }
    };
    std::vector<Long> cell_depth;
    { // the depth d0 of the cells, each refined uniformly to its deepest leaf: uniform where it fits in max_bytes, else the fewest pieces
      const Long d_max = *std::max_element(leaf_depth.begin(), leaf_depth.end());
      Long d0 = d_max;
      if ((((Long)1) << d_max) * row * (Long)sizeof(Real) > max_bytes) {
        Long best = -1;
        for (Long d = 0; d <= d_max && (((Long)1) << d) <= max_leaves; d++) {
          cell_depths(cell_depth, d);
          Long total = 0;
          for (const Long e : cell_depth) total += ((Long)1) << e;
          if (best < 0 || total < best) {
            best = total;
            d0 = d;
          }
        }
        if (max_pieces >= 0 && best > max_pieces) return false;
      }
      cell_depths(cell_depth, d0);
    }

    const Long n_cell = (Long)cell_depth.size();
    const double h = (ub - lb) / (double)n_cell;
    std::vector<Real> coef, sub(n_cell), off(n_cell);
    double m[32];
    for (Long ci = 0; ci < n_cell; ci++) { // the pieces of each cell, the cell refined further where a piece misses tol
      bool ok = false;
      while (!ok) {
        const Long n_sub = ((Long)1) << cell_depth[ci];
        if (max_pieces >= 0 && (Long)coef.size() / row + n_sub > max_pieces) return false;
        const Long c_off = (Long)coef.size();
        ok = true;
        for (Long j = 0; j < n_sub && ok; j++) {
          const double a = lb + h * (double)ci + h * (double)j / (double)n_sub;
          ok = fit(m, a, a + h / (double)n_sub);
          for (Integer i = 0; i < row; i++) coef.push_back(i < n ? (Real)m[i] : (Real)0);
        }
        if (!ok) {
          coef.resize(c_off);
          if (++cell_depth[ci] > max_depth) return false;
          continue;
        }
        sub[ci] = (Real)n_sub;
        off[ci] = (Real)(c_off / row);
      }
    }

    T.lb_ = (Real)lb;
    T.inv_h_ = (Real)(1 / h);
    T.n_cell_ = n_cell;
    T.degree_ = p;
    T.row_ = row;
    T.uniform_ = (Long)coef.size() == n_cell * row;
    T.sub_.ReInit(n_cell);
    T.off_.ReInit(n_cell);
    T.coef_.ReInit((Long)coef.size());
    for (Long ci = 0; ci < n_cell; ci++) {
      T.sub_[ci] = sub[ci];
      T.off_[ci] = off[ci];
    }
    for (Long i = 0; i < (Long)coef.size(); i++) T.coef_[i] = coef[i];
    return true;
  }

  template <class Real> inline Real FunctionTable<Real>::operator()(Real x) const {
    const Real u = (x - lb_) * inv_h_;
    Real c = std::floor(u);
    c = (c >= 0 ? c : 0); // also for NaN
    c = (c > (Real)(n_cell_ - 1) ? (Real)(n_cell_ - 1) : c);
    const Long ci = (Long)c;
    const Real sub = sub_[ci];
    const Real v = (u - c) * sub;
    Real k = std::floor(v);
    k = (k >= 0 ? k : 0);
    k = (k > sub - 1 ? sub - 1 : k);
    const Real t = (v - k) * 2 - 1;
    const Real* m = &coef_[(Long)(off_[ci] + k) * row_];
    Real r = m[degree_];
    for (Integer i = degree_ - 1; i >= 0; i--) r = r * t + m[i];
    return r;
  }

  template <class Real> template <Integer N> __attribute__((always_inline)) inline Vec<Real,N> FunctionTable<Real>::operator()(const Vec<Real,N>& x) const {
    using VecR = Vec<Real,N>;
#if defined(__AVX512DQ__) && defined(__AVX512VL__)
    using Idx = typename std::conditional<(N >= 4 && sizeof(Real) == 4), int32_t, int64_t>::type; // native conversions, vcvttpd2qq for double
#else
    using Idx = typename std::conditional<(N >= 4), int32_t, int64_t>::type; // native conversions
#endif
    using VecI = Vec<Idx,N>;
    const VecR u = FMA(x, VecR(inv_h_), VecR(-lb_ * inv_h_));
    const VecR uc = max(min(u, VecR((Real)(n_cell_ - 1))), VecR::Zero()); // min gives n_cell_ - 1 for NaN
    const VecI ci = Convert<VecI>(uc);
    VecR t; // the local variable in [-1, 1]
    alignas(sizeof(VecI)) Idx idx[N]; // the piece of each lane
    if (uniform_) {
      t = FMA(u - floor(uc), VecR((Real)2), VecR((Real)-1));
      ci.StoreAligned(idx);
    } else {
      const VecR sub = VecR::Gather(&sub_[0], ci);
      const VecR v = (u - floor(uc)) * sub;
      const VecR k = floor(max(min(v, sub - (Real)1), VecR::Zero()));
      t = FMA(v - k, VecR((Real)2), VecR((Real)-1));
      Convert<VecI>(VecR::Gather(&off_[0], ci) + k).StoreAligned(idx);
    }
    switch (row_) {
      case 8: return detail_function_table::eval_rows<8>(&coef_[0], idx, t);
      case 16: return detail_function_table::eval_rows<16>(&coef_[0], idx, t);
      case 24: return detail_function_table::eval_rows<24>(&coef_[0], idx, t);
      default: return detail_function_table::eval_rows<32>(&coef_[0], idx, t);
    }
  }

  template <class Real> Integer FunctionTable<Real>::Degree() const {
    return degree_;
  }

  template <class Real> Long FunctionTable<Real>::Pieces() const {
    return coef_.Dim() / row_;
  }

}

#endif // _SCTL_FUNCTION_TABLE_TXX_
