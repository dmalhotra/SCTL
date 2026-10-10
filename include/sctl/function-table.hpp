#ifndef _SCTL_FUNCTION_TABLE_HPP_
#define _SCTL_FUNCTION_TABLE_HPP_

#include "sctl/common.hpp"  // for Integer, Long, sctl
#include "sctl/vector.hpp"  // for Vector
#include "sctl/vec.hpp"     // for Vec

namespace sctl {

  /**
   * A piecewise Chebyshev approximation of a function on [lb, ub], built once to a given number of digits and
   * evaluated for scalars and Vec.
   *
   * @tparam Real float or double.
   */
  template <class Real> class FunctionTable {
    public:
      FunctionTable() = default;

      /**
       * Build the table of f on [lb, ub] with an error of at most 10^-digits times the largest |f| on each piece, or
       * the change of f from rounding x where larger.
       *
       * @param f the function, accurate to the requested digits, called with a Real and returning a value convertible
       * to double.
       * @param lb lower end of the interval.
       * @param ub upper end of the interval.
       * @param digits number of correct digits, at most 15 (float: 6).
       * @param degree polynomial degree of the pieces, below 32, or -1 for the fastest of 7, 15 and 23 whose table fits
       * in 256 KB, else 23.
       */
      template <class Fn> FunctionTable(const Fn& f, Real lb, Real ub, Integer digits, Integer degree = -1);

      /** The value at x, by the polynomial of the nearest piece outside [lb, ub]. */
      Real operator()(Real x) const;

      /** The values at the elements of x, by the polynomial of the nearest piece outside [lb, ub]. */
      template <Integer N> Vec<Real,N> operator()(const Vec<Real,N>& x) const;

      /** The polynomial degree of the pieces. */
      Integer Degree() const;

      /** The number of pieces. */
      Long Pieces() const;

    private:
      template <class Fn> static bool Build(FunctionTable& T, const Fn& f, double lb, double ub, double tol, Integer p, Long bytes, bool uniform);

      static constexpr Long max_bytes = 256 * 1024; // size limit of the tables of a chosen degree and of uniform tables

      Real lb_ = 0;
      Real inv_h_ = 0; // 1/width of a cell
      Long n_cell_ = 0; // number of cells
      Integer degree_ = 0;
      Integer row_ = 0; // number of coefficients of a piece, a multiple of 8 with zeros beyond the degree
      bool uniform_ = true; // one piece per cell
      Vector<Real> sub_; // number of pieces of each cell, a power of 2
      Vector<Real> off_; // index of the first piece of each cell, the pieces being the rows of coef_
      Vector<Real> coef_; // monomial coefficients in t of each piece, lowest degree first, t in [-1, 1] on the piece
  };

}

#endif // _SCTL_FUNCTION_TABLE_HPP_
