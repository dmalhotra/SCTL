#ifndef _SCTL_MORTON_HPP_
#define _SCTL_MORTON_HPP_

#include <ostream>          // for ostream
#include <array>            // for array
#include <cstdint>          // for uint8_t, int8_t, uint32_t

#include "sctl/common.hpp"  // for Integer, Long, SCTL_NAMESPACE

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 15
#endif

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;

/**
 * @brief Morton class template representing a Morton index in a space-filling curve.
 *
 * @tparam DIM Dimensionality of the Morton index. Defaults to 3.
 */
template <Integer DIM = 3> class Morton {

 public:
  /**
   * @brief Unsigned integer type for Morton index representation.
   *
   * The size of this type depends on the maximum depth of the Morton index.
   */
  #if SCTL_MAX_DEPTH < 7
  typedef uint8_t UINT_T;
  #elif SCTL_MAX_DEPTH < 15
  typedef uint16_t UINT_T;
  #elif SCTL_MAX_DEPTH < 31
  typedef uint32_t UINT_T;
  #elif SCTL_MAX_DEPTH < 63
  typedef uint64_t UINT_T;
  #endif

  /**
   * @brief Maximum depth of the Morton index.
   */
  static constexpr Integer MAX_DEPTH = SCTL_MAX_DEPTH;

  /**
   * @brief Get the maximum depth of the Morton index.
   *
   * @return The maximum depth of the Morton index.
   */
  static constexpr Integer MaxDepth();

  /**
   * @brief Default constructor for Morton.
   */
  Morton();

  /**
   * @brief Constructor for Morton using coordinate iterators.
   *
   * @param coord ConstIterator to the coordinates.
   * @param depth_ Depth of the Morton index. Defaults to maximum depth.
   */
  template <class T> explicit Morton(ConstIterator<T> coord, uint8_t depth_ = MAX_DEPTH);

  /**
   * @brief Get the depth of the Morton index.
   *
   * @return The depth of the Morton index.
   */
  int8_t Depth() const;

  /**
   * @brief Get the coordinates of the origin of a Morton box.
   *
   * @tparam ArrayType Type of the array to store coordinates.
   * @param coord Array to store coordinates.
   */
  template <class ArrayType> void Coord(ArrayType&& coord) const;

  /**
   * @brief Get the coordinates of the origin of a Morton box.
   *
   * @tparam Real Floating point type of the coordinates.
   * @return Array containing coordinates.
   */
  template <class Real> std::array<Real,DIM> Coord() const;

  /**
   * @brief Get the Morton index of the next box.
   *
   * @return Morton index of the next box.
   */
  Morton Next() const;

  /**
   * @brief Get the Morton index of the ancestor box at a given level.
   *
   * @param ancestor_level Level of the ancestor box.
   * @return Morton index of the ancestor box.
   */
  Morton Ancestor(uint8_t ancestor_level) const;

  /**
   * @brief Get the Morton index of the deepest first descendant box.
   *
   * @param level Depth level.
   * @return Morton index of the deepest first descendant.
   */
  Morton DFD(uint8_t level = MAX_DEPTH) const;

  /**
   * @brief Get a list of the 3^DIM neighbor Morton IDs. If a neighbor doesn't
   * exist then the corresponding vector element has negative depth.
   *
   * @param nbrs Vector to store neighboring Morton indices.
   * @param level Depth level.
   * @param periodic Flag indicating periodic boundary conditions.
   */
  void NbrList(Vector<Morton>& nbrs, uint8_t level, bool periodic) const;

  /**
   * @brief Get the Morton indices of the children boxes.
   *
   * @param nlst Vector to store Morton indices of children boxes.
   */
  void Children(Vector<Morton> &nlst) const;

  /**
   * @brief Less than comparison operator.
   *
   * @param m Morton index to compare with.
   * @return True if this Morton index is less than the given Morton index, false otherwise.
   */
  bool operator<(const Morton &m) const;

  /**
   * @brief Greater than comparison operator.
   *
   * @param m Morton index to compare with.
   * @return True if this Morton index is greater than the given Morton index, false otherwise.
   */
  bool operator>(const Morton &m) const;

  /**
   * @brief Inequality comparison operator.
   *
   * @param m Morton index to compare with.
   * @return True if this Morton index is not equal to the given Morton index, false otherwise.
   */
  bool operator!=(const Morton &m) const;

  /**
   * @brief Equality comparison operator.
   *
   * @param m Morton index to compare with.
   * @return True if this Morton index is equal to the given Morton index, false otherwise.
   */
  bool operator==(const Morton &m) const;

  /**
   * @brief Less than or equal to comparison operator.
   *
   * @param m Morton index to compare with.
   * @return True if this Morton index is less than or equal to the given Morton index, false otherwise.
   */
  bool operator<=(const Morton &m) const;

  /**
   * @brief Greater than or equal to comparison operator.
   *
   * @param m Morton index to compare with.
   * @return True if this Morton index is greater than or equal to the given Morton index, false otherwise.
   */
  bool operator>=(const Morton &m) const;

  /**
   * @brief Check if this Morton index is an ancestor of another Morton index.
   *
   * @param descendant Morton index to check against.
   * @return True if this Morton index is an ancestor of the given Morton index, false otherwise.
   */
  bool isAncestor(Morton const &descendant) const;

  /**
   * @brief Compute the difference in Morton indices.
   *
   * @param I Morton index to subtract.
   * @return Difference in Morton indices.
   */
  Long operator-(const Morton<DIM> &I) const;

  /**
   * @brief Overloaded stream insertion operator.
   *
   * @param out Output stream.
   * @param mid Morton index to output.
   * @return Reference to the output stream.
   */
  template <Integer D> friend std::ostream& operator<<(std::ostream &out, const Morton<D> &mid);

 private:

  /**
   * @brief Maximum coordinate value.
   */
  static constexpr UINT_T maxCoord = ((UINT_T)1) << (MAX_DEPTH);

  /**
   * @brief Array storing coordinates.
   */
  UINT_T x[DIM];
  // StaticArray<UINT_T,DIM> x;

  /**
   * @brief Depth of the Morton index.
   */
  int8_t depth;
};

}

#endif // _SCTL_MORTON_HPP_
