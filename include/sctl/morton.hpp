#ifndef _SCTL_MORTON_
#define _SCTL_MORTON_

#include <sctl/common.hpp>
#include <cstdint>

#ifndef SCTL_MAX_DEPTH
#define SCTL_MAX_DEPTH 15
#endif

namespace SCTL_NAMESPACE {

template <Integer DIM = 3> class Morton {

 public:
  #if SCTL_MAX_DEPTH < 7
  typedef uint8_t UINT_T;
  #elif SCTL_MAX_DEPTH < 15
  typedef uint16_t UINT_T;
  #elif SCTL_MAX_DEPTH < 31
  typedef uint32_t UINT_T;
  #elif SCTL_MAX_DEPTH < 63
  typedef uint64_t UINT_T;
  #endif

  static constexpr Integer MAX_DEPTH = SCTL_MAX_DEPTH;

  static constexpr Integer MaxDepth();

  Morton();

  template <class T> explicit Morton(ConstIterator<T> coord, uint8_t depth_ = MAX_DEPTH);

  int8_t Depth() const;

  /**
   * Gives the ccoordinates of the origin of a Morton box.
   */
  template <class ArrayType> void Coord(ArrayType&& coord) const;
  template <class Real> std::array<Real,DIM> Coord() const;

  Morton Next() const;

  Morton Ancestor(uint8_t ancestor_level) const;

  /**
   * \brief Returns the deepest first descendant.
   */
  Morton DFD(uint8_t level = MAX_DEPTH) const;

  /**
   * Return 3^DIM neighbor Morton IDs. If a neighbor doesn't exist then the
   * returned Morton ID has negative depth.
   */
  void NbrList(Vector<Morton>& nbrs, uint8_t level, bool periodic) const;

  void Children(Vector<Morton> &nlst) const;

  bool operator<(const Morton &m) const;

  bool operator>(const Morton &m) const;

  bool operator!=(const Morton &m) const;

  bool operator==(const Morton &m) const;

  bool operator<=(const Morton &m) const;

  bool operator>=(const Morton &m) const;

  bool isAncestor(Morton const &descendant) const;

  Long operator-(const Morton<DIM> &I) const;

  template <Integer D> friend std::ostream& operator<<(std::ostream &out, const Morton<D> &mid);

 private:

  static constexpr UINT_T maxCoord = ((UINT_T)1) << (MAX_DEPTH);

  // StaticArray<UINT_T,DIM> x;
  UINT_T x[DIM];
  int8_t depth;
};

}

#include SCTL_INCLUDE(morton.txx)

#endif  //_SCTL_MORTON_
