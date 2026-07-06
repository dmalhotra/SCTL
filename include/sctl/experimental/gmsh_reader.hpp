#ifndef _SCTL_GMSH_READER_HPP_
#define _SCTL_GMSH_READER_HPP_

#include <string>
#include <sctl.hpp>
#include "sctl/experimental/quad_element.hpp"

namespace sctl {

  /**
   * Import a gmsh (MSH 4.1 ASCII) surface mesh of quadrilateral patches and
   * turn it into the AoS nodal-coordinate array consumed by QuadElemList.
   *
   * Each gmsh quad patch (Lagrange quadrangle of order p, i.e. (p+1)x(p+1)
   * nodes on equispaced reference points, hierarchical gmsh node ordering) is
   * resampled, via its Lagrange shape functions, onto a tensor-product
   * Gauss-Legendre grid of a caller-chosen `target_order`. The geometry order
   * of the mesh and the quadrature order of the resulting QuadElemList are thus
   * decoupled. Non-quad surface elements (e.g. triangles) and lower-dimensional
   * (line/point) elements are ignored.
   *
   * @see QuadElemList
   */
  template <class Real> struct GmshReader {

    /**
     * Read all quad patches and resample each onto a target_order GL grid.
     * @param[in] fname path to an MSH 4.1 ASCII file.
     * @param[in] target_order GL order (target_order x target_order nodes/elem).
     * @return AoS node coords {x,y,z,...}, lexicographic (u,v) u-slow per
     * element, all elements concatenated. Feed to QuadElemList(target_order, .).
     */
    static Vector<Real> ReadQuadCoord(const std::string& fname, Integer target_order);

    /**
     * Convenience: build a ready QuadElemList from a gmsh file.
     */
    static QuadElemList<Real> LoadQuadElemList(const std::string& fname, Integer target_order);

    /**
     * Parse the raw quad patches (no resampling). All quads must share the same
     * order. Node coordinates are returned already remapped from gmsh's
     * hierarchical ordering into tensor (i,j) order, u-slow (flat i*nside+j).
     * @param[in] fname path to an MSH 4.1 ASCII file.
     * @param[out] nside nodes per side (p+1) of each patch.
     * @param[out] src AoS coords {x,y,z,...}, nside*nside nodes/patch, u-slow.
     * @return number of quad patches read.
     */
    static Long ReadQuadPatches(const std::string& fname, Integer& nside, Vector<Real>& src);

    /** gmsh element type -> nodes per side of the full-tensor Lagrange quad (0 if not such a quad). */
    static Integer QuadTypeToNside(Integer gmsh_type);

    /**
     * Permutation from gmsh-local quad node index to tensor flat index i*nside+j
     * (i = u-index, j = v-index). Recursive corners->edges->interior ordering.
     * @param[in] nside nodes per side (>=2).
     */
    static Vector<Long> GmshQuadTensorPerm(Integer nside);
  };

}

#endif // _SCTL_GMSH_READER_HPP_
