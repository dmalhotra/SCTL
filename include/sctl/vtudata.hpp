#ifndef _SCTL_VTUDATA_
#define _SCTL_VTUDATA_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)

namespace SCTL_NAMESPACE {

class Comm;
template <class ValueType> class Vector;
template <class ValueType> class Matrix;

/**
 * @brief Struct for storing data in the VTK (Visualization Toolkit) unstructured mesh format.
 * Refer to "File Formats for VTK Version 4.2".
 *
 * This struct provides facilities for storing point and cell data in the VTK unstructured mesh format,
 * which is commonly used for visualization purposes in scientific computing.
 */
class VTUData {
  public:
    /**
     * @brief Type alias for the real type used in VTK.
     */
    typedef float VTKReal;

    // Point data
    Vector<VTKReal> coord;  //!< Vector storing 3D coordinates of points.
    Vector<VTKReal> value;  //!< Vector storing values associated with points.

    // Cell data
    Vector<int32_t> connect;  //!< Vector storing connectivity information for cells.
    Vector<int32_t> offset;   //!< Vector storing offset information for cells.
    Vector<uint8_t> types;    //!< Vector storing cell types.

    /**
     * @brief Write the VTU data to a VTK file.
     *
     * @param fname File name for the output VTK file.
     * @param comm MPI communicator.
     */
    void WriteVTK(const std::string& fname, const Comm& comm = Comm::Self()) const;

    /**
     * Example code showing how to use the VTUData class.
     */
    static void test() {
      VTUData vtu_data;

      // Add 7-particles
      for (long i = 0; i < 7; i++) { // particle i
        for (long k = 0; k < 3; k++) { // coordinate k
          vtu_data.coord.PushBack((VTKReal)drand48());
        }
        vtu_data.value.PushBack((VTKReal)drand48());
      }

      // Add tetrahedron
      vtu_data.types.PushBack(10); // VTK_TETRA (=10)
      for (long i = 0; i < 4; i++) vtu_data.connect.PushBack(i);
      vtu_data.offset.PushBack(vtu_data.connect.Dim());

      // Add triangle
      vtu_data.types.PushBack(5); // VTK_TRIANGLE(=5)
      for (long i = 4; i < 7; i++) vtu_data.connect.PushBack(i);
      vtu_data.offset.PushBack(vtu_data.connect.Dim());

      vtu_data.WriteVTK("vtudata-test");
    }

    template <class ElemLst> void AddElems(const ElemLst elem_lst, Integer order, const Comm& comm = Comm::Self()); // TODO: move to boundary_integral.hpp
    template <class ElemLst, class ValueBasis> void AddElems(const ElemLst elem_lst, const Vector<ValueBasis>& elem_value, Integer order, const Comm& comm = Comm::Self()); // TODO: move to boundary_integral.hpp

  private:
    template <class CoordType, Integer ELEM_DIM> static Matrix<CoordType> VTK_Nodes(Integer order); // TODO: move to boundary_integral.hpp
};

}

#include SCTL_INCLUDE(vtudata.txx)

#endif //_SCTL_VTUDATA_
