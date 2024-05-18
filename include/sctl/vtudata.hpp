#ifndef _SCTL_VTUDATA_
#define _SCTL_VTUDATA_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(vector.hpp)

namespace SCTL_NAMESPACE {

template <class ValueType> class Matrix;

/**
 * @brief Struct for storing data in the VTK (Visualization Toolkit) unstructured grid format.
 * Refer to <a href="https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#unstructured-grid">VTK File Formats - Unstructured grid</a> and
 * <a href="https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png">cell-types</a>.
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
    static void test();

    template <class ElemLst> void AddElems(const ElemLst elem_lst, Integer order, const Comm& comm = Comm::Self()); // TODO: move to boundary_integral.hpp
    template <class ElemLst, class ValueBasis> void AddElems(const ElemLst elem_lst, const Vector<ValueBasis>& elem_value, Integer order, const Comm& comm = Comm::Self()); // TODO: move to boundary_integral.hpp

  private:
    template <class CoordType, Integer ELEM_DIM> static Matrix<CoordType> VTK_Nodes(Integer order); // TODO: move to boundary_integral.hpp
};

}

#include SCTL_INCLUDE(vtudata.txx)

#endif //_SCTL_VTUDATA_
