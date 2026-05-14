#ifndef _SCTL_VTUDATA_HPP_
#define _SCTL_VTUDATA_HPP_

#include <cstdint>            // for int32_t, uint8_t
#include <string>             // for string

#include "sctl/common.hpp"    // for Integer, sctl
#include "sctl/comm.hpp"      // for Comm
#include "sctl/vector.hpp"    // for Vector

namespace sctl {

/**
 * This struct provides facilities for storing point and cell data in the VTK (Visualization Toolkit) unstructured mesh
 * format, which is commonly used for visualization purposes in scientific computing.
 * Refer to <a href="https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#unstructured-grid">VTK File Formats - Unstructured grid</a> and
 * <a href="https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png">cell-types</a>.
 */
class VTUData {
  public:
    /**
     * Type alias for the real type used in VTK output. Fixed to `float` for
     * ParaView/VTK compatibility; higher-precision inputs are silently
     * narrowed when written into `coord`/`value`. Fine for visualization,
     * not for numerical round-trips.
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
     * Write the VTU data to a VTK file.
     *
     * @param fname File name for the output VTK file.
     * @param comm MPI communicator.
     */
    void WriteVTK(const std::string& fname, const Comm& comm = Comm::Self()) const;

    /**
     * Example code showing how to use the VTUData class.
     */
    static void test();
};

}

#endif // _SCTL_VTUDATA_HPP_
