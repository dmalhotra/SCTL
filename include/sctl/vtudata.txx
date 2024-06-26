#ifndef _SCTL_VTUDATA_TXX_
#define _SCTL_VTUDATA_TXX_

#include <stdlib.h>               // for drand48
#include <cstdint>                // for int32_t, uint32_t, uint8_t, uint16_t
#include <fstream>                // for basic_ofstream, basic_ostream, oper...
#include <iomanip>                // for operator<<, setfill, setw
#include <sstream>                // for basic_stringstream
#include <string>                 // for char_traits, allocator, basic_string

#include "sctl/common.hpp"        // for Integer, Long, SCTL_ASSERT, SCTL_NA...
#include "sctl/vtudata.hpp"       // for VTUData
#include "sctl/comm.hpp"          // for Comm, CommOp
#include "sctl/comm.txx"          // for Comm::Rank, Comm::Allreduce, Comm::...
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator
#include "sctl/iterator.txx"      // for Iterator::Iterator<ValueType>, Iter...
#include "sctl/math_utils.hpp"    // for const_pi, cos
#include "sctl/math_utils.txx"    // for pow
#include "sctl/matrix.hpp"        // for Matrix
#include "sctl/static-array.hpp"  // for StaticArray
#include "sctl/static-array.txx"  // for StaticArray::operator+, StaticArray...
#include "sctl/vector.hpp"        // for Vector
#include "sctl/vector.txx"        // for Vector::Dim, Vector::PushBack, Vect...

namespace sctl {

inline void VTUData::test() {
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

inline void VTUData::WriteVTK(const std::string& fname, const Comm& comm) const {
  typedef typename VTUData::VTKReal VTKReal;
  Long value_dof = 0;
  {  // Write vtu file.
    std::ofstream vtufile;
    {  // Open file for writing.
      std::stringstream vtufname;
      vtufname << fname << std::setfill('0') << std::setw(6) << comm.Rank() << ".vtu";
      vtufile.open(vtufname.str().c_str());
      if (vtufile.fail()) return;
    }
    {  // Write to file.
      Long pt_cnt = coord.Dim() / 3;
      Long cell_cnt = types.Dim();
      { // Set value_dof
        StaticArray<Long,2> pts_cnt{pt_cnt,0};
        StaticArray<Long,2> val_cnt{value.Dim(),0};
        comm.Allreduce(pts_cnt+0, pts_cnt+1, 1, CommOp::SUM);
        comm.Allreduce(val_cnt+0, val_cnt+1, 1, CommOp::SUM);
        value_dof = (pts_cnt[1] ? val_cnt[1] / pts_cnt[1] : 0);
      }

      Vector<int32_t> mpi_rank;
      {  // Set  mpi_rank
        Integer new_myrank = comm.Rank();
        mpi_rank.ReInit(pt_cnt);
        for (Long i = 0; i < mpi_rank.Dim(); i++) mpi_rank[i] = new_myrank;
      }

      bool isLittleEndian;
      {  // Set isLittleEndian
        uint16_t number = 0x1;
        uint8_t *numPtr = (uint8_t *)&number;
        isLittleEndian = (numPtr[0] == 1);
      }

      Long data_size = 0;
      vtufile << "<?xml version=\"1.0\"?>\n";
      vtufile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"" << (isLittleEndian ? "LittleEndian" : "BigEndian") << "\">\n";
      // ===========================================================================
      vtufile << "  <UnstructuredGrid>\n";
      vtufile << "    <Piece NumberOfPoints=\"" << pt_cnt << "\" NumberOfCells=\"" << cell_cnt << "\">\n";
      //---------------------------------------------------------------------------
      vtufile << "      <Points>\n";
      vtufile << "        <DataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"3\" Name=\"Position\" format=\"appended\" offset=\"" << data_size << "\" />\n";
      data_size += sizeof(uint32_t) + coord.Dim() * sizeof(VTKReal);
      vtufile << "      </Points>\n";
      //---------------------------------------------------------------------------
      vtufile << "      <PointData>\n";
      if (value_dof) {  // value
        vtufile << "        <DataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"" << value_dof << "\" Name=\"value\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + value.Dim() * sizeof(VTKReal);
      }
      {  // mpi_rank
        vtufile << "        <DataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + pt_cnt * sizeof(int32_t);
      }
      vtufile << "      </PointData>\n";
      //---------------------------------------------------------------------------
      //---------------------------------------------------------------------------
      vtufile << "      <Cells>\n";
      vtufile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << data_size << "\" />\n";
      data_size += sizeof(uint32_t) + connect.Dim() * sizeof(int32_t);
      vtufile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << data_size << "\" />\n";
      data_size += sizeof(uint32_t) + offset.Dim() * sizeof(int32_t);
      vtufile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"" << data_size << "\" />\n";
      //data_size += sizeof(uint32_t) + types.Dim() * sizeof(uint8_t);
      vtufile << "      </Cells>\n";
      //---------------------------------------------------------------------------
      vtufile << "    </Piece>\n";
      vtufile << "  </UnstructuredGrid>\n";
      // ===========================================================================
      vtufile << "  <AppendedData encoding=\"raw\">\n";
      vtufile << "    _";

      int32_t block_size;
      {  // coord
        block_size = coord.Dim() * sizeof(VTKReal);
        vtufile.write((char *)&block_size, sizeof(int32_t));
        if (coord.Dim()) vtufile.write((char *)&coord[0], coord.Dim() * sizeof(VTKReal));
      }
      if (value_dof) {  // value
        block_size = value.Dim() * sizeof(VTKReal);
        vtufile.write((char *)&block_size, sizeof(int32_t));
        if (value.Dim()) vtufile.write((char *)&value[0], value.Dim() * sizeof(VTKReal));
      }
      {  // mpi_rank
        block_size = mpi_rank.Dim() * sizeof(int32_t);
        vtufile.write((char *)&block_size, sizeof(int32_t));
        if (mpi_rank.Dim()) vtufile.write((char *)&mpi_rank[0], mpi_rank.Dim() * sizeof(int32_t));
      }
      {  // block_size
        block_size = connect.Dim() * sizeof(int32_t);
        vtufile.write((char *)&block_size, sizeof(int32_t));
        if (connect.Dim()) vtufile.write((char *)&connect[0], connect.Dim() * sizeof(int32_t));
      }
      {  // offset
        block_size = offset.Dim() * sizeof(int32_t);
        vtufile.write((char *)&block_size, sizeof(int32_t));
        if (offset.Dim()) vtufile.write((char *)&offset[0], offset.Dim() * sizeof(int32_t));
      }
      {  // types
        block_size = types.Dim() * sizeof(uint8_t);
        vtufile.write((char *)&block_size, sizeof(int32_t));
        if (types.Dim()) vtufile.write((char *)&types[0], types.Dim() * sizeof(uint8_t));
      }

      vtufile << "\n";
      vtufile << "  </AppendedData>\n";
      // ===========================================================================
      vtufile << "</VTKFile>\n";
    }
    vtufile.close();  // close file
  }
  if (!comm.Rank()) {  // Write pvtu file
    std::ofstream pvtufile;
    {  // Open file for writing
      std::stringstream pvtufname;
      pvtufname << fname << ".pvtu";
      pvtufile.open(pvtufname.str().c_str());
      if (pvtufile.fail()) return;
    }
    {  // Write to file.
      pvtufile << "<?xml version=\"1.0\"?>\n";
      pvtufile << "<VTKFile type=\"PUnstructuredGrid\">\n";
      pvtufile << "  <PUnstructuredGrid GhostLevel=\"0\">\n";
      pvtufile << "      <PPoints>\n";
      pvtufile << "        <PDataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"3\" Name=\"Position\"/>\n";
      pvtufile << "      </PPoints>\n";
      pvtufile << "      <PPointData>\n";
      if (value_dof) {  // value
        pvtufile << "        <PDataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"" << value_dof << "\" Name=\"value\"/>\n";
      }
      {  // mpi_rank
        pvtufile << "        <PDataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\"/>\n";
      }
      pvtufile << "      </PPointData>\n";
      {
        // Extract filename from path.
        std::stringstream vtupath;
        vtupath << '/' << fname;
        std::string pathname = vtupath.str();
        std::string fname_ = pathname.substr(pathname.find_last_of("/\\") + 1);
        // char *fname_ = (char*)strrchr(vtupath.str().c_str(), '/') + 1;
        // std::string fname_ =
        // boost::filesystem::path(fname).filename().string().
        for (Integer i = 0; i < comm.Size(); i++) pvtufile << "      <Piece Source=\"" << fname_ << std::setfill('0') << std::setw(6) << i << ".vtu\"/>\n";
      }
      pvtufile << "  </PUnstructuredGrid>\n";
      pvtufile << "</VTKFile>\n";
    }
    pvtufile.close();  // close file
  }
};

template <class ElemLst> inline void VTUData::AddElems(const ElemLst elem_lst, Integer order, const Comm& comm) {
  constexpr Integer COORD_DIM = ElemLst::CoordDim();
  constexpr Integer ElemDim = ElemLst::ElemDim();
  using CoordBasis = typename ElemLst::CoordBasis;
  using CoordType = typename ElemLst::CoordType;
  Long N0 = coord.Dim() / COORD_DIM;
  Long NElem = elem_lst.NElem();

  Matrix<CoordType> nodes = VTK_Nodes<CoordType, ElemDim>(order);
  Integer Nnodes = sctl::pow<ElemDim,Integer>(order);
  SCTL_ASSERT(nodes.Dim(0) == ElemDim);
  SCTL_ASSERT(nodes.Dim(1) == Nnodes);
  { // Set coord
    Matrix<CoordType> vtk_coord;
    auto M = CoordBasis::SetupEval(nodes);
    CoordBasis::Eval(vtk_coord, elem_lst.ElemVector(), M);
    for (Long k = 0; k < NElem; k++) {
      for (Integer i = 0; i < Nnodes; i++) {
        constexpr Integer dim = (COORD_DIM < 3 ? COORD_DIM : 3);
        for (Integer j = 0; j < dim; j++) {
          coord.PushBack((VTUData::VTKReal)vtk_coord[k*COORD_DIM+j][i]);
        }
        for (Integer j = dim; j < 3; j++) {
          coord.PushBack((VTUData::VTKReal)0);
        }
      }
    }
  }

  if (ElemLst::ElemDim() == 2) {
    for (Long k = 0; k < NElem; k++) {
      for (Integer i = 0; i < order-1; i++) {
        for (Integer j = 0; j < order-1; j++) {
          Long idx = k*Nnodes + i*order + j;
          connect.PushBack(N0+idx);
          connect.PushBack(N0+idx+1);
          connect.PushBack(N0+idx+order+1);
          connect.PushBack(N0+idx+order);
          offset.PushBack(connect.Dim());
          types.PushBack(9);
        }
      }
    }
  } else {
    // TODO
    SCTL_ASSERT(false);
  }
}
template <class ElemLst, class ValueBasis> inline void VTUData::AddElems(const ElemLst elem_lst, const Vector<ValueBasis>& elem_value, Integer order, const Comm& comm) {
  constexpr Integer ElemDim = ElemLst::ElemDim();
  using ValueType = typename ValueBasis::ValueType;
  Long NElem = elem_lst.NElem();

  Integer dof = (NElem==0 ? 0 : elem_value.Dim() / NElem);
  SCTL_ASSERT(elem_value.Dim() == NElem * dof);
  AddElems(elem_lst, order, comm);

  Matrix<ValueType> nodes = VTK_Nodes<ValueType, ElemDim>(order);
  Integer Nnodes = sctl::pow<ElemDim,Integer>(order);
  SCTL_ASSERT(nodes.Dim(0) == ElemDim);
  SCTL_ASSERT(nodes.Dim(1) == Nnodes);

  { // Set value
    Matrix<ValueType> vtk_value;
    auto M = ValueBasis::SetupEval(nodes);
    ValueBasis::Eval(vtk_value, elem_value, M);
    for (Long k = 0; k < NElem; k++) {
      for (Integer i = 0; i < Nnodes; i++) {
        for (Integer j = 0; j < dof; j++) {
          value.PushBack((VTUData::VTKReal)vtk_value[k*dof+j][i]);
        }
      }
    }
  }
}

template <class CoordType, Integer ELEM_DIM> inline Matrix<CoordType> VTUData::VTK_Nodes(Integer order) {
  Matrix<CoordType> nodes;
  if (ELEM_DIM == 2) {
    Integer Nnodes = order*order;
    nodes.ReInit(ELEM_DIM, Nnodes);
    for (Integer i = 0; i < order; i++) {
      for (Integer j = 0; j < order; j++) {
        //nodes[0][i*order+j] = i / (CoordType)(order-1);
        //nodes[1][i*order+j] = j / (CoordType)(order-1);
        nodes[0][i*order+j] = 0.5 - 0.5 * sctl::cos<CoordType>((2*i+1) * const_pi<CoordType>() / (2*order));
        nodes[1][i*order+j] = 0.5 - 0.5 * sctl::cos<CoordType>((2*j+1) * const_pi<CoordType>() / (2*order));
      }
    }
  } else {
    // TODO
    SCTL_ASSERT(false);
  }
  return nodes;
}

}

#endif // _SCTL_VTUDATA_TXX_
