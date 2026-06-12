// Per-function tests for sctl/vtudata.{hpp,txx}.
//
// VTUData has a small public surface: 5 Vector<> fields (coord, value,
// connect, offset, types) and WriteVTK. We exercise field initialization,
// write to disk, smoke-check the produced .pvtu file is valid XML, then
// remove the artifacts.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "sctl/common.hpp"
#include "sctl/comm.hpp"
#include "sctl/comm.txx"
#include "sctl/iterator.hpp"
#include "sctl/iterator.txx"
#include "sctl/vector.hpp"
#include "sctl/vector.txx"
#include "sctl/vtudata.hpp"
#include "sctl/vtudata.txx"

#include "test-utils.hpp"

using sctl::Long;
using sctl::VTUData;
using sctl::Comm;

static std::string read_file(const std::string& fname) {
  std::ifstream f(fname);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);

  // --- empty VTUData: fields are default-constructed Vectors ---
  std::printf("default ctor :\n");
  {
    VTUData vtu;
    CHECK(vtu.coord  .Dim() == 0);
    CHECK(vtu.value  .Dim() == 0);
    CHECK(vtu.connect.Dim() == 0);
    CHECK(vtu.offset .Dim() == 0);
    CHECK(vtu.types  .Dim() == 0);
  }

  // --- assemble a tiny single-tetrahedron mesh and WriteVTK ---
  // VTK_TETRA = cell type 10. 4 corners.
  std::printf("WriteVTK (single tet) :\n");
  {
    VTUData vtu;
    // 4 points in 3D
    vtu.coord = sctl::Vector<float>({0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f,
                                     0.0f, 1.0f, 0.0f,
                                     0.0f, 0.0f, 1.0f});
    vtu.value = sctl::Vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    vtu.connect = sctl::Vector<int32_t>({0, 1, 2, 3});
    vtu.offset  = sctl::Vector<int32_t>({4});
    vtu.types   = sctl::Vector<uint8_t>({10});

    const std::string fname = "/tmp/sctl-test-vtudata";
    vtu.WriteVTK(fname);

    // WriteVTK produces "<fname>.pvtu" and "<fname>_0.vtu" (rank 0 piece).
    const std::string pvtu_path = fname + ".pvtu";
    std::ifstream f(pvtu_path);
    CHECK(f.good());

    const std::string xml = read_file(pvtu_path);
    CHECK(!xml.empty());
    CHECK(xml.find("VTKFile") != std::string::npos);

    // remove the artifacts
    std::remove(pvtu_path.c_str());
    std::remove((fname + "000000.vtu").c_str());
  }

  // --- WriteVTK on empty data is a no-op (or at least doesn't crash) ---
  std::printf("WriteVTK (empty) :\n");
  {
    VTUData empty;
    const std::string fname = "/tmp/sctl-test-vtudata-empty";
    empty.WriteVTK(fname);
    // Either no file is produced, or an empty-mesh file is. Either way, no crash.
    CHECK(true);
    std::remove((fname + ".pvtu").c_str());
    std::remove((fname + "000000.vtu").c_str());
  }

  TEST_SUMMARY_RETURN();

  Comm::MPI_Finalize();
}
