/**
 * This demo code shows how to use the class sctl::QuadElemList to build a
 * cubed-sphere geometry and write it to VTK for visualization.
 *
 * To compile and run the code, start in the SCTL root directory and run:
 * make bin/test-quad-elem && ./bin/test-quad-elem
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
using namespace sctl;

namespace {

template <class Real> void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R) {
  switch (face) {
    case 0: x =  1; y =  a; z =  b; break;
    case 1: x = -1; y = -a; z =  b; break;
    case 2: x =  a; y =  1; z = -b; break;
    case 3: x =  a; y = -1; z =  b; break;
    case 4: x =  a; y =  b; z =  1; break;
    case 5: x = -a; y =  b; z = -1; break;
    default: SCTL_ASSERT(false);
  }
  const Real r = sqrt<Real>(x * x + y * y + z * z);
  x *= R / r;
  y *= R / r;
  z *= R / r;
}

}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);

  {
    const Comm comm = Comm::World();
    SCTL_ASSERT_MSG(comm.Size() == 1, "\
        This demo is sequential. In a distributed memory implementation, each process\n\
        would build only its local section of the geometry.");

    const Long ElemOrder = 8;
    const Long PatchPerFace = 3;
    const double Radius = 1.0;
    const Long Nelem = 6 * PatchPerFace * PatchPerFace;

    Vector<double> X;

    const Vector<double>& nds = QuadElemList<double>::ParamNodes(ElemOrder);
    for (Integer face = 0; face < 6; face++) {
      for (Long iu = 0; iu < PatchPerFace; iu++) {
        for (Long iv = 0; iv < PatchPerFace; iv++) {
          for (Long i = 0; i < ElemOrder; i++) {
            const double u = (iu + nds[i]) / (double)PatchPerFace;
            const double a = 2 * u - 1;
            for (Long j = 0; j < ElemOrder; j++) {
              const double v = (iv + nds[j]) / (double)PatchPerFace;
              const double b = 2 * v - 1;

              double x, y, z;
              FacePoint(x, y, z, face, a, b, Radius);
              X.PushBack(x);
              X.PushBack(y);
              X.PushBack(z);
            }
          }
        }
      }
    }

    QuadElemList<double> elem_lst(ElemOrder, X);

    elem_lst.Write("cubed-sphere.geom", comm);
    elem_lst.Read<double>("cubed-sphere.geom", comm);

    Vector<double> Xsurf, Xn;
    Vector<Long> element_wise_node_cnt;
    elem_lst.GetNodeCoord(&Xsurf, &Xn, &element_wise_node_cnt);

    Vector<double> dXdu, dXdv;
    for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
      const auto& nodes = QuadElemList<double>::ParamNodes(ElemOrder);
      Vector<double> dXdu_elem, dXdv_elem;
      elem_lst.GetGeom(nullptr, nullptr, nullptr, &dXdu_elem, &dXdv_elem, nodes, nodes, elem_idx);
      for (const auto& v : dXdu_elem) dXdu.PushBack(v);
      for (const auto& v : dXdv_elem) dXdv.PushBack(v);
    }

    elem_lst.WriteVTK("cubed-sphere", Xn, comm);
  }

  Comm::MPI_Finalize();
  return 0;
}
