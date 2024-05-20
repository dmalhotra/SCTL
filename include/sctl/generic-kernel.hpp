#ifndef _SCTL_GENERIC_KERNEL_HPP_
#define _SCTL_GENERIC_KERNEL_HPP_

#include "sctl/common.hpp"  // for Integer, SCTL_NAMESPACE
#include SCTL_INCLUDE(vec.hpp)     // for Vec

namespace SCTL_NAMESPACE {

template <class ValueType> class Vector;
template <class ValueType> class Matrix;

template <class uKernel, Integer KDIM0, Integer KDIM1, Integer DIM, Integer N_DIM> struct uKerHelper {
  template <Integer digits, class VecType> static void MatEval(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const VecType (&n)[N_DIM], const void* ctx_ptr) {
    uKernel::template uKerMatrix<digits>(u, r, n, ctx_ptr);
  }
};
template <class uKernel, Integer KDIM0, Integer KDIM1, Integer DIM> struct uKerHelper<uKernel,KDIM0,KDIM1,DIM,0> {
  template <Integer digits, class VecType, class NormalType> static void MatEval(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const NormalType& n, const void* ctx_ptr) {
    uKernel::template uKerMatrix<digits>(u, r, ctx_ptr);
  }
};

template <class uKernel> class GenericKernel : public uKernel {

    template <class VecType, Integer K0, Integer K1, Integer D, class ...T> static constexpr Integer get_DIM  (void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], T... args)) { return D;  }
    template <class VecType, Integer K0, Integer K1, Integer D, class ...T> static constexpr Integer get_KDIM0(void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], T... args)) { return K0; }
    template <class VecType, Integer K0, Integer K1, Integer D, class ...T> static constexpr Integer get_KDIM1(void (*uKer)(VecType (&u)[K0][K1], const VecType (&r)[D], T... args)) { return K1; }

    static constexpr Integer DIM   = get_DIM  (uKernel::template uKerMatrix<0,Vec<double,1>>);
    static constexpr Integer KDIM0 = get_KDIM0(uKernel::template uKerMatrix<0,Vec<double,1>>);
    static constexpr Integer KDIM1 = get_KDIM1(uKernel::template uKerMatrix<0,Vec<double,1>>);


    template <Integer cnt> static constexpr Integer argsize_helper() { return 0; }
    template <Integer cnt, class T, class ...T1> static constexpr Integer argsize_helper() { return (cnt == 0 ? sizeof(T) : 0) + argsize_helper<cnt-1, T1...>(); }
    template <Integer idx, class ...T1> static constexpr Integer argsize(void (uKer)(T1... args)) { return argsize_helper<idx, T1...>(); }

    template <Integer cnt> static constexpr Integer argcount_helper() { return cnt; }
    template <Integer cnt, class T, class ...T1> static constexpr Integer argcount_helper() { return argcount_helper<cnt+1, T1...>(); }
    template <class ...T1> static constexpr Integer argcount(void (uKer)(T1... args)) { return argcount_helper<0, T1...>(); }

    static constexpr Integer ARGCNT = argcount(uKernel::template uKerMatrix<0,Vec<double,1>>);
    static constexpr Integer N_DIM = (ARGCNT > 3 ? argsize<2>(uKernel::template uKerMatrix<0,Vec<double,1>>)/sizeof(Vec<double,1>) : 0);
    static constexpr Integer N_DIM_ = (N_DIM?N_DIM:1); // non-zero

  public:

    GenericKernel() : ctx_ptr(nullptr) {}

    static constexpr Integer CoordDim() {
      return DIM;
    }
    static constexpr Integer NormalDim() {
      return N_DIM;
    }
    static constexpr Integer SrcDim() {
      return KDIM0;
    }
    static constexpr Integer TrgDim() {
      return KDIM1;
    }

    const void* GetCtxPtr() const {
      return ctx_ptr;
    }

    template <class Real, bool enable_openmp> static void Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

    template <class Real, bool enable_openmp=false, Integer digits=-1> void Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src) const;

    template <class Real, bool enable_openmp=false, Integer digits=-1> void KernelMatrix(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn) const;

    template <Integer digits, class VecType, class NormalType> static void uKerMatrix(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const NormalType& n, const void* ctx_ptr);

  private:
    void* ctx_ptr;
};

}  // end namespace

#endif // _SCTL_GENERIC_KERNEL_HPP_
