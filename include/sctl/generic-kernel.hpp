#ifndef _SCTL_GENERIC_KERNEL_HPP_
#define _SCTL_GENERIC_KERNEL_HPP_

#include "sctl/common.hpp"  // for Integer, sctl
#include "sctl/vec.hpp"     // for Vec

namespace sctl {

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

/**
 * @class GenericKernel
 * This class is designed to help create new custom kernel objects. Kernel objects for Laplace and Stokes in 3D are defined in `kernel_functions.hpp` and can be used as a template.
 *
 * @tparam uKernel The base class for the kernel.
 *
 * @see kernel_functions.hpp
 */
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

    /**
     * Default constructor.
     */
    GenericKernel();

    /**
     * Returns the coordinate dimension.
     * @return The coordinate dimension.
     */
    static constexpr Integer CoordDim();

    /**
     * Returns the dimensions of the normal vector. It will return zero if the kernel function does
     * not require a normal vector; otherwise it will return DIM.
     *
     * @return The normal dimension.
     */
    static constexpr Integer NormalDim();

    /**
     * Returns the source dimension.
     * @return The source dimension.
     */
    static constexpr Integer SrcDim();

    /**
     * Returns the target dimension.
     * @return The target dimension.
     */
    static constexpr Integer TrgDim();

    /**
     * Set the pointer to the context data.
     */
    void SetCtxPtr(void* ctx);

    /**
     * Returns a constant pointer to the context.
     * @return A constant pointer to the context.
     */
    const void* GetCtxPtr() const;

    /**
     * Evaluates the kernel and stores the result in `v_trg`.
     * @tparam Real The type of the real numbers used.
     * @tparam enable_openmp A boolean flag to enable OpenMP.
     * @param v_trg The vector to store the potential result.
     * @param r_trg The vector of target point coordinates.
     * @param r_src The vector of source point coordinates.
     * @param n_src The vector of source normals.
     * @param v_src The vector of source densities.
     * @param digits The number of significant digits for evaluation.
     * @param self A constant iterator pointing to the self interaction flag.
     */
    template <class Real, bool enable_openmp> static void Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src, Integer digits, ConstIterator<char> self);

    /**
     * Evaluates the kernel and stores the result in `v_trg`.
     * @tparam Real The type of the real numbers used.
     * @tparam enable_openmp A boolean flag to enable OpenMP. Default is false.
     * @tparam digits The number of significant digits for evaluation. Default is -1 for machine-precision.
     * @param v_trg The vector to store the potential result.
     * @param r_trg The vector of target point coordinates.
     * @param r_src The vector of source point coordinates.
     * @param n_src The vector of source normals.
     * @param v_src The vector of source densities.
     */
    template <class Real, bool enable_openmp=false, Integer digits=-1> void Eval(Vector<Real>& v_trg, const Vector<Real>& r_trg, const Vector<Real>& r_src, const Vector<Real>& n_src, const Vector<Real>& v_src) const;

    /**
     * Computes the kernel matrix and stores it in `M`.
     * @tparam Real The type of the real numbers used.
     * @tparam enable_openmp A boolean flag to enable OpenMP. Default is false.
     * @tparam digits The number of significant digits for evaluation. Default is -1.
     * @param M The matrix to store the kernel matrix.
     * @param Xt The vector of target point coordinates.
     * @param Xs The vector of source point coordinates.
     * @param Xn The vector of source normals.
     */
    template <class Real, bool enable_openmp=false, Integer digits=-1> void KernelMatrix(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& Xs, const Vector<Real>& Xn) const;

    /**
     * Static method for kernel matrix computation.
     * @tparam digits The number of significant digits for evaluation.
     * @tparam VecType The vector type.
     * @tparam NormalType The normal type.
     * @param u The output kernel matrix.
     * @param r The input coordinates.
     * @param n The input normals.
     * @param ctx_ptr The context pointer.
     */
    template <Integer digits, class VecType, class NormalType> static void uKerMatrix(VecType (&u)[KDIM0][KDIM1], const VecType (&r)[DIM], const NormalType& n, const void* ctx_ptr);

  private:
    void* ctx_ptr;
};

}  // end namespace

#endif // _SCTL_GENERIC_KERNEL_HPP_
