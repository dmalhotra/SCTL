#ifndef _SCTL_FMM_WRAPPER_HPP_
#define _SCTL_FMM_WRAPPER_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(kernel_functions.hpp)

#include <map>
#include <string>

#ifdef SCTL_HAVE_PVFMM
namespace pvfmm {
  template <class Real> struct Kernel;
  template <class Real> class MPI_Node;
  template <class Node> class FMM_Node;
  template <class FMM_Node> class FMM_Pts;
  template <class FMM_Mat> class FMM_Tree;
  template <class Real> using PtFMM_Node = FMM_Node<MPI_Node<Real>>;
  template <class Real> using PtFMM      = FMM_Pts<PtFMM_Node<Real>>;
  template <class Real> using PtFMM_Tree = FMM_Tree<PtFMM<Real>>;
}
#endif

namespace SCTL_NAMESPACE {

class Comm;
template <class ValueType> class Vector;

/**
 * Evaluate potentials from particle sources using PVFMM.
 */
template <class Real, Integer DIM> class ParticleFMM {
  public:

    // Delete copy constructor and assignment operator
    ParticleFMM(const ParticleFMM&) = delete;
    ParticleFMM& operator= (const ParticleFMM&) = delete;

    /**
     * Constructor.
     *
     * @param[in] comm Communicator for distributed memory parallelism.
     */
    ParticleFMM(const Comm& comm = Comm::Self());

    /**
     * Destructor
     */
    ~ParticleFMM();

    /**
     * Set communicator.
     *
     * @param[in] comm Communicator for distributed memory parallelism.
     */
    void SetComm(const Comm& comm);

    /**
     * Set FMM accuracy
     *
     * @param[in] digits number of digits of accuracy.
     */
    void SetAccuracy(Integer digits);

    /**
     * Set kernel objects for KIFMM.
     *
     * @param[in] ker_m2m kernel for multipole-to-multipole translations.
     * @param[in] ker_m2l kernel for multipole-to-local translations.
     * @param[in] ker_l2l kernel for local-to-local translations.
     */
    template <class KerM2M, class KerM2L, class KerL2L> void SetKernels(const KerM2M& ker_m2m, const KerM2L& ker_m2l, const KerL2L& ker_l2l);

    /**
     * Add a source type.
     *
     * @param[in] name name for the source type.
     * @param[in] ker_s2m kernel for source-to-multipole translations.
     * @param[in] ker_s2l kernel for source-to-local translations.
     */
    template <class KerS2M, class KerS2L> void AddSrc(const std::string& name, const KerS2M& ker_s2m, const KerS2L& ker_s2l);

    /**
     * Add a target type.
     *
     * @param[in] name name for the target type.
     * @param[in] ker_m2t kernel for multipole-to-target translations.
     * @param[in] ker_l2t kernel for local-to-target translations.
     */
    template <class KerM2T, class KerL2T> void AddTrg(const std::string& name, const KerM2T& ker_m2t, const KerL2T& ker_l2t);

    /**
     * Set kernel function for source-to-target interactions.
     *
     * @param[in] src_name name for the source type.
     * @param[in] trg_name name for the target type.
     * @param[in] ker_s2t kernel for source-to-target translations.
     */
    template <class KerS2T> void SetKernelS2T(const std::string& src_name, const std::string& trg_name, const KerS2T& ker_s2t);

    /**
     * Delete a source type.
     *
     * @param[in] name name for the source type.
     */
    void DeleteSrc(const std::string& name);

    /**
     * Delete a target type.
     *
     * @param[in] name name for the target type.
     */
    void DeleteTrg(const std::string& name);

    /**
     * Set coordinates for a source type.
     *
     * @param[in] name name for the source type.
     * @param[in] src_coord coordinates for the source particles in AoS order.
     * @param[in] src_normal normal vectors at each source if the kernel requires it.
     */
    void SetSrcCoord(const std::string& name, const Vector<Real>& src_coord, const Vector<Real>& src_normal = Vector<Real>());

    /**
     * Set densities for a source type.
     *
     * @param[in] name name for the source type.
     * @param[in] src_density density for the source particles in AoS order.
     */
    void SetSrcDensity(const std::string& name, const Vector<Real>& src_density);

    /**
     * Set coordinates for a target type.
     *
     * @param[in] name name for the target type.
     * @param[in] trg_coord coordinates for the target particles in AoS order.
     */
    void SetTrgCoord(const std::string& name, const Vector<Real>& trg_coord);

    /**
     * Evaluate the potential for a target type using FMM. Defaults to direct evaluation when FMM not available.
     *
     * @param[out] U the computed potential.
     * @param[in] trg_name name for the target type.
     */
    void Eval(Vector<Real>& U, const std::string& trg_name) const;

    /**
     * Evaluate the potential for a target type using direct evaluation.
     *
     * @param[out] U the computed potential.
     * @param[in] trg_name name for the target type.
     */
    void EvalDirect(Vector<Real>& U, const std::string& trg_name) const;

    /**
     * Example code showing usage of class ParticleFMM.
     */
    static void test(const Comm& comm);

  private:

    struct FMMKernels;
    struct SrcData;
    struct TrgData;
    struct S2TData;

    static void BuildSrcTrgScal(const S2TData& s2t_data, bool verbose);

    template <class Ker> static void DeleteKer(Iterator<char> ker);

    void CheckKernelDims() const;

    void DeleteS2T(const std::string& src_name, const std::string& trg_name);

    #ifdef SCTL_HAVE_PVFMM
    template <class SCTLKernel, bool use_dummy_normal=false> struct PVFMMKernelFn; // construct PVFMMKernel from SCTLKernel

    void EvalPVFMM(Vector<Real>& U, const std::string& trg_name) const;
    #endif

    FMMKernels fmm_ker;
    std::map<std::string, SrcData> src_map;
    std::map<std::string, TrgData> trg_map;
    std::map<std::pair<std::string,std::string>, S2TData> s2t_map;

    Comm comm_;
    Integer digits_;
};

}  // end namespace

#include SCTL_INCLUDE(fmm-wrapper.txx)

#endif  //_SCTL_FMM_WRAPPER_HPP_
