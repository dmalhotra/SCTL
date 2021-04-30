#ifndef _SCTL_BOUNDARY_INTEGRAL_HPP_
#define _SCTL_BOUNDARY_INTEGRAL_HPP_

#include SCTL_INCLUDE(quadrule.hpp)
#include SCTL_INCLUDE(fft_wrapper.hpp)
#include SCTL_INCLUDE(tensor.hpp)
#include SCTL_INCLUDE(tree.hpp)
#include SCTL_INCLUDE(kernel_functions.hpp)
#include SCTL_INCLUDE(matrix.hpp)
#include SCTL_INCLUDE(ompUtils.hpp)
#include SCTL_INCLUDE(morton.hpp)
#include SCTL_INCLUDE(profile.hpp)
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(comm.hpp)
#include SCTL_INCLUDE(legendre_rule.hpp)
#include SCTL_INCLUDE(common.hpp)

#include <map>
#include <set>
#include <functional>

namespace SCTL_NAMESPACE {

  template <class Real, class Kernel> class BoundaryIntegralOp {
      static constexpr Integer KDIM0 = Kernel::SrcDim();
      static constexpr Integer KDIM1 = Kernel::TrgDim();
      static constexpr Integer COORD_DIM = 3;

    public:

      BoundaryIntegralOp() = delete;
      BoundaryIntegralOp(const BoundaryIntegralOp&) = delete;
      BoundaryIntegralOp& operator= (const BoundaryIntegralOp&) = delete;

      BoundaryIntegralOp(const Kernel& ker, const Comm& comm = Comm::Self()) : tol_(1e-10), ker_(ker), comm_(comm) {
        ClearSetup();
      }

      ~BoundaryIntegralOp() {
        Vector<std::string> elem_lst_name;
        for (auto& it : elem_lst_map) elem_lst_name.PushBack(it.first);
        for (const auto& name : elem_lst_name) DeleteElemList(name);
      }

      void SetAccuracy(Real tol) {
        setup_far_flag = false;
        setup_self_flag = false;
        setup_near_flag = false;
        tol_ = tol;
      }

      template <class ElemType> void AddElemList(const Vector<ElemType>& elem_lst, const std::string& name = std::to_string(typeid(ElemType).hash_code())) {
        static_assert(std::is_trivial<ElemType>::value, "ElemeType must be trivial.");
        SCTL_ASSERT_MSG(elem_lst_map.find(name) == elem_lst_map.end(), "Element list already exists.");
        //SCTL_ASSERT_MSG(is_same_string_across_processes(name, comm_), "List name must match on all processes.");

        elem_lst_map[name].ReInit(elem_lst.Dim()*sizeof(ElemType), (Iterator<char>)(ConstIterator<char>)elem_lst.begin());
        elem_data_map[name].typeid_hash = typeid(ElemType).hash_code();
        elem_data_map[name].size = sizeof(ElemType);
        elem_data_map[name].NodeCount = ElemType::NodeCount();
        elem_data_map[name].FarFieldQuadNodeCount = ElemType::FarFieldQuadNodeCount();
        elem_data_map[name].GetNodeCoord = ElemType::template GetAllNodeCoord<char>;
        elem_data_map[name].GetFarFieldQuadNodes = ElemType::template GetAllFarFieldQuadNodes<char>;
        elem_data_map[name].GetFarFieldQuadDensity = ElemType::template GetAllFarFieldQuadDensity<KDIM0>;
        elem_data_map[name].SelfInterac = ElemType::template AllSelfInterac<Kernel,char>;
        elem_data_map[name].NearInterac = ElemType::template AllNearInterac<Kernel,char>;
        ClearSetup();
      }

      template <class ElemType> void GetElemList(Vector<ElemType>& elem_lst, const std::string& name = std::to_string(typeid(ElemType).hash_code())) const {
        SCTL_ASSERT_MSG(elem_lst_map.find(name) != elem_lst_map.end(), "Element list does not exist.");
        SCTL_ASSERT_MSG(elem_data_map.at(name).typeid_hash == typeid(ElemType).hash_code(), "Element type is invalid.");

        const auto& elem_lst_ = elem_lst_map.at(name);
        elem_lst.ReInit(elem_lst_.Dim()/sizeof(ElemType), (Iterator<ElemType>)elem_lst_.begin());
      }

      void DeleteElemList(const std::string& name) {
        SCTL_ASSERT_MSG(elem_lst_map.find(name) != elem_lst_map.end(), "Element list does not exist.");

        elem_data_map.erase(name);
        elem_lst_map.erase(name);
        ClearSetup();
      }

      template <class ElemType> void DeleteElemList() {
        DeleteElemList(std::to_string(typeid(ElemType).hash_code()));
      }

      Long Dim(Integer k) const {
        SetupBasic();
        if (k == 0) {
          const Long Nelem = elem_nds_cnt.Dim();
          return (elem_nds_dsp[Nelem-1] + elem_nds_cnt[Nelem-1]) * KDIM0;
        }
        if (k == 1) {
          return (Xtrg.Dim()/COORD_DIM) * KDIM1;
        }
        SCTL_ASSERT(false);
        return -1;
      }

      void Setup() const {
        if (setup_flag && setup_far_flag && setup_self_flag && setup_near_flag) return;
        Profile::Tic("Setup", &comm_);
        SetupBasic();
        SetupFar();
        SetupSelf();
        SetupNear();
        Profile::Toc();
      }

      void ClearSetup() const {
        setup_flag = false;
        setup_far_flag = false;
        setup_self_flag = false;
        setup_near_flag = false;
      }

      void ComputePotential(Vector<Real>& U, const Vector<Real>& F) const {
        Setup();
        Profile::Tic("Eval", &comm_);
        ComputeFarField(U, F);
        ComputeNearInterac(U, F);
        Profile::Toc();
      }

    private:

      static bool is_same_string_across_processes(const std::string& S, const Comm& comm) {
        StaticArray<Long,1> max_len, min_len, len = {(Long)S.size()};
        comm.Allreduce((ConstIterator<Long>)len, (Iterator<Long>)max_len, 1, Comm::CommOp::MAX);
        comm.Allreduce((ConstIterator<Long>)len, (Iterator<Long>)min_len, 1, Comm::CommOp::MIN);
        if (max_len[0] != len[0] || min_len[0] != len[0]) return false;

        Vector<char> S_(len[0]), S_max(len[0]), S_min(len[0]);
        for (Long i = 0; i < len[0]; i++) S_[i] = S[i];
        comm.Allreduce(S_.begin(), S_max.begin(), len[0], Comm::CommOp::MAX);
        comm.Allreduce(S_.begin(), S_min.begin(), len[0], Comm::CommOp::MIN);
        for (Long i = 0; i < len[0]; i++) {
          if (S_[i] != S_max[i]) return false;
          if (S_[i] != S_min[i]) return false;
        }
        return true;
      }
      template <class VType> static void concat_vecs(Vector<VType>& v, const Vector<Vector<VType>>& vec_lst) {
        const Long N = vec_lst.Dim();
        Vector<Long> dsp(N+1); dsp[0] = 0;
        for (Long i = 0; i < N; i++) {
          dsp[i+1] = dsp[i] + vec_lst[i].Dim();
        }
        if (v.Dim() != dsp[N]) v.ReInit(dsp[N]);
        for (Long i = 0; i < N; i++) {
          Vector<VType> v_(vec_lst[i].Dim(), v.begin()+dsp[i], false);
          v_ = vec_lst[i];
        }
      }
      static void BuildNearList(Vector<Real>& Xtrg_near, Vector<Long>& near_scatter_index, Vector<Long>& near_trg_cnt, Vector<Long>& near_elem_cnt, Vector<Long>& near_trg_dsp, Vector<Long>& near_elem_dsp, const Vector<Real>& Xtrg, const Vector<Real>& Xsrc, const Vector<Real>& src_radius, const Vector<Long>& src_elem_nds_cnt, const Vector<Long>& src_elem_nds_dsp, const Comm& comm) {
        // Input: Xtrg, Xsrc, src_radius, src_elem_nds_cnt, src_elem_nds_dsp, comm
        // Output: Xtrg_near, near_scatter_index, near_trg_cnt, near_elem_cnt, near_trg_dsp, near_elem_dsp

        struct NodeData {
          Long idx;
          Real rad;
          StaticArray<Real,COORD_DIM> X;
          Morton<COORD_DIM> mid;
          Long elem_idx;
          Long pid;
        };
        auto comp_node_mid = [](const NodeData& A, const NodeData& B) {
          return A.mid < B.mid;
        };
        auto comp_node_eid_idx = [](const NodeData& A, const NodeData& B) {
          return A.elem_idx<B.elem_idx || (A.elem_idx==B.elem_idx && A.idx<B.idx);
        };
        auto node_dist2 = [](const NodeData& A, const NodeData& B) {
          Real dist2 = 0;
          for (Long i = 0; i < COORD_DIM; i++) {
            Real dX = A.X[i] - B.X[i];
            dist2 += dX * dX;
          }
          return dist2;
        };

        const Long Ntrg = Xtrg.Dim()/COORD_DIM;
        const Long Nsrc = Xsrc.Dim()/COORD_DIM;
        const Long Nelem = src_elem_nds_cnt.Dim();

        Long trg_offset, src_offset, elem_offset;
        { // set trg_offset, src_offset, elem_offset
          StaticArray<Long,3> send_buff={Ntrg, Nsrc, Nelem}, recv_buff={0,0,0};
          comm.Scan((ConstIterator<Long>)send_buff, (Iterator<Long>)recv_buff, 3, Comm::CommOp::SUM);
          trg_offset  = recv_buff[0] - send_buff[0];
          src_offset  = recv_buff[1] - send_buff[1];
          elem_offset = recv_buff[2] - send_buff[2];
        }

        Vector<NodeData> trg_nodes(Ntrg), src_nodes(Nsrc);
        { // Set trg_nodes, src_nodes
          Real BBlen_inv;
          StaticArray<Real,COORD_DIM> BBX0;
          { // Determine bounding-box
            StaticArray<Real,COORD_DIM> X0_local;
            if (Ntrg) {
              for (Long k = 0; k < COORD_DIM; k++) {
                X0_local[k] = Xtrg[k];
              }
            } else if (Nsrc) {
              for (Long k = 0; k < COORD_DIM; k++) {
                X0_local[k] = Xsrc[k];
              }
            } else {
              SCTL_ASSERT_MSG(false, "Local source and target vectors cannot both be empty!");
            }
            for (Long i = 0; i < Ntrg; i++) {
              for (Long k = 0; k < COORD_DIM; k++) {
                X0_local[k] = std::min<Real>(X0_local[k], Xtrg[i*COORD_DIM+k]);
              }
            }
            for (Long i = 0; i < Nsrc; i++) {
              for (Long k = 0; k < COORD_DIM; k++) {
                X0_local[k] = std::min<Real>(X0_local[k], Xsrc[i*COORD_DIM+k]);
              }
            }
            comm.Allreduce<Real>(X0_local, BBX0, COORD_DIM, Comm::CommOp::MIN);

            Real BBlen, len_local = 0;
            for (Long i = 0; i < Ntrg; i++) {
              for (Long k = 0; k < COORD_DIM; k++) {
                len_local = std::max<Real>(len_local, Xtrg[i*COORD_DIM+k]-BBX0[k]);
              }
            }
            for (Long i = 0; i < Nsrc; i++) {
              for (Long k = 0; k < COORD_DIM; k++) {
                len_local = std::max<Real>(len_local, Xsrc[i*COORD_DIM+k]-BBX0[k]);
              }
            }
            comm.Allreduce<Real>(Ptr2ConstItr<Real>(&len_local,1), Ptr2Itr<Real>(&BBlen,1), 1, Comm::CommOp::MAX);
            BBlen_inv = 1/BBlen;
          }
          { // Expand bounding-box so that no points are on the boundary
            for (Long i = 0; i < COORD_DIM; i++) {
              BBX0[i] -= 0.05/BBlen_inv;
            }
            BBlen_inv /= 1.1;
          }

          for (Long i = 0; i < Ntrg; i++) { // Set trg_nodes
            StaticArray<Real,COORD_DIM> Xmid;
            trg_nodes[i].idx = trg_offset + i;
            trg_nodes[i].rad = 0;
            for (Long k = 0; k < COORD_DIM; k++) {
              trg_nodes[i].X[k] = Xtrg[i*COORD_DIM+k];
              Xmid[k] = (Xtrg[i*COORD_DIM+k]-BBX0[k]) * BBlen_inv;
            }
            trg_nodes[i].mid = Morton<COORD_DIM>((ConstIterator<Real>)Xmid);
            trg_nodes[i].elem_idx = 0;
            trg_nodes[i].pid = comm.Rank();
          }
          for (Long i = 0; i < Nsrc; i++) { // Set src_nodes
            Integer depth = (Integer)(log(src_radius[i]*BBlen_inv)/log(0.5));
            StaticArray<Real,COORD_DIM> Xmid;
            src_nodes[i].idx = src_offset + i;
            src_nodes[i].rad = src_radius[i];
            for (Long k = 0; k < COORD_DIM; k++) {
              src_nodes[i].X[k] = Xsrc[i*COORD_DIM+k];
              Xmid[k] = (Xsrc[i*COORD_DIM+k]-BBX0[k]) * BBlen_inv;
            }
            src_nodes[i].mid = Morton<COORD_DIM>((ConstIterator<Real>)Xmid, depth);
            src_nodes[i].pid = comm.Rank();
          }
          for (Long i = 0; i < Nelem; i++) { // Set src_nodes.elem_idx
            for (Long j = 0; j < src_elem_nds_cnt[i]; j++) {
              src_nodes[src_elem_nds_dsp[i]+j].elem_idx = elem_offset + i;
            }
          }
        }

        Vector<NodeData> trg_nodes0, src_nodes0, splitter_nodes(comm.Size());
        { // Set trg_nodes0 <- sort(trg_nodes), src_nodes0 <- sort(src_nodes)
          comm.HyperQuickSort(src_nodes, src_nodes0, comp_node_mid);
          comm.HyperQuickSort(trg_nodes, trg_nodes0, comp_node_mid);

          SCTL_ASSERT(src_nodes.Dim());
          comm.Allgather(src_nodes.begin(), 1, splitter_nodes.begin(), 1);
          comm.PartitionS(trg_nodes0, src_nodes0[0], comp_node_mid);
        }

        Vector<NodeData> src_nodes1;
        { // Set src_nodes1 <- src_nodes0 + halo // TODO: replace allgather with halo-exchange
          const Long Np = comm.Size();
          Vector<Long> cnt0(1), cnt(Np), dsp(Np);
          cnt0[0] = src_nodes0.Dim(); dsp[0] = 0;
          comm.Allgather(cnt0.begin(), 1, cnt.begin(), 1);
          omp_par::scan(cnt.begin(), dsp.begin(), Np);

          src_nodes1.ReInit(dsp[Np-1] + cnt[Np-1]);
          comm.Allgatherv(src_nodes0.begin(), src_nodes0.Dim(), src_nodes1.begin(), cnt.begin(), dsp.begin());
        }

        Vector<NodeData> near_lst;
        if (src_nodes1.Dim()) { // Set near_lst
          // sort by elem_idx and mid
          auto comp_elem_idx_mid = [](const NodeData& A, const NodeData& B) {
            return (A.elem_idx<B.elem_idx) || (A.elem_idx==B.elem_idx && A.mid<B.mid);
          };
          omp_par::merge_sort(src_nodes1.begin(), src_nodes1.end(), comp_elem_idx_mid);

          // Preallocate memory // TODO: parallelize
          Vector<Morton<COORD_DIM>> src_mid_lst, trg_mid_lst, nbr_lst;
          Vector<std::pair<Long,Long>> trg_src_near_mid;
          std::set<Morton<COORD_DIM>> trg_mid_set;
          Vector<Long> src_range, trg_range;

          Long eid0 = src_nodes1[0].elem_idx;
          Long eid1 = src_nodes1[src_nodes1.Dim()-1].elem_idx + 1;
          for (Long eid = eid0; eid < eid1; eid++) { // loop over all elements
            Long src_idx0, src_idx1;
            { // Set (src_idx0, src_idx1) the index range of nodes with elem_idx eid
              NodeData srch_node;
              srch_node.elem_idx = eid;
              src_idx0 = std::lower_bound(src_nodes1.begin(), src_nodes1.end(), srch_node, [](const NodeData& A, const NodeData& B){return A.elem_idx<B.elem_idx;}) - src_nodes1.begin();
              src_idx1 = std::upper_bound(src_nodes1.begin(), src_nodes1.end(), srch_node, [](const NodeData& A, const NodeData& B){return A.elem_idx<B.elem_idx;}) - src_nodes1.begin();
            }
            { // build near-list for element eid
              trg_src_near_mid.ReInit(0);
              src_mid_lst.ReInit(0);
              trg_mid_lst.ReInit(0);
              src_range.ReInit(0);
              trg_range.ReInit(0);
              trg_mid_set.clear();
              { // build src_mid_lst, src_range
                Long src_idx = src_idx0;
                while (src_idx < src_idx1) {
                  NodeData nxt_node;
                  nxt_node.mid = src_nodes1[src_idx].mid.Next();
                  Long src_idx_new = std::lower_bound(src_nodes1.begin()+src_idx, src_nodes1.begin()+src_idx1, nxt_node, comp_node_mid) - src_nodes1.begin();
                  src_mid_lst.PushBack(src_nodes1[src_idx].mid);
                  src_range.PushBack(src_idx    );
                  src_range.PushBack(src_idx_new);
                  src_idx = src_idx_new;
                }
              }
              { // build trg_mid_lst, trg_range
                Morton<COORD_DIM> nxt_node;
                for (const auto& src_mid : src_mid_lst) {
                  src_mid.NbrList(nbr_lst, src_mid.Depth(), false);
                  for (const auto& mid : nbr_lst) {
                    trg_mid_set.insert(mid);
                  }
                }
                for (const auto& trg_mid : trg_mid_set) {
                  if (trg_mid >= nxt_node) {
                    nxt_node = trg_mid.Next();
                    NodeData node0, node1;
                    node0.mid = trg_mid;
                    node1.mid = nxt_node;
                    Long trg_range0 = std::lower_bound(trg_nodes0.begin(), trg_nodes0.end(), node0, comp_node_mid) - trg_nodes0.begin();
                    Long trg_range1 = std::lower_bound(trg_nodes0.begin(), trg_nodes0.end(), node1, comp_node_mid) - trg_nodes0.begin();
                    if (trg_range1 > trg_range0) {
                      trg_range.PushBack(trg_range0);
                      trg_range.PushBack(trg_range1);
                      trg_mid_lst.PushBack(trg_mid);
                    }
                  }
                }
              }
              { // build interaction list trg_src_near_mid
                for (Long i = 0; i < src_mid_lst.Dim(); i++) {
                  src_mid_lst[i].NbrList(nbr_lst, src_mid_lst[i].Depth(), false);
                  for (const auto& mid : nbr_lst) {
                    Long j = std::upper_bound(trg_mid_lst.begin(), trg_mid_lst.end(), mid) - trg_mid_lst.begin() - 1;
                    if (j>=0 && trg_mid_lst[j].isAncestor(mid.DFD())) {
                      trg_src_near_mid.PushBack(std::pair<Long,Long>(j,i));
                    }
                  }
                }
                std::sort(trg_src_near_mid.begin(), trg_src_near_mid.end());
              }
              { // build near_lst
                for (Long i = 0; i < trg_mid_lst.Dim(); i++) { // loop over trg_mid
                  Long j0 = std::lower_bound(trg_src_near_mid.begin(), trg_src_near_mid.end(), std::pair<Long,Long>(i+0,0)) - trg_src_near_mid.begin();
                  Long j1 = std::lower_bound(trg_src_near_mid.begin(), trg_src_near_mid.end(), std::pair<Long,Long>(i+1,0)) - trg_src_near_mid.begin();
                  for (Long ii = trg_range[2*i+0]; ii < trg_range[2*i+1]; ii++) { // loop over trg_nodes0
                    const NodeData& trg_node = trg_nodes0[ii];
                    bool is_near = false;
                    for (Long j = j0; j < j1; j++) { // loop over near src_mid
                      Long jj = trg_src_near_mid[j].second;
                      if (j==j0 || trg_src_near_mid[j-1].second!=jj) {
                        for (Long jjj = src_range[jj*2+0]; jjj < src_range[jj*2+1]; jjj++) { // loop over src_nodes1
                          const NodeData& src_node = src_nodes1[jjj];
                          is_near = (node_dist2(src_node,trg_node) < src_node.rad*src_node.rad);
                          if (is_near) break;
                        }
                      }
                      if (is_near) break;
                    }
                    if (is_near) {
                      NodeData node = trg_node;
                      node.elem_idx = eid;
                      near_lst.PushBack(node);
                    }
                  }
                }
              }
            }
          }
        }
        { // sort and partition by elem-ID, remove duplicates
          Vector<NodeData> near_lst0;
          { // near_lst0 <-- partition(dist_sort(near_lst), elem_offset)
            NodeData split_node;
            split_node.idx=0;
            split_node.elem_idx=elem_offset;
            comm.HyperQuickSort(near_lst, near_lst0, comp_node_eid_idx);
            comm.PartitionS(near_lst0, split_node, comp_node_eid_idx);
          }
          if (near_lst0.Dim()) { // near_lst <-- remove_duplicates(near_lst0)
            const Long N0 = near_lst0.Dim();
            Vector<Long> cnt(N0), dsp(N0); dsp[0] = 0;
            #pragma omp parallel for schedule(static)
            for (Long i = 0; i < N0; i++) {
              if (i==0 || near_lst0[i-1].elem_idx!=near_lst0[i].elem_idx || near_lst0[i-1].idx!=near_lst0[i].idx) {
                cnt[i] = 1;
              } else {
                cnt[i] = 0;
              }
            }
            omp_par::scan(cnt.begin(), dsp.begin(), N0);

            const Long N1 = dsp[N0-1] + cnt[N0-1];
            near_lst.ReInit(N1);
            #pragma omp parallel for schedule(static)
            for (Long i = 0; i < N1; i++) {
              if(cnt[i]) {
                near_lst[i] = near_lst0[i];
              }
            }
          } else {
            near_lst.ReInit(0);
          }
        }

        { // Set Xtrg_near
          Xtrg_near.ReInit(near_lst.Dim()*COORD_DIM);
          #pragma omp parallel for schedule(static)
          for (Long i = 0; i < near_lst.Dim(); i++) {
            for (Long k = 0; k < COORD_DIM; k++) {
              Xtrg_near[i*COORD_DIM+k] = near_lst[i].X[k];
            }
          }
        }
        { // Set near_elem_cnt, near_elem_dsp
          near_elem_cnt.ReInit(Nelem);
          near_elem_dsp.ReInit(Nelem);
          #pragma omp parallel
          { // Set near_elem_cnt, near_elem_dsp
            const Integer tid = omp_get_thread_num();
            const Integer omp_p = omp_get_num_threads();
            const Long elem_idx0 = Nelem*(tid+0)/omp_p;
            const Long elem_idx1 = Nelem*(tid+1)/omp_p;
            for (Long i = elem_idx0; i < elem_idx1; i++) {
              near_elem_cnt[i] = 0;
              near_elem_dsp[i] = 0;
            }

            Long idx0, idx1;
            { // Set index range [idx0, idx1] in near_lst for this thread
              NodeData srch_node0, srch_node1;
              srch_node0.elem_idx = elem_offset + elem_idx0; srch_node0.idx = 0;
              srch_node1.elem_idx = elem_offset + elem_idx1; srch_node1.idx = 0;
              idx0 = std::lower_bound(near_lst.begin(), near_lst.end(), srch_node0, comp_node_eid_idx) - near_lst.begin();
              idx1 = std::lower_bound(near_lst.begin(), near_lst.end(), srch_node1, comp_node_eid_idx) - near_lst.begin();
            }
            for (Long i = idx0; i < idx1;) {
              Long elem_idx_ = near_lst[i].elem_idx, dsp = i, cnt = 0;
              for (; i<idx1 && near_lst[i].elem_idx==elem_idx_; i++) cnt++;
              near_elem_dsp[elem_idx_-elem_offset] = dsp;
              near_elem_cnt[elem_idx_-elem_offset] = cnt;
            }
          }
        }

        { // Set scatter_index, near_trg_cnt, near_trg_dsp
          Vector<Long> trg_idx(near_lst.Dim());
          #pragma omp parallel for schedule(static)
          for (Long i = 0; i < trg_idx.Dim(); i++) {
            trg_idx[i] = near_lst[i].idx;
          }
          comm.SortScatterIndex(trg_idx, near_scatter_index, &trg_offset);
          comm.ScatterForward(trg_idx, near_scatter_index);

          near_trg_cnt.ReInit(Ntrg);
          near_trg_dsp.ReInit(Ntrg);
          #pragma omp parallel
          { // Set near_trg_cnt, near_trg_dsp
            const Integer tid = omp_get_thread_num();
            const Integer omp_p = omp_get_num_threads();
            const Long trg_idx0 = Ntrg*(tid+0)/omp_p;
            const Long trg_idx1 = Ntrg*(tid+1)/omp_p;
            for (Long i = trg_idx0; i < trg_idx1; i++) {
              near_trg_cnt[i] = 0;
              near_trg_dsp[i] = 0;
            }

            Long idx0 = std::lower_bound(trg_idx.begin(), trg_idx.end(), trg_offset + trg_idx0) - trg_idx.begin();
            Long idx1 = std::lower_bound(trg_idx.begin(), trg_idx.end(), trg_offset + trg_idx1) - trg_idx.begin();
            for (Long i = idx0; i < idx1;) {
              Long trg_idx_ = trg_idx[i], dsp = i, cnt = 0;
              for (; i<idx1 && trg_idx[i]==trg_idx_; i++) cnt++;
              near_trg_dsp[trg_idx_-trg_offset] = dsp;
              near_trg_cnt[trg_idx_-trg_offset] = cnt;
            }
          }
        }
      }

      void SetupBasic() const {
        if (setup_flag) return;
        elem_lst_name.ReInit(0);
        elem_lst_cnt.ReInit(0);
        elem_lst_dsp.ReInit(0);
        elem_nds_cnt.ReInit(0);
        elem_nds_dsp.ReInit(0);
        Xsurf.ReInit(0);
        Xtrg.ReInit(0);

        const Long Nlst = elem_lst_map.size();
        { // Set elem_lst_name, elem_lst_cnt, elem_lst_dsp
          elem_lst_cnt.ReInit(Nlst);
          elem_lst_dsp.ReInit(Nlst);
          elem_lst_name.ReInit(Nlst);
          auto it = elem_lst_map.begin();
          for (Long i = 0; i < Nlst; i++) {
            const auto& name = it->first;
            const auto& elem_data = elem_data_map.at(name);
            elem_lst_cnt[i] = it->second.Dim()/elem_data.size;
            elem_lst_name[i] = name;
            it++;
          }
          if (Nlst) elem_lst_dsp[0] = 0;
          omp_par::scan(elem_lst_cnt.begin(), elem_lst_dsp.begin(), Nlst);
        }
        const Long Nelem = (Nlst ? elem_lst_dsp[Nlst-1] + elem_lst_cnt[Nlst-1] : 0);
        { // Set elem_nds_cnt, elem_nds_dsp
          elem_nds_cnt.ReInit(Nelem);
          elem_nds_dsp.ReInit(Nelem);
          for (Long i = 0; i < Nlst; i++) {
            const auto& name = elem_lst_name[i];
            const auto& elem_data = elem_data_map.at(name);
            for (Long j = 0; j < elem_lst_cnt[i]; j++) {
              const Long elem_idx = elem_lst_dsp[i] + j;
              elem_nds_cnt[elem_idx] = elem_data.NodeCount;
            }
          }
          if (Nelem) elem_nds_dsp[0] = 0;
          omp_par::scan(elem_nds_cnt.begin(), elem_nds_dsp.begin(), Nelem);
        }

        Vector<Vector<Real>> Xsurf_(Nlst);
        for (Long i = 0; i < Nlst; i++) {
          const auto& name = elem_lst_name[i];
          const auto& elem_lst = elem_lst_map.at(name);
          const auto& elem_data = elem_data_map.at(name);
          elem_data.GetNodeCoord(&Xsurf_[i], nullptr, elem_lst);
        }
        concat_vecs(Xsurf, Xsurf_);
        Xtrg = Xsurf; // TODO: allow off-surf trgs

        setup_flag = true;
      }
      void SetupFar() const {
        if (setup_far_flag) return;
        X_far.ReInit(0);
        Xn_far.ReInit(0);
        wts_far.ReInit(0);
        dist_far.ReInit(0);
        elem_nds_cnt_far.ReInit(0);
        elem_nds_dsp_far.ReInit(0);
        SetupBasic();

        const Long Nlst = elem_lst_map.size();
        const Long Nelem = elem_nds_cnt.Dim();
        { // Set elem_nds_cnt_far, elem_nds_dsp_far
          elem_nds_cnt_far.ReInit(Nelem);
          elem_nds_dsp_far.ReInit(Nelem);
          for (Long i = 0; i < Nlst; i++) {
            const auto& name = elem_lst_name[i];
            const auto& elem_data = elem_data_map.at(name);
            for (Long j = 0; j < elem_lst_cnt[i]; j++) {
              const Long elem_idx = elem_lst_dsp[i] + j;
              elem_nds_cnt_far[elem_idx] = elem_data.FarFieldQuadNodeCount;
            }
          }
          if (Nelem) elem_nds_dsp_far[0] = 0;
          omp_par::scan(elem_nds_cnt_far.begin(), elem_nds_dsp_far.begin(), Nelem);
        }

        Profile::Tic("SetupFarField", &comm_);
        Vector<Vector<Real>> X_far_(Nlst);
        Vector<Vector<Real>> Xn_far_(Nlst);
        Vector<Vector<Real>> wts_far_(Nlst);
        Vector<Vector<Real>> dist_far_(Nlst);
        for (Long i = 0; i < Nlst; i++) {
          const auto& name = elem_lst_name[i];
          const auto& elem_lst = elem_lst_map.at(name);
          const auto& elem_data = elem_data_map.at(name);
          elem_data.GetFarFieldQuadNodes(X_far_[i], Xn_far_[i], wts_far_[i], dist_far_[i], tol_, elem_lst);
        }
        concat_vecs(X_far   , X_far_   );
        concat_vecs(Xn_far  , Xn_far_  );
        concat_vecs(wts_far , wts_far_ );
        concat_vecs(dist_far, dist_far_);
        Profile::Toc();

        setup_far_flag = true;
      }
      void SetupSelf() const {
        if (setup_self_flag) return;
        K_self.ReInit(0);
        SetupBasic();

        Profile::Tic("SetupSingular", &comm_);
        const Long Nlst = elem_lst_map.size();
        Vector<Vector<Matrix<Real>>> K_self_(Nlst);
        for (Long i = 0; i < Nlst; i++) {
          const auto& name = elem_lst_name[i];
          const auto& elem_lst = elem_lst_map.at(name);
          const auto& elem_data = elem_data_map.at(name);
          elem_data.SelfInterac(K_self_[i], elem_lst, ker_, tol_);
        }
        concat_vecs(K_self, K_self_);
        Profile::Toc();

        setup_self_flag = true;
      }
      void SetupNear() const {
        if (setup_near_flag) return;
        Xtrg_near.ReInit(0);
        near_scatter_index.ReInit(0);
        near_trg_cnt.ReInit(0);
        near_trg_dsp.ReInit(0);
        near_elem_cnt.ReInit(0);
        near_elem_dsp.ReInit(0);
        K_near_cnt.ReInit(0);
        K_near_dsp.ReInit(0);
        K_near.ReInit(0);
        SetupBasic();
        SetupFar();
        SetupSelf();

        Profile::Tic("SetupNear", &comm_);
        BuildNearList(Xtrg_near, near_scatter_index, near_trg_cnt, near_elem_cnt, near_trg_dsp, near_elem_dsp, Xtrg, X_far, dist_far, elem_nds_cnt_far, elem_nds_dsp_far, comm_);
        { // Set K_near_cnt, K_near_dsp, K_near
          const Long Nlst = elem_lst_map.size();
          const Long Nelem = near_elem_cnt.Dim();
          SCTL_ASSERT(Nelem == elem_nds_cnt.Dim());
          if (Nelem) { // Set K_near_cnt, K_near_dsp
            K_near_cnt.ReInit(Nelem);
            K_near_dsp.ReInit(Nelem);
            if (Nelem) K_near_dsp[0] = 0;
            for (Long i = 0; i < Nelem; i++) {
              K_near_cnt[i] = elem_nds_cnt[i]*near_elem_cnt[i];
            }
            omp_par::scan(K_near_cnt.begin(), K_near_dsp.begin(), Nelem);
          }
          if (Nelem) { // Set K_near
            K_near.ReInit(K_near_dsp[Nelem-1]+K_near_cnt[Nelem-1]);
            for (Long i = 0; i < Nlst; i++) {
              const auto& name = elem_lst_name[i];
              const auto& elem_lst = elem_lst_map.at(name);
              const auto& elem_data = elem_data_map.at(name);
              for (Long j = 0; j < elem_lst_cnt[i]; j++) {
                const Long elem_idx = elem_lst_dsp[i]+j;
                const Long Ntrg = near_elem_cnt[elem_idx];
                ConstIterator<char> elem_ptr = elem_lst.begin()+j*elem_data.size;
                const Vector<Real> Xsurf_(elem_nds_cnt[elem_idx]*COORD_DIM, Xsurf.begin()+elem_nds_dsp[elem_idx]*COORD_DIM, false);
                Matrix<Real> K_near_(elem_nds_cnt[elem_idx]*KDIM0,near_elem_cnt[elem_idx]*KDIM1, K_near.begin()+K_near_dsp[elem_idx]*KDIM0*KDIM1, false);
                for (Long k = 0; k < Ntrg; k++) {
                  Long min_Xt = -1, min_Xsurf = -1;
                  const Vector<Real> Xt(COORD_DIM, Xtrg_near.begin()+(near_elem_dsp[elem_idx]+k)*COORD_DIM, false);
                  auto compute_min_dist2 = [](Long& min_idx, Long& min_idy, const Vector<Real>& X, const Vector<Real>& Y) {
                    const Long Nx = X.Dim() / COORD_DIM;
                    const Long Ny = Y.Dim() / COORD_DIM;
                    Real min_r2 = -1;
                    for (Long i = 0 ; i < Nx; i++) {
                      for (Long j = 0 ; j < Ny; j++) {
                        Real r2 = 0;
                        for (Long k = 0; k < COORD_DIM; k++) {
                          Real d = X[i*COORD_DIM+k] - Y[j*COORD_DIM+k];
                          r2 += d*d;
                        }
                        if (min_r2<0 || r2<min_r2) {
                          min_idx = i;
                          min_idy = j;
                          min_r2 = r2;
                        }
                      }
                    }
                    return min_r2;
                  };
                  Real trg_elem_dist2 = compute_min_dist2(min_Xt, min_Xsurf, Xt, Xsurf_);
                  SCTL_ASSERT(min_Xt >= 0 && min_Xsurf >= 0);

                  if (trg_elem_dist2 == 0) { // Set K_near0
                    Matrix<Real> K_near0(K_self[elem_idx].Dim(0),K_self[elem_idx].Dim(1), K_self[elem_idx].begin(), false);
                    for (Long l = 0; l < K_near0.Dim(0); l++) {
                      for (Long k1 = 0; k1 < KDIM1; k1++) {
                        K_near_[l][k*KDIM1+k1] = K_near0[l][min_Xsurf*KDIM1+k1];
                      }
                    }
                  } else {
                    Matrix<Real> K_near0;
                    elem_data.NearInterac(K_near0, Xt, elem_ptr, ker_, tol_);
                    for (Long l = 0; l < K_near0.Dim(0); l++) {
                      for (Long k1 = 0; k1 < KDIM1; k1++) {
                        K_near_[l][k*KDIM1+k1] = K_near0[l][k1];
                      }
                    }
                  }
                }
              }
            }
          }

          for (Long i = 0; i < Nlst; i++) { // Subtract direct-interaction part from K_near
            const auto& name = elem_lst_name[i];
            const auto& elem_data = elem_data_map.at(name);
            auto F_upsample_op = [](const ElemLstData& elem_data) {
              const Long N0 = elem_data.NodeCount*KDIM0;
              const Long N1 = elem_data.FarFieldQuadNodeCount*KDIM0;
              Matrix<Real> M_upsample(N0, N1);
              Vector<Real> V(N0); V.SetZero();
              for (Long i = 0; i < N0; i++) {
                V[i] = 1;
                Vector<Real> Vout(N1, M_upsample[i], false);
                elem_data.GetFarFieldQuadDensity(Vout, V);
                V[i] = 0;
              }
              return M_upsample;
            };
            Matrix<Real> M_upsample = F_upsample_op(elem_data);

            for (Long j = 0; j < elem_lst_cnt[i]; j++) { // subtract direct sum
              const Long elem_idx = elem_lst_dsp[i]+j;
              const Long trg_cnt = near_elem_cnt[elem_idx];
              const Long trg_dsp = near_elem_dsp[elem_idx];
              const Vector<Real> Xtrg_near_(trg_cnt*COORD_DIM, Xtrg_near.begin()+trg_dsp*COORD_DIM, false);
              if (!trg_cnt) continue;

              const Long far_src_cnt = elem_nds_cnt_far[elem_idx];
              const Long far_src_dsp = elem_nds_dsp_far[elem_idx];
              const Vector<Real> X (far_src_cnt*COORD_DIM,  X_far.begin() + far_src_dsp*COORD_DIM, false);
              const Vector<Real> Xn(far_src_cnt*COORD_DIM, Xn_far.begin() + far_src_dsp*COORD_DIM, false);
              const Vector<Real> wts(far_src_cnt, wts_far.begin() + far_src_dsp, false);

              SCTL_ASSERT(K_near_cnt[elem_idx] == elem_nds_cnt[elem_idx]*trg_cnt);
              Matrix<Real> K_near_(elem_nds_cnt[elem_idx]*KDIM0, trg_cnt*KDIM1, K_near.begin()+K_near_dsp[elem_idx]*KDIM0*KDIM1, false);
              { // Set K_near_
                Matrix<Real> Mker;
                ker_.KernelMatrix(Mker, Xtrg_near_, X, Xn);
                for (Long k0 = 0; k0 < far_src_cnt; k0++) {
                  for (Long k1 = 0; k1 < KDIM0; k1++) {
                    for (Long l = 0; l < trg_cnt*KDIM1; l++) {
                      Mker[k0*KDIM0+k1][l] *= -wts[k0];
                    }
                  }
                }
                Matrix<Real>::GEMM(K_near_, M_upsample, Mker, (Real)1);
              }
            }
          }
        }
        Profile::Toc();

        setup_near_flag = true;
      }

      void ComputeFarField(Vector<Real>& U, const Vector<Real>& F) const {
        const Long Nsrc = X_far.Dim()/COORD_DIM;
        const Long Ntrg = Xtrg.Dim()/COORD_DIM;

        { // Set F_far
          Long offset = 0, offset_far = 0;
          const Long Nlst = elem_lst_map.size();
          if (F_far.Dim() != Nsrc*KDIM0) F_far.ReInit(Nsrc*KDIM0);
          for (Long i = 0; i < Nlst; i++) { // Init F_far
            const auto& name = elem_lst_name[i];
            const auto& elem_lst = elem_lst_map.at(name);
            const auto& elem_data = elem_data_map.at(name);
            const Long Nelem = elem_lst.Dim()/elem_data.size;

            SCTL_ASSERT(F.Dim() >= offset + Nelem*elem_data.NodeCount*KDIM0);
            SCTL_ASSERT(F_far.Dim() >= offset_far + Nelem*elem_data.FarFieldQuadNodeCount*KDIM0);
            const Vector<Real> F_(Nelem*elem_data.NodeCount*KDIM0, (Iterator<Real>)F.begin() + offset, false);
            Vector<Real> F_far_(Nelem*elem_data.FarFieldQuadNodeCount*KDIM0, F_far.begin() + offset_far, false);
            elem_data.GetFarFieldQuadDensity(F_far_, F_);
            offset_far += F_far_.Dim();
            offset += F_.Dim();
          }
          for (Long i = 0; i < Nsrc; i++) { // apply wts_far
            for (Long j = 0; j < KDIM0; j++) {
              F_far[i*KDIM0+j] *= wts_far[i];
            }
          }
        }
        if (U.Dim() != Ntrg*KDIM1) {
          U.ReInit(Ntrg*KDIM1);
          U.SetZero();
        }
        fmm.Eval(U, Xtrg, X_far, Xn_far, F_far, ker_, comm_);
      }
      void ComputeNearInterac(Vector<Real>& U, const Vector<Real>& F) const {
        const Long Ntrg = Xtrg.Dim()/COORD_DIM;
        const Long Nelem = near_elem_cnt.Dim();
        if (U.Dim() != Ntrg*KDIM1) {
          U.ReInit(Ntrg*KDIM1);
          U.SetZero();
        }

        Vector<Real> U_near((near_elem_dsp[Nelem-1]+near_elem_cnt[Nelem-1])*KDIM1);
        for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) { // subtract direct sum
          const Long src_dof = elem_nds_cnt[elem_idx]*KDIM0;
          const Long trg_dof = near_elem_cnt[elem_idx]*KDIM1;
          const Matrix<Real> K_near_(src_dof, trg_dof, K_near.begin() + K_near_dsp[elem_idx]*KDIM0*KDIM1, false);
          const Matrix<Real> F_(1, src_dof, (Iterator<Real>)F.begin() + elem_nds_dsp[elem_idx]*KDIM0, false);
          Matrix<Real> U_(1, trg_dof, U_near.begin() + near_elem_dsp[elem_idx]*KDIM1, false);
          Matrix<Real>::GEMM(U_, F_, K_near_);
        }

        SCTL_ASSERT(near_trg_cnt.Dim() == Ntrg);
        comm_.ScatterForward(U_near, near_scatter_index);
        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < Ntrg; i++) { // Accumulate result to U
          Long near_cnt = near_trg_cnt[i];
          Long near_dsp = near_trg_dsp[i];
          for (Long j = 0; j < near_cnt; j++) {
            U[i] += U_near[near_dsp+j];
          }
        }
      }

      struct ElemLstData {
        Long typeid_hash, size;
        Long NodeCount, FarFieldQuadNodeCount;

        void (*GetNodeCoord)(Vector<Real>*, Vector<Real>*, const Vector<char>&);
        void (*GetFarFieldQuadNodes)(Vector<Real>&, Vector<Real>&, Vector<Real>&, Vector<Real>&, Real, const Vector<char>&);
        void (*GetFarFieldQuadDensity)(Vector<Real>&, const Vector<Real>&);
        void (*SelfInterac)(Vector<Matrix<Real>>&, const Vector<char>&, const Kernel&, Real);
        void (*NearInterac)(Matrix<Real>&, const Vector<Real>&, ConstIterator<char>&, const Kernel&, Real);
      };
      std::map<std::string,Vector<char>> elem_lst_map;
      std::map<std::string,ElemLstData> elem_data_map;
      Real tol_;
      Kernel ker_;
      Comm comm_;

      mutable bool setup_flag;
      mutable Vector<std::string> elem_lst_name; // name of each element-list (size=Nlst)
      mutable Vector<Long> elem_lst_cnt, elem_lst_dsp; // cnt and dsp of elements for each elem_lst (size=Nlst)
      mutable Vector<Long> elem_nds_cnt, elem_nds_dsp; // cnt and dsp of nodes for each element (size=Nelem)
      mutable Vector<Real> Xsurf; // Position of surface node points (target points for on-surface evaluation)
      mutable Vector<Real> Xtrg; // Position of target points

      mutable bool setup_far_flag;
      mutable ParticleFMM<Real,COORD_DIM> fmm;
      mutable Vector<Long> elem_nds_cnt_far, elem_nds_dsp_far; // cnt and dsp of far-nodes for each element (size=Nelem)
      mutable Vector<Real> X_far, Xn_far, wts_far; // position, normal and weights for far-field quadrature
      mutable Vector<Real> dist_far; // minimum distance of target points for far-field evaluation
      mutable Vector<Real> F_far; // pre-allocated memory for density in far-field evaluation

      mutable bool setup_near_flag;
      mutable Vector<Real> Xtrg_near; // position of near-interaction target points sorted by element (size=Nnear*COORD_DIM)
      mutable Vector<Long> near_scatter_index; // prmutation vector that takes near-interactions sorted by elem-idx to sorted by trg-idx (size=Nnear)
      mutable Vector<Long> near_trg_cnt, near_trg_dsp; // cnt and dsp of near-interactions for each target (size=Ntrg)
      mutable Vector<Long> near_elem_cnt, near_elem_dsp; // cnt and dsp of near-interaction for each element (size=Nelem)
      mutable Vector<Long> K_near_cnt, K_near_dsp; // cnt and dsp of element wise near-interaction matrix (size=Nelem)
      mutable Vector<Real> K_near;

      mutable bool setup_self_flag;
      mutable Vector<Matrix<Real>> K_self;
  };

  template <class Real, class ElemType> class GenericBoundaryElement {
    public:

      template <class EType> static void GetAllNodeCoord(Vector<Real>* X, Vector<Real>* Xn, const Vector<EType>& elem_lst) {
        const Long Nelem = elem_lst.Dim()*sizeof(EType)/sizeof(ElemType);
        ConstIterator<ElemType> elem_lst_ = (ConstIterator<ElemType>)elem_lst.begin();

        static constexpr Integer COORD_DIM = ElemType::CoordDim();
        static constexpr Long node_cnt = ElemType::NodeCount();
        if (X  != nullptr && X ->Dim() != Nelem*node_cnt*COORD_DIM) X ->ReInit(Nelem*node_cnt*COORD_DIM);
        if (Xn != nullptr && Xn->Dim() != Nelem*node_cnt*COORD_DIM) Xn->ReInit(Nelem*node_cnt*COORD_DIM);
        for (Long i = 0; i < Nelem; i++) {
          Vector<Real> X_, Xn_;
          if (X  != nullptr) X_.ReInit(node_cnt*COORD_DIM, X->begin()+i*node_cnt*COORD_DIM, false);
          if (Xn != nullptr) Xn_.ReInit(node_cnt*COORD_DIM, Xn->begin()+i*node_cnt*COORD_DIM, false);
          elem_lst_[i].GetNodeCoord((X==nullptr?nullptr:&X_), (Xn==nullptr?nullptr:&Xn_));
        }
      }

      template <class EType> static void GetAllFarFieldQuadNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Real tol, const Vector<EType>& elem_lst) {
        const Long Nelem = elem_lst.Dim()*sizeof(EType)/sizeof(ElemType);
        ConstIterator<ElemType> elem_lst_ = (ConstIterator<ElemType>)elem_lst.begin();

        static constexpr Integer COORD_DIM = ElemType::CoordDim();
        static constexpr Long node_cnt = ElemType::FarFieldQuadNodeCount();
        if (X       .Dim() != Nelem*node_cnt*COORD_DIM) X       .ReInit(Nelem*node_cnt*COORD_DIM);
        if (Xn      .Dim() != Nelem*node_cnt*COORD_DIM) Xn      .ReInit(Nelem*node_cnt*COORD_DIM);
        if (wts     .Dim() != Nelem*node_cnt          ) wts     .ReInit(Nelem*node_cnt          );
        if (dist_far.Dim() != Nelem*node_cnt          ) dist_far.ReInit(Nelem*node_cnt          );
        for (Long i = 0; i < Nelem; i++) {
          Vector<Real>        X_(node_cnt*COORD_DIM,        X.begin()+i*node_cnt*COORD_DIM, false);
          Vector<Real>       Xn_(node_cnt*COORD_DIM,       Xn.begin()+i*node_cnt*COORD_DIM, false);
          Vector<Real>      wts_(node_cnt          ,      wts.begin()+i*node_cnt          , false);
          Vector<Real> dist_far_(node_cnt          , dist_far.begin()+i*node_cnt          , false);
          elem_lst_[i].GetFarFieldQuadNodes(X_, Xn_, wts_, dist_far_, tol);
        }
      }
      template <Integer DOF> static void GetAllFarFieldQuadDensity(Vector<Real>& Fout, const Vector<Real>& Fin) {
        const Long Nelem = Fin.Dim()/(ElemType::NodeCount()*DOF);
        if (Fout.Dim() != Nelem*ElemType::FarFieldQuadNodeCount()*DOF) Fout.ReInit(Nelem*ElemType::FarFieldQuadNodeCount()*DOF);
        for (Long i = 0; i < Nelem; i++) {
          const Vector<Real> Fin_(ElemType::NodeCount()*DOF, (Iterator<Real>)Fin.begin() + i*ElemType::NodeCount()*DOF, false);
          Vector<Real> Fout_(ElemType::FarFieldQuadNodeCount()*DOF, Fout.begin() + i*ElemType::FarFieldQuadNodeCount()*DOF, false);
          ElemType::template GetFarFieldQuadDensity<DOF>(Fout_, Fin_);
        }
      }

      template <class Kernel, class EType> static void AllSelfInterac(Vector<Matrix<Real>>& M_lst, const Vector<EType>& elem_lst, const Kernel& ker, Real tol) {
        const Long Nelem = elem_lst.Dim()*sizeof(EType)/sizeof(ElemType);
        ConstIterator<ElemType> elem_lst_ = (ConstIterator<ElemType>)elem_lst.begin();

        if (M_lst.Dim() != Nelem) M_lst.ReInit(Nelem);
        for (Long i = 0; i < Nelem; i++) {
          M_lst[i] = elem_lst_[i].template SelfInterac<Kernel>(ker, tol);
        }
      }

      template <class Kernel, class EType> static void AllNearInterac(Matrix<Real>& M, const Vector<Real>& Xt, ConstIterator<EType>& elem_ptr, const Kernel& ker, Real tol) {
        static constexpr Integer COORD_DIM = ElemType::CoordDim();
        static constexpr Long node_cnt = ElemType::NodeCount();
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();

        const Long Ntrg = Xt.Dim() / COORD_DIM;
        const ElemType& elem = *(ConstIterator<ElemType>)elem_ptr;
        if (M.Dim(0) != node_cnt*KDIM0 || M.Dim(1) != Ntrg*KDIM1) {
          M.ReInit(node_cnt*KDIM0, Ntrg*KDIM1);
        }
        for (Long i = 0; i < Ntrg; i++) {
          Tensor<Real,true,COORD_DIM,1> x_trg;
          for (Long k = 0; k < COORD_DIM; k++) x_trg(k,0) = Xt[i*COORD_DIM+k];
          Matrix<Real> M_ = elem.template NearInterac<Kernel>(x_trg, ker, tol);
          for (Long j = 0; j < node_cnt*KDIM0; j++) {
            for (Long k = 0; k < KDIM1; k++) {
              M[j][i*KDIM1+k] = M_[j][k];
            }
          }
        }
      }

      template <class EType> static void GetVTUData(VTUData& vtu_data, const Vector<EType>& elem_lst, const Vector<Real>& F) {
        const Long Nelem = elem_lst.Dim()*sizeof(EType)/sizeof(ElemType);
        ConstIterator<ElemType> elem_lst_ = (ConstIterator<ElemType>)elem_lst.begin();
        for (Long i = 0; i < Nelem; i++) {
          Vector<Real> F_(F.Dim()/Nelem, (Iterator<Real>)F.begin()+i*F.Dim()/Nelem, false);
          elem_lst_[i].GetVTUData(vtu_data, F_);
        }
      }

      template <class EType> static void WriteVTK(std::string fname, const Vector<EType>& elem_lst, const Vector<Real>& F, const Comm& comm = Comm::Self()) {
        VTUData vtu_data;
        GetVTUData(vtu_data, elem_lst, F);
        vtu_data.WriteVTK(fname, comm);
      }
  };





  template <class Real> class LagrangeInterp {
    public:

      static void Interpolate(Vector<Real>& wts, const Vector<Real>& src_nds, const Vector<Real>& trg_nds) {
        if (1) {
          Long Nsrc = src_nds.Dim();
          Long Ntrg = trg_nds.Dim();
          if (wts.Dim() != Nsrc*Ntrg) wts.ReInit(Nsrc*Ntrg);

          Matrix<Real> M(Nsrc, Ntrg, wts.begin(), false);
          for (Long i1 = 0; i1 < Ntrg; i1++) {
            Real x = trg_nds[i1];
            for (Integer j = 0; j < Nsrc; j++) {
              Real y = 1;
              for (Integer k = 0; k < Nsrc; k++) {
                y *= (j==k ? 1 : (src_nds[k] - x) / (src_nds[k] - src_nds[j]));
              }
              M[j][i1] = y;
            }
          }
        }
        if (0) { // Barycentric, numerically unstable (TODO: diagnose)
          Long Nsrc = src_nds.Dim();
          Long Ntrg = trg_nds.Dim();
          if (wts.Dim() != Nsrc*Ntrg) wts.ReInit(Nsrc*Ntrg);
          if (!wts.Dim()) return;
          for (Long t = 0; t < Ntrg; t++) {
            Real scal = 0;
            Long overlap = -1;
            for (Long s = 0; s < Nsrc; s++) {
              if (src_nds[s] == trg_nds[t]) overlap = s;
              scal += 1.0/(src_nds[s]-trg_nds[t]) * (s%2?1.0:-1.0) * ((s==0)||(s==Nsrc-1)?0.5:1.0);
            }
            scal = 1.0 / scal;

            if (overlap == -1) {
              for (Long s = 0; s < Nsrc; s++) {
                wts[s*Ntrg+t] = 1.0/(src_nds[s]-trg_nds[t]) * (s%2?1.0:-1.0) * ((s==0)||(s==Nsrc-1)?0.5:1.0) * scal;
              }
            } else {
              for (Long s = 0; s < Nsrc; s++) wts[s*Ntrg+t] = 0;
              wts[overlap*Ntrg+t] = 1;
            }
          }
        }
      }

      static void Derivative(Vector<Real>& df, const Vector<Real>& f, const Vector<Real>& nds) {
        Long N = nds.Dim();
        Long dof = f.Dim() / N;
        SCTL_ASSERT(f.Dim() == N * dof);
        if (df.Dim() != N * dof) df.ReInit(N * dof);
        if (N*dof == 0) return;

        auto dp = [&nds,&N](Real x, Long i) {
          Real scal = 1;
          for (Long j = 0; j < N; j++) {
            if (i!=j) scal *= (nds[i] - nds[j]);
          }
          scal = 1/scal;
          Real wt = 0;
          for (Long k = 0; k < N; k++) {
            Real wt_ = 1;
            if (k!=i) {
              for (Long j = 0; j < N; j++) {
                if (j!=k && j!=i) wt_ *= (x - nds[j]);
              }
              wt += wt_;
            }
          }
          return wt * scal;
        };
        for (Long k = 0; k < dof; k++) {
          for (Long i = 0; i < N; i++) {
            Real df_ = 0;
            for (Long j = 0; j < N; j++) {
              df_ += f[k*N+j] * dp(nds[i],j);
            }
            df[k*N+i] = df_;
          }
        }
      }

      static void test() { // TODO: cleanup
        Matrix<Real> f(1,3);
        f[0][0] = 0; f[0][1] = 1; f[0][2] = 0.5;

        Vector<Real> src, trg;
        for (Long i = 0; i < 3; i++) src.PushBack(i);
        for (Long i = 0; i < 11; i++) trg.PushBack(i*0.2);
        Vector<Real> wts;
        Interpolate(wts,src,trg);
        Matrix<Real> Mwts(src.Dim(), trg.Dim(), wts.begin(), false);
        Matrix<Real> ff = f * Mwts;
        std::cout<<ff<<'\n';

        Vector<Real> df;
        Derivative(df, Vector<Real>(f.Dim(0)*f.Dim(1),f.begin()), src);
        std::cout<<df<<'\n';
      }

    private:
  };

  template <class Real, Integer ORDER> static const std::pair<Vector<Real>,Vector<Real>>& LogSingularityQuadRule() {
    auto compute_nds_wts = []() {
      using ValueType = QuadReal;
      Vector<ValueType> nds, wts;
      auto integrands = [](const Vector<ValueType>& nds) {
        const Integer K = ORDER;
        const Long N = nds.Dim();
        Matrix<ValueType> M(N,K);
        for (Long j = 0; j < N; j++) {
          for (Long i = 0; i < K/2; i++) {
            M[j][i] = pow<ValueType,Long>(nds[j],i);
          }
          for (Long i = K/2; i < K; i++) {
            M[j][i] = pow<ValueType,Long>(nds[j],K-i-1) * log<ValueType>(nds[j]);
          }
        }
        return M;
      };
      InterpQuadRule<ValueType>::Build(nds, wts, integrands, 1e-20, ORDER, 2e-4, 0.9998); // TODO: diagnose accuracy issues

      std::pair<Vector<Real>,Vector<Real>> nds_wts;
      nds_wts.first.ReInit(nds.Dim());
      nds_wts.second.ReInit(wts.Dim());
      for (Long i = 0; i < nds.Dim(); i++) {
        nds_wts.first[i] = (Real)nds[i];
        nds_wts.second[i] = (Real)wts[i];
      }
      return nds_wts;
    };
    static auto nds_wts = compute_nds_wts();
    return nds_wts;
  }

  template <class Real, Integer Nm = 12, Integer Nr = 20, Integer Nt = 16> class ToroidalGreensFn {
      static constexpr Integer COORD_DIM = 3;
      static constexpr Real min_dist = 0.0;
      static constexpr Real max_dist = 0.2;

    public:

      ToroidalGreensFn() {
      }

      template <class Kernel> void Setup(const Kernel& ker, Real R0) {
        PrecompToroidalGreensFn<QuadReal>(ker, R0);
      }

      template <class Kernel> void BuildOperatorModal(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const {
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Integer Nmm = (Nm/2+1)*2;
        static constexpr Integer Ntt = (Nt/2+1)*2;

        StaticArray<Real,2*Nr> buff0;
        StaticArray<Real,Ntt> buff1;
        Vector<Real> r_basis(Nr,buff0,false);
        Vector<Real> interp_r(Nr,buff0+Nr,false);
        Vector<Real> interp_Ntt(Ntt,buff1,false);
        if (M.Dim(0) != KDIM0*Nmm || M.Dim(1) != KDIM1) M.ReInit(KDIM0*Nmm,KDIM1);
        { // Set M
          const Real r = sqrt<Real>(x0*x0 + x1*x1);
          const Real rho = sqrt<Real>((r-R0_)*(r-R0_) + x2*x2);
          if (rho < max_dist*R0_) {
            const Real r_inv = 1/r;
            const Real rho_inv = 1/rho;
            const Real cos_theta = x0*r_inv;
            const Real sin_theta = x1*r_inv;
            const Real cos_phi = x2*rho_inv;
            const Real sin_phi = (r-R0_)*rho_inv;

            { // Set interp_r
              interp_r = 0;
              const Real rho0 = (rho/R0_-min_dist)/(max_dist-min_dist);
              BasisFn<Real>::EvalBasis(r_basis, rho0);
              for (Long i = 0; i < Nr; i++) {
                Real fn_val = 0;
                for (Long j = 0; j < Nr; j++) {
                  fn_val += Mnds2coeff1[0][i*Nr+j] * r_basis[j];
                }
                for (Long j = 0; j < Nr; j++) {
                  interp_r[j] += Mnds2coeff0[0][i*Nr+j] * fn_val;
                }
              }
            }
            { // Set interp_Ntt
              interp_Ntt[0] = 0.5;
              interp_Ntt[1] = 0.0;
              Complex<Real> exp_t(cos_phi, sin_phi);
              Complex<Real> exp_jt(cos_phi, sin_phi);
              for (Long j = 1; j < Ntt/2; j++) {
                interp_Ntt[j*2+0] = exp_jt.real;
                interp_Ntt[j*2+1] =-exp_jt.imag;
                exp_jt *= exp_t;
              }
            }

            M = 0;
            for (Long j = 0; j < Nr; j++) {
              for (Long k = 0; k < Ntt; k++) {
                Real interp_wt = interp_r[j] * interp_Ntt[k];
                ConstIterator<Real> Ut_ = Ut.begin() + (j*Ntt+k)*KDIM0*Nmm*KDIM1;
                for (Long i = 0; i < KDIM0*Nmm*KDIM1; i++) { // Set M
                  M[0][i] += Ut_[i] * interp_wt;
                }
              }
            }
            { // Rotate by theta
              Complex<Real> exp_iktheta(1,0), exp_itheta(cos_theta, -sin_theta);
              for (Long k = 0; k < Nmm/2; k++) {
                for (Long i = 0; i < KDIM0; i++) {
                  for (Long j = 0; j < KDIM1; j++) {
                    Complex<Real> c(M[i*Nmm+2*k+0][j],M[i*Nmm+2*k+1][j]);
                    c *= exp_iktheta;
                    M[i*Nmm+2*k+0][j] = c.real;
                    M[i*Nmm+2*k+1][j] = c.imag;
                  }
                }
                exp_iktheta *= exp_itheta;
              }
            }
          } else if (rho < max_dist*R0_*1.25) {
            BuildOperatorModalDirect<110>(M, x0, x1, x2, ker);
          } else if (rho < max_dist*R0_*1.67) {
            BuildOperatorModalDirect<88>(M, x0, x1, x2, ker);
          } else if (rho < max_dist*R0_*2.5) {
            BuildOperatorModalDirect<76>(M, x0, x1, x2, ker);
          } else if (rho < max_dist*R0_*5) {
            BuildOperatorModalDirect<50>(M, x0, x1, x2, ker);
          } else if (rho < max_dist*R0_*10) {
            BuildOperatorModalDirect<25>(M, x0, x1, x2, ker);
          } else if (rho < max_dist*R0_*20) {
            BuildOperatorModalDirect<14>(M, x0, x1, x2, ker);
          } else {
            BuildOperatorModalDirect<Nm>(M, x0, x1, x2, ker);
          }
        }
      }

    private:

      template <class Kernel> Vector<Real> Eval(const Vector<Real>& F, const Vector<Real>& x, const Kernel& ker) const {
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Integer Nmm = (Nm/2+1)*2;
        const Long N = x.Dim() / COORD_DIM;
        SCTL_ASSERT(x.Dim() == N * COORD_DIM);
        SCTL_ASSERT(F.Dim() == Nm*KDIM0);

        Vector<Real> F_fourier_coeff;
        { // Set F_fourier_coeff <-- FFT(F)
          Vector<Real> F_ = F;
          { // Transpose F_
            Matrix<Real> FF(Nm,KDIM0,F_.begin(), false);
            FF = FF.Transpose();
          }
          fft_Nm_R2C.Execute(F_, F_fourier_coeff);
        }
        const Matrix<Real> F_(1, KDIM0*Nmm, (Iterator<Real>)F_fourier_coeff.begin(), false);

        Matrix<Real> M;
        Vector<Real> Utrg(N*KDIM1);
        for (Long l = 0; l < N; l++) {
          BuildOperatorModal(M, x[l*COORD_DIM+0], x[l*COORD_DIM+1], x[l*COORD_DIM+2], ker);
          Matrix<Real> Utrg_(1, KDIM1, Utrg.begin()+l*KDIM1, false);
          Utrg_ = F_ * M;
        }
        return Utrg;
      }
      template <class Kernel> Vector<Real> Eval_(const Vector<Real>& F, const Vector<Real>& x, const Kernel& ker) const {
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Integer Nmm = (Nm/2+1)*2;
        static constexpr Integer Ntt = (Nt/2+1)*2;
        const Long N = x.Dim() / COORD_DIM;
        SCTL_ASSERT(x.Dim() == N * COORD_DIM);
        SCTL_ASSERT(F.Dim() == Nm*KDIM0);

        Vector<Real> F_fourier_coeff, FF(KDIM0*Nmm);
        { // Set F_fourier_coeff <-- FFT(F)
          Vector<Real> F_ = F;
          { // Transpose F_
            Matrix<Real> FF(Nm,KDIM0,F_.begin(), false);
            FF = FF.Transpose();
          }
          fft_Nm_R2C.Execute(F_, F_fourier_coeff);
        }

        Vector<Real> Utrg(N*KDIM1);
        Vector<Real> interp_r(Nr), interp_Ntt(Ntt), r_basis(Nr);
        Vector<Real> Ucoeff0(Nr*KDIM1*Ntt), Ucoeff1(KDIM1*Ntt);
        for (Long l = 0; l < N; l++) {
          const Real r = sqrt<Real>(x[l*COORD_DIM+0]*x[l*COORD_DIM+0] + x[l*COORD_DIM+1]*x[l*COORD_DIM+1]);
          const Real rho = sqrt<Real>((r-R0_)*(r-R0_) + x[l*COORD_DIM+2]*x[l*COORD_DIM+2]);
          const Real r_inv = 1/r;
          const Real rho_inv = 1/rho;
          const Real cos_theta = x[l*COORD_DIM+0]*r_inv;
          const Real sin_theta = x[l*COORD_DIM+1]*r_inv;
          const Real cos_phi = x[l*COORD_DIM+2]*rho_inv;
          const Real sin_phi = (r-R0_)*rho_inv;

          { // Set interp_r
            interp_r = 0;
            const Real rho0 = (rho/R0_-min_dist)/(max_dist-min_dist);
            BasisFn<Real>::EvalBasis(r_basis, rho0);
            for (Long i = 0; i < Nr; i++) {
              Real fn_val = 0;
              for (Long j = 0; j < Nr; j++) {
                fn_val += Mnds2coeff1[0][i*Nr+j] * r_basis[j];
              }
              for (Long j = 0; j < Nr; j++) {
                interp_r[j] += Mnds2coeff0[0][i*Nr+j] * fn_val;
              }
            }
          }
          { // Set interp_Ntt
            Complex<Real> exp_t(cos_phi, sin_phi);
            Complex<Real> exp_jt(1.0,0.0);
            for (Long j = 0; j < Ntt/2; j++) {
              interp_Ntt[j*2+0] = exp_jt.real * (j==0?1:2);
              interp_Ntt[j*2+1] =-exp_jt.imag * (j==0?1:2);
              exp_jt *= exp_t;
            }
          }

          { // Set Ucoeff0 (eval in theta)
            Ucoeff0 = 0;
            { // FF <-- Rotate F_fourier_coeff by theta
              Complex<Real> exp_iktheta(1,0), exp_itheta(cos_theta, sin_theta);
              for (Long k = 0; k < Nmm/2; k++) {
                for (Long i = 0; i < KDIM0; i++) {
                  Complex<Real> c(F_fourier_coeff[i*Nmm+2*k+0],F_fourier_coeff[i*Nmm+2*k+1]);
                  c *= exp_iktheta;
                  FF[i*Nmm+2*k+0] = c.real;
                  FF[i*Nmm+2*k+1] = c.imag;
                }
                exp_iktheta *= exp_itheta;
              }
            }
            for (Long i = 0; i < KDIM0*Nmm; i++) {
              for (Long j = 0; j < KDIM1*Nr*Ntt; j++) {
                Ucoeff0[j] += U[i*KDIM1*Nr*Ntt+j] * FF[i];
              }
            }
          }
          { // Set Ucoeff1 (eval in r)
            Ucoeff1 = 0;
            for (Long i = 0; i < Nr; i++) {
              for (Long j = 0; j < KDIM1*Ntt; j++) {
                Ucoeff1[j] += Ucoeff0[i*KDIM1*Ntt+j] * interp_r[i];
              }
            }
          }
          for (Long k = 0; k < KDIM1; k++) { // Set Utrg (eval in phi)
            Utrg[l*KDIM1+k] = 0;
            for (Long i = 0; i < Ntt; i++) {
              Utrg[l*KDIM1+k] += Ucoeff1[k*Ntt+i] * interp_Ntt[i];
            }
          }
        }
        return Utrg;
      }
      static void test() {
        if (1) {
          using RefValType = QuadReal;

          Real R0 = 1.0;
          ToroidalGreensFn tor_sl, tor_dl;
          GenericKernel<Laplace3D_FxU> laplace_sl;
          GenericKernel<Laplace3D_DxU> laplace_dl;
          tor_sl.Setup(laplace_sl, R0);
          tor_dl.Setup(laplace_dl, R0);

          static constexpr Integer KDIM0 = laplace_sl.SrcDim();
          Vector<Real> F(Nm * KDIM0);
          for (auto& x : F) x = drand48();

          if (1) {
            Long Ntrg = 1000;
            Vector<RefValType> Xtrg;
            for (Long i = 0; i < Ntrg; i++) {
              Real rho_min = 1e-4;
              Real rho_max = 0.5;
              Real rho = rho_min + (rho_max-rho_min)*drand48(); //*1e-3;
              //Real rho = min_dist + 0.01*(max_dist-min_dist)*(0.5-cos<Real>(2*const_pi<Real>()*(2*i+1)/(2*Ntrg))*0.5);
              Real theta = 2*const_pi<Real>()*drand48()*0 + 2*const_pi<Real>()*(i%13)/(Real)13;
              Real phi = 2*const_pi<Real>()*drand48();

              Real x = R0*(1+rho*sin<Real>(phi))*cos(theta);
              Real y = R0*(1+rho*sin<Real>(phi))*sin(theta);
              Real z = R0*(0+rho*cos<Real>(phi));

              Xtrg.PushBack(x);
              Xtrg.PushBack(y);
              Xtrg.PushBack(z);
            }

            Vector<Real> Usl0, Udl0, Usl, Udl;
            Profile::Tic("AdapInteg");
            { // Reference solution
              Vector<RefValType> F_, Usl_, Udl_;
              for (const auto& x : F) F_.PushBack((Real)x);
              ComputePotential<RefValType>(Usl_, Xtrg, R0, F_, laplace_sl, 1e-16);
              ComputePotential<RefValType>(Udl_, Xtrg, R0, F_, laplace_dl, 1e-16);
              for (const auto& x : Usl_) Usl0.PushBack((Real)x);
              for (const auto& x : Udl_) Udl0.PushBack((Real)x);
            }
            Profile::Toc();

            Profile::Tic("Interp");
            {
              Vector<Real> Xtrg_;
              for (const auto& x : Xtrg) Xtrg_.PushBack((Real)x);
              auto Usl_ = tor_sl.Eval(F, Xtrg_, laplace_sl);
              auto Udl_ = tor_dl.Eval(F, Xtrg_, laplace_dl);
              for (const auto& x : Usl_) Usl.PushBack((Real)x);
              for (const auto& x : Udl_) Udl.PushBack((Real)x);
            }
            Profile::Toc();

            auto print_err = [](const Vector<Real>& A, const Vector<Real>& B) {
              Real max_err = 0, max_val = 0;
              SCTL_ASSERT(A.Dim() == B.Dim());
              for (Long i = 0 ; i< A.Dim(); i++) {
                max_err = std::max<Real>(max_err, fabs(A[i]-B[i]));
                max_val = std::max<Real>(max_val, (fabs(A[i]) + fabs(B[i]))*0.5);
              }
              std::cout<<max_err<<' '<<max_val<<'\n';
            };
            print_err(Usl0, Usl);
            print_err(Udl0, Udl);
          }
        }
      }

      template <class ValueType> class BasisFn { // p(x) log(x) + q(x) + 1/x
        public:
          BasisFn(const Vector<ValueType> coeff_ = Vector<ValueType>()) {
            coeff = coeff_;
          }
          ValueType operator()(ValueType x) const {
            return Eval(coeff, x);
          }
          static ValueType Eval(const Vector<ValueType>& coeff, ValueType x) {
            if (1) {
              ValueType sum = 0;
              ValueType log_x = log(x);
              Long Nsplit = std::max<Long>(0,(coeff.Dim()-1)/2);
              ValueType x_i = 1;
              for (Long i = 0; i < Nsplit; i++) {
                sum += coeff[i] * x_i;
                x_i *= x;
              }
              x_i = 1;
              for (Long i = coeff.Dim()-2; i >= Nsplit; i--) {
                sum += coeff[i] * log_x * x_i;
                x_i *= x;
              }
              if (coeff.Dim()-1 >= 0) sum += coeff[coeff.Dim()-1] / x;
              return sum;
            }
            if (0) {
              ValueType sum = 0;
              Long Nsplit = coeff.Dim()/2;
              for (Long i = 0; i < Nsplit; i++) {
                sum += coeff[i] * sctl::pow<ValueType,Long>(x,i);
              }
              for (Long i = Nsplit; i < coeff.Dim(); i++) {
                sum += coeff[i] * log(x) * sctl::pow<ValueType,Long>(x,coeff.Dim()-1-i);
              }
              return sum;
            }
          }
          static void EvalBasis(Vector<ValueType>& f, ValueType x) {
            const Long N = f.Dim();
            const Long Nsplit = std::max<Long>(0,(N-1)/2);

            ValueType xi = 1;
            for (Long i = 0; i < Nsplit; i++) {
              f[i] = xi;
              xi *= x;
            }

            ValueType xi_logx = log(x);
            for (Long i = N-2; i >= Nsplit; i--) {
              f[i] = xi_logx;
              xi_logx *= x;
            }

            if (N-1 >= 0) f[N-1] = 1/x;
          }
          static const Vector<ValueType>& nds_(Integer ORDER) { // deprecated
            static const sctl::StaticArray<ValueType,5 > nds5  {1.747857969640005453202124299968203e-3,  1.220161911248954696010511948148562e-2,  7.708347852903332055274361090030625e-2,  5.420940122333937028469511670276969e-1,  9.691634047222954983305786471744474e-1};
            static const sctl::StaticArray<ValueType,7 > nds7  {1.143705162923377692130498810603695e-3,  5.348223589601553570404767339452919e-3,  2.254021947715960451687217561016805e-2,  1.566240520732313604322465088473462e-1,  4.696355030223110629166955081670463e-1,  8.324555624341443002017210500673981e-1,  9.964283501903625529089037217052480e-1};
            static const sctl::StaticArray<ValueType,9 > nds9  {7.750297910176441700771677483296586e-4,  2.948245803669999326127432279023414e-3,  9.407021093036879645485359067786471e-3,  5.663847523943087905004102439288421e-2,  1.860182576725104007129098350048472e-1,  3.937284685630997393751846064685136e-1,  6.471882401289929631440107971297982e-1,  8.659483403255844158659688350378772e-1,  9.807231826305770691048253064348460e-1};
            static const sctl::StaticArray<ValueType,11> nds11 {5.844870016838243645052031439634601e-4,  1.832284228725583632179450165779897e-3,  5.100661350197020914109371988609083e-3,  1.885362712493337020622430513414042e-2,  6.909742414410318574697257133406199e-2,  1.840865274392665113719254511203209e-1,  3.260602801017889821406150045149584e-1,  5.137815367978860885809095671712842e-1,  7.065189006305751583411080827318289e-1,  8.689542085707831903654255764774800e-1,  9.729139736445992085428801338024995e-1};
            static const sctl::StaticArray<ValueType,13> nds13 {4.770931924274411228706882006615248e-4,  1.344091924315536908000176641072776e-3,  3.119372799692294857763945299322239e-3,  9.398485031170282308589834661322113e-3,  4.719525719290589287955889215321132e-2,  1.124181398375037404603249304733991e-1,  2.140941459611401291847908023004893e-1,  3.564764848574302722434161702618062e-1,  5.027111286080193748745243448378650e-1,  6.553491888196598620124179801533293e-1,  7.926977600259585713882591004224329e-1,  9.077878284355496971740221161352449e-1,  9.827965982415868641663159439715085e-1};
            static const sctl::StaticArray<ValueType,15> nds15 {3.207350856908672585001712293289862e-4,  1.062861630232329968179721292188536e-3,  1.488431910762773502772223330822387e-3,  1.053782476117089217618840482416415e-2,  3.518486408168319451497497279281785e-2,  8.147973287588877869952789689667240e-2,  1.585451098594837202599358153274382e-1,  2.665252692482519453659171223958681e-1,  3.874944456395143547688360013617529e-1,  5.001921332145012628232204406192892e-1,  6.789348123428514882817413551131659e-1,  7.879709479756661085740624165868839e-1,  9.005321447263247126052542232534276e-1,  9.750597427989895514247038099492040e-1,  9.999989999999999999756930546546129e-1};
            static const sctl::StaticArray<ValueType,17> nds17 {2.005234456012493820097748515796567e-4,  6.611390820952148654271903390853912e-4,  2.259884726595600675743780084930427e-3,  9.671162292588652925183267573232594e-3,  2.819556286550932587162862699598129e-2,  8.660727003197378114210304897741483e-2,  1.095676197510256949544066862768835e-1,  2.035829472431950724529673889820619e-1,  3.312634046560660180161324339294858e-1,  4.028248249984212517648533932867765e-1,  5.222182857815142399675607284307161e-1,  6.171949239213912235082911696981259e-1,  7.083761303151333065737379381517697e-1,  7.890791501553583908098350031206721e-1,  8.525161614165805375505019241829482e-1,  9.523683787738955314947062410975243e-1,  9.854924457398927187169459190632464e-1};
            static const sctl::StaticArray<ValueType,19> nds19 {4.274248712815090836640726116319210e-4,  1.238034498250004862417959686980028e-3,  1.581651489241347337927983733101702e-3,  1.694885815811070592356360626895470e-3,  1.686213799221140580661230106098879e-2,  4.684693074572490318959593543029356e-2,  1.213780493442497151663107244840269e-1,  2.226498309528670042290401375420620e-1,  2.657701978841803882483342685395724e-1,  3.794489268575985544589603603778710e-1,  4.461012225086163520843413476122930e-1,  5.156768507322298650818280783162499e-1,  6.336816402697930765363082833583332e-1,  7.173005910700513702584266966741015e-1,  7.876770429901608996893804760424656e-1,  8.923482018183432473883003402176196e-1,  9.558774519142369224409080455157667e-1,  9.879767139199415027570970813245340e-1,  9.999989999999999999756930546546129e-1};

            SCTL_ASSERT(ORDER>=5 && ORDER<=19);
            static const Vector<ValueType> Vnds7 (7 , (Iterator<ValueType>)(ConstIterator<ValueType>)nds7 , false);
            static const Vector<ValueType> Vnds9 (9 , (Iterator<ValueType>)(ConstIterator<ValueType>)nds9 , false);
            static const Vector<ValueType> Vnds11(11, (Iterator<ValueType>)(ConstIterator<ValueType>)nds11, false);
            static const Vector<ValueType> Vnds13(13, (Iterator<ValueType>)(ConstIterator<ValueType>)nds13, false);
            static const Vector<ValueType> Vnds15(15, (Iterator<ValueType>)(ConstIterator<ValueType>)nds15, false);
            static const Vector<ValueType> Vnds17(17, (Iterator<ValueType>)(ConstIterator<ValueType>)nds17, false);
            static const Vector<ValueType> Vnds19(19, (Iterator<ValueType>)(ConstIterator<ValueType>)nds19, false);

            if (ORDER == 7 ) return Vnds7 ;
            if (ORDER == 9 ) return Vnds9 ;
            if (ORDER == 11) return Vnds11;
            if (ORDER == 13) return Vnds13;
            if (ORDER == 15) return Vnds15;
            if (ORDER == 17) return Vnds17;
            if (ORDER == 19) return Vnds19;
            return ChebQuadRule<ValueType>::nds(ORDER);
          }
          static const Vector<ValueType>& nds(Integer ORDER) {
            ValueType fn_start = 1e-7, fn_end = 1.0;
            auto compute_nds = [&ORDER,&fn_start,&fn_end]() {
              Vector<ValueType> nds, wts;
              auto integrands = [&ORDER,&fn_start,&fn_end](const Vector<ValueType>& nds) {
                const Integer K = ORDER;
                const Long N = nds.Dim();
                Matrix<ValueType> M(N,K);
                for (Long j = 0; j < N; j++) {
                  Vector<ValueType> f(K,M[j],false);
                  EvalBasis(f, nds[j]*(fn_end-fn_start)+fn_start);
                }
                return M;
              };
              InterpQuadRule<ValueType>::Build(nds, wts, integrands, sqrt(machine_eps<ValueType>()), ORDER);
              return nds*(fn_end-fn_start)+fn_start;
            };
            static Vector<ValueType> nds = compute_nds();
            return nds;
          }
        private:
          Vector<ValueType> coeff;
      };

      template <class ValueType, class Kernel> void PrecompToroidalGreensFn(const Kernel& ker, ValueType R0) {
        SCTL_ASSERT(ker.CoordDim() == COORD_DIM);
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Long Nmm = (Nm/2+1)*2;
        static constexpr Long Ntt = (Nt/2+1)*2;
        R0_ = (Real)R0;

        const auto& nds = BasisFn<ValueType>::nds(Nr);
        { // Set Mnds2coeff0, Mnds2coeff1
          Matrix<ValueType> M(Nr,Nr);
          Vector<ValueType> coeff(Nr); coeff = 0;
          for (Long i = 0; i < Nr; i++) {
            coeff[i] = 1;
            for (Long j = 0; j < Nr; j++) {
              M[i][j] = BasisFn<ValueType>::Eval(coeff, nds[j]);
            }
            coeff[i] = 0;
          }

          Matrix<ValueType> U, S, Vt;
          M.SVD(U, S, Vt);
          for (Long i = 0; i < S.Dim(0); i++) {
            S[i][i] = 1/S[i][i];
          }
          auto Mnds2coeff0_ = S * Vt;
          auto Mnds2coeff1_ = U.Transpose();
          Mnds2coeff0.ReInit(Mnds2coeff0_.Dim(0), Mnds2coeff0_.Dim(1));
          Mnds2coeff1.ReInit(Mnds2coeff1_.Dim(0), Mnds2coeff1_.Dim(1));
          for (Long i = 0; i < Mnds2coeff0.Dim(0)*Mnds2coeff0.Dim(1); i++) Mnds2coeff0[0][i] = (Real)Mnds2coeff0_[0][i];
          for (Long i = 0; i < Mnds2coeff1.Dim(0)*Mnds2coeff1.Dim(1); i++) Mnds2coeff1[0][i] = (Real)Mnds2coeff1_[0][i];
        }
        { // Setup fft_Nm_R2C
          Vector<Long> dim_vec(1);
          dim_vec[0] = Nm;
          fft_Nm_R2C.Setup(FFT_Type::R2C, KDIM0, dim_vec);
          fft_Nm_C2R.Setup(FFT_Type::C2R, KDIM0*KDIM1, dim_vec);
        }

        Vector<ValueType> Xtrg(Nr*Nt*COORD_DIM);
        for (Long i = 0; i < Nr; i++) {
          for (Long j = 0; j < Nt; j++) {
            Xtrg[(i*Nt+j)*COORD_DIM+0] = R0 * (1.0 + (min_dist+(max_dist-min_dist)*nds[i]) * sin<ValueType>(j*2*const_pi<ValueType>()/Nt));
            Xtrg[(i*Nt+j)*COORD_DIM+1] = R0 * (0.0);
            Xtrg[(i*Nt+j)*COORD_DIM+2] = R0 * (0.0 + (min_dist+(max_dist-min_dist)*nds[i]) * cos<ValueType>(j*2*const_pi<ValueType>()/Nt));
          }
        }

        Vector<ValueType> U0(KDIM0*Nmm*Nr*KDIM1*Nt);
        { // Set U0
          FFT<ValueType> fft_Nm_C2R;
          { // Setup fft_Nm_C2R
            Vector<Long> dim_vec(1);
            dim_vec[0] = Nm;
            fft_Nm_C2R.Setup(FFT_Type::C2R, KDIM0, dim_vec);
          }
          Vector<ValueType> Fcoeff(KDIM0*Nmm), F, U_;
          for (Long i = 0; i < KDIM0*Nmm; i++) {
            Fcoeff = 0; Fcoeff[i] = 1;
            { // Set F
              fft_Nm_C2R.Execute(Fcoeff, F);
              Matrix<ValueType> FF(KDIM0,Nm,F.begin(), false);
              FF = FF.Transpose();
            }
            ComputePotential<ValueType>(U_, Xtrg, R0, F, ker);
            SCTL_ASSERT(U_.Dim() == Nr*Nt*KDIM1);

            for (Long j = 0; j < Nr; j++) {
              for (Long l = 0; l < Nt; l++) {
                for (Long k = 0; k < KDIM1; k++) {
                  U0[((i*Nr+j)*KDIM1+k)*Nt+l] = U_[(j*Nt+l)*KDIM1+k];
                }
              }
            }
          }
        }

        Vector<ValueType> U1(KDIM0*Nmm*Nr*KDIM1*Ntt);
        { // U1 <-- fft_Nt(U0)
          FFT<ValueType> fft_Nt;
          Vector<Long> dim_vec(1); dim_vec = Nt;
          fft_Nt.Setup(FFT_Type::R2C, KDIM0*Nmm*Nr*KDIM1, dim_vec);
          fft_Nt.Execute(U0, U1);
          if (Nt%2==0 && Nt) {
            for (Long i = Ntt-2; i < U1.Dim(); i += Ntt) {
              U1[i] *= 0.5;
            }
          }
          U1 *= 1.0/sqrt<ValueType>(Nt);
        }

        U.ReInit(KDIM0*Nmm*KDIM1*Nr*Ntt);
        { // U <-- rearrange(U1)
          for (Long i0 = 0; i0 < KDIM0*Nmm; i0++) {
            for (Long i1 = 0; i1 < Nr; i1++) {
              for (Long i2 = 0; i2 < KDIM1; i2++) {
                for (Long i3 = 0; i3 < Ntt; i3++) {
                  U[((i0*Nr+i1)*KDIM1+i2)*Ntt+i3] = (Real)U1[((i0*KDIM1+i2)*Nr+i1)*Ntt+i3];
                }
              }
            }
          }
        }

        Ut.ReInit(Nr*Ntt*KDIM0*Nmm*KDIM1);
        { // Set Ut
          Matrix<Real> Ut_(Nr*Ntt,KDIM0*Nmm*KDIM1, Ut.begin(), false);
          Matrix<Real> U_(KDIM0*Nmm*KDIM1,Nr*Ntt, U.begin(), false);
          Ut_ = U_.Transpose()*2.0;
        }
      }

      template <class ValueType, class Kernel> static void ComputePotential(Vector<ValueType>& U, const Vector<ValueType>& Xtrg, ValueType R0, const Vector<ValueType>& F_, const Kernel& ker, ValueType tol = 1e-18) {
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        Vector<ValueType> F_fourier_coeff;
        const Long Nt_ = F_.Dim() / KDIM0; // number of Fourier modes
        SCTL_ASSERT(F_.Dim() == Nt_ * KDIM0);

        { // Transpose F_
          Matrix<ValueType> FF(Nt_,KDIM0,(Iterator<ValueType>)F_.begin(), false);
          FF = FF.Transpose();
        }
        { // Set F_fourier_coeff
          FFT<ValueType> fft_plan;
          Vector<Long> dim_vec(1); dim_vec[0] = Nt_;
          fft_plan.Setup(FFT_Type::R2C, KDIM0, dim_vec);
          fft_plan.Execute(F_, F_fourier_coeff);
          if (Nt_%2==0 && F_fourier_coeff.Dim()) {
            F_fourier_coeff[F_fourier_coeff.Dim()-2] *= 0.5;
          }
        }
        auto EvalFourierExp = [&Nt_](Vector<ValueType>& F, const Vector<ValueType>& F_fourier_coeff, Integer dof, const Vector<ValueType>& theta) {
          const Long N = F_fourier_coeff.Dim() / dof / 2;
          SCTL_ASSERT(F_fourier_coeff.Dim() == dof * N * 2);
          const Long Ntheta = theta.Dim();
          if (F.Dim() != Ntheta*dof) F.ReInit(Ntheta*dof);
          for (Integer k = 0; k < dof; k++) {
            for (Long j = 0; j < Ntheta; j++) {
              Complex<ValueType> F_(0,0);
              for (Long i = 0; i < N; i++) {
                Complex<ValueType> c(F_fourier_coeff[(k*N+i)*2+0],F_fourier_coeff[(k*N+i)*2+1]);
                Complex<ValueType> exp_t(cos<ValueType>(theta[j]*i), sin<ValueType>(theta[j]*i));
                F_ += exp_t * c * (i==0?1:2);
              }
              F[j*dof+k] = F_.real/sqrt<ValueType>(Nt_);
            }
          }
        };

        static constexpr Integer QuadOrder = 18;
        std::function<Vector<ValueType>(ValueType,ValueType,ValueType)>  compute_potential = [&](ValueType a, ValueType b, ValueType tol) -> Vector<ValueType> {
          auto GetGeomCircle = [&R0] (Vector<ValueType>& Xsrc, Vector<ValueType>& Nsrc, const Vector<ValueType>& nds) {
            Long N = nds.Dim();
            if (Xsrc.Dim() != N * COORD_DIM) Xsrc.ReInit(N*COORD_DIM);
            if (Nsrc.Dim() != N * COORD_DIM) Nsrc.ReInit(N*COORD_DIM);
            for (Long i = 0; i < N; i++) {
              Xsrc[i*COORD_DIM+0] = R0 * cos<ValueType>(nds[i]);
              Xsrc[i*COORD_DIM+1] = R0 * sin<ValueType>(nds[i]);
              Xsrc[i*COORD_DIM+2] = R0 * 0;
              Nsrc[i*COORD_DIM+0] = cos<ValueType>(nds[i]);
              Nsrc[i*COORD_DIM+1] = sin<ValueType>(nds[i]);
              Nsrc[i*COORD_DIM+2] = 0;
            }
          };

          const auto& nds0 = ChebQuadRule<ValueType>::nds(QuadOrder+1);
          const auto& wts0 = ChebQuadRule<ValueType>::wts(QuadOrder+1);
          const auto& nds1 = ChebQuadRule<ValueType>::nds(QuadOrder+0);
          const auto& wts1 = ChebQuadRule<ValueType>::wts(QuadOrder+0);

          Vector<ValueType> U0;
          Vector<ValueType> Xsrc, Nsrc, Fsrc;
          GetGeomCircle(Xsrc, Nsrc, a+(b-a)*nds0);
          EvalFourierExp(Fsrc, F_fourier_coeff, KDIM0, a+(b-a)*nds0);
          for (Long i = 0; i < nds0.Dim(); i++) {
            for (Long j = 0; j < KDIM0; j++) {
              Fsrc[i*KDIM0+j] *= ((b-a) * wts0[i]);
            }
          }
          ker.Eval(U0, Xtrg, Xsrc, Nsrc, Fsrc);

          Vector<ValueType> U1;
          GetGeomCircle(Xsrc, Nsrc, a+(b-a)*nds1);
          EvalFourierExp(Fsrc, F_fourier_coeff, KDIM0, a+(b-a)*nds1);
          for (Long i = 0; i < nds1.Dim(); i++) {
            for (Long j = 0; j < KDIM0; j++) {
              Fsrc[i*KDIM0+j] *= ((b-a) * wts1[i]);
            }
          }
          ker.Eval(U1, Xtrg, Xsrc, Nsrc, Fsrc);

          ValueType err = 0, max_val = 0;
          for (Long i = 0; i < U1.Dim(); i++) {
            err = std::max<ValueType>(err, fabs(U0[i]-U1[i]));
            max_val = std::max<ValueType>(max_val, fabs(U0[i]));
          }
          if (err < tol || (b-a)<tol) {
          //if ((a != 0 && b != 2*const_pi<ValueType>()) || (b-a)<tol) {
            std::cout<<a<<' '<<b-a<<' '<<err<<' '<<tol<<'\n';
            return U1;
          } else {
            U0 = compute_potential(a, (a+b)*0.5, tol);
            U1 = compute_potential((a+b)*0.5, b, tol);
            return U0 + U1;
          }
        };
        U = compute_potential(0, 2*const_pi<ValueType>(), tol);
      };

      template <Integer Nnds, class Kernel> void BuildOperatorModalDirect(Matrix<Real>& M, const Real x0, const Real x1, const Real x2, const Kernel& ker) const {
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Integer Nmm = (Nm/2+1)*2;

        auto get_sin_theta = [](Long N){
          Vector<Real> sin_theta(N);
          for (Long i = 0; i < N; i++) {
            sin_theta[i] = sin(2*const_pi<Real>()*i/N);
          }
          return sin_theta;
        };
        auto get_cos_theta = [](Long N){
          Vector<Real> cos_theta(N);
          for (Long i = 0; i < N; i++) {
            cos_theta[i] = cos(2*const_pi<Real>()*i/N);
          }
          return cos_theta;
        };
        auto get_circle_coord = [](Long N, Real R0){
          Vector<Real> X(N*COORD_DIM);
          for (Long i = 0; i < N; i++) {
            X[i*COORD_DIM+0] = R0*cos(2*const_pi<Real>()*i/N);
            X[i*COORD_DIM+1] = R0*sin(2*const_pi<Real>()*i/N);
            X[i*COORD_DIM+2] = 0;
          }
          return X;
        };
        static Real scal = 2/sqrt<Real>(Nm);

        static const Vector<Real> sin_nds = get_sin_theta(Nnds);
        static const Vector<Real> cos_nds = get_cos_theta(Nnds);
        static const Vector<Real> Xn = get_circle_coord(Nnds,1);

        StaticArray<Real,Nnds*COORD_DIM> buff0;
        Vector<Real> Xs(Nnds*COORD_DIM,buff0,false);
        Xs = Xn * R0_;

        StaticArray<Real,COORD_DIM> Xt = {x0,x1,x2};
        StaticArray<Real,KDIM0*KDIM1*Nnds> mem_buff2;
        Matrix<Real> Mker(KDIM0*Nnds, KDIM1, mem_buff2, false);
        ker.KernelMatrix(Mker, Vector<Real>(COORD_DIM,(Iterator<Real>)Xt,false), Xs, Xn);

        StaticArray<Real,4*Nnds> mem_buff3;
        Vector<Complex<Real>> exp_itheta(Nnds, (Iterator<Complex<Real>>)(mem_buff3+0*Nnds), false);
        Vector<Complex<Real>> exp_iktheta_da(Nnds, (Iterator<Complex<Real>>)(mem_buff3+2*Nnds), false);
        for (Integer j = 0; j < Nnds; j++) {
          exp_itheta[j].real = cos_nds[j];
          exp_itheta[j].imag =-sin_nds[j];
          exp_iktheta_da[j].real = 2*const_pi<Real>()/Nnds*scal;
          exp_iktheta_da[j].imag = 0;
        }
        for (Integer k = 0; k < Nmm/2; k++) { // apply Mker to complex exponentials
          // TODO: FFT might be faster since points are uniform
          Tensor<Real,true,KDIM0,KDIM1> Mk0, Mk1;
          for (Integer i0 = 0; i0 < KDIM0; i0++) {
            for (Integer i1 = 0; i1 < KDIM1; i1++) {
              Mk0(i0,i1) = 0;
              Mk1(i0,i1) = 0;
            }
          }
          for (Integer j = 0; j < Nnds; j++) {
            Tensor<Real,false,KDIM0,KDIM1> Mker_(Mker[j*KDIM0]);
            Mk0 = Mk0 + Mker_ * exp_iktheta_da[j].real;
            Mk1 = Mk1 + Mker_ * exp_iktheta_da[j].imag;
          }
          for (Integer i0 = 0; i0 < KDIM0; i0++) {
            for (Integer i1 = 0; i1 < KDIM1; i1++) {
              M[i0*Nmm+(k*2+0)][i1] = Mk0(i0,i1);
              M[i0*Nmm+(k*2+1)][i1] = Mk1(i0,i1);
            }
          }
          exp_iktheta_da *= exp_itheta;
        }
        for (Integer i0 = 0; i0 < KDIM0; i0++) {
          for (Integer i1 = 0; i1 < KDIM1; i1++) {
            M[i0*Nmm+0][i1] *= 0.5;
            M[i0*Nmm+1][i1] *= 0.5;
            if (Nm%2 == 0) {
              M[(i0+1)*Nmm-2][i1] *= 0.5;
              M[(i0+1)*Nmm-1][i1] *= 0.5;
            }
          }
        }
      }

      Real R0_;
      FFT<Real> fft_Nm_R2C, fft_Nm_C2R;
      Matrix<Real> Mnds2coeff0, Mnds2coeff1;
      Vector<Real> U; // KDIM0*Nmm*KDIM1*Nr*Ntt
      Vector<Real> Ut; // Nr*Ntt*KDIM0*Nmm*KDIM1
  };

  template <class Real, Integer ChebOrder = 10, Integer FourierOrder = 8> class SlenderElement : public GenericBoundaryElement<Real,SlenderElement<Real,ChebOrder,FourierOrder>> {
      static constexpr Integer COORD_DIM = 3;
      using Vec3 = Tensor<Real,true,COORD_DIM,1>;

      static constexpr Integer FARFIELD_UPSAMPLE = 1;

    public:

      SlenderElement() = default;
      SlenderElement(const Vector<Real>& coord, const Vector<Real>& radius) {
        Init(coord, radius);
      }
      void Init(const Vector<Real>& coord_, const Vector<Real>& radius_) {
        SCTL_ASSERT_MSG(coord_.Dim() == ChebOrder*COORD_DIM, "Length of the coordinate vector does not match the element order");
        SCTL_ASSERT_MSG(radius_.Dim() == ChebOrder, "Length of the radius vector does not match the element order");
        Vector<Real>(ChebOrder*COORD_DIM,coord,false) = coord_;
        for (Long i = 0; i < ChebOrder; i++) {
          for (Integer k = 0; k < COORD_DIM; k++) {
            coord[k*ChebOrder+i] = coord_[i*COORD_DIM+k];
          }
          radius[i] = radius_[i];
        }
        { // Set dr, ds, d2s
          Vector<Real> dr_(ChebOrder,dr,false);
          Vector<Real> dx_(COORD_DIM*ChebOrder,dx,false);
          Vector<Real> d2x_(COORD_DIM*ChebOrder,d2x,false);
          LagrangeInterp<Real>::Derivative(dr_, Vector<Real>(ChebOrder,radius,false), CenterlineNodes());
          LagrangeInterp<Real>::Derivative(dx_, Vector<Real>(COORD_DIM*ChebOrder,coord,false), CenterlineNodes());
          LagrangeInterp<Real>::Derivative(d2x_, dx_, CenterlineNodes());
        }
        { // Set e1
          Integer orient_dir = 0;
          for (Integer i = 0; i < COORD_DIM; i++) {
            e1[i*ChebOrder+0] = 0;
            if (fabs(dx[i*ChebOrder+0]) < fabs(dx[orient_dir*ChebOrder+0])) orient_dir = i;
          }
          e1[orient_dir*ChebOrder+0] = 1;
          for (Long i = 0; i < ChebOrder; i++) {
            Vec3 e1_vec, dx_vec;
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1_vec(k,0) = (i==0 ? e1[k*ChebOrder] : e1[k*ChebOrder+i-1]);
              dx_vec(k,0) = dx[k*ChebOrder+i];
            }
            e1_vec = e1_vec - dx_vec*(dot_prod(dx_vec,e1_vec)/dot_prod(dx_vec,dx_vec));
            Real scal = (1.0/sqrt<Real>(dot_prod(e1_vec,e1_vec)));
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1[k*ChebOrder+i] = e1_vec(k,0) * scal;
            }
          }
        }
      }

      // static functions
      static constexpr Integer CoordDim() {
        return COORD_DIM;
      }

      // geometry functions
      static constexpr Integer NodeCount() {
        return ChebOrder * FourierOrder;
      }
      void GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn) const {
        GetGeom(X,Xn,nullptr,nullptr,nullptr, CenterlineNodes(), sin_theta<FourierOrder>(), cos_theta<FourierOrder>());
        //static auto& sin_theta_ = sin_theta();
        //static auto& cos_theta_ = cos_theta();
        //if (X.Dim() != NodeCount()*COORD_DIM) X.ReInit(NodeCount()*COORD_DIM);
        //for (Long i = 0; i < ChebOrder; i++) {
        //  Vec3 x_, dx_, e1_;
        //  for (Long k = 0; k < COORD_DIM; k++) {
        //    x_(k,0) = coord[k*ChebOrder+i];
        //    dx_(k,0) = dx[k*ChebOrder+i];
        //    e1_(k,0) = e1[k*ChebOrder+i];
        //  }
        //  Real inv_dx2 = 1/dot_prod(dx_,dx_);
        //  e1_ = e1_ - dx_ * dot_prod(e1_, dx_) * inv_dx2;
        //  e1_ = e1_ * (1/sqrt<Real>(dot_prod(e1_,e1_)));

        //  Vec3 e2_ = cross_prod(e1_,dx_);
        //  e2_ = e2_ * (1/sqrt<Real>(dot_prod(e2_,e2_)));
        //  for (Long j = 0; j < FourierOrder; j++) {
        //    Vec3 y = x_ + e1_ * (radius[i]*cos_theta_[j]) + e2_ * (radius[i]*sin_theta_[j]);
        //    for (Long k = 0; k < COORD_DIM; k++) {
        //      X[(i*FourierOrder+j)*COORD_DIM+k] = y(k,0);
        //    }
        //  }
        //}
      }

      // far-field quadratures
      static constexpr Integer FarFieldQuadNodeCount() {
        return ChebOrder*FARFIELD_UPSAMPLE*FourierOrder*FARFIELD_UPSAMPLE;
      }
      template <Integer DOF> static void GetFarFieldQuadDensity(Vector<Real>& Fout, const Vector<Real>& Fin) {
        if (Fout.Dim() != FarFieldQuadNodeCount()*DOF) Fout.ReInit(FarFieldQuadNodeCount()*DOF);
        SCTL_ASSERT(Fin.Dim() == NodeCount()*DOF);
        auto compute_Mfourier_upsample = []() {
          static constexpr Integer FourierModes = FourierOrder/2+1;
          Matrix<Real> Mfourier_inv = fourier_matrix_inv<FourierOrder,FourierModes>();
          Matrix<Real> Mfourier = fourier_matrix<FourierModes,FourierOrder*FARFIELD_UPSAMPLE>();
          return Mfourier_inv * Mfourier;
        };
        auto compute_Mcheb_upsample = []() {
          Matrix<Real> Minterp(ChebOrder, ChebOrder*FARFIELD_UPSAMPLE);
          const auto& nds = CenterlineNodes();
          const auto& leg_nds = LegendreQuadRule<Real,ChebOrder*FARFIELD_UPSAMPLE>().first;
          Vector<Real> Vinterp(ChebOrder*ChebOrder*FARFIELD_UPSAMPLE, Minterp.begin(), false);
          LagrangeInterp<Real>::Interpolate(Vinterp, nds, leg_nds);
          return Minterp;
        };
        static const Matrix<Real> Mfourier_upsample = compute_Mfourier_upsample();
        static const Matrix<Real> Mcheb_upsample = compute_Mcheb_upsample().Transpose();

        StaticArray<Real,ChebOrder*FourierOrder*FARFIELD_UPSAMPLE*DOF> F0;
        Matrix<Real> F0_(ChebOrder,FourierOrder*FARFIELD_UPSAMPLE*DOF, F0, false);
        const Matrix<Real> Fin_(ChebOrder,FourierOrder*DOF, (Iterator<Real>)Fin.begin(), false);
        Matrix<Real> Fout_(ChebOrder*FARFIELD_UPSAMPLE,FourierOrder*FARFIELD_UPSAMPLE*DOF, Fout.begin(), false);
        for (Long i = 0; i < ChebOrder; i++) { // Set F0
          for (Long k = 0; k < DOF; k++) {
            for (Long j0 = 0; j0 < FourierOrder*FARFIELD_UPSAMPLE; j0++) {
              Real f = 0;
              for (Long j1 = 0; j1 < FourierOrder; j1++) {
                f += Fin_[i][j1*DOF+k] * Mfourier_upsample[j1][j0];
              }
              F0_[i][j0*DOF+k] = f;
            }
          }
        }
        Matrix<Real>::GEMM(Fout_, Mcheb_upsample, F0_);
      }
      void GetFarFieldQuadNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Real tol) const {
        if (wts     .Dim() != FarFieldQuadNodeCount()) wts     .ReInit(FarFieldQuadNodeCount());
        if (dist_far.Dim() != FarFieldQuadNodeCount()) dist_far.ReInit(FarFieldQuadNodeCount());
        const auto& leg_nds = LegendreQuadRule<Real,ChebOrder*FARFIELD_UPSAMPLE>().first;
        const auto& leg_wts = LegendreQuadRule<Real,ChebOrder*FARFIELD_UPSAMPLE>().second;

        StaticArray<Real,FarFieldQuadNodeCount()*COORD_DIM> dX_ds_, dX_dt_;
        Vector<Real> dX_ds(FarFieldQuadNodeCount()*COORD_DIM, (Iterator<Real>)dX_ds_, false);
        Vector<Real> dX_dt(FarFieldQuadNodeCount()*COORD_DIM, (Iterator<Real>)dX_dt_, false);
        GetGeom(&X, &Xn, &wts, &dX_ds, &dX_dt, leg_nds, sin_theta<FourierOrder*FARFIELD_UPSAMPLE>(), cos_theta<FourierOrder*FARFIELD_UPSAMPLE>());

        static constexpr Real theta_quad_wt = 2*const_pi<Real>()/(FourierOrder*FARFIELD_UPSAMPLE);
        for (Long i = 0; i < ChebOrder*FARFIELD_UPSAMPLE; i++) { // Set wts *= leg_wts * 2*pi/(FourierOrder*FARFIELD_UPSAMPLE)
          Real quad_wt = leg_wts[i] * theta_quad_wt;
          for (Long j = 0; j < FourierOrder*FARFIELD_UPSAMPLE; j++) {
            wts[i*FourierOrder*FARFIELD_UPSAMPLE+j] *= quad_wt;
          }
        }
        { // Set dist_far
          for (Long i = 0; i < FarFieldQuadNodeCount(); i++) {
            Real dxds = sqrt<Real>(dX_ds[i*COORD_DIM+0]*dX_ds[i*COORD_DIM+0] + dX_ds[i*COORD_DIM+1]*dX_ds[i*COORD_DIM+1] + dX_ds[i*COORD_DIM+2]*dX_ds[i*COORD_DIM+2])*const_pi<Real>()/2;
            Real dxdt = sqrt<Real>(dX_dt[i*COORD_DIM+0]*dX_dt[i*COORD_DIM+0] + dX_dt[i*COORD_DIM+1]*dX_dt[i*COORD_DIM+1] + dX_dt[i*COORD_DIM+2]*dX_dt[i*COORD_DIM+2])*const_pi<Real>()*2;
            Real h = std::max<Real>(dxds/(ChebOrder*FARFIELD_UPSAMPLE), dxdt/(FourierOrder*FARFIELD_UPSAMPLE));
            dist_far[i] = -0.15 * log(tol)*h; // TODO: use better estimate
          }
        }
      }

      // near-singlar quadratures
      template <class Kernel> Matrix<Real> NearInterac(const Vec3& Xt, const Kernel& ker, Real tol) const {
        static constexpr Integer FourierModes = FourierOrder/2+1;
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Integer Nbuff = 10000; // TODO
        Integer digits = (Integer)(log(tol)/log(0.1)+0.5);

        Matrix<Real> Mt(KDIM1, KDIM0*ChebOrder*FourierModes*2);
        { // Set Mt
          Vector<Real> quad_nds, quad_wts; // Quadrature rule in s
          auto adap_quad_rule = [this,tol](Vector<Real>& quad_nds, Vector<Real>& quad_wts, const Vec3& x_trg) {
            static constexpr Long LegQuadOrder = 1*ChebOrder;
            const auto& leg_nds = LegendreQuadRule<Real,LegQuadOrder>().first;
            const auto& leg_wts = LegendreQuadRule<Real,LegQuadOrder>().second;
            auto adap_ref = [&leg_nds,&leg_wts](Vector<Real>& nds, Vector<Real>& wts, Real a, Real b, Integer levels) {
              if (nds.Dim() != levels * LegQuadOrder) nds.ReInit(levels*LegQuadOrder);
              if (wts.Dim() != levels * LegQuadOrder) wts.ReInit(levels*LegQuadOrder);
              Vector<Real> nds_(nds.Dim(), nds.begin(), false);
              Vector<Real> wts_(wts.Dim(), wts.begin(), false);

              while (levels) {
                Vector<Real> nds0(LegQuadOrder, nds_.begin(), false);
                Vector<Real> wts0(LegQuadOrder, wts_.begin(), false);
                Vector<Real> nds1((levels-1)*LegQuadOrder, nds_.begin()+LegQuadOrder, false);
                Vector<Real> wts1((levels-1)*LegQuadOrder, wts_.begin()+LegQuadOrder, false);

                Real end_point = (levels==1 ? b : (a+b)*0.5);
                nds0 = leg_nds * (end_point-a) + a;
                wts0 = leg_wts * fabs<Real>(end_point-a);

                nds_.Swap(nds1);
                wts_.Swap(wts1);
                a = end_point;
                levels--;
              }
            };

            // TODO: develop special quadrature rule instead of adaptive integration
            if (0) { // adaptive/dyadic refinement
              static constexpr Integer levels = 6;
              quad_nds.ReInit(2*levels*LegQuadOrder);
              quad_wts.ReInit(2*levels*LegQuadOrder);
              Vector<Real> nds0(levels*LegQuadOrder,quad_nds.begin(),false);
              Vector<Real> wts0(levels*LegQuadOrder,quad_wts.begin(),false);
              Vector<Real> nds1(levels*LegQuadOrder,quad_nds.begin()+levels*LegQuadOrder,false);
              Vector<Real> wts1(levels*LegQuadOrder,quad_wts.begin()+levels*LegQuadOrder,false);
              adap_ref(nds0, wts0, 0.5, 0.0, levels);
              adap_ref(nds1, wts1, 0.5, 1.0, levels);
            } else {
              Real dist_min, s_min, dxds;
              { // Set dist_min, s_min, dxds
                auto get_dist = [this] (const Vec3& x_trg, Real s) -> Real {
                  StaticArray<Real,ChebOrder> buff;
                  Vector<Real> interp_wts(ChebOrder,buff,false);
                  LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(), Vector<Real>(1,Ptr2Itr<Real>(&s,1),false));

                  Real r0 = 0;
                  Vec3 x0, dx_ds0;
                  for (Long i = 0; i < COORD_DIM; i++) {
                    x0(i,0) = 0;
                    dx_ds0(i,0) = 0;
                  }
                  for (Long i = 0; i < ChebOrder; i++) {
                    r0 += radius[i] * interp_wts[i];
                    x0(0,0) += this->coord[0*ChebOrder+i] * interp_wts[i];
                    x0(1,0) += this->coord[1*ChebOrder+i] * interp_wts[i];
                    x0(2,0) += this->coord[2*ChebOrder+i] * interp_wts[i];
                    dx_ds0(0,0) += this->dx[0*ChebOrder+i] * interp_wts[i];
                    dx_ds0(1,0) += this->dx[1*ChebOrder+i] * interp_wts[i];
                    dx_ds0(2,0) += this->dx[2*ChebOrder+i] * interp_wts[i];
                  }
                  Vec3 dx = x0 - x_trg;
                  Vec3 n0 = dx_ds0 * sqrt<Real>(1/dot_prod(dx_ds0, dx_ds0));
                  Real dz = dot_prod(dx, n0);
                  Vec3 dr = dx - n0*dz;
                  Real dR = sqrt<Real>(dot_prod(dr,dr)) - r0;
                  return sqrt<Real>(dR*dR + dz*dz);
                };
                StaticArray<Real,2> dist;
                StaticArray<Real,2> s_val = {0,1};
                dist[0] = get_dist(x_trg, s_val[0]);
                dist[1] = get_dist(x_trg, s_val[1]);
                for (Long i = 0; i < 20; i++) { // Binary search: set dist, s_val
                  Real ss = (s_val[0] + s_val[1]) * 0.5;
                  Real dd = get_dist(x_trg, ss);
                  if (dist[0] > dist[1]) {
                    dist[0] = dd;
                    s_val[0] = ss;
                  } else {
                    dist[1] = dd;
                    s_val[1] = ss;
                  }
                }
                if (dist[0] < dist[1]) { // Set dis_min, s_min
                  dist_min = dist[0];
                  s_min = s_val[0];
                } else {
                  dist_min = dist[1];
                  s_min = s_val[1];
                }
                { // Set dx_ds;
                  StaticArray<Real,ChebOrder> buff;
                  Vector<Real> interp_wts(ChebOrder,buff,false);
                  LagrangeInterp<Real>::Interpolate(interp_wts, CenterlineNodes(), Vector<Real>(1,Ptr2Itr<Real>(&s_min,1),false));

                  Vec3 dxds_vec;
                  for (Long i = 0; i < COORD_DIM; i++) {
                    dxds_vec(i,0) = 0;
                  }
                  for (Long i = 0; i < ChebOrder; i++) {
                    dxds_vec(0,0) += this->dx[0*ChebOrder+i] * interp_wts[i];
                    dxds_vec(1,0) += this->dx[1*ChebOrder+i] * interp_wts[i];
                    dxds_vec(2,0) += this->dx[2*ChebOrder+i] * interp_wts[i];
                  }
                  dxds = sqrt<Real>(dot_prod(dxds_vec,dxds_vec))*const_pi<Real>()/2;
                }
              }
              Real h0 =   (s_min)*dxds/LegQuadOrder;
              Real h1 = (1-s_min)*dxds/LegQuadOrder;
              Real dist_far0 = -0.15 * log(tol)*h0; // TODO: use better estimate
              Real dist_far1 = -0.15 * log(tol)*h1; // TODO: use better estimate
              Integer adap_levels0 = (s_min==0 ? 0 : std::max<Integer>(0,(Integer)(log(dist_far0/dist_min)/log(2.0)+0.5))+1);
              Integer adap_levels1 = (s_min==1 ? 0 : std::max<Integer>(0,(Integer)(log(dist_far1/dist_min)/log(2.0)+0.5))+1);

              Long N0 = adap_levels0 * LegQuadOrder;
              Long N1 = adap_levels1 * LegQuadOrder;
              quad_nds.ReInit(N0+N1);
              quad_wts.ReInit(N0+N1);
              Vector<Real> nds0(N0, quad_nds.begin(), false);
              Vector<Real> wts0(N0, quad_wts.begin(), false);
              Vector<Real> nds1(N1, quad_nds.begin()+N0, false);
              Vector<Real> wts1(N1, quad_wts.begin()+N0, false);
              adap_ref(nds0, wts0, 0, s_min, adap_levels0);
              adap_ref(nds1, wts1, 1, s_min, adap_levels1);
            }
          };
          adap_quad_rule(quad_nds, quad_wts, Xt);
          SCTL_ASSERT(quad_nds.Dim() <= Nbuff);

          Matrix<Real> Minterp_quad_nds;
          { // Set Minterp_quad_nds
            Minterp_quad_nds.ReInit(ChebOrder, quad_nds.Dim());
            Vector<Real> Vinterp_quad_nds(ChebOrder*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
            LagrangeInterp<Real>::Interpolate(Vinterp_quad_nds, CenterlineNodes(), quad_nds);
          }

          Vec3 x_trg = Xt;
          Matrix<Real> r_src, dr_src, x_src, dx_src, d2x_src, e1_src, e2_src, de1_src, de2_src;
          r_src  .ReInit(        1,quad_nds.Dim());
          dr_src .ReInit(        1,quad_nds.Dim());
          x_src  .ReInit(COORD_DIM,quad_nds.Dim());
          dx_src .ReInit(COORD_DIM,quad_nds.Dim());
          d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
          e1_src .ReInit(COORD_DIM,quad_nds.Dim());
          e2_src .ReInit(COORD_DIM,quad_nds.Dim());
          de1_src.ReInit(COORD_DIM,quad_nds.Dim());
          de2_src.ReInit(COORD_DIM,quad_nds.Dim());
          { // Set x_src, x_trg (improve numerical stability)
            Matrix<Real> x_nodes(COORD_DIM,ChebOrder, (Iterator<Real>)(ConstIterator<Real>)coord, true);
            for (Long j = 0; j < ChebOrder; j++) {
              for (Integer k = 0; k < COORD_DIM; k++) {
                x_nodes[k][j] -= x_trg(k,0);
              }
            }
            Matrix<Real>::GEMM(  x_src, x_nodes, Minterp_quad_nds);
            for (Integer k = 0; k < COORD_DIM; k++) {
              x_trg(k,0) = 0;
            }
          }
          //Matrix<Real>::GEMM(  x_src, Matrix<Real>(COORD_DIM,ChebOrder, coord,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( dx_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dx,false), Minterp_quad_nds);
          Matrix<Real>::GEMM(d2x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)   d2x,false), Minterp_quad_nds);
          Matrix<Real>::GEMM(  r_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)radius,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( dr_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dr,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( e1_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    e1,false), Minterp_quad_nds);
          for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
            Vec3 e1, dx, d2x;
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1(k,0) = e1_src[k][j];
              dx(k,0) = dx_src[k][j];
              d2x(k,0) = d2x_src[k][j];
            }
            Real inv_dx2 = 1/dot_prod(dx,dx);
            e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
            e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

            Vec3 e2 = cross_prod(e1, dx);
            e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
            Vec3 de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
            Vec3 de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1_src[k][j] = e1(k,0);
              e2_src[k][j] = e2(k,0);
              de1_src[k][j] = de1(k,0);
              de2_src[k][j] = de2(k,0);
            }
          }

          const Vec3 y_trg = x_trg;
          StaticArray<Real,Nbuff*FourierModes*2*KDIM0*KDIM1> mem_buff0;
          Matrix<Real> M_tor(quad_nds.Dim(), FourierModes*2*KDIM0 * KDIM1, mem_buff0, false);
          auto toroidal_greens_fn_batched = [this,&ker,digits](Matrix<Real>& M, const Vec3& y_trg, const Matrix<Real>& x_src, const Matrix<Real>& dx_src, const Matrix<Real>& d2x_src, const Matrix<Real>& r_src, const Matrix<Real>& dr_src, const Matrix<Real>& e1_src, const Matrix<Real>& e2_src, const Matrix<Real>& de1_src, const Matrix<Real>& de2_src){
            const Long BatchSize = M.Dim(0);
            SCTL_ASSERT(  x_src.Dim(1) == BatchSize);
            SCTL_ASSERT( dx_src.Dim(1) == BatchSize);
            SCTL_ASSERT(d2x_src.Dim(1) == BatchSize);
            SCTL_ASSERT(  r_src.Dim(1) == BatchSize);
            SCTL_ASSERT( dr_src.Dim(1) == BatchSize);
            SCTL_ASSERT( e1_src.Dim(1) == BatchSize);
            SCTL_ASSERT( e2_src.Dim(1) == BatchSize);
            SCTL_ASSERT(de1_src.Dim(1) == BatchSize);
            SCTL_ASSERT(de2_src.Dim(1) == BatchSize);
            SCTL_ASSERT(M.Dim(1) == FourierModes*2*KDIM0 * KDIM1);
            for (Long ii = 0; ii < BatchSize; ii++) {
              Real r = r_src[0][ii], dr = dr_src[0][ii];
              Vec3 x, dx, d2x, e1, e2, de1, de2;
              { // Set x, dx, d2x, e1, e2, de1, de2
                for (Integer k = 0; k < COORD_DIM; k++) {
                  x  (k,0) =   x_src[k][ii];
                  dx (k,0) =  dx_src[k][ii];
                  d2x(k,0) = d2x_src[k][ii];
                  e1 (k,0) =  e1_src[k][ii];
                  e2 (k,0) =  e2_src[k][ii];
                  de1(k,0) = de1_src[k][ii];
                  de2(k,0) = de2_src[k][ii];
                }
              }

              auto toroidal_greens_fn = [this,&ker,digits](Matrix<Real>& M, const Vec3& Xt, const Vec3& x, const Vec3& dx, const Vec3& d2x, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const Real r, const Real dr) {
                SCTL_ASSERT(M.Dim(0) == KDIM0*FourierModes*2);
                SCTL_ASSERT(M.Dim(1) == KDIM1);

                StaticArray<Real,3*Nbuff> mem_buff0;
                Vector<Real> wts, sin_nds, cos_nds;
                ToroidalSpecialQuadRule<Real,FourierModes+1>(sin_nds, cos_nds, wts, mem_buff0, 3*Nbuff, Xt-x, e1, e2, cross_prod(e1,e2), r, digits);
                const Long Nnds = wts.Dim();

                StaticArray<Real,(COORD_DIM*2+1)*Nbuff> mem_buff1;
                Vector<Real> y(Nnds*COORD_DIM, mem_buff1+0*COORD_DIM*Nbuff, false);
                Vector<Real> n(Nnds*COORD_DIM, mem_buff1+1*COORD_DIM*Nbuff, false);
                Vector<Real> da(         Nnds, mem_buff1+2*COORD_DIM*Nbuff, false);
                for (Integer j = 0; j < Nnds; j++) { // Set x, n, da
                  Real sint = sin_nds[j];
                  Real cost = cos_nds[j];

                  Vec3 dy_ds = dx + e1*(dr*cost) + e2*(dr*sint) + de1*(r*cost) + de2*(r*sint);
                  Vec3 dy_dt = e1*(-r*sint) + e2*(r*cost);

                  Vec3 y_ = x + e1*(r*cost) + e2*(r*sint);
                  Vec3 n_ = cross_prod(dy_ds, dy_dt);
                  Real da_ = sqrt<Real>(dot_prod(n_,n_));
                  n_ = n_ * (1/da_);

                  for (Integer k = 0; k < COORD_DIM; k++) {
                    y[j*COORD_DIM+k] = y_(k,0);
                    n[j*COORD_DIM+k] = n_(k,0);
                  }
                  da[j] = da_;
                }

                StaticArray<Real,KDIM0*KDIM1*Nbuff> mem_buff2;
                Matrix<Real> Mker(KDIM0*Nnds, KDIM1, mem_buff2, false);
                ker.KernelMatrix(Mker, Vector<Real>(COORD_DIM,(Iterator<Real>)Xt.begin(),false), y, n);

                StaticArray<Real,4*Nbuff> mem_buff3;
                Vector<Complex<Real>> exp_itheta(Nnds, (Iterator<Complex<Real>>)(mem_buff3+0*Nbuff), false);
                Vector<Complex<Real>> exp_iktheta_da(Nnds, (Iterator<Complex<Real>>)(mem_buff3+2*Nbuff), false);
                for (Integer j = 0; j < Nnds; j++) {
                  exp_itheta[j].real = cos_nds[j];
                  exp_itheta[j].imag = sin_nds[j];
                  exp_iktheta_da[j].real = da[j] * wts[j];
                  exp_iktheta_da[j].imag = 0;
                }
                for (Integer k = 0; k < FourierModes; k++) {
                  Tensor<Real,true,KDIM0,KDIM1> Mk0, Mk1;
                  for (Integer i0 = 0; i0 < KDIM0; i0++) {
                    for (Integer i1 = 0; i1 < KDIM1; i1++) {
                      Mk0(i0,i1) = 0;
                      Mk1(i0,i1) = 0;
                    }
                  }
                  for (Integer j = 0; j < Nnds; j++) {
                    Tensor<Real,false,KDIM0,KDIM1> Mker_(Mker[j*KDIM0]);
                    Mk0 = Mk0 + Mker_ * exp_iktheta_da[j].real;
                    Mk1 = Mk1 + Mker_ * exp_iktheta_da[j].imag;
                  }
                  for (Integer i0 = 0; i0 < KDIM0; i0++) {
                    for (Integer i1 = 0; i1 < KDIM1; i1++) {
                      M[i0*(FourierModes*2)+(k*2+0)][i1] = Mk0(i0,i1);
                      M[i0*(FourierModes*2)+(k*2+1)][i1] = Mk1(i0,i1);
                    }
                  }
                  exp_iktheta_da *= exp_itheta;
                }
              };
              Matrix<Real> M_toroidal_greens_fn(KDIM0*FourierModes*2, KDIM1, M[ii], false);
              toroidal_greens_fn(M_toroidal_greens_fn, y_trg, x, dx, d2x, e1, e2, de1, de2, r, dr);
            }
          };
          toroidal_greens_fn_batched(M_tor, y_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, e2_src, de1_src, de2_src);

          StaticArray<Real,ChebOrder*FourierModes*2*KDIM0*KDIM1> mem_buff1;
          Matrix<Real> M_(ChebOrder, FourierModes*2*KDIM0 * KDIM1, mem_buff1, false);
          for (Long ii = 0; ii < M_tor.Dim(0); ii++) {
            Matrix<Real> M_tor_(M_tor.Dim(1), KDIM1, M_tor[ii], false);
            M_tor_ *= quad_wts[ii];
          }
          Matrix<Real>::GEMM(M_, Minterp_quad_nds, M_tor);

          for (Long ii = 0; ii < ChebOrder*FourierModes*2; ii++) { // Mt <-- M_
            for (Long k0 = 0; k0 < KDIM0; k0++) {
              for (Long k1 = 0; k1 < KDIM1; k1++) {
                Mt[k1][k0*ChebOrder*FourierModes*2+ii] = M_[0][(ii*KDIM0+k0)*KDIM1+k1];
              }
            }
          }
        }

        Matrix<Real> Mt_(KDIM1, KDIM0*ChebOrder*FourierOrder);
        { // Set Mt_
          static const Matrix<Real> M_fourier_inv = fourier_matrix_inv<FourierOrder,FourierModes>().Transpose();
          Matrix<Real> M_nodal(KDIM1*KDIM0*ChebOrder, FourierOrder, Mt_.begin(), false);
          Matrix<Real> M_modal(KDIM1*KDIM0*ChebOrder, FourierModes*2, Mt.begin(), false);
          Matrix<Real>::GEMM(M_nodal, M_modal, M_fourier_inv);
        }
        return Mt_.Transpose();
      }

      // singular quadratures
      template <class Kernel> Matrix<Real> SelfInterac(const Kernel& ker, Real tol) const {
        if (tol < 1e-14) {
          return SelfInteracHelper<15,Kernel>(ker);
        } else if (tol < 1e-13) {
          return SelfInteracHelper<14,Kernel>(ker);
        } else if (tol < 1e-12) {
          return SelfInteracHelper<13,Kernel>(ker);
        } else if (tol < 1e-11) {
          return SelfInteracHelper<12,Kernel>(ker);
        } else if (tol < 1e-10) {
          return SelfInteracHelper<11,Kernel>(ker);
        } else if (tol < 1e-09) {
          return SelfInteracHelper<10,Kernel>(ker);
        } else if (tol < 1e-08) {
          return SelfInteracHelper< 9,Kernel>(ker);
        } else if (tol < 1e-07) {
          return SelfInteracHelper< 8,Kernel>(ker);
        } else if (tol < 1e-06) {
          return SelfInteracHelper< 7,Kernel>(ker);
        } else if (tol < 1e-05) {
          return SelfInteracHelper< 6,Kernel>(ker);
        } else if (tol < 1e-04) {
          return SelfInteracHelper< 5,Kernel>(ker);
        } else if (tol < 1e-03) {
          return SelfInteracHelper< 4,Kernel>(ker);
        } else if (tol < 1e-02) {
          return SelfInteracHelper< 3,Kernel>(ker);
        } else if (tol < 1e-01) {
          return SelfInteracHelper< 2,Kernel>(ker);
        } else if (tol < 1e-00) {
          return SelfInteracHelper< 1,Kernel>(ker);
        } else{
          return SelfInteracHelper< 0,Kernel>(ker);
        }
      }

      // visualization
      void GetVTUData(VTUData& vtu_data, const Vector<Real>& F) const {
        Vector<Real> X;
        GetGeom(&X,nullptr,nullptr,nullptr,nullptr, CenterlineNodes(), sin_theta<FourierOrder>(), cos_theta<FourierOrder>());

        Long point_offset = vtu_data.coord.Dim() / COORD_DIM;
        for (const auto& x : X) vtu_data.coord.PushBack((VTUData::VTKReal)x);
        for (const auto& f : F) vtu_data.value.PushBack((VTUData::VTKReal)f);
        for (Long i = 0; i < ChebOrder-1; i++) {
          for (Long j = 0; j <= FourierOrder; j++) {
            vtu_data.connect.PushBack(point_offset + (i+0)*FourierOrder+(j%FourierOrder));
            vtu_data.connect.PushBack(point_offset + (i+1)*FourierOrder+(j%FourierOrder));
          }
          vtu_data.offset.PushBack(vtu_data.connect.Dim());
          vtu_data.types.PushBack(6);
        }
      }

      static const Vector<Real>& CenterlineNodes() {
        return ChebQuadRule<Real>::template nds<ChebOrder>();
      }

      static void test() {
        //sctl::Profile::Enable(true);
        //ComputeSpecialQuadRule<ChebOrder,FourierOrder/2+1>(); //////////////////////////
        //Profile::Tic("Test");
        //ComputeSpecialQuadRule<ChebOrder,FourierOrder/2+1>(); //////////////////////////
        //Profile::Toc();

        if (Comm::World().Rank()) return; // execute on one MPI process

        Vector<SlenderElement> elem_lst0(8);
        for (Long k = 0; k < elem_lst0.Dim(); k++) { // Init elem_lst0
          const auto& nds = SlenderElement::CenterlineNodes();
          Vector<Real> coord(nds.Dim()*COORD_DIM), radius(nds.Dim());
          for (Long i = 0; i < nds.Dim(); i++) {
            Real theta = 2*const_pi<Real>()*(k+nds[i])/(Real)elem_lst0.Dim();
            coord[i*COORD_DIM+0] = cos<Real>(theta);
            coord[i*COORD_DIM+1] = sin<Real>(theta);
            coord[i*COORD_DIM+2] = 0.1*sin<Real>(2*theta);
            radius[i] = 0.01*(2+sin<Real>(theta+sqrt<Real>(2)));
          }
          elem_lst0[k].Init(coord,radius);
        }

        GenericKernel<sctl::Laplace3D_DxU> laplace_dl;
        BoundaryIntegralOp<Real,GenericKernel<sctl::Laplace3D_DxU>> LapDL(laplace_dl);
        LapDL.AddElemList(elem_lst0);

        // Warm-up run
        Vector<Real> F(LapDL.Dim(0)), U; F = 1;
        LapDL.ComputePotential(U,F);
        LapDL.ClearSetup();

        sctl::Profile::Enable(true);
        Profile::Tic("Setup+Eval");
        LapDL.ComputePotential(U,F);
        Profile::Toc();

        U = 0;
        Profile::Tic("Eval");
        LapDL.ComputePotential(U,F);
        Profile::Toc();

        Vector<Real> Uerr = U*(1/(4*const_pi<Real>())) - 0.5;
        SlenderElement<Real>::WriteVTK("Uerr", elem_lst0, Uerr); // Write VTK
        { // Print error
          Real max_err = 0;
          for (auto x : Uerr) max_err = std::max<Real>(max_err, fabs(x));
          std::cout<<"Error = "<<max_err<<'\n';
        }
        sctl::Profile::Enable(false);
        sctl::Profile::print();
      }
      static void test_greens_identity() {
        const auto concat_vecs = [](Vector<Real>& v, const Vector<Vector<Real>>& vec_lst) {
          const Long N = vec_lst.Dim();
          Vector<Long> dsp(N+1); dsp[0] = 0;
          for (Long i = 0; i < N; i++) {
            dsp[i+1] = dsp[i] + vec_lst[i].Dim();
          }
          if (v.Dim() != dsp[N]) v.ReInit(dsp[N]);
          for (Long i = 0; i < N; i++) {
            Vector<Real> v_(vec_lst[i].Dim(), v.begin()+dsp[i], false);
            v_ = vec_lst[i];
          }
        };
        auto loop_geom = [](Real& x, Real& y, Real& z, Real& r, const Real theta){
          x = cos<Real>(theta);
          y = sin<Real>(theta);
          z = 0;//0.1*sin<Real>(theta-sqrt<Real>(2));
          r = 0.01*(2+sin<Real>(theta+sqrt<Real>(2)));
        };
        const Comm comm = Comm::World();

        Vector<SlenderElement<Real,10,14>> elem_lst0;
        Vector<SlenderElement<Real,8,14>> elem_lst1;
        { // Set elem_lst0, elem_lst1
          const Long Nelem = 16;
          const Long Np = comm.Size();
          const Long rank = comm.Rank();
          const Long idx0 = Nelem*(rank+0)/Np;
          const Long idx1 = Nelem*(rank+1)/Np;

          elem_lst0.ReInit(idx1-idx0);
          elem_lst1.ReInit(idx1-idx0);
          for (Long k = idx0; k < idx1; k++) { // Init elem_lst0
            const auto& nds = decltype(elem_lst0)::value_type::CenterlineNodes();
            Vector<Real> coord(nds.Dim()*COORD_DIM), radius(nds.Dim());
            for (Long i = 0; i < nds.Dim(); i++) {
              Real theta = const_pi<Real>()*(k+nds[i])/Nelem;
              loop_geom(coord[i*COORD_DIM+0], coord[i*COORD_DIM+1], coord[i*COORD_DIM+2], radius[i], theta);
            }
            elem_lst0[k-idx0].Init(coord,radius);
          }
          for (Long k = idx0; k < idx1; k++) { // Init elem_lst1
            const auto& nds = decltype(elem_lst1)::value_type::CenterlineNodes();
            Vector<Real> coord(nds.Dim()*COORD_DIM), radius(nds.Dim());
            for (Long i = 0; i < nds.Dim(); i++) {
              Real theta = const_pi<Real>()*(1+(k+nds[i])/Nelem);
              loop_geom(coord[i*COORD_DIM+0], coord[i*COORD_DIM+1], coord[i*COORD_DIM+2], radius[i], theta);
            }
            elem_lst1[k-idx0].Init(coord,radius);
          }
        }

        GenericKernel<sctl::Laplace3D_FxU> laplace_sl;
        GenericKernel<sctl::Laplace3D_DxU> laplace_dl;
        GenericKernel<sctl::Laplace3D_FxdU> laplace_grad;
        BoundaryIntegralOp<Real,GenericKernel<sctl::Laplace3D_FxU>> LapSL(laplace_sl, comm);
        BoundaryIntegralOp<Real,GenericKernel<sctl::Laplace3D_DxU>> LapDL(laplace_dl, comm);
        LapSL.AddElemList(elem_lst0);
        LapSL.AddElemList(elem_lst1);
        LapDL.AddElemList(elem_lst0);
        LapDL.AddElemList(elem_lst1);

        Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
        { // Get X, Xn
          Vector<Vector<Real>> X_(2), Xn_(2);
          decltype(elem_lst0)::value_type::GetAllNodeCoord(&X_[0], &Xn_[0], elem_lst0);
          decltype(elem_lst1)::value_type::GetAllNodeCoord(&X_[1], &Xn_[1], elem_lst1);
          concat_vecs(X, X_);
          concat_vecs(Xn, Xn_);
        }
        { // Set Fs, Fd, Uref
          Vector<Real> X0{0.3,0.6,0.2}, Xn0{0,0,0}, F0{1}, dU;
          laplace_sl.Eval(Uref, X, X0, Xn0, F0);
          laplace_grad.Eval(dU, X, X0, Xn0, F0);

          Fd = Uref;
          { // Set Fs <-- -dot_prod(dU, Xn)
            Fs.ReInit(X.Dim()/COORD_DIM);
            for (Long i = 0; i < Fs.Dim(); i++) {
              Real dU_dot_Xn = 0;
              for (Long k = 0; k < COORD_DIM; k++) {
                dU_dot_Xn += dU[i*COORD_DIM+k] * Xn[i*COORD_DIM+k];
              }
              Fs[i] = -dU_dot_Xn;
            }
          }
        }

        // Warm-up run
        LapSL.ComputePotential(Us,Fs);
        LapDL.ComputePotential(Ud,Fd);
        LapSL.ClearSetup();
        LapDL.ClearSetup();

        sctl::Profile::Enable(true);
        Profile::Tic("Setup+Eval", &comm);
        LapSL.ComputePotential(Us,Fs);
        LapDL.ComputePotential(Ud,Fd);
        Profile::Toc();

        Us = 0; Ud = 0;
        Profile::Tic("Eval", &comm);
        LapSL.ComputePotential(Us,Fs);
        LapDL.ComputePotential(Ud,Fd);
        Profile::Toc();

        sctl::Profile::print(&comm);
        sctl::Profile::Enable(false);
        { // Write VTK
          Vector<Real> Uerr = Fd*0.5 + (Us + Ud)*(1/(4*const_pi<Real>())) - Uref;
          const Long N0 = elem_lst0.Dim() * decltype(elem_lst0)::value_type::NodeCount();
          const Long N1 = elem_lst1.Dim() * decltype(elem_lst1)::value_type::NodeCount();
          decltype(elem_lst0)::value_type::WriteVTK("Uerr0", elem_lst0, Vector<Real>(N0,Uerr.begin()+ 0,false), comm);
          decltype(elem_lst1)::value_type::WriteVTK("Uerr1", elem_lst1, Vector<Real>(N1,Uerr.begin()+N0,false), comm);
        }
      }

    private:

      template <class ValueType> static void ReadFile(Vector<Vector<ValueType>>& data, const std::string fname) {
        FILE* f = fopen(fname.c_str(), "r");
        if (f == nullptr) {
          std::cout << "Unable to open file for reading:" << fname << '\n';
        } else {
          uint64_t data_len;
          Long readlen = fread(&data_len, sizeof(uint64_t), 1, f);
          SCTL_ASSERT(readlen == 1);
          if (data_len) {
            data.ReInit(data_len);
            for (Long i = 0; i < data.Dim(); i++) {
              readlen = fread(&data_len, sizeof(uint64_t), 1, f);
              SCTL_ASSERT(readlen == 1);
              data[i].ReInit(data_len);
              if (data_len) {
                readlen = fread(&data[i][0], sizeof(ValueType), data_len, f);
                SCTL_ASSERT(readlen == (Long)data_len);
              }
            }
          }
          fclose(f);
        }
      }
      template <class ValueType> static void WriteFile(const Vector<Vector<ValueType>>& data, const std::string fname) {
        FILE* f = fopen(fname.c_str(), "wb+");
        if (f == nullptr) {
          std::cout << "Unable to open file for writing:" << fname << '\n';
          exit(0);
        }
        uint64_t data_len = data.Dim();
        fwrite(&data_len, sizeof(uint64_t), 1, f);

        for (Integer i = 0; i < data.Dim(); i++) {
          data_len = data[i].Dim();
          fwrite(&data_len, sizeof(uint64_t), 1, f);
          if (data_len) fwrite(&data[i][0], sizeof(ValueType), data_len, f);
        }
        fclose(f);
      }

      template <class ValueType> static ValueType dot_prod(const Tensor<ValueType,true,COORD_DIM,1>& u, const Tensor<ValueType,true,COORD_DIM,1>& v) {
        ValueType u_dot_v = 0;
        u_dot_v += u(0,0) * v(0,0);
        u_dot_v += u(1,0) * v(1,0);
        u_dot_v += u(2,0) * v(2,0);
        return u_dot_v;
      };
      template <class ValueType> static Tensor<ValueType,true,COORD_DIM,1> cross_prod(const Tensor<ValueType,true,COORD_DIM,1>& u, const Tensor<ValueType,true,COORD_DIM,1>& v) {
        Tensor<ValueType,true,COORD_DIM,1> uxv;
        uxv(0,0) = u(1,0) * v(2,0) - u(2,0) * v(1,0);
        uxv(1,0) = u(2,0) * v(0,0) - u(0,0) * v(2,0);
        uxv(2,0) = u(0,0) * v(1,0) - u(1,0) * v(0,0);
        return uxv;
      };

      template <Integer ORDER> static const Vector<Real>& sin_theta() {
        auto compute_sin_theta = [](){
          Vector<Real> sin_theta(ORDER);
          for (Long i = 0; i < ORDER; i++) {
            sin_theta[i] = sin<Real>(2*const_pi<Real>()*i/ORDER);
          }
          return sin_theta;
        };
        static const Vector<Real> sin_theta_ = compute_sin_theta();
        return sin_theta_;
      }
      template <Integer ORDER> static const Vector<Real>& cos_theta() {
        auto compute_cos_theta = [](){
          Vector<Real> cos_theta(ORDER);
          for (Long i = 0; i < ORDER; i++) {
            cos_theta[i] = cos<Real>(2*const_pi<Real>()*i/ORDER);
          }
          return cos_theta;
        };
        static const Vector<Real> cos_theta_ = compute_cos_theta();
        return cos_theta_;
      }
      template <Integer Nmodes, Integer Nnodes> static Matrix<Real> fourier_matrix() {
        Matrix<Real> M_fourier(2*Nmodes,Nnodes);
        for (Long i = 0; i < Nnodes; i++) {
          Real theta = 2*const_pi<Real>()*i/Nnodes;
          for (Long k = 0; k < Nmodes; k++) {
            M_fourier[k*2+0][i] = cos(k*theta);
            M_fourier[k*2+1][i] = sin(k*theta);
          }
        }
        return M_fourier;
      };
      template <Integer Nnodes, Integer Nmodes> static Matrix<Real> fourier_matrix_inv() {
        static_assert(Nmodes <= Nnodes/2+1);
        Matrix<Real> M_fourier_inv(Nnodes,2*Nmodes);
        for (Long i = 0; i < Nnodes; i++) {
          Real theta = 2*const_pi<Real>()*i/Nnodes;
          for (Long k = 0; k < Nmodes; k++) {
            static constexpr Real scal = 2/(Real)Nnodes;
            M_fourier_inv[i][k*2+0] = cos(k*theta)*scal;
            M_fourier_inv[i][k*2+1] = sin(k*theta)*scal;
          }
        }
        for (Long i = 0; i < Nnodes; i++) {
          M_fourier_inv[i][0] *= 0.5;
        }
        if (Nnodes == (Nmodes-1)*2) {
          for (Long i = 0; i < Nnodes; i++) {
            M_fourier_inv[i][Nnodes] *= 0.5;
          }
        }
        return M_fourier_inv;
      };

      template <class ValueType> static const std::pair<Vector<ValueType>,Vector<ValueType>> LegendreQuadRule(Integer ORDER) {
        std::pair<Vector<ValueType>,Vector<ValueType>> nds_wts;
        auto& x_ = nds_wts.first;
        auto& w_ = nds_wts.second;
        if (std::is_same<double,ValueType>::value || std::is_same<float,ValueType>::value) {
          Vector<double> xd(ORDER);
          Vector<double> wd(ORDER);
          int kind = 1;
          double alpha = 0.0, beta = 0.0, a = -1.0, b = 1.0;
          cgqf(ORDER, kind, (double)alpha, (double)beta, (double)a, (double)b, &xd[0], &wd[0]);
          for (Integer i = 0; i < ORDER; i++) {
            x_.PushBack((ValueType)(0.5 * xd[i] + 0.5));
            w_.PushBack((ValueType)(0.5 * wd[i]));
          }
        } else {
          x_ = ChebQuadRule<ValueType>::nds(ORDER);
          w_ = ChebQuadRule<ValueType>::wts(ORDER);
        }
        return nds_wts;
      };
      template <class ValueType, Integer ORDER> static const std::pair<Vector<ValueType>,Vector<ValueType>>& LegendreQuadRule() {
        static const auto nds_wts = LegendreQuadRule<ValueType>(ORDER);
        return nds_wts;
      }

      template <class RealType, Integer Nmodes> static Vector<Vector<RealType>> BuildToroidalSpecialQuadRules() {
        static constexpr Integer max_adap_depth = 30; // build quadrature rules for points up to 2*pi*0.5^max_adap_depth from source loop
        static constexpr Integer crossover_adap_depth = 4;
        static constexpr Integer max_digits = 20;

        using ValueType = QuadReal;
        Vector<Vector<ValueType>> data;
        static const std::string fname = std::string("toroidal_quad_rule_m") + std::to_string(Nmodes);
        ReadFile(data, fname);
        if (data.Dim() != max_adap_depth*max_digits) { // If file is not-found then compute quadrature rule and write to file
          data.ReInit(max_adap_depth * max_digits);
          for (Integer idx = 0; idx < max_adap_depth; idx++) {
            Vector<Vector<ValueType>> quad_nds,  quad_wts;
            { // generate special quadrature rule
              Vector<ValueType> nds, wts;
              Matrix<ValueType> Mintegrands;
              auto discretize_basis_functions = [](Matrix<ValueType>& Mintegrands, Vector<ValueType>& nds, Vector<ValueType>& wts, ValueType dist, const std::pair<Vector<ValueType>,Vector<ValueType>>& panel_quad_nds_wts) {
                auto trg_coord = [](ValueType dist) {
                  Vector<ValueType> Xtrg; //(2*Nmodes*2*Nmodes*COORD_DIM);
                  for (Long i = 0; i < 2*Nmodes; i++) {
                    for (Long j = 0; j < 2*Nmodes; j++) {
                      ValueType theta = i*2*const_pi<ValueType>()/(2*Nmodes);
                      ValueType r = (0.5 + i*0.5/(2*Nmodes)) * dist;
                      ValueType x0 = r*cos<ValueType>(theta)+1;
                      ValueType x1 = 0;
                      ValueType x2 = r*sin<ValueType>(theta);
                      if (x0 > 0) {
                        Xtrg.PushBack(x0);
                        Xtrg.PushBack(x1);
                        Xtrg.PushBack(x2);
                      }
                    }
                  }
                  return Xtrg;
                };
                Vector<ValueType> Xtrg = trg_coord(dist);
                Long Ntrg = Xtrg.Dim()/COORD_DIM;

                auto adap_nds_wts = [&panel_quad_nds_wts](Vector<ValueType>& nds, Vector<ValueType>& wts, Integer levels){
                  const auto& leg_nds = panel_quad_nds_wts.first;
                  const auto& leg_wts = panel_quad_nds_wts.second;
                  SCTL_ASSERT(levels);
                  Long N = 2*levels;
                  ValueType l = 0.5;
                  nds.ReInit(N*leg_nds.Dim());
                  wts.ReInit(N*leg_nds.Dim());
                  for (Integer idx = 0; idx < levels-1; idx++) {
                    l *= 0.5;
                    Vector<ValueType> nds0(leg_nds.Dim(), nds.begin()+(idx*2+0)*leg_nds.Dim(), false);
                    Vector<ValueType> nds1(leg_nds.Dim(), nds.begin()+(idx*2+1)*leg_nds.Dim(), false);
                    Vector<ValueType> wts0(leg_wts.Dim(), wts.begin()+(idx*2+0)*leg_wts.Dim(), false);
                    Vector<ValueType> wts1(leg_wts.Dim(), wts.begin()+(idx*2+1)*leg_wts.Dim(), false);
                    for (Long i = 0; i < leg_nds.Dim(); i++) {
                      ValueType s = leg_nds[i]*l + l;
                      nds0[i] = s;
                      nds1[i] = 1-s;
                      wts0[i] = leg_wts[i]*l;
                      wts1[i] = wts0[i];
                    }
                  }
                  { // set nds, wts
                    Long idx = levels-1;
                    Vector<ValueType> nds0(leg_nds.Dim(), nds.begin()+(idx*2+0)*leg_nds.Dim(), false);
                    Vector<ValueType> nds1(leg_nds.Dim(), nds.begin()+(idx*2+1)*leg_nds.Dim(), false);
                    Vector<ValueType> wts0(leg_wts.Dim(), wts.begin()+(idx*2+0)*leg_wts.Dim(), false);
                    Vector<ValueType> wts1(leg_wts.Dim(), wts.begin()+(idx*2+1)*leg_wts.Dim(), false);
                    for (Long i = 0; i < leg_nds.Dim(); i++) {
                      ValueType s = leg_nds[i]*l;
                      nds0[i] = s;
                      nds1[i] = 1-s;
                      wts0[i] = leg_wts[i]*l;
                      wts1[i] = wts0[i];
                    }
                  }
                };
                adap_nds_wts(nds, wts, std::max<Integer>(1,(Integer)(log(dist/2/const_pi<ValueType>())/log(0.5)+0.5)));

                Long Nnds = nds.Dim();
                Vector<Complex<ValueType>> exp_itheta(Nnds), exp_iktheta(Nnds);
                Vector<ValueType> Xsrc(Nnds*COORD_DIM);
                for (Long i = 0; i < Nnds; i++) {
                  ValueType cos_i = cos(2*const_pi<ValueType>()*nds[i]);
                  ValueType sin_i = sin(2*const_pi<ValueType>()*nds[i]);
                  exp_itheta[i].real = cos_i;
                  exp_itheta[i].imag = sin_i;
                  exp_iktheta[i].real = 1;
                  exp_iktheta[i].imag = 0;
                  Xsrc[i*COORD_DIM+0] = cos_i;
                  Xsrc[i*COORD_DIM+1] = sin_i;
                  Xsrc[i*COORD_DIM+2] = 0;
                }

                Matrix<ValueType> Mker_sl, Mker_dl;
                GenericKernel<Laplace3D_FxU> laplace_sl;
                GenericKernel<Laplace3D_DxU> laplace_dl;
                laplace_sl.KernelMatrix(Mker_sl, Xtrg, Xsrc, Xsrc);
                laplace_dl.KernelMatrix(Mker_dl, Xtrg, Xsrc, Xsrc);
                SCTL_ASSERT(Mker_sl.Dim(0) == Nnds);
                SCTL_ASSERT(Mker_sl.Dim(1) == Ntrg);
                SCTL_ASSERT(Mker_dl.Dim(0) == Nnds);
                SCTL_ASSERT(Mker_dl.Dim(1) == Ntrg);

                Mintegrands.ReInit(Nnds, (Nmodes*2) * 2 * Ntrg);
                for (Long k = 0; k < Nmodes; k++) {
                  for (Long i = 0; i < Nnds; i++) {
                    for (Long j = 0; j < Ntrg; j++) {
                      Mintegrands[i][((k*2+0)*2+0)*Ntrg+j] = Mker_sl[i][j] * exp_iktheta[i].real;
                      Mintegrands[i][((k*2+1)*2+0)*Ntrg+j] = Mker_sl[i][j] * exp_iktheta[i].imag;
                      Mintegrands[i][((k*2+0)*2+1)*Ntrg+j] = Mker_dl[i][j] * exp_iktheta[i].real;
                      Mintegrands[i][((k*2+1)*2+1)*Ntrg+j] = Mker_dl[i][j] * exp_iktheta[i].imag;
                    }
                  }
                  for (Long i = 0; i < Nnds; i++) {
                    exp_iktheta[i] *= exp_itheta[i];
                  }
                }
              };
              ValueType dist = 4*const_pi<ValueType>()*pow<ValueType,Long>(0.5,idx); // distance of target points from the source loop (which is a unit circle)
              discretize_basis_functions(Mintegrands, nds, wts, dist, LegendreQuadRule<ValueType,45>()); // TODO: adaptively select Legendre order

              Vector<ValueType> eps_vec;
              for (Long k = 0; k < max_digits; k++) eps_vec.PushBack(pow<ValueType,Long>(0.1,k));
              auto cond_num_vec = InterpQuadRule<ValueType>::Build(quad_nds, quad_wts,  Mintegrands, nds, wts, eps_vec);
            }
            for (Integer digits = 0; digits < max_digits; digits++) {
              Long N = quad_nds[digits].Dim();
              data[idx*max_digits+digits].ReInit(3*N);
              for (Long i = 0; i < N; i++) {
                data[idx*max_digits+digits][i*3+0] = cos<ValueType>(2*const_pi<ValueType>()*quad_nds[digits][i]);
                data[idx*max_digits+digits][i*3+1] = sin<ValueType>(2*const_pi<ValueType>()*quad_nds[digits][i]);
                data[idx*max_digits+digits][i*3+2] = (2*const_pi<ValueType>()*quad_wts[digits][i]);
              }
            }
          }
          WriteFile(data, fname);
        }
        for (Integer idx = 0; idx < crossover_adap_depth; idx++) { // Use trapezoidal rule up to crossover_adap_depth
          for (Integer digits = 0; digits < max_digits; digits++) {
            Long N = std::max<Long>(digits*pow<Long,Long>(2,idx), Nmodes); // TODO: determine optimal order by testing error or adaptively
            data[idx*max_digits+digits].ReInit(3*N);
            for (Long i = 0; i < N; i++) {
              ValueType quad_nds = i/(ValueType)N;
              ValueType quad_wts = 1/(ValueType)N;
              data[idx*max_digits+digits][i*3+0] = cos<ValueType>(2*const_pi<ValueType>()*quad_nds);
              data[idx*max_digits+digits][i*3+1] = sin<ValueType>(2*const_pi<ValueType>()*quad_nds);
              data[idx*max_digits+digits][i*3+2] = (2*const_pi<ValueType>()*quad_wts);
            }
          }
        }

        Vector<Vector<RealType>> quad_rule_lst;
        quad_rule_lst.ReInit(data.Dim()*3);
        for (Integer i = 0; i < data.Dim(); i++) {
          uint64_t data_len = data[i].Dim()/3;
          quad_rule_lst[i*3+0].ReInit(data_len);
          quad_rule_lst[i*3+1].ReInit(data_len);
          quad_rule_lst[i*3+2].ReInit(data_len);
          for (Long j = 0; j < (Long)data_len; j++) {
            quad_rule_lst[i*3+0][j] = (RealType)data[i][j*3+0];
            quad_rule_lst[i*3+1][j] = (RealType)data[i][j*3+1];
            quad_rule_lst[i*3+2][j] = (RealType)data[i][j*3+2];
          }
        }
        return quad_rule_lst;
      };
      template <class RealType, Integer Nmodes> static void ToroidalSpecialQuadRule(Vector<RealType>& sin_nds, Vector<RealType>& cos_nds, Vector<RealType>& wts, Iterator<RealType> mem_buff, Long Nbuff, const Tensor<RealType,true,COORD_DIM,1>& Xt_X0, const Tensor<RealType,true,COORD_DIM,1>& e1, const Tensor<RealType,true,COORD_DIM,1>& e2, const Tensor<RealType,true,COORD_DIM,1>& e1xe2, RealType R0, Integer digits) {
        static constexpr Integer max_adap_depth = 30; // build quadrature rules for points up to 2*pi*0.5^max_adap_depth from source loop
        static constexpr Integer crossover_adap_depth = 4;
        static constexpr Integer max_digits = 20;
        SCTL_ASSERT(digits<max_digits);

        const RealType XX = dot_prod(Xt_X0, e1);
        const RealType YY = dot_prod(Xt_X0, e2);
        const RealType ZZ = dot_prod(Xt_X0, e1xe2);
        const RealType R = sqrt<RealType>(XX*XX+YY*YY);
        const RealType Rinv = 1/R;
        const RealType dtheta = sqrt<RealType>((R-R0)*(R-R0) + ZZ*ZZ)/R0;
        const Complex<RealType> exp_theta0(XX*Rinv, YY*Rinv);
        Long adap_depth = 0;
        { // Set adap_depth
          for (RealType s = dtheta; s<2*const_pi<RealType>(); s*=2) adap_depth++;
          SCTL_ASSERT(adap_depth < max_adap_depth);
        }

        static const Vector<Vector<RealType>> quad_rules = BuildToroidalSpecialQuadRules<RealType,Nmodes>();
        { // Set sin_nds, cos_nds, wts
          const auto& cos_nds0 = quad_rules[(adap_depth*max_digits+digits)*3+0];
          const auto& sin_nds0 = quad_rules[(adap_depth*max_digits+digits)*3+1];
          const auto& wts0 = quad_rules[(adap_depth*max_digits+digits)*3+2];
          const Long N = wts0.Dim();

          if (Nbuff>=3*N && mem_buff != NullIterator<RealType>()) {
            cos_nds.ReInit(N, mem_buff+0*N, false);
            sin_nds.ReInit(N, mem_buff+1*N, false);
            wts.ReInit(N, mem_buff+2*N, false);
          } else {
            cos_nds.ReInit(N);
            sin_nds.ReInit(N);
            wts.ReInit(N);
          }
          if (adap_depth >= crossover_adap_depth) {
            for (Long i = 0; i < cos_nds.Dim(); i++) { // rotate cos_nds, sin_nds
              Complex<RealType> nds(cos_nds0[i], sin_nds0[i]);
              nds *= exp_theta0;
              cos_nds[i] = nds.real;
              sin_nds[i] = nds.imag;
              wts[i] = wts0[i];
            }
          } else {
            cos_nds = cos_nds0;
            sin_nds = sin_nds0;
            wts = wts0;
          }
        }
      }

      template <Integer Ncheb, Integer FourierModes, Integer digits, class ValueType, class Kernel> static void SpecialQuadBuildBasisMatrix(Matrix<ValueType>& M, Vector<ValueType>& quad_nds, Vector<ValueType>& quad_wts, ValueType elem_length, Integer RefLevels, const Kernel& ker) {
        using Vec3 = Tensor<ValueType,true,COORD_DIM,1>;
        // TODO: cleanup

        static constexpr Long LegQuadOrder = 2*digits;
        static constexpr Long LogQuadOrder = 18; // this has non-negative weights
        static constexpr Integer Nbuff = 10000; // TODO

        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();

        Vec3 y_trg;
        y_trg(0,0) = 1;
        y_trg(1,0) = 0;
        y_trg(2,0) = 0;

        StaticArray<ValueType,Ncheb> radius;
        StaticArray<ValueType,COORD_DIM*Ncheb> coord;
        StaticArray<ValueType,Ncheb> dr;
        StaticArray<ValueType,COORD_DIM*Ncheb> dx;
        StaticArray<ValueType,COORD_DIM*Ncheb> d2x;
        StaticArray<ValueType,COORD_DIM*Ncheb> e1;
        for (Long i = 0; i < Ncheb; i++) {
          radius[i] = 1;
          dr[i] = 0;

          coord[0*Ncheb+i] = 0;
          coord[1*Ncheb+i] = 0;
          coord[2*Ncheb+i] = ChebQuadRule<Real>::template nds<Ncheb>()[i] * elem_length;

          dx[0*Ncheb+i] = 0;
          dx[1*Ncheb+i] = 0;
          dx[2*Ncheb+i] = elem_length;

          d2x[0*Ncheb+i] = 0;
          d2x[1*Ncheb+i] = 0;
          d2x[2*Ncheb+i] = 0;

          e1[0*Ncheb+i] = 1;
          e1[1*Ncheb+i] = 0;
          e1[2*Ncheb+i] = 0;
        }

        auto adap_ref = [](Vector<ValueType>& nds, Vector<ValueType>& wts, ValueType a, ValueType b, Integer levels) {
          const auto& log_quad_nds = LogSingularityQuadRule<ValueType,LogQuadOrder>().first;
          const auto& log_quad_wts = LogSingularityQuadRule<ValueType,LogQuadOrder>().second;
          const auto& leg_nds = LegendreQuadRule<ValueType,LegQuadOrder>().first;
          const auto& leg_wts = LegendreQuadRule<ValueType,LegQuadOrder>().second;

          Long N = std::max<Integer>(levels-1,0)*LegQuadOrder + LogQuadOrder;
          if (nds.Dim() != N) nds.ReInit(N);
          if (wts.Dim() != N) wts.ReInit(N);

          Vector<ValueType> nds_(nds.Dim(), nds.begin(), false);
          Vector<ValueType> wts_(wts.Dim(), wts.begin(), false);
          while (levels>1) {
            Vector<ValueType> nds0(LegQuadOrder, nds_.begin(), false);
            Vector<ValueType> wts0(LegQuadOrder, wts_.begin(), false);
            Vector<ValueType> nds1(nds_.Dim()-LegQuadOrder, nds_.begin()+LegQuadOrder, false);
            Vector<ValueType> wts1(wts_.Dim()-LegQuadOrder, wts_.begin()+LegQuadOrder, false);

            ValueType end_point = (a+b)/2;
            nds0 = leg_nds * (a-end_point) + end_point;
            wts0 = leg_wts * fabs<ValueType>(a-end_point);

            nds_.Swap(nds1);
            wts_.Swap(wts1);
            a = end_point;
            levels--;
          }
          nds_ = log_quad_nds * (a-b) + b;
          wts_ = log_quad_wts * fabs<ValueType>(a-b);
        };
        adap_ref(quad_nds, quad_wts, (ValueType)1, (ValueType)0, RefLevels); // adaptive quadrature rule
        SCTL_ASSERT(quad_nds.Dim() <= Nbuff);

        Matrix<ValueType> Minterp_quad_nds;
        { // Set Minterp_quad_nds
          Minterp_quad_nds.ReInit(Ncheb, quad_nds.Dim());
          Vector<ValueType> Vinterp_quad_nds(Ncheb*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
          LagrangeInterp<ValueType>::Interpolate(Vinterp_quad_nds, ChebQuadRule<ValueType>::template nds<Ncheb>(), quad_nds);
        }

        Matrix<ValueType> r_src, dr_src, x_src, dx_src, d2x_src, e1_src, e2_src, de1_src, de2_src;
        r_src  .ReInit(        1,quad_nds.Dim());
        dr_src .ReInit(        1,quad_nds.Dim());
        x_src  .ReInit(COORD_DIM,quad_nds.Dim());
        dx_src .ReInit(COORD_DIM,quad_nds.Dim());
        d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
        e1_src .ReInit(COORD_DIM,quad_nds.Dim());
        e2_src .ReInit(COORD_DIM,quad_nds.Dim());
        de1_src.ReInit(COORD_DIM,quad_nds.Dim());
        de2_src.ReInit(COORD_DIM,quad_nds.Dim());
        Matrix<ValueType>::GEMM(  x_src, Matrix<ValueType>(COORD_DIM,Ncheb,(Iterator<ValueType>)(ConstIterator<ValueType>) coord,false), Minterp_quad_nds);
        Matrix<ValueType>::GEMM( dx_src, Matrix<ValueType>(COORD_DIM,Ncheb,(Iterator<ValueType>)(ConstIterator<ValueType>)    dx,false), Minterp_quad_nds);
        Matrix<ValueType>::GEMM(d2x_src, Matrix<ValueType>(COORD_DIM,Ncheb,(Iterator<ValueType>)(ConstIterator<ValueType>)   d2x,false), Minterp_quad_nds);
        Matrix<ValueType>::GEMM(  r_src, Matrix<ValueType>(        1,Ncheb,(Iterator<ValueType>)(ConstIterator<ValueType>)radius,false), Minterp_quad_nds);
        Matrix<ValueType>::GEMM( dr_src, Matrix<ValueType>(        1,Ncheb,(Iterator<ValueType>)(ConstIterator<ValueType>)    dr,false), Minterp_quad_nds);
        Matrix<ValueType>::GEMM( e1_src, Matrix<ValueType>(COORD_DIM,Ncheb,(Iterator<ValueType>)(ConstIterator<ValueType>)    e1,false), Minterp_quad_nds);
        for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
          Vec3 e1, dx, d2x;
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1(k,0) = e1_src[k][j];
            dx(k,0) = dx_src[k][j];
            d2x(k,0) = d2x_src[k][j];
          }
          ValueType inv_dx2 = 1/dot_prod(dx,dx);
          e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
          e1 = e1 * (1/sqrt<ValueType>(dot_prod(e1,e1)));

          Vec3 e2 = cross_prod(e1, dx);
          e2 = e2 * (1/sqrt<ValueType>(dot_prod(e2,e2)));
          Vec3 de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
          Vec3 de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
          for (Integer k = 0; k < COORD_DIM; k++) {
            e1_src[k][j] = e1(k,0);
            e2_src[k][j] = e2(k,0);
            de1_src[k][j] = de1(k,0);
            de2_src[k][j] = de2(k,0);
          }
        }

        StaticArray<ValueType,Nbuff*FourierModes*2*KDIM0*KDIM1> mem_buff0;
        Matrix<ValueType> M_tor(quad_nds.Dim(), FourierModes*2*KDIM0 * KDIM1, mem_buff0, false);
        auto toroidal_greens_fn_batched = [&ker](Matrix<ValueType>& M, const Vec3& y_trg, const Matrix<ValueType>& x_src, const Matrix<ValueType>& dx_src, const Matrix<ValueType>& d2x_src, const Matrix<ValueType>& r_src, const Matrix<ValueType>& dr_src, const Matrix<ValueType>& e1_src, const Matrix<ValueType>& e2_src, const Matrix<ValueType>& de1_src, const Matrix<ValueType>& de2_src){
          const Long BatchSize = M.Dim(0);
          SCTL_ASSERT(  x_src.Dim(1) == BatchSize);
          SCTL_ASSERT( dx_src.Dim(1) == BatchSize);
          SCTL_ASSERT(d2x_src.Dim(1) == BatchSize);
          SCTL_ASSERT(  r_src.Dim(1) == BatchSize);
          SCTL_ASSERT( dr_src.Dim(1) == BatchSize);
          SCTL_ASSERT( e1_src.Dim(1) == BatchSize);
          SCTL_ASSERT( e2_src.Dim(1) == BatchSize);
          SCTL_ASSERT(de1_src.Dim(1) == BatchSize);
          SCTL_ASSERT(de2_src.Dim(1) == BatchSize);
          SCTL_ASSERT(M.Dim(1) == FourierModes*2*KDIM0 * KDIM1);
          for (Long ii = 0; ii < BatchSize; ii++) {
            ValueType r = r_src[0][ii], dr = dr_src[0][ii];
            Vec3 x, dx, d2x, e1, e2, de1, de2;
            { // Set x, dx, d2x, e1, e2, de1, de2
              for (Integer k = 0; k < COORD_DIM; k++) {
                x  (k,0) =   x_src[k][ii];
                dx (k,0) =  dx_src[k][ii];
                d2x(k,0) = d2x_src[k][ii];
                e1 (k,0) =  e1_src[k][ii];
                e2 (k,0) =  e2_src[k][ii];
                de1(k,0) = de1_src[k][ii];
                de2(k,0) = de2_src[k][ii];
              }
            }

            auto toroidal_greens_fn = [&ker](Matrix<ValueType>& M, const Vec3& Xt, const Vec3& x, const Vec3& dx, const Vec3& d2x, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const ValueType r, const ValueType dr) {
              SCTL_ASSERT(M.Dim(0) == KDIM0*FourierModes*2);
              SCTL_ASSERT(M.Dim(1) == KDIM1);

              StaticArray<ValueType,3*Nbuff> mem_buff0;
              Vector<ValueType> wts, sin_nds, cos_nds;
              ToroidalSpecialQuadRule<ValueType,FourierModes+1>(sin_nds, cos_nds, wts, mem_buff0, 3*Nbuff, Xt-x, e1, e2, cross_prod(e1,e2), r, digits);
              const Long Nnds = wts.Dim();

              StaticArray<ValueType,(COORD_DIM*2+1)*Nbuff> mem_buff1;
              Vector<ValueType> y(Nnds*COORD_DIM, mem_buff1+0*COORD_DIM*Nbuff, false);
              Vector<ValueType> n(Nnds*COORD_DIM, mem_buff1+1*COORD_DIM*Nbuff, false);
              Vector<ValueType> da(         Nnds, mem_buff1+2*COORD_DIM*Nbuff, false);
              for (Integer j = 0; j < Nnds; j++) { // Set x, n, da
                ValueType sint = sin_nds[j];
                ValueType cost = cos_nds[j];

                Vec3 dy_ds = dx + e1*(dr*cost) + e2*(dr*sint) + de1*(r*cost) + de2*(r*sint);
                Vec3 dy_dt = e1*(-r*sint) + e2*(r*cost);

                Vec3 y_ = x + e1*(r*cost) + e2*(r*sint);
                Vec3 n_ = cross_prod(dy_ds, dy_dt);
                ValueType da_ = sqrt<ValueType>(dot_prod(n_,n_));
                n_ = n_ * (1/da_);

                for (Integer k = 0; k < COORD_DIM; k++) {
                  y[j*COORD_DIM+k] = y_(k,0);
                  n[j*COORD_DIM+k] = n_(k,0);
                }
                da[j] = da_;
              }

              StaticArray<ValueType,KDIM0*KDIM1*Nbuff> mem_buff2;
              Matrix<ValueType> Mker(KDIM0*Nnds, KDIM1, mem_buff2, false);
              ker.KernelMatrix(Mker, Vector<ValueType>(COORD_DIM,(Iterator<ValueType>)Xt.begin(),false), y, n);

              StaticArray<ValueType,4*Nbuff> mem_buff3;
              Vector<Complex<ValueType>> exp_itheta(Nnds, (Iterator<Complex<ValueType>>)(mem_buff3+0*Nbuff), false);
              Vector<Complex<ValueType>> exp_iktheta_da(Nnds, (Iterator<Complex<ValueType>>)(mem_buff3+2*Nbuff), false);
              for (Integer j = 0; j < Nnds; j++) {
                exp_itheta[j].real = cos_nds[j];
                exp_itheta[j].imag = sin_nds[j];
                exp_iktheta_da[j].real = da[j] * wts[j];
                exp_iktheta_da[j].imag = 0;
              }
              for (Integer k = 0; k < FourierModes; k++) {
                Tensor<ValueType,true,KDIM0,KDIM1> Mk0, Mk1;
                for (Integer i0 = 0; i0 < KDIM0; i0++) {
                  for (Integer i1 = 0; i1 < KDIM1; i1++) {
                    Mk0(i0,i1) = 0;
                    Mk1(i0,i1) = 0;
                  }
                }
                for (Integer j = 0; j < Nnds; j++) {
                  Tensor<ValueType,false,KDIM0,KDIM1> Mker_(Mker[j*KDIM0]);
                  Mk0 = Mk0 + Mker_ * exp_iktheta_da[j].real;
                  Mk1 = Mk1 + Mker_ * exp_iktheta_da[j].imag;
                }
                for (Integer i0 = 0; i0 < KDIM0; i0++) {
                  for (Integer i1 = 0; i1 < KDIM1; i1++) {
                    M[i0*(FourierModes*2)+(k*2+0)][i1] = Mk0(i0,i1);
                    M[i0*(FourierModes*2)+(k*2+1)][i1] = Mk1(i0,i1);
                  }
                }
                exp_iktheta_da *= exp_itheta;
              }
            };
            Matrix<ValueType> M_toroidal_greens_fn(KDIM0*FourierModes*2, KDIM1, M[ii], false);
            toroidal_greens_fn(M_toroidal_greens_fn, y_trg, x, dx, d2x, e1, e2, de1, de2, r, dr);
          }
        };
        toroidal_greens_fn_batched(M_tor, y_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, e2_src, de1_src, de2_src);

        M.ReInit(quad_nds.Dim(), Ncheb*FourierModes*2*KDIM0*KDIM1);
        for (Long i = 0; i < quad_nds.Dim(); i++) {
          for (Long j = 0; j < Ncheb; j++) {
            for (Long k = 0; k < FourierModes*2*KDIM0*KDIM1; k++) {
              M[i][j*FourierModes*2*KDIM0*KDIM1+k] = Minterp_quad_nds[j][i] * M_tor[i][k];
            }
          }
        }
      }
      template <class ValueType, Integer Ncheb, Integer FourierModes> static Vector<Vector<ValueType>> BuildSpecialQuadRules(ValueType elem_length) {
        static constexpr Integer Nlen = 10; // number of length samples in [elem_length/sqrt(2), elem_length*sqrt(2)]
        static constexpr Integer max_digits = 19;
        Integer depth = (Integer)(log<ValueType>(elem_length)/log<ValueType>(2)+4);

        GenericKernel<Laplace3D_FxU> laplace_sl; // TODO
        GenericKernel<Laplace3D_DxU> laplace_dl; // TODO

        Vector<ValueType> nds, wts;
        Matrix<ValueType> Mintegrands;
        { // Set nds, wts, Mintegrands
          Vector<Matrix<ValueType>> Msl(Nlen), Mdl(Nlen);
          Vector<Vector<ValueType>> nds_(Nlen), wts_(Nlen);
          #pragma omp parallel for schedule(static)
          for (Long k = 0; k < Nlen; k++) {
            ValueType length = elem_length/sqrt<ValueType>(2.0)*k/(Nlen-1) + elem_length*sqrt<ValueType>(2.0)*(Nlen-k-1)/(Nlen-1);
            SpecialQuadBuildBasisMatrix<Ncheb,FourierModes,max_digits>(Msl[k], nds_[k], wts_[k], length, depth, laplace_sl);
            SpecialQuadBuildBasisMatrix<Ncheb,FourierModes,max_digits>(Mdl[k], nds_[k], wts_[k], length, depth, laplace_dl);
          }
          nds = nds_[0];
          wts = wts_[0];

          const Long N0 = nds.Dim();
          Vector<Long> cnt(Nlen*2), dsp(Nlen*2); dsp[0] = 0;
          for (Long k = 0; k < Nlen; k++) {
            SCTL_ASSERT(Msl[k].Dim(0) == N0);
            SCTL_ASSERT(Mdl[k].Dim(0) == N0);
            cnt[k*2+0] = Msl[k].Dim(1);
            cnt[k*2+1] = Mdl[k].Dim(1);
          }
          omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());

          Mintegrands.ReInit(N0, dsp[Nlen*2-1] + cnt[Nlen*2-1]);
          #pragma omp parallel for schedule(static)
          for (Long k = 0; k < Nlen; k++) {
            for (Long i = 0; i < N0; i++) {
              for (Long j = 0; j < cnt[k*2+0]; j++) {
                Mintegrands[i][dsp[k*2+0]+j] = Msl[k][i][j];
              }
            }
            for (Long i = 0; i < N0; i++) {
              for (Long j = 0; j < cnt[k*2+1]; j++) {
                Mintegrands[i][dsp[k*2+1]+j] = Mdl[k][i][j];
              }
            }
          }
        }

        Vector<Vector<ValueType>> nds_wts(max_digits*2);
        { // Set nds_wts
          Vector<ValueType> eps_vec;
          Vector<Vector<ValueType>> quad_nds, quad_wts;
          for (Long k = 0; k < max_digits; k++) eps_vec.PushBack(pow<ValueType,Long>(0.1,k));
          InterpQuadRule<ValueType>::Build(quad_nds, quad_wts,  Mintegrands, nds, wts, eps_vec);
          SCTL_ASSERT(quad_nds.Dim() == max_digits);
          SCTL_ASSERT(quad_wts.Dim() == max_digits);
          for (Long k = 0; k < max_digits; k++) {
            nds_wts[k*2+0] = quad_nds[k];
            nds_wts[k*2+1] = quad_wts[k];
          }
        }
        return nds_wts;
      }
      template <Integer digits, bool adap_quad = false> static void SpecialQuadRule(Vector<Real>& nds, Vector<Real>& wts, Real s, Real elem_radius, Real elem_length) {
        static constexpr Integer max_adap_depth = 11; // TODO
        static constexpr Integer max_digits = 19;

        static constexpr Integer LogSingularQuadOrder = 2*digits; // TODO: determine optimal order
        static constexpr Integer LegQuadOrder = 2*digits; // TODO: determine optimal order

        auto one_sided_rule = [](Real radius, Real length) -> std::pair<Vector<Real>,Vector<Real>> {
          auto load_special_quad_rule = [](){
            static const std::string fname = std::string(("special_quad_q")+std::to_string(ChebOrder));
            using ValueType = QuadReal;

            Vector<Vector<ValueType>> data;
            ReadFile(data, fname);
            if (data.Dim() != max_adap_depth*max_digits*2) { // build quadrature rules
              data.ReInit(max_adap_depth*max_digits*2);
              ValueType length = 60.0; // TODO
              for (Integer i = 0; i < max_adap_depth; i++) {
                auto nds_wts = BuildSpecialQuadRules<ValueType, ChebOrder, FourierOrder/2+1>(length);
                for (Long j = 0; j < max_digits; j++) {
                  data[(i*max_digits+j)*2+0] = nds_wts[j*2+0];
                  data[(i*max_digits+j)*2+1] = nds_wts[j*2+1];
                }
                length *= (ValueType)0.5;
              }
              WriteFile(data, fname);
            }

            Vector<std::pair<Vector<Real>,Vector<Real>>> nds_wts_lst(max_adap_depth);
            for (Long i = 0; i < max_adap_depth; i++) { // Set nds_wts_lst
              const auto& nds_ = data[(i*max_digits+digits)*2+0];
              const auto& wts_ = data[(i*max_digits+digits)*2+1];
              nds_wts_lst[i]. first.ReInit(nds_.Dim());
              nds_wts_lst[i].second.ReInit(wts_.Dim());
              for (Long j = 0; j < nds_.Dim(); j++) {
                nds_wts_lst[i]. first[j] = (Real)nds_[j];
                nds_wts_lst[i].second[j] = (Real)wts_[j];
              }
            }
            return nds_wts_lst;
          };
          const auto& leg_nds = LegendreQuadRule<Real, LegQuadOrder>().first;
          const auto& leg_wts = LegendreQuadRule<Real, LegQuadOrder>().second;
          const auto& log_sing_nds_wts = LogSingularityQuadRule<Real,LogSingularQuadOrder>();

          std::pair<Vector<Real>,Vector<Real>> nds_wts;
          if (adap_quad) {
            if (length < 0.8*radius) { // log-singular quadrature
              nds_wts = log_sing_nds_wts;
            } else { // adaptive with scale-dependent quadrature
              Real s = 1.0;
              while (length*s>0.8*radius) {
                s*=0.5;
                for (Long i = 0; i < leg_nds.Dim(); i++) {
                  nds_wts.first.PushBack(leg_nds[i]*s+s);
                  nds_wts.second.PushBack(leg_wts[i]*s);
                }
              }
              { // add rule for singular part
                const auto& sing_nds = log_sing_nds_wts.first;
                const auto& sing_wts = log_sing_nds_wts.second;
                for (Long i = 0; i < sing_nds.Dim(); i++) {
                  nds_wts.first.PushBack(sing_nds[i]*s);
                  nds_wts.second.PushBack(sing_wts[i]*s);
                }
              }
            }
          } else {
            static const Vector<std::pair<Vector<Real>,Vector<Real>>> nds_wts_lst = load_special_quad_rule();
            if (length < 0.8*radius) { // log-singular quadrature
              nds_wts = log_sing_nds_wts;
            } else if (length < 80*radius) { // scale-dependent quadrature
              Long quad_idx = 0;
              { // Set quad_idx
                Real min_dist = 1e10;
                Real r = 0.0125, l = 1.0;
                for (Integer i = 0; i < nds_wts_lst.Dim(); i++) {
                  if (min_dist > fabs(r/l - radius/length)) {
                    min_dist = fabs(r/l - radius/length);
                    quad_idx = i;
                  }
                  l *= 0.5;
                }
              }
              nds_wts = nds_wts_lst[quad_idx];
            } else { // adaptive with scale-dependent quadrature
              Real s = 1.0;
              while (length*s>radius*80) {
                s*=0.5;
                for (Long i = 0; i < leg_nds.Dim(); i++) {
                  nds_wts.first.PushBack(leg_nds[i]*s+s);
                  nds_wts.second.PushBack(leg_wts[i]*s);
                }
              }
              { // add rule for singular part
                const auto& sing_nds = nds_wts_lst[0].first;
                const auto& sing_wts = nds_wts_lst[0].second;
                for (Long i = 0; i < sing_nds.Dim(); i++) {
                  nds_wts.first.PushBack(sing_nds[i]*s);
                  nds_wts.second.PushBack(sing_wts[i]*s);
                }
              }
            }
          }
          return nds_wts;
        };
        const auto nds_wts0 = one_sided_rule(elem_radius, elem_length*s);
        const auto nds_wts1 = one_sided_rule(elem_radius, elem_length*(1-s));
        const Long N0 = nds_wts0.first.Dim();
        const Long N1 = nds_wts1.first.Dim();

        nds.ReInit(N0 + N1);
        wts.ReInit(N0 + N1);
        Vector<Real> nds0(N0, nds.begin() + 0*N0, false);
        Vector<Real> wts0(N0, wts.begin() + 0*N0, false);
        Vector<Real> nds1(N1, nds.begin() + 1*N0, false);
        Vector<Real> wts1(N1, wts.begin() + 1*N0, false);
        nds0 = (nds_wts0.first*(-1)+1)*s;
        wts0 = (nds_wts0.second      )*s;
        nds1 = (nds_wts1.first )*(1-s)+s;
        wts1 = (nds_wts1.second)*(1-s);
      }



      void GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_ds, Vector<Real>* dX_dt, const Vector<Real>& s_param, const Vector<Real>& sin_theta_, const Vector<Real>& cos_theta_) const {
        const Long Nt = sin_theta_.Dim();
        const Long Ns = s_param.Dim();
        const Long N = Ns * Nt;

        if (X  && X ->Dim() != N*COORD_DIM) X ->ReInit(N*COORD_DIM);
        if (Xn && Xn->Dim() != N*COORD_DIM) Xn->ReInit(N*COORD_DIM);
        if (Xa && Xa->Dim() != N          ) Xa->ReInit(N);
        if (dX_ds && dX_ds->Dim() != N*COORD_DIM) dX_ds->ReInit(N*COORD_DIM);
        if (dX_dt && dX_dt->Dim() != N*COORD_DIM) dX_dt->ReInit(N*COORD_DIM);

        Matrix<Real> M_lagrange_interp;
        { // Set M_lagrange_interp
          M_lagrange_interp.ReInit(ChebOrder, Ns);
          Vector<Real> V_lagrange_interp(ChebOrder*Ns, M_lagrange_interp.begin(), false);
          LagrangeInterp<Real>::Interpolate(V_lagrange_interp, CenterlineNodes(), s_param);
        }

        Matrix<Real> r_, dr_, x_, dx_, d2x_, e1_;
        r_  .ReInit(        1,Ns);
        dr_ .ReInit(        1,Ns);
        x_  .ReInit(COORD_DIM,Ns);
        dx_ .ReInit(COORD_DIM,Ns);
        d2x_.ReInit(COORD_DIM,Ns);
        e1_ .ReInit(COORD_DIM,Ns);
        Matrix<Real>::GEMM(  x_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>) coord,false), M_lagrange_interp);
        Matrix<Real>::GEMM( dx_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dx,false), M_lagrange_interp);
        Matrix<Real>::GEMM(d2x_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)   d2x,false), M_lagrange_interp);
        Matrix<Real>::GEMM(  r_, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)radius,false), M_lagrange_interp);
        Matrix<Real>::GEMM( dr_, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dr,false), M_lagrange_interp);
        Matrix<Real>::GEMM( e1_, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    e1,false), M_lagrange_interp);
        auto compute_coord = [](Vec3& y, const Vec3& x, const Vec3& e1, const Vec3& e2, const Real r, const Real sint, const Real cost) {
          y = x + e1*(r*cost) + e2*(r*sint);
        };
        auto compute_normal_area_elem_tangents = [](Vec3& n, Real& da, Vec3& dy_ds, Vec3& dy_dt, const Vec3& dx, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const Real r, const Real dr, const Real sint, const Real cost) {
          dy_ds = dx + e1*(dr*cost) + e2*(dr*sint) + de1*(r*cost) + de2*(r*sint);
          dy_dt = e1*(-r*sint) + e2*(r*cost);

          n = cross_prod(dy_ds, dy_dt);
          da = sqrt<Real>(dot_prod(n,n));
          n = n * (1/da);
        };

        for (Long j = 0; j < Ns; j++) {
          Real r = r_[0][j], dr = dr_[0][j];
          Vec3 x, dx, d2x, e1, e2, de1, de2;
          for (Integer k = 0; k < COORD_DIM; k++) { // Set x, dx, d2x, e1
            x(k,0)  = x_[k][j];
            dx(k,0) = dx_[k][j];
            d2x(k,0) = d2x_[k][j];
            e1(k,0) = e1_[k][j];
          }
          { // Set e1 (orthonormalize), e2, de1, de2
            Real inv_dx2 = 1/dot_prod(dx,dx);
            e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
            e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

            e2 = cross_prod(e1, dx);
            e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
            de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
            de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
          }

          if (X) {
            for (Integer i = 0; i < Nt; i++) { // Set X
              Vec3 y;
              compute_coord(y, x, e1, e2, r, sin_theta_[i], cos_theta_[i]);
              for (Integer k = 0; k < COORD_DIM; k++) {
                (*X)[(j*Nt+i)*COORD_DIM+k] = y(k,0);
              }
            }
          }
          if (Xn || Xa) {
            for (Integer i = 0; i < Nt; i++) { // Set X, Xn, Xa, dX_ds, dX_dt
              Real da;
              Vec3 n, dx_ds, dx_dt;
              compute_normal_area_elem_tangents(n, da, dx_ds, dx_dt, dx, e1, e2, de1, de2, r, dr, sin_theta_[i], cos_theta_[i]);
              if (Xn) {
                for (Integer k = 0; k < COORD_DIM; k++) {
                  (*Xn)[(j*Nt+i)*COORD_DIM+k] = n(k,0);
                }
              }
              if (Xa) {
                (*Xa)[j*Nt+i] = da;
              }
              if (dX_ds) {
                for (Integer k = 0; k < COORD_DIM; k++) {
                  (*dX_ds)[(j*Nt+i)*COORD_DIM+k] = dx_ds(k,0);
                }
              }
              if (dX_dt) {
                for (Integer k = 0; k < COORD_DIM; k++) {
                  (*dX_dt)[(j*Nt+i)*COORD_DIM+k] = dx_dt(k,0);
                }
              }
            }
          }
        }
      }


      template <Integer digits, class Kernel> Matrix<Real> SelfInteracHelper_(const Kernel& ker) const { // constant radius
        static constexpr Integer KerScaleExp=-2; // for laplace double-layer // TODO: determine this automatically
        static ToroidalGreensFn<Real,FourierOrder> tor_greens_fn;
        { // Setup tor_greens_fn
          static bool first_time = true;
          #pragma omp critical
          if (first_time) {
            tor_greens_fn.Setup(ker,1.0);
            first_time = false;
          }
        }

        static constexpr Integer FourierModes = FourierOrder/2+1;
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Integer Nbuff = 10000; // TODO

        static const Real dtheta = 2*const_pi<Real>()/FourierOrder;
        static const Complex<Real> exp_dtheta(cos<Real>(dtheta), sin<Real>(dtheta));

        Matrix<Real> Mt(KDIM1*ChebOrder*FourierOrder, KDIM0*ChebOrder*FourierModes*2);
        for (Long i = 0; i < ChebOrder; i++) {
          Real r_trg = radius[i];
          Real s_trg = CenterlineNodes()[i];
          Vec3 x_trg, dx_trg, e1_trg, e2_trg;
          { // Set x_trg, e1_trg, e2_trg
            for (Integer k = 0; k < COORD_DIM; k++) {
              x_trg (k,0) = coord[k*ChebOrder+i];
              e1_trg(k,0) = e1[k*ChebOrder+i];
              dx_trg(k,0) = dx[k*ChebOrder+i];
            }
            e2_trg = cross_prod(e1_trg, dx_trg);
            e2_trg = e2_trg * (1/sqrt<Real>(dot_prod(e2_trg,e2_trg)));
          }

          Vector<Real> quad_nds, quad_wts; // Quadrature rule in s
          SpecialQuadRule<digits>(quad_nds, quad_wts, s_trg, r_trg, sqrt<Real>(dot_prod(dx_trg, dx_trg)));
          SCTL_ASSERT(quad_nds.Dim() <= Nbuff);

          Matrix<Real> Minterp_quad_nds;
          { // Set Minterp_quad_nds
            Minterp_quad_nds.ReInit(ChebOrder, quad_nds.Dim());
            Vector<Real> Vinterp_quad_nds(ChebOrder*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
            LagrangeInterp<Real>::Interpolate(Vinterp_quad_nds, CenterlineNodes(), quad_nds);
          }

          Matrix<Real> r_src, dr_src, x_src, dx_src, d2x_src, e1_src, e2_src, de1_src, de2_src;
          r_src  .ReInit(        1,quad_nds.Dim());
          dr_src .ReInit(        1,quad_nds.Dim());
          x_src  .ReInit(COORD_DIM,quad_nds.Dim());
          dx_src .ReInit(COORD_DIM,quad_nds.Dim());
          d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
          e1_src .ReInit(COORD_DIM,quad_nds.Dim());
          e2_src .ReInit(COORD_DIM,quad_nds.Dim());
          de1_src.ReInit(COORD_DIM,quad_nds.Dim());
          de2_src.ReInit(COORD_DIM,quad_nds.Dim());
          { // Set x_src, x_trg (improve numerical stability)
            Matrix<Real> x_nodes(COORD_DIM,ChebOrder, (Iterator<Real>)(ConstIterator<Real>)coord, true);
            for (Long j = 0; j < ChebOrder; j++) {
              for (Integer k = 0; k < COORD_DIM; k++) {
                x_nodes[k][j] -= x_trg(k,0);
              }
            }
            Matrix<Real>::GEMM(  x_src, x_nodes, Minterp_quad_nds);
            for (Integer k = 0; k < COORD_DIM; k++) {
              x_trg(k,0) = 0;
            }
          }
          //Matrix<Real>::GEMM(  x_src, Matrix<Real>(COORD_DIM,ChebOrder, coord,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( dx_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dx,false), Minterp_quad_nds);
          Matrix<Real>::GEMM(d2x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)   d2x,false), Minterp_quad_nds);
          Matrix<Real>::GEMM(  r_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)radius,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( dr_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dr,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( e1_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    e1,false), Minterp_quad_nds);
          for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
            Vec3 e1, dx, d2x;
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1(k,0) = e1_src[k][j];
              dx(k,0) = dx_src[k][j];
              d2x(k,0) = d2x_src[k][j];
            }
            Real inv_dx2 = 1/dot_prod(dx,dx);
            e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
            e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

            Vec3 e2 = cross_prod(e1, dx);
            e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
            Vec3 de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
            Vec3 de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1_src[k][j] = e1(k,0);
              e2_src[k][j] = e2(k,0);
              de1_src[k][j] = de1(k,0);
              de2_src[k][j] = de2(k,0);
            }
          }

          Complex<Real> exp_theta_trg(1,0);
          for (Long j = 0; j < FourierOrder; j++) {
            const Vec3 y_trg = x_trg + e1_trg*r_trg*exp_theta_trg.real + e2_trg*r_trg*exp_theta_trg.imag;

            StaticArray<Real,Nbuff*FourierModes*2*KDIM0*KDIM1> mem_buff0;
            Matrix<Real> M_tor(quad_nds.Dim(), FourierModes*2*KDIM0 * KDIM1, mem_buff0, false);
            auto toroidal_greens_fn_batched = [this,&ker](Matrix<Real>& M, const Vec3& y_trg, const Matrix<Real>& x_src, const Matrix<Real>& dx_src, const Matrix<Real>& d2x_src, const Matrix<Real>& r_src, const Matrix<Real>& dr_src, const Matrix<Real>& e1_src, const Matrix<Real>& e2_src, const Matrix<Real>& de1_src, const Matrix<Real>& de2_src){
              const Long BatchSize = M.Dim(0);
              SCTL_ASSERT(  x_src.Dim(1) == BatchSize);
              SCTL_ASSERT( dx_src.Dim(1) == BatchSize);
              SCTL_ASSERT(d2x_src.Dim(1) == BatchSize);
              SCTL_ASSERT(  r_src.Dim(1) == BatchSize);
              SCTL_ASSERT( dr_src.Dim(1) == BatchSize);
              SCTL_ASSERT( e1_src.Dim(1) == BatchSize);
              SCTL_ASSERT( e2_src.Dim(1) == BatchSize);
              SCTL_ASSERT(de1_src.Dim(1) == BatchSize);
              SCTL_ASSERT(de2_src.Dim(1) == BatchSize);
              SCTL_ASSERT(M.Dim(1) == FourierModes*2*KDIM0 * KDIM1);
              for (Long ii = 0; ii < BatchSize; ii++) {
                Real r = r_src[0][ii]; //, dr = dr_src[0][ii];
                Vec3 x, dx, d2x, e1, e2, de1, de2;
                { // Set x, dx, d2x, e1, e2, de1, de2
                  for (Integer k = 0; k < COORD_DIM; k++) {
                    x  (k,0) =   x_src[k][ii];
                    dx (k,0) =  dx_src[k][ii];
                    d2x(k,0) = d2x_src[k][ii];
                    e1 (k,0) =  e1_src[k][ii];
                    e2 (k,0) =  e2_src[k][ii];
                    de1(k,0) = de1_src[k][ii];
                    de2(k,0) = de2_src[k][ii];
                  }
                }

                Matrix<Real> M_toroidal_greens_fn(KDIM0*FourierModes*2, KDIM1, M[ii], false);
                //auto toroidal_greens_fn = [this,&ker](Matrix<Real>& M, const Vec3& Xt, const Vec3& x, const Vec3& dx, const Vec3& d2x, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const Real r, const Real dr) {
                //  SCTL_ASSERT(M.Dim(0) == KDIM0*FourierModes*2);
                //  SCTL_ASSERT(M.Dim(1) == KDIM1);

                //  StaticArray<Real,3*Nbuff> mem_buff0;
                //  Vector<Real> wts, sin_nds, cos_nds;
                //  ToroidalSpecialQuadRule<Real,FourierModes+1>(sin_nds, cos_nds, wts, mem_buff0, 3*Nbuff, Xt-x, e1, e2, cross_prod(e1,e2), r, digits);
                //  const Long Nnds = wts.Dim();

                //  StaticArray<Real,(COORD_DIM*2+1)*Nbuff> mem_buff1;
                //  Vector<Real> y(Nnds*COORD_DIM, mem_buff1+0*COORD_DIM*Nbuff, false);
                //  Vector<Real> n(Nnds*COORD_DIM, mem_buff1+1*COORD_DIM*Nbuff, false);
                //  Vector<Real> da(         Nnds, mem_buff1+2*COORD_DIM*Nbuff, false);
                //  for (Integer j = 0; j < Nnds; j++) { // Set x, n, da
                //    Real sint = sin_nds[j];
                //    Real cost = cos_nds[j];

                //    Vec3 dy_ds = dx + e1*(dr*cost) + e2*(dr*sint) + de1*(r*cost) + de2*(r*sint);
                //    Vec3 dy_dt = e1*(-r*sint) + e2*(r*cost);

                //    Vec3 y_ = x + e1*(r*cost) + e2*(r*sint);
                //    Vec3 n_ = cross_prod(dy_ds, dy_dt);
                //    Real da_ = sqrt<Real>(dot_prod(n_,n_));
                //    n_ = n_ * (1/da_);

                //    for (Integer k = 0; k < COORD_DIM; k++) {
                //      y[j*COORD_DIM+k] = y_(k,0);
                //      n[j*COORD_DIM+k] = n_(k,0);
                //    }
                //    da[j] = da_;
                //  }

                //  StaticArray<Real,KDIM0*KDIM1*Nbuff> mem_buff2;
                //  Matrix<Real> Mker(KDIM0*Nnds, KDIM1, mem_buff2, false);
                //  ker.KernelMatrix(Mker, Vector<Real>(COORD_DIM,(Iterator<Real>)Xt.begin(),false), y, n);

                //  StaticArray<Real,4*Nbuff> mem_buff3;
                //  Vector<Complex<Real>> exp_itheta(Nnds, (Iterator<Complex<Real>>)(mem_buff3+0*Nbuff), false);
                //  Vector<Complex<Real>> exp_iktheta_wts(Nnds, (Iterator<Complex<Real>>)(mem_buff3+2*Nbuff), false);
                //  for (Integer j = 0; j < Nnds; j++) {
                //    exp_itheta[j].real = cos_nds[j];
                //    exp_itheta[j].imag = sin_nds[j];
                //    exp_iktheta_wts[j].real = wts[j];
                //    exp_iktheta_wts[j].imag = 0;
                //  }
                //  for (Integer k = 0; k < FourierModes; k++) {
                //    Tensor<Real,true,KDIM0,KDIM1> Mk0, Mk1;
                //    for (Integer i0 = 0; i0 < KDIM0; i0++) {
                //      for (Integer i1 = 0; i1 < KDIM1; i1++) {
                //        Mk0(i0,i1) = 0;
                //        Mk1(i0,i1) = 0;
                //      }
                //    }
                //    for (Integer j = 0; j < Nnds; j++) {
                //      Tensor<Real,false,KDIM0,KDIM1> Mker_(Mker[j*KDIM0]);
                //      Mk0 = Mk0 + Mker_ * exp_iktheta_wts[j].real;
                //      Mk1 = Mk1 + Mker_ * exp_iktheta_wts[j].imag;
                //    }
                //    for (Integer i0 = 0; i0 < KDIM0; i0++) {
                //      for (Integer i1 = 0; i1 < KDIM1; i1++) {
                //        M[i0*(FourierModes*2)+(k*2+0)][i1] = Mk0(i0,i1);
                //        M[i0*(FourierModes*2)+(k*2+1)][i1] = Mk1(i0,i1);
                //      }
                //    }
                //    exp_iktheta_wts *= exp_itheta;
                //  }
                //};
                //toroidal_greens_fn(M_toroidal_greens_fn, y_trg, x, dx, d2x, e1, e2, de1, de2, r, dr);

                tor_greens_fn.BuildOperatorModal(M_toroidal_greens_fn, dot_prod(y_trg-x,e1)/r, dot_prod(y_trg-x,e2)/r, dot_prod(y_trg-x,cross_prod(e1,e2))/r, ker);
                { // Scale M_toroidal_greens_fn
                  Real scale = sqrt(2.0) * sctl::pow<KerScaleExp>(r);
                  for (Long i = 0; i < KDIM0; i++) {
                    for (Long k = 0; k < FourierModes; k++) {
                      for (Long j = 0; j < KDIM1; j++) {
                        M_toroidal_greens_fn[i*FourierModes*2+k*2+0][j] *= scale;
                        M_toroidal_greens_fn[i*FourierModes*2+k*2+1][j] *=-scale;
                      }
                    }
                  }
                  for (Long i = 0; i < KDIM0; i++) {
                    for (Long j = 0; j < KDIM1; j++) {
                      M_toroidal_greens_fn[i*FourierModes*2+0][j] *= 2;
                      M_toroidal_greens_fn[i*FourierModes*2+1][j] *= 2;
                      if (FourierOrder%2 == 0) {
                        M_toroidal_greens_fn[(i+1)*FourierModes*2-2][j] *= 2;
                        M_toroidal_greens_fn[(i+1)*FourierModes*2-1][j] *= 2;
                      }
                    }
                  }
                }
              }
            };
            toroidal_greens_fn_batched(M_tor, y_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, e2_src, de1_src, de2_src);

            StaticArray<Real,ChebOrder*FourierModes*2*KDIM0*KDIM1> mem_buff1;
            Matrix<Real> M_(ChebOrder, FourierModes*2*KDIM0 * KDIM1, mem_buff1, false);
            for (Long ii = 0; ii < M_tor.Dim(0); ii++) {
              Matrix<Real> M_tor_(M_tor.Dim(1), KDIM1, M_tor[ii], false);
              M_tor_ *= quad_wts[ii];
            }
            Matrix<Real>::GEMM(M_, Minterp_quad_nds, M_tor);

            for (Long ii = 0; ii < ChebOrder*FourierModes*2; ii++) { // Mt <-- M_
              for (Long k0 = 0; k0 < KDIM0; k0++) {
                for (Long k1 = 0; k1 < KDIM1; k1++) {
                  Mt[(k1*ChebOrder+i)*FourierOrder+j][k0*ChebOrder*FourierModes*2+ii] = M_[0][(ii*KDIM0+k0)*KDIM1+k1];
                }
              }
            }
            exp_theta_trg *= exp_dtheta;
          }
        }

        Matrix<Real> Mt_(KDIM1*ChebOrder*FourierOrder, KDIM0*ChebOrder*FourierOrder);
        { // Set Mt_
          static const Matrix<Real> M_fourier_inv = fourier_matrix_inv<FourierOrder,FourierModes>().Transpose();
          const Matrix<Real> M_modal(KDIM1*ChebOrder*FourierOrder*KDIM0*ChebOrder, FourierModes*2, Mt.begin(), false);
          Matrix<Real> M_nodal(KDIM1*ChebOrder*FourierOrder*KDIM0*ChebOrder, FourierOrder, Mt_.begin(), false);
          Matrix<Real>::GEMM(M_nodal, M_modal, M_fourier_inv);
        }
        { // Mt_ <-- Mt_ * Xa
          Vector<Real> Xa;
          GetGeom(nullptr, nullptr, &Xa, nullptr, nullptr, CenterlineNodes(), sin_theta<FourierOrder>(), cos_theta<FourierOrder>());
          SCTL_ASSERT(Xa.Dim() == ChebOrder*FourierOrder);
          for (Long k = 0; k < KDIM1*ChebOrder*FourierOrder; k++) {
            for (Long i = 0; i < KDIM0; i++) {
              for (Long j = 0; j < ChebOrder*FourierOrder; j++) {
                Mt_[k][i*ChebOrder*FourierOrder+j] *= Xa[j];
              }
            }
          }
        }
        return Mt_.Transpose();
      }
      template <Integer digits, class Kernel> Matrix<Real> SelfInteracHelper(const Kernel& ker) const {
        static constexpr Integer FourierModes = FourierOrder/2+1;
        static constexpr Integer KDIM0 = Kernel::SrcDim();
        static constexpr Integer KDIM1 = Kernel::TrgDim();
        static constexpr Integer Nbuff = 10000; // TODO

        static const Real dtheta = 2*const_pi<Real>()/FourierOrder;
        static const Complex<Real> exp_dtheta(cos<Real>(dtheta), sin<Real>(dtheta));

        Matrix<Real> Mt(KDIM1*ChebOrder*FourierOrder, KDIM0*ChebOrder*FourierModes*2);
        for (Long i = 0; i < ChebOrder; i++) {
          Real r_trg = radius[i];
          Real s_trg = CenterlineNodes()[i];
          Vec3 x_trg, dx_trg, e1_trg, e2_trg;
          { // Set x_trg, e1_trg, e2_trg
            for (Integer k = 0; k < COORD_DIM; k++) {
              x_trg (k,0) = coord[k*ChebOrder+i];
              e1_trg(k,0) = e1[k*ChebOrder+i];
              dx_trg(k,0) = dx[k*ChebOrder+i];
            }
            e2_trg = cross_prod(e1_trg, dx_trg);
            e2_trg = e2_trg * (1/sqrt<Real>(dot_prod(e2_trg,e2_trg)));
          }

          Vector<Real> quad_nds, quad_wts; // Quadrature rule in s
          SpecialQuadRule<digits>(quad_nds, quad_wts, s_trg, r_trg, sqrt<Real>(dot_prod(dx_trg, dx_trg)));
          SCTL_ASSERT(quad_nds.Dim() <= Nbuff);

          Matrix<Real> Minterp_quad_nds;
          { // Set Minterp_quad_nds
            Minterp_quad_nds.ReInit(ChebOrder, quad_nds.Dim());
            Vector<Real> Vinterp_quad_nds(ChebOrder*quad_nds.Dim(), Minterp_quad_nds.begin(), false);
            LagrangeInterp<Real>::Interpolate(Vinterp_quad_nds, CenterlineNodes(), quad_nds);
          }

          Matrix<Real> r_src, dr_src, x_src, dx_src, d2x_src, e1_src, e2_src, de1_src, de2_src;
          r_src  .ReInit(        1,quad_nds.Dim());
          dr_src .ReInit(        1,quad_nds.Dim());
          x_src  .ReInit(COORD_DIM,quad_nds.Dim());
          dx_src .ReInit(COORD_DIM,quad_nds.Dim());
          d2x_src.ReInit(COORD_DIM,quad_nds.Dim());
          e1_src .ReInit(COORD_DIM,quad_nds.Dim());
          e2_src .ReInit(COORD_DIM,quad_nds.Dim());
          de1_src.ReInit(COORD_DIM,quad_nds.Dim());
          de2_src.ReInit(COORD_DIM,quad_nds.Dim());
          { // Set x_src, x_trg (improve numerical stability)
            Matrix<Real> x_nodes(COORD_DIM,ChebOrder, (Iterator<Real>)(ConstIterator<Real>)coord, true);
            for (Long j = 0; j < ChebOrder; j++) {
              for (Integer k = 0; k < COORD_DIM; k++) {
                x_nodes[k][j] -= x_trg(k,0);
              }
            }
            Matrix<Real>::GEMM(  x_src, x_nodes, Minterp_quad_nds);
            for (Integer k = 0; k < COORD_DIM; k++) {
              x_trg(k,0) = 0;
            }
          }
          //Matrix<Real>::GEMM(  x_src, Matrix<Real>(COORD_DIM,ChebOrder, coord,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( dx_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dx,false), Minterp_quad_nds);
          Matrix<Real>::GEMM(d2x_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)   d2x,false), Minterp_quad_nds);
          Matrix<Real>::GEMM(  r_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)radius,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( dr_src, Matrix<Real>(        1,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    dr,false), Minterp_quad_nds);
          Matrix<Real>::GEMM( e1_src, Matrix<Real>(COORD_DIM,ChebOrder,(Iterator<Real>)(ConstIterator<Real>)    e1,false), Minterp_quad_nds);
          for (Long j = 0; j < quad_nds.Dim(); j++) { // Set e2_src
            Vec3 e1, dx, d2x;
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1(k,0) = e1_src[k][j];
              dx(k,0) = dx_src[k][j];
              d2x(k,0) = d2x_src[k][j];
            }
            Real inv_dx2 = 1/dot_prod(dx,dx);
            e1 = e1 - dx * dot_prod(e1, dx) * inv_dx2;
            e1 = e1 * (1/sqrt<Real>(dot_prod(e1,e1)));

            Vec3 e2 = cross_prod(e1, dx);
            e2 = e2 * (1/sqrt<Real>(dot_prod(e2,e2)));
            Vec3 de1 = dx*(-dot_prod(e1,d2x) * inv_dx2);
            Vec3 de2 = dx*(-dot_prod(e2,d2x) * inv_dx2);
            for (Integer k = 0; k < COORD_DIM; k++) {
              e1_src[k][j] = e1(k,0);
              e2_src[k][j] = e2(k,0);
              de1_src[k][j] = de1(k,0);
              de2_src[k][j] = de2(k,0);
            }
          }

          Complex<Real> exp_theta_trg(1,0);
          for (Long j = 0; j < FourierOrder; j++) {
            const Vec3 y_trg = x_trg + e1_trg*r_trg*exp_theta_trg.real + e2_trg*r_trg*exp_theta_trg.imag;

            StaticArray<Real,Nbuff*FourierModes*2*KDIM0*KDIM1> mem_buff0;
            Matrix<Real> M_tor(quad_nds.Dim(), FourierModes*2*KDIM0 * KDIM1, mem_buff0, false);
            auto toroidal_greens_fn_batched = [this,&ker](Matrix<Real>& M, const Vec3& y_trg, const Matrix<Real>& x_src, const Matrix<Real>& dx_src, const Matrix<Real>& d2x_src, const Matrix<Real>& r_src, const Matrix<Real>& dr_src, const Matrix<Real>& e1_src, const Matrix<Real>& e2_src, const Matrix<Real>& de1_src, const Matrix<Real>& de2_src){
              const Long BatchSize = M.Dim(0);
              SCTL_ASSERT(  x_src.Dim(1) == BatchSize);
              SCTL_ASSERT( dx_src.Dim(1) == BatchSize);
              SCTL_ASSERT(d2x_src.Dim(1) == BatchSize);
              SCTL_ASSERT(  r_src.Dim(1) == BatchSize);
              SCTL_ASSERT( dr_src.Dim(1) == BatchSize);
              SCTL_ASSERT( e1_src.Dim(1) == BatchSize);
              SCTL_ASSERT( e2_src.Dim(1) == BatchSize);
              SCTL_ASSERT(de1_src.Dim(1) == BatchSize);
              SCTL_ASSERT(de2_src.Dim(1) == BatchSize);
              SCTL_ASSERT(M.Dim(1) == FourierModes*2*KDIM0 * KDIM1);
              for (Long ii = 0; ii < BatchSize; ii++) {
                Real r = r_src[0][ii], dr = dr_src[0][ii];
                Vec3 x, dx, d2x, e1, e2, de1, de2;
                { // Set x, dx, d2x, e1, e2, de1, de2
                  for (Integer k = 0; k < COORD_DIM; k++) {
                    x  (k,0) =   x_src[k][ii];
                    dx (k,0) =  dx_src[k][ii];
                    d2x(k,0) = d2x_src[k][ii];
                    e1 (k,0) =  e1_src[k][ii];
                    e2 (k,0) =  e2_src[k][ii];
                    de1(k,0) = de1_src[k][ii];
                    de2(k,0) = de2_src[k][ii];
                  }
                }

                auto toroidal_greens_fn = [this,&ker](Matrix<Real>& M, const Vec3& Xt, const Vec3& x, const Vec3& dx, const Vec3& d2x, const Vec3& e1, const Vec3& e2, const Vec3& de1, const Vec3& de2, const Real r, const Real dr) {
                  SCTL_ASSERT(M.Dim(0) == KDIM0*FourierModes*2);
                  SCTL_ASSERT(M.Dim(1) == KDIM1);

                  StaticArray<Real,3*Nbuff> mem_buff0;
                  Vector<Real> wts, sin_nds, cos_nds;
                  ToroidalSpecialQuadRule<Real,FourierModes+1>(sin_nds, cos_nds, wts, mem_buff0, 3*Nbuff, Xt-x, e1, e2, cross_prod(e1,e2), r, digits);
                  const Long Nnds = wts.Dim();

                  StaticArray<Real,(COORD_DIM*2+1)*Nbuff> mem_buff1;
                  Vector<Real> y(Nnds*COORD_DIM, mem_buff1+0*COORD_DIM*Nbuff, false);
                  Vector<Real> n(Nnds*COORD_DIM, mem_buff1+1*COORD_DIM*Nbuff, false);
                  Vector<Real> da(         Nnds, mem_buff1+2*COORD_DIM*Nbuff, false);
                  for (Integer j = 0; j < Nnds; j++) { // Set x, n, da
                    Real sint = sin_nds[j];
                    Real cost = cos_nds[j];

                    Vec3 dy_ds = dx + e1*(dr*cost) + e2*(dr*sint) + de1*(r*cost) + de2*(r*sint);
                    Vec3 dy_dt = e1*(-r*sint) + e2*(r*cost);

                    Vec3 y_ = x + e1*(r*cost) + e2*(r*sint);
                    Vec3 n_ = cross_prod(dy_ds, dy_dt);
                    Real da_ = sqrt<Real>(dot_prod(n_,n_));
                    n_ = n_ * (1/da_);

                    for (Integer k = 0; k < COORD_DIM; k++) {
                      y[j*COORD_DIM+k] = y_(k,0);
                      n[j*COORD_DIM+k] = n_(k,0);
                    }
                    da[j] = da_;
                  }

                  StaticArray<Real,KDIM0*KDIM1*Nbuff> mem_buff2;
                  Matrix<Real> Mker(KDIM0*Nnds, KDIM1, mem_buff2, false);
                  ker.KernelMatrix(Mker, Vector<Real>(COORD_DIM,(Iterator<Real>)Xt.begin(),false), y, n);

                  StaticArray<Real,4*Nbuff> mem_buff3;
                  Vector<Complex<Real>> exp_itheta(Nnds, (Iterator<Complex<Real>>)(mem_buff3+0*Nbuff), false);
                  Vector<Complex<Real>> exp_iktheta_da(Nnds, (Iterator<Complex<Real>>)(mem_buff3+2*Nbuff), false);
                  for (Integer j = 0; j < Nnds; j++) {
                    exp_itheta[j].real = cos_nds[j];
                    exp_itheta[j].imag = sin_nds[j];
                    exp_iktheta_da[j].real = da[j] * wts[j];
                    exp_iktheta_da[j].imag = 0;
                  }
                  for (Integer k = 0; k < FourierModes; k++) {
                    Tensor<Real,true,KDIM0,KDIM1> Mk0, Mk1;
                    for (Integer i0 = 0; i0 < KDIM0; i0++) {
                      for (Integer i1 = 0; i1 < KDIM1; i1++) {
                        Mk0(i0,i1) = 0;
                        Mk1(i0,i1) = 0;
                      }
                    }
                    for (Integer j = 0; j < Nnds; j++) {
                      Tensor<Real,false,KDIM0,KDIM1> Mker_(Mker[j*KDIM0]);
                      Mk0 = Mk0 + Mker_ * exp_iktheta_da[j].real;
                      Mk1 = Mk1 + Mker_ * exp_iktheta_da[j].imag;
                    }
                    for (Integer i0 = 0; i0 < KDIM0; i0++) {
                      for (Integer i1 = 0; i1 < KDIM1; i1++) {
                        M[i0*(FourierModes*2)+(k*2+0)][i1] = Mk0(i0,i1);
                        M[i0*(FourierModes*2)+(k*2+1)][i1] = Mk1(i0,i1);
                      }
                    }
                    exp_iktheta_da *= exp_itheta;
                  }
                };
                Matrix<Real> M_toroidal_greens_fn(KDIM0*FourierModes*2, KDIM1, M[ii], false);
                toroidal_greens_fn(M_toroidal_greens_fn, y_trg, x, dx, d2x, e1, e2, de1, de2, r, dr);
              }
            };
            toroidal_greens_fn_batched(M_tor, y_trg, x_src, dx_src, d2x_src, r_src, dr_src, e1_src, e2_src, de1_src, de2_src);

            StaticArray<Real,ChebOrder*FourierModes*2*KDIM0*KDIM1> mem_buff1;
            Matrix<Real> M_(ChebOrder, FourierModes*2*KDIM0 * KDIM1, mem_buff1, false);
            for (Long ii = 0; ii < M_tor.Dim(0); ii++) {
              Matrix<Real> M_tor_(M_tor.Dim(1), KDIM1, M_tor[ii], false);
              M_tor_ *= quad_wts[ii];
            }
            Matrix<Real>::GEMM(M_, Minterp_quad_nds, M_tor);

            for (Long ii = 0; ii < ChebOrder*FourierModes*2; ii++) { // Mt <-- M_
              for (Long k0 = 0; k0 < KDIM0; k0++) {
                for (Long k1 = 0; k1 < KDIM1; k1++) {
                  Mt[(k1*ChebOrder+i)*FourierOrder+j][k0*ChebOrder*FourierModes*2+ii] = M_[0][(ii*KDIM0+k0)*KDIM1+k1];
                }
              }
            }
            exp_theta_trg *= exp_dtheta;
          }
        }

        Matrix<Real> Mt_(KDIM1*ChebOrder*FourierOrder, KDIM0*ChebOrder*FourierOrder);
        { // Set Mt_
          static const Matrix<Real> M_fourier_inv = fourier_matrix_inv<FourierOrder,FourierModes>().Transpose();
          Matrix<Real> M_nodal(KDIM1*ChebOrder*FourierOrder*KDIM0*ChebOrder, FourierOrder, Mt_.begin(), false);
          Matrix<Real> M_modal(KDIM1*ChebOrder*FourierOrder*KDIM0*ChebOrder, FourierModes*2, Mt.begin(), false);
          Matrix<Real>::GEMM(M_nodal, M_modal, M_fourier_inv);
        }
        return Mt_.Transpose();
      }

      StaticArray<Real,ChebOrder> radius;
      StaticArray<Real,COORD_DIM*ChebOrder> coord;

      // dependent quantities
      StaticArray<Real,ChebOrder> dr;
      StaticArray<Real,COORD_DIM*ChebOrder> dx;
      StaticArray<Real,COORD_DIM*ChebOrder> d2x;
      StaticArray<Real,COORD_DIM*ChebOrder> e1;
  };

}

#endif //_SCTL_BOUNDARY_INTEGRAL_HPP_
