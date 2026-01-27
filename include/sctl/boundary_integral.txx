#ifndef _SCTL_BOUNDARY_INTEGRAL_TXX_
#define _SCTL_BOUNDARY_INTEGRAL_TXX_

#include <omp.h>                       // for omp_get_num_threads, omp_get_t...
#include <algorithm>                   // for lower_bound, max, min, upper_b...
#include <map>                         // for map
#include <set>                         // for set, __tree_const_iterator
#include <string>                      // for basic_string, string, to_string
#include <type_traits>                 // for is_copy_constructible
#include <typeinfo>                    // for type_info
#include <utility>                     // for pair, make_pair

#include "sctl/common.hpp"             // for Long, Integer, SCTL_ASSERT
#include "sctl/boundary_integral.hpp"  // for BoundaryIntegralOp, BuildNearList
#include "sctl/comm.hpp"               // for Comm, CommOp
#include "sctl/comm.txx"               // for Comm::Allreduce, Comm::Size
#include "sctl/iterator.hpp"           // for Iterator, ConstIterator
#include "sctl/iterator.txx"           // for Iterator::Iterator<ValueType>
#include "sctl/math_utils.hpp"         // for log, sqrt
#include "sctl/matrix.hpp"             // for Matrix
#include "sctl/morton.hpp"             // for Morton
#include "sctl/ompUtils.txx"           // for scan, merge_sort
#include "sctl/profile.hpp"            // for Profile
#include "sctl/profile.txx"            // for Profile::Tic, Profile::Toc
#include "sctl/static-array.hpp"       // for StaticArray
#include "sctl/static-array.txx"       // for StaticArray::operator[], Stati...
#include "sctl/tree.hpp"               // for Morton
#include "sctl/vector.hpp"             // for Vector
#include "sctl/vector.txx"             // for Vector::operator[], Vector::begin

namespace sctl {

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

  template <class Real, Integer COORD_DIM> void BuildNearList(Vector<Real>& Xtrg_near, Vector<Real>& Xn_trg_near, Vector<Long>& near_elem_cnt, Vector<Long>& near_elem_dsp, Vector<Long>& near_scatter_index, Vector<Long>& near_trg_cnt, Vector<Long>& near_trg_dsp, const Vector<Real>& Xtrg, const Vector<Real>& Xn_trg, const Vector<Real>& Xsrc, const Vector<Real>& src_radius, const Vector<Long>& src_elem_nds_cnt, const Vector<Long>& src_elem_nds_dsp, const Comm& comm) {
    // Input: Xtrg, Xn_trg, Xsrc, src_radius, src_elem_nds_cnt, src_elem_nds_dsp, comm
    // Output: Xtrg_near, Xn_trg_near, near_elem_cnt, near_elem_dsp, near_scatter_index, near_trg_cnt, near_trg_dsp

    struct NodeData {
      Long idx;
      Real rad;
      StaticArray<Real,COORD_DIM> X;
      StaticArray<Real,COORD_DIM> Xn;
      Morton<COORD_DIM> mid;
      Long elem_idx;
      Long pid;
    };
    auto comp_node_mid = [](const NodeData& A, const NodeData& B) {
      return A.mid < B.mid;
    };
    auto comp_node_eid_idx = [](const NodeData& A, const NodeData& B) {
      return (A.elem_idx<B.elem_idx) || (A.elem_idx==B.elem_idx && A.idx<B.idx);
    };
    auto node_dist2 = [](const NodeData& A, const NodeData& B) {
      Real dist2 = 0;
      for (Long i = 0; i < COORD_DIM; i++) {
        Real dX = A.X[i] - B.X[i];
        dist2 += dX * dX;
      }
      return dist2;
    };

    bool have_trg_normal;
    { // Set have_trg_normal
      StaticArray<Long,1> Nloc{Xn_trg.Dim()}, Nglb{0};
      comm.Allreduce<Long>(Nloc, Nglb, 1, CommOp::SUM);
      have_trg_normal = (Nglb[0] > 0);
      SCTL_ASSERT(!have_trg_normal || (Xn_trg.Dim() == Xtrg.Dim()));
    }

    const Long Ntrg = Xtrg.Dim()/COORD_DIM;
    const Long Nsrc = Xsrc.Dim()/COORD_DIM;
    const Long Nelem = src_elem_nds_cnt.Dim();
    Comm comm_ = comm.Split(Nsrc || Ntrg);
    if (!Nsrc && !Ntrg) return;
    const Long np = comm_.Size();
    const Long rank = comm_.Rank();

    Long trg_offset, src_offset, elem_offset;
    { // set trg_offset, src_offset, elem_offset
      StaticArray<Long,3> send_buff{Ntrg, Nsrc, Nelem}, recv_buff{0,0,0};
      comm_.Scan((ConstIterator<Long>)send_buff, (Iterator<Long>)recv_buff, 3, CommOp::SUM);
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
        if (Ntrg) { // Init X0_local
          for (Long k = 0; k < COORD_DIM; k++) {
            X0_local[k] = Xtrg[k];
          }
        } else {
          SCTL_ASSERT(Nsrc);
          for (Long k = 0; k < COORD_DIM; k++) {
            X0_local[k] = Xsrc[k];
          }
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
        comm_.Allreduce<Real>(X0_local, BBX0, COORD_DIM, CommOp::MIN);

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
        comm_.Allreduce<Real>(Ptr2ConstItr<Real>(&len_local,1), Ptr2Itr<Real>(&BBlen,1), 1, CommOp::MAX);
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
          trg_nodes[i].Xn[k] = (have_trg_normal ? Xn_trg[i*COORD_DIM+k] : 0);
          Xmid[k] = (Xtrg[i*COORD_DIM+k]-BBX0[k]) * BBlen_inv;
        }
        trg_nodes[i].mid = Morton<COORD_DIM>((ConstIterator<Real>)Xmid);
        trg_nodes[i].elem_idx = 0;
        trg_nodes[i].pid = rank;
      }
      for (Long i = 0; i < Nsrc; i++) { // Set src_nodes
        Integer depth = (Integer)(log(src_radius[i]*BBlen_inv+machine_eps<Real>())/log(0.5));
        depth = std::min(Morton<COORD_DIM>::MaxDepth(), std::max<Integer>(depth,0));
        StaticArray<Real,COORD_DIM> Xmid;
        src_nodes[i].idx = src_offset + i;
        src_nodes[i].rad = src_radius[i];
        for (Long k = 0; k < COORD_DIM; k++) {
          src_nodes[i].X[k] = Xsrc[i*COORD_DIM+k];
          Xmid[k] = (Xsrc[i*COORD_DIM+k]-BBX0[k]) * BBlen_inv;
        }
        src_nodes[i].mid = Morton<COORD_DIM>((ConstIterator<Real>)Xmid, depth);
        src_nodes[i].pid = rank;
      }
      for (Long i = 0; i < Nelem; i++) { // Set src_nodes.elem_idx
        for (Long j = 0; j < src_elem_nds_cnt[i]; j++) {
          src_nodes[src_elem_nds_dsp[i]+j].elem_idx = elem_offset + i;
        }
      }
    }

    Vector<NodeData> trg_nodes0, src_nodes0, splitter_nodes(comm_.Size());
    { // Set trg_nodes0 <- sort(trg_nodes), src_nodes0 <- sort(src_nodes)
      comm_.HyperQuickSort(src_nodes, src_nodes0, comp_node_mid);
      comm_.HyperQuickSort(trg_nodes, trg_nodes0, comp_node_mid);

      SCTL_ASSERT(src_nodes0.Dim());
      StaticArray<NodeData,1> splitter_node{src_nodes0[0]};
      if (!rank) splitter_node[0].mid = Morton<COORD_DIM>();
      while (splitter_node[0].mid.Depth()) { // find coarsest ancestor with same coordinates
        auto& mid = splitter_node[0].mid;
        const Long depth = mid.Depth();
        if (mid == mid.Ancestor(depth-1).Ancestor(depth)) {
          mid = mid.Ancestor(depth-1);
        } else break;
      }

      comm_.Allgather((ConstIterator<NodeData>)splitter_node, 1, splitter_nodes.begin(), 1);
      comm_.PartitionS(src_nodes0, splitter_node[0], comp_node_mid);
      comm_.PartitionS(trg_nodes0, splitter_node[0], comp_node_mid);
    }

    Vector<NodeData> src_nodes1;
    if (1) { // Set src_nodes1 <- src_nodes0 + halo
      Vector<std::pair<Long,Long>> proc_srcidx_lst;
      std::set<Long> user_proc_set; // tmp
      Vector<Morton<COORD_DIM>> nbr_lst; // tmp
      for (Long i = 0; i < src_nodes0.Dim(); i++) {
        user_proc_set.clear();
        src_nodes0[i].mid.NbrList(nbr_lst, src_nodes0[i].mid.Depth(), false);
        for (const auto nbr : nbr_lst) if (nbr.Depth() >= 0) {
          const auto proc_split_srch = [&splitter_nodes,&comp_node_mid](const Morton<COORD_DIM>& m) {
            NodeData srch_node; srch_node.mid = m;
            return  std::upper_bound(splitter_nodes.begin(), splitter_nodes.end(), srch_node, comp_node_mid) - splitter_nodes.begin() - 1;
          };
          Long p0 = proc_split_srch(nbr);
          Long p1 = proc_split_srch(nbr.Next());
          if (p1 < comm_.Size() && splitter_nodes[p1].mid < nbr.Next()) p1++;
          for (Long k = p0; k < p1; k++) {
            if (k != rank) user_proc_set.insert(k);
          }
        }
        for (const auto& p : user_proc_set) {
          proc_srcidx_lst.PushBack(std::make_pair(p, i));
        }
      }
      omp_par::merge_sort(proc_srcidx_lst.begin(), proc_srcidx_lst.end());

      Vector<Long> scnt(np), sdsp(np); scnt = 0; sdsp = 0;
      Vector<Long> rcnt(np), rdsp(np); rcnt = 0; rdsp = 0;
      Vector<NodeData> sbuff(proc_srcidx_lst.Dim());
      for (Long i = 0; i < sbuff.Dim(); i++) {
        sbuff[i] = src_nodes0[proc_srcidx_lst[i].second];
        scnt[proc_srcidx_lst[i].first]++;
      }
      omp_par::scan(scnt.begin(), sdsp.begin(), np);
      comm_.Alltoall<Long>(scnt.begin(), 1, rcnt.begin(), 1);
      omp_par::scan(rcnt.begin(), rdsp.begin(), np);

      // Exchange data
      Vector<NodeData> rbuff(rdsp[np-1] + rcnt[np-1]);
      void* req_ptr = comm_.Ialltoallv_sparse(sbuff.begin(), scnt.begin(), sdsp.begin(), rbuff.begin(), rcnt.begin(), rdsp.begin());

      // Set src_nodes1
      src_nodes1.ReInit(rbuff.Dim() + src_nodes0.Dim());
      for (Long i = 0; i < src_nodes0.Dim(); i++) {
        src_nodes1[rdsp[rank]+i] = src_nodes0[i];
      }
      comm_.Wait(req_ptr);
      for (Long i = 0; i < rdsp[rank]; i++) {
        src_nodes1[i] = rbuff[i];
      }
      for (Long i = rdsp[rank]; i < rbuff.Dim(); i++) {
        src_nodes1[src_nodes0.Dim()+i] = rbuff[i];
      }
    } else { // src_nodes1 <- Allgather(src_nodes0)
      const Long Np = comm_.Size();
      Vector<Long> cnt0(1), cnt(Np), dsp(Np);
      cnt0[0] = src_nodes0.Dim(); dsp[0] = 0;
      comm_.Allgather(cnt0.begin(), 1, cnt.begin(), 1);
      omp_par::scan(cnt.begin(), dsp.begin(), Np);

      src_nodes1.ReInit(dsp[Np-1] + cnt[Np-1]);
      comm_.Allgatherv(src_nodes0.begin(), src_nodes0.Dim(), src_nodes1.begin(), cnt.begin(), dsp.begin());
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
          trg_src_near_mid.ReInit(0); // list of neighbor pairs from (trg_mid_lst x src_mid_lst), sorted by trg-mid first and then src-mid
          src_mid_lst.ReInit(0); // unique covering nodes of element-eid
          trg_mid_lst.ReInit(0); // unique neighbor nodes of src_mid_lst
          src_range.ReInit(0); // range of src-spheres (src_nodes1) contained in each src_mid_lst
          trg_range.ReInit(0); // range of trg-points (trg_nodes0) contained in each trg_mid_lst
          trg_mid_set.clear(); // tmp
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
              for (const auto& mid : nbr_lst) if (mid.Depth() >= 0) {
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
              for (const auto& mid : nbr_lst) if (mid.Depth() >= 0) {
                Long j = std::upper_bound(trg_mid_lst.begin(), trg_mid_lst.end(), mid) - trg_mid_lst.begin() - 1;
                if (j>=0 && mid.Ancestor(trg_mid_lst[j].Depth()) == trg_mid_lst[j]) {
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
    { // sort and partition by elem-ID
      Vector<NodeData> near_lst0;
      { // near_lst0 <-- partition(dist_sort(near_lst), elem_offset)
        NodeData split_node;
        split_node.idx=0;
        split_node.elem_idx=elem_offset;
        comm_.HyperQuickSort(near_lst, near_lst0, comp_node_eid_idx);
        comm_.PartitionS(near_lst0, split_node, comp_node_eid_idx);
      }
      near_lst.Swap(near_lst0);
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
    if (have_trg_normal) { // Set Xn_trg_near
      Xn_trg_near.ReInit(near_lst.Dim()*COORD_DIM);
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < near_lst.Dim(); i++) {
        for (Long k = 0; k < COORD_DIM; k++) {
          Xn_trg_near[i*COORD_DIM+k] = near_lst[i].Xn[k];
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
          Long elem_idx_ = near_lst[i].elem_idx, cnt = 0; //, dsp = i;
          for (; i<idx1 && near_lst[i].elem_idx==elem_idx_; i++) cnt++;
          //near_elem_dsp[elem_idx_-elem_offset] = dsp; // skips for elements with cnt == 0
          near_elem_cnt[elem_idx_-elem_offset] = cnt;
        }
      }
      omp_par::scan(near_elem_cnt.begin(), near_elem_dsp.begin(), Nelem);
    }

    { // Set scatter_index, near_trg_cnt, near_trg_dsp
      Vector<Long> trg_idx(near_lst.Dim());
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < trg_idx.Dim(); i++) {
        trg_idx[i] = near_lst[i].idx;
      }
      comm_.SortScatterIndex(trg_idx, near_scatter_index, &trg_offset);
      comm_.ScatterForward(trg_idx, near_scatter_index);

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




  template <class Real> void ElementListBase<Real>::GetFarFieldDensity(Vector<Real>& Fout, const Vector<Real>& Fin) const {
    if (Fout.Dim() != 0) Fout.ReInit(0);
  }

  template <class Real> void ElementListBase<Real>::FarFieldDensityOperatorTranspose(Matrix<Real>& Mout, const Matrix<Real>& Min, const Long elem_idx) const {
    if (Mout.Dim(0) != 0 || Mout.Dim(1) != 0) Mout.ReInit(0,0);
  }


  template <class Real> template <class Kernel> void ElementListBase<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    if (M_lst.Dim() != 0) M_lst.ReInit(0);
  }

  template <class Real> template <class Kernel> void ElementListBase<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    if (M.Dim(0) != 0 || M.Dim(1) != 0) M.ReInit(0,0);
  }

  template <class Real> template <class Kernel> void ElementListBase<Real>::EvalNearInterac(Vector<Real>& u, const Vector<Real>& f, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    if (u.Dim() != 0) u.ReInit(0);
  }

  template <class Real> bool ElementListBase<Real>::MatrixFree() const {
    return false;
  }



  template <class Real, class Kernel> BoundaryIntegralOp<Real,Kernel>::BoundaryIntegralOp(const Kernel& ker, bool trg_normal_dot_prod, const Comm& comm) : tol_(1e-10), ker_(ker), trg_normal_dot_prod_(trg_normal_dot_prod), comm_(comm), fmm(comm) {
    SCTL_ASSERT(!trg_normal_dot_prod_ || (KDIM1 % COORD_DIM == 0));
    ClearSetup();

    fmm.SetKernels(ker, ker, ker);
    fmm.AddSrc("Src", ker, ker);
    fmm.AddTrg("Trg", ker, ker);
    fmm.SetKernelS2T("Src", "Trg", ker);
    fmm.SetAccuracy((Integer)(log(tol_)/log(0.1))+1);
  }

  template <class Real, class Kernel> BoundaryIntegralOp<Real,Kernel>::~BoundaryIntegralOp() {
    Vector<std::string> elem_lst_name;
    for (auto& it : elem_lst_map) elem_lst_name.PushBack(it.first);
    for (const auto& name : elem_lst_name) DeleteElemList(name);
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetPeriodicity(Periodicity p, Real L) {
    if (p == periodicity_ && L == period_length_) return;
    setup_far_flag = false;
    setup_near_flag = false;
    periodicity_ = p;
    period_length_ = L;
    fmm.SetPeriodicity(p, L);
  }

  template <class Real, class Kernel> Periodicity BoundaryIntegralOp<Real,Kernel>::GetPeriodicity() const {
    return periodicity_;
  }

  template <class Real, class Kernel> Real BoundaryIntegralOp<Real,Kernel>::GetPeriodLength() const {
    return period_length_;
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetAccuracy(Real tol) {
    if (tol == tol_) return;
    setup_far_flag = false;
    setup_self_flag = false;
    setup_near_flag = false;
    tol_ = tol;
    fmm.SetAccuracy((Integer)(log(tol_)/log(0.1))+1);
  }

  template <class Real, class Kernel> Real BoundaryIntegralOp<Real,Kernel>::GetAccuracy() const {
    return tol_;
  }

  template <class Real, class Kernel> template <class KerS2M, class KerS2L, class KerS2T, class KerM2M, class KerM2L, class KerM2T, class KerL2L, class KerL2T> void BoundaryIntegralOp<Real,Kernel>::SetFMMKer(const KerS2M& k_s2m, const KerS2L& k_s2l, const KerS2T& k_s2t, const KerM2M& k_m2m, const KerM2L& k_m2l, const KerM2T& k_m2t, const KerL2L& k_l2l, const KerL2T& k_l2t, const typename ParticleFMM<Real,COORD_DIM>::VolPotenT m2l_vol_poten, const typename ParticleFMM<Real,COORD_DIM>::VolPotenT m2t_vol_poten) {
    fmm.DeleteSrc("Src");
    fmm.DeleteTrg("Trg");

    fmm.SetKernels(k_m2m, k_m2l, k_l2l, m2l_vol_poten);
    fmm.AddSrc("Src", k_s2m, k_s2l);
    fmm.AddTrg("Trg", k_m2t, k_l2t, (m2t_vol_poten ? m2t_vol_poten : m2l_vol_poten));
    fmm.SetKernelS2T("Src", "Trg", k_s2t);
  }

  template <class Real, class Kernel> template <class ElemLstType> void BoundaryIntegralOp<Real,Kernel>::AddElemList(const ElemLstType& elem_lst, const std::string& name) {
    static_assert(std::is_copy_constructible<ElemLstType>::value, "ElemeType must be copy-constructible.");
    //SCTL_ASSERT_MSG(elem_lst_map.find(name) == elem_lst_map.end(), "Element list already exists.");
    if (elem_lst_map.find(name) != elem_lst_map.end()) DeleteElemList(name);

    elem_lst_map[name] = dynamic_cast<ElementListBase<Real>*>(new ElemLstType(elem_lst));
    elem_data_map[name].SelfInterac = ElemLstType::template SelfInterac<Kernel>;
    elem_data_map[name].NearInterac = ElemLstType::template NearInterac<Kernel>;
    elem_data_map[name].EvalNearInterac = ElemLstType::template EvalNearInterac<Kernel>;
    ClearSetup();
  }

  template <class Real, class Kernel> template <class ElemLstType> const ElemLstType& BoundaryIntegralOp<Real,Kernel>::GetElemList(const std::string& name) const {
    SCTL_ASSERT_MSG(elem_lst_map.find(name) != elem_lst_map.end(), "Element list does not exist.");
    return *dynamic_cast<const ElemLstType*>(elem_lst_map.at(name));
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::DeleteElemList(const std::string& name) {
    //SCTL_ASSERT_MSG(elem_lst_map.find(name) != elem_lst_map.end(), "Element list does not exist.");
    if (elem_lst_map.find(name) == elem_lst_map.end()) return;

    delete (ElementListBase<Real>*)elem_lst_map[name];
    elem_data_map.erase(name);
    elem_lst_map.erase(name);
    ClearSetup();
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetTargetCoord(const Vector<Real>& Xtrg) {
    Xt = Xtrg;
    setup_flag = false;
    setup_far_flag = false;
    setup_near_flag = false;
  }
  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetTargetNormal(const Vector<Real>& Xn_trg) {
    Xnt = Xn_trg;
    setup_flag = false;
    setup_near_flag = false;
  }

  template <class Real, class Kernel> template <class ElemLstType> void BoundaryIntegralOp<Real,Kernel>::DeleteElemList() {
    DeleteElemList(std::to_string(typeid(ElemLstType).hash_code()));
  }

  template <class Real, class Kernel> Long BoundaryIntegralOp<Real,Kernel>::Dim(Integer k) const {
    SetupBasic();
    if (k == 0) {
      const Long Nelem = elem_nds_cnt.Dim();
      return (Nelem ? (elem_nds_dsp[Nelem-1] + elem_nds_cnt[Nelem-1]) * KDIM0 : 0);
    }
    if (k == 1) {
      return (Xtrg.Dim()/COORD_DIM) * (trg_normal_dot_prod_ ? KDIM1/COORD_DIM : KDIM1);
    }
    SCTL_ASSERT(false);
    return -1;
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::Setup() const {
    if (setup_flag && setup_far_flag && setup_self_flag && setup_near_flag) return;
    Profile::Tic("Setup", &comm_, true, 5);
    SetupBasic();
    SetupFar();
    SetupSelf();
    SetupNear();
    Profile::Toc();
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::ClearSetup() const {
    setup_flag = false;
    setup_far_flag = false;
    setup_self_flag = false;
    setup_near_flag = false;
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::ComputePotential(Vector<Real>& U, const Vector<Real>& F) const {
    Setup();
    Profile::Tic("Eval", &comm_, true, 5);
    ComputeFarField(U, F);
    ComputeNearInterac(U, F);
    Profile::Toc();
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SqrtScaling(Vector<Real>& U) const {
    const Long Nsrc = (elem_nds_cnt.Dim() ? elem_nds_dsp.end()[-1] + elem_nds_cnt.end()[-1] : 0);
    const Long dof = (Nsrc ? U.Dim() / Nsrc : 0);
    SCTL_ASSERT(U.Dim() == Nsrc * dof);

    const Long Nlst = elem_lst_map.size();
    for (Long i = 0; i < Nlst; i++) { // Init F_far
      const Long elem_idx0 = elem_lst_dsp[i];
      const Long elem_idx1 = elem_lst_dsp[i]+elem_lst_cnt[i];

      Vector<Real> X, Xn, wts, dist_far;
      Vector<Long> element_wise_node_cnt, element_wise_node_dsp(elem_idx1-elem_idx0); element_wise_node_dsp = 0;
      elem_lst_map.at(elem_lst_name[i])->GetFarFieldNodes(X, Xn, wts, dist_far, element_wise_node_cnt, 1);
      SCTL_ASSERT(element_wise_node_cnt.Dim() == elem_lst_cnt[i]);
      for (Long j = 1; j < element_wise_node_cnt.Dim(); j++) {
        element_wise_node_dsp[j] = element_wise_node_dsp[j-1] + element_wise_node_cnt[j-1];
      }

      for (Long elem = elem_idx0; elem < elem_idx1; elem++) {
        const Long j0 = elem_nds_dsp[elem];
        const Long j1 = elem_nds_dsp[elem] + elem_nds_cnt[elem];
        Vector<Real> U_((j1-j0)*dof, U.begin() + j0*dof, false);

        Real area = 0;
        for (Long j = 0; j < element_wise_node_cnt[elem-elem_idx0]; j++) {
          area += wts[element_wise_node_dsp[elem-elem_idx0]+j];
        }
        U_ *= sqrt<Real>(area);
      }
    }
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::InvSqrtScaling(Vector<Real>& U) const {
    const Long Nsrc = (elem_nds_cnt.Dim() ? elem_nds_dsp.end()[-1] + elem_nds_cnt.end()[-1] : 0);
    const Long dof = (Nsrc ? U.Dim() / Nsrc : 0);
    SCTL_ASSERT(U.Dim() == Nsrc * dof);

    const Long Nlst = elem_lst_map.size();
    for (Long i = 0; i < Nlst; i++) { // Init F_far
      const Long elem_idx0 = elem_lst_dsp[i];
      const Long elem_idx1 = elem_lst_dsp[i]+elem_lst_cnt[i];

      Vector<Real> X, Xn, wts, dist_far;
      Vector<Long> element_wise_node_cnt, element_wise_node_dsp(elem_idx1-elem_idx0); element_wise_node_dsp = 0;
      elem_lst_map.at(elem_lst_name[i])->GetFarFieldNodes(X, Xn, wts, dist_far, element_wise_node_cnt, 1);
      SCTL_ASSERT(element_wise_node_cnt.Dim() == elem_lst_cnt[i]);
      for (Long j = 1; j < element_wise_node_cnt.Dim(); j++) {
        element_wise_node_dsp[j] = element_wise_node_dsp[j-1] + element_wise_node_cnt[j-1];
      }

      for (Long elem = elem_idx0; elem < elem_idx1; elem++) {
        const Long j0 = elem_nds_dsp[elem];
        const Long j1 = elem_nds_dsp[elem] + elem_nds_cnt[elem];
        Vector<Real> U_((j1-j0)*dof, U.begin() + j0*dof, false);

        Real area = 0;
        for (Long j = 0; j < element_wise_node_cnt[elem-elem_idx0]; j++) {
          area += wts[element_wise_node_dsp[elem-elem_idx0]+j];
        }
        U_ *= 1/sqrt<Real>(area);
      }
    }
  }



  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetupBasic() const {
    if (setup_flag) return;
    elem_lst_name.ReInit(0);
    elem_lst_cnt.ReInit(0);
    elem_lst_dsp.ReInit(0);
    elem_nds_cnt.ReInit(0);
    elem_nds_dsp.ReInit(0);
    Xsurf.ReInit(0);
    Xtrg.ReInit(0);
    Xn_trg.ReInit(0);

    const Long Nlst = elem_lst_map.size();
    { // Set elem_lst_name
      elem_lst_name.ReInit(0);
      for (const auto& x : elem_lst_map) {
        elem_lst_name.PushBack(x.first);
      }
    }
    { // Set elem_lst_cnt, elem_nds_cnt, Xsurf
      elem_lst_cnt .ReInit(Nlst);
      Vector<Vector<Real>> Xsurf_(Nlst);
      Vector<Vector<Real>> Xn_surf_(Nlst);
      Vector<Vector<Long>> elem_nds_cnt_(Nlst);
      for (Long i = 0; i < Nlst; i++) {
        elem_lst_map.at(elem_lst_name[i])->GetNodeCoord(&Xsurf_[i], &Xn_surf_[i], &elem_nds_cnt_[i]);
        elem_lst_cnt[i] = elem_nds_cnt_[i].Dim();
      }
      concat_vecs(Xsurf, Xsurf_);
      concat_vecs(Xn_surf, Xn_surf_);
      concat_vecs(elem_nds_cnt, elem_nds_cnt_);
    }
    { // Set elem_lst_dsp, elem_nds_dsp
      const Long Nelem = elem_nds_cnt.Dim();
      if (elem_lst_dsp.Dim() != Nlst ) elem_lst_dsp.ReInit(Nlst);
      if (elem_nds_dsp.Dim() != Nelem) elem_nds_dsp.ReInit(Nelem);
      if (Nlst ) elem_lst_dsp[0] = 0;
      if (Nelem) elem_nds_dsp[0] = 0;
      omp_par::scan(elem_lst_cnt.begin(), elem_lst_dsp.begin(), Nlst);
      omp_par::scan(elem_nds_cnt.begin(), elem_nds_dsp.begin(), Nelem);
      SCTL_ASSERT(Nelem == (Nlst ? elem_lst_dsp[Nlst-1] + elem_lst_cnt[Nlst-1] : 0));
    }
    { // Set Xtrg
      StaticArray<Long,2> Xt_size{Xt.Dim(),0};
      comm_.Allreduce(Xt_size+0, Xt_size+1, 1, CommOp::SUM);
      if (Xt_size[1]) { // Set Xtrg
        Xtrg = Xt;
      } else {
        Xtrg = Xsurf;
      }
    }
    if (trg_normal_dot_prod_) {
      if (Xnt.Dim()) {
        Xn_trg = Xnt;
        SCTL_ASSERT_MSG(Xn_trg.Dim() == Xtrg.Dim(), "Invalid normal vector at targets.");
      } else {
        Xn_trg = Xn_surf;
      }
    }

    setup_flag = true;
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetupFar() const {
    if (setup_far_flag) return;
    X_far.ReInit(0);
    Xn_far.ReInit(0);
    wts_far.ReInit(0);
    dist_far.ReInit(0);
    elem_nds_cnt_far.ReInit(0);
    elem_nds_dsp_far.ReInit(0);
    SetupBasic();

    Profile::Tic("SetupFarField", &comm_, false, 6);
    const Long Nlst = elem_lst_map.size();
    Vector<Vector<Real>> X_far_(Nlst);
    Vector<Vector<Real>> Xn_far_(Nlst);
    Vector<Vector<Real>> wts_far_(Nlst);
    Vector<Vector<Real>> dist_far_(Nlst);
    Vector<Vector<Long>> elem_nds_cnt_far_(Nlst);
    for (Long i = 0; i < Nlst; i++) {
      elem_lst_map.at(elem_lst_name[i])->GetFarFieldNodes(X_far_[i], Xn_far_[i], wts_far_[i], dist_far_[i], elem_nds_cnt_far_[i], tol_);
    }
    concat_vecs(X_far   , X_far_   );
    concat_vecs(Xn_far  , Xn_far_  );
    concat_vecs(wts_far , wts_far_ );
    concat_vecs(dist_far, dist_far_);
    concat_vecs(elem_nds_cnt_far, elem_nds_cnt_far_);
    { // Set elem_nds_dsp_far
      const Long Nelem = elem_nds_cnt.Dim();
      SCTL_ASSERT(elem_nds_cnt_far.Dim() == Nelem);
      if (elem_nds_dsp_far.Dim() != Nelem) elem_nds_dsp_far.ReInit(Nelem);
      if (Nelem) elem_nds_dsp_far[0] = 0;
      omp_par::scan(elem_nds_cnt_far.begin(), elem_nds_dsp_far.begin(), Nelem);
    }

    fmm.SetSrcCoord("Src", X_far, Xn_far);
    fmm.SetTrgCoord("Trg", Xtrg);
    Profile::Toc();

    setup_far_flag = true;
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetupSelf() const {
    // TODO: skip SetupSelf when no on-surface targets.
    if (setup_self_flag) return;
    SetupBasic();

    Profile::Tic("SetupSingular", &comm_, false, 6);
    { // Set K_self
      const Long Nlst = elem_lst_map.size();
      Vector<Long> elem_cnt(Nlst), elem_dsp(Nlst);
      for (Long i = 0; i < Nlst; i++) {
        const auto& name = elem_lst_name[i];
        const auto& elem_lst = elem_lst_map.at(name);
        elem_cnt[i] = (elem_lst->MatrixFree() ? 0 : elem_lst->Size());
        elem_dsp[i] = (i==0?0:elem_dsp[i-1]+elem_cnt[i-1]);
      }

      if (K_self.Dim() != elem_dsp[Nlst-1]+elem_cnt[Nlst-1]) K_self.ReInit(elem_dsp[Nlst-1]+elem_cnt[Nlst-1]);
      // TODO: also pre-allocate elements of K_self from a memory pool.
      for (Long i = 0; i < Nlst; i++) {
        const auto& name = elem_lst_name[i];
        const auto& elem_lst = elem_lst_map.at(name);
        const auto& elem_data = elem_data_map.at(name);
        if (elem_lst->MatrixFree()) continue;
        Vector<Matrix<Real>> K_self_(elem_cnt[i], K_self.begin() + elem_dsp[i], false);
        elem_data.SelfInterac(K_self_, ker_, tol_, trg_normal_dot_prod_, elem_lst);
      }
    }
    Profile::Toc();

    setup_self_flag = true;
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::SetupNear() const {
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

    Profile::Tic("SetupNear", &comm_, true, 6);
    Profile::Tic("BuildNearLst", &comm_, true, 7);
    if (periodicity_ == Periodicity::NONE) {
      BuildNearList<Real,COORD_DIM>(Xtrg_near, Xn_trg_near, near_elem_cnt, near_elem_dsp, near_scatter_index, near_trg_cnt, near_trg_dsp, Xtrg, Xn_trg, X_far, dist_far, elem_nds_cnt_far, elem_nds_dsp_far, comm_);
    } else {
      // TODO: build periodicity into BuildNearList, will be more efficient than duplicating targets
      const Long Ncopy = [this](){
        Long n = 1;
        for (Long k = 0; k < COORD_DIM; k++) n *= ( static_cast<uint8_t>(periodicity_)&(1<<k) ? 3 : 1);
        return n;
      }();

      Matrix<Real> periodic_shift(Ncopy, COORD_DIM);
      { // Set periodic_shift
        Long count = 1;
        periodic_shift = 0;
        for (Long k = 0; k < COORD_DIM; k++) {
          if (static_cast<uint8_t>(periodicity_)&(1<<k)) {
            for (Long i = count; i < 3*count; i++) {
              for (Long kk = 0; kk < k; kk++) periodic_shift[i][kk] = periodic_shift[i%count][kk];
              periodic_shift[i][k] = (i/count==2 ? -1 : i/count);
            }
            count *= 3;
          }
        }
        periodic_shift *= period_length_;
      }

      const Long Ntrg = Xtrg.Dim()/COORD_DIM;
      Vector<Real> Xtrg_(Ntrg*COORD_DIM*Ncopy), Xn_trg_;
      for (Long i = 0; i < Ntrg; i++) { // Set Xtrg_
        for (Long j = 0; j < Ncopy; j++) {
          for (Long k = 0; k < COORD_DIM; k++) {
            Xtrg_[(i*Ncopy+j)*COORD_DIM+k] = Xtrg[i*COORD_DIM+k] + periodic_shift[j][k];
          }
        }
      }
      if (Xn_trg.Dim()) { // Set Xn_trg_
        Xn_trg_.ReInit(Ntrg*COORD_DIM*Ncopy);
        for (Long i = 0; i < Ntrg; i++) {
          for (Long j = 0; j < Ncopy; j++) {
            for (Long k = 0; k < COORD_DIM; k++) {
              Xn_trg_[(i*Ncopy+j)*COORD_DIM+k] = Xn_trg[i*COORD_DIM+k];
            }
          }
        }
      }

      Vector<Long> near_trg_cnt_, near_trg_dsp_;
      BuildNearList<Real,COORD_DIM>(Xtrg_near, Xn_trg_near, near_elem_cnt, near_elem_dsp, near_scatter_index, near_trg_cnt_, near_trg_dsp_, Xtrg_, Xn_trg_, X_far, dist_far, elem_nds_cnt_far, elem_nds_dsp_far, comm_);
      SCTL_ASSERT(near_trg_cnt_.Dim() == Ntrg*Ncopy);

      near_trg_cnt.ReInit(Ntrg);
      near_trg_dsp.ReInit(Ntrg);
      for (Long i = 0; i < Ntrg; i++) {
        near_trg_cnt[i] = 0;
        for (Long j = 0; j < Ncopy; j++) {
          if (!near_trg_cnt[i]) near_trg_dsp[i] = near_trg_dsp_[i*Ncopy+j]; // set dsp of first non-zero count
          near_trg_cnt[i] += near_trg_cnt_[i*Ncopy+j];
        }
      }
    }
    Profile::Toc();

    { // Set K_near_cnt, K_near_dsp, K_near
      const Integer KDIM1_ = (trg_normal_dot_prod_ ? KDIM1/COORD_DIM : KDIM1);
      const Long Nlst = elem_lst_map.size();
      const Long Nelem = near_elem_cnt.Dim();
      SCTL_ASSERT(Nelem == elem_nds_cnt.Dim());
      if (Nelem) { // Set K_near_cnt, K_near_dsp
        K_near_cnt.ReInit(Nelem);
        K_near_dsp.ReInit(Nelem);
        if (Nelem) K_near_dsp[0] = 0;
        #pragma omp parallel for schedule(static)
        for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
          const Long elem_lst_idx = std::lower_bound(elem_lst_dsp.begin(), elem_lst_dsp.end(), elem_idx+1) - elem_lst_dsp.begin() - 1;
          const auto& elem_lst = elem_lst_map.at(elem_lst_name[elem_lst_idx]);
          if (elem_lst->MatrixFree()) {
            K_near_cnt[elem_idx] = 0;
          } else {
            K_near_cnt[elem_idx] = elem_nds_cnt[elem_idx]*near_elem_cnt[elem_idx];
          }
        }
        omp_par::scan(K_near_cnt.begin(), K_near_dsp.begin(), Nelem);
      }
      if (Nelem) { // Set K_near
        K_near.ReInit((K_near_dsp[Nelem-1]+K_near_cnt[Nelem-1])*KDIM0*KDIM1_);

        constexpr Long cache_line_size = 512;
        const Long N_near = near_elem_dsp[Nelem-1] + near_elem_cnt[Nelem-1];
        const Long omp_chunk_size = std::max(N_near/omp_get_max_threads()/32, (cache_line_size+KDIM1_-1)/KDIM1_);
        #pragma omp parallel for schedule(dynamic,omp_chunk_size)
        for (Long i = 0; i < N_near; i++) { // loop over all pairs of elements and their near targets
          const Long elem_idx = std::lower_bound(near_elem_dsp.begin(), near_elem_dsp.end(), i+1) - near_elem_dsp.begin() - 1;
          const Long elem_lst_idx = std::lower_bound(elem_lst_dsp.begin(), elem_lst_dsp.end(), elem_idx+1) - elem_lst_dsp.begin() - 1;

          const auto& name = elem_lst_name[elem_lst_idx];
          const auto& elem_lst = elem_lst_map.at(name);
          const auto& elem_data = elem_data_map.at(name);

          if (!elem_lst->MatrixFree()) {
            const Long j = elem_idx - elem_lst_dsp[elem_lst_idx]; // element index in elem_lst

            const Long N0 = elem_nds_cnt[elem_idx]*KDIM0;
            const Vector<Real> Xsurf_(elem_nds_cnt[elem_idx]*COORD_DIM, Xsurf.begin()+elem_nds_dsp[elem_idx]*COORD_DIM, false);
            Matrix<Real> K_near_(N0,near_elem_cnt[elem_idx]*KDIM1_, K_near.begin()+K_near_dsp[elem_idx]*KDIM0*KDIM1_, false);

            {
              const Long k = i - near_elem_dsp[elem_idx]; // target index in near-list of elem_idx
              Long min_Xt = -1, min_Xsurf = -1;
              const Vector<Real> Xt(COORD_DIM, Xtrg_near.begin()+(near_elem_dsp[elem_idx]+k)*COORD_DIM, false);
              const Vector<Real> Xn((trg_normal_dot_prod_ ? COORD_DIM : 0), Xn_trg_near.begin()+(near_elem_dsp[elem_idx]+k)*COORD_DIM, false);
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
              const Real trg_elem_dist2 = compute_min_dist2(min_Xt, min_Xsurf, Xt, Xsurf_);
              SCTL_ASSERT(min_Xt >= 0 && min_Xsurf >= 0);

              if (trg_elem_dist2 == 0) { // Set K_near_
                if (K_self.Dim() && K_self[elem_idx].Dim(0) && K_self[elem_idx].Dim(1)) {
                  const auto& K_near0 = K_self[elem_idx];
                  SCTL_ASSERT(K_near0.Dim(0) == N0);
                  for (Long l = 0; l < N0; l++) {
                    for (Long k1 = 0; k1 < KDIM1_; k1++) {
                      K_near_[l][k*KDIM1_+k1] = K_near0[l][min_Xsurf*KDIM1_+k1];
                    }
                  }
                } else {
                  for (Long l = 0; l < N0; l++) {
                    for (Long k1 = 0; k1 < KDIM1_; k1++) {
                      K_near_[l][k*KDIM1_+k1] = 0;
                    }
                  }
                }
              } else {
                StaticArray<Real,10000> buff0;
                Matrix<Real> K_near0(N0, KDIM1_, (N0*KDIM1_>10000?NullIterator<Real>():buff0), (N0*KDIM1_>10000));
                elem_data.NearInterac(K_near0, Xt, Xn, ker_, tol_, j, elem_lst);

                if (K_near0.Dim(0) != 0 && K_near0.Dim(1) != 0) {
                  for (Long l = 0; l < N0; l++) {
                    for (Long k1 = 0; k1 < KDIM1_; k1++) {
                      K_near_[l][k*KDIM1_+k1] = K_near0[l][k1];
                    }
                  }
                } else {
                  for (Long l = 0; l < N0; l++) {
                    for (Long k1 = 0; k1 < KDIM1_; k1++) {
                      K_near_[l][k*KDIM1_+k1] = 0;
                    }
                  }
                }
              }
            }
          }
        }
      }

      for (Long i = 0; i < Nlst; i++) { // Subtract direct-interaction part from K_near
        const auto& elem_lst = elem_lst_map.at(elem_lst_name[i]);
        if (elem_lst->MatrixFree()) continue;
        #pragma omp parallel for if(elem_lst_cnt[i] > omp_get_max_threads()) schedule(dynamic)
        for (Long j = 0; j < elem_lst_cnt[i]; j++) { // subtract direct sum
          const Long elem_idx = elem_lst_dsp[i]+j;
          const Long trg_cnt = near_elem_cnt[elem_idx];
          const Long trg_dsp = near_elem_dsp[elem_idx];
          const Vector<Real> Xtrg_near_(trg_cnt*COORD_DIM, Xtrg_near.begin()+trg_dsp*COORD_DIM, false);
          const Vector<Real> Xn_trg_near_((trg_normal_dot_prod_ ? trg_cnt*COORD_DIM : 0), Xn_trg_near.begin()+trg_dsp*COORD_DIM, false);
          if (!trg_cnt) continue;

          const Long far_src_cnt = elem_nds_cnt_far[elem_idx];
          const Long far_src_dsp = elem_nds_dsp_far[elem_idx];
          const Vector<Real> X (far_src_cnt*COORD_DIM,  X_far.begin() + far_src_dsp*COORD_DIM, false);
          const Vector<Real> Xn(far_src_cnt*COORD_DIM, Xn_far.begin() + far_src_dsp*COORD_DIM, false);
          const Vector<Real> wts(far_src_cnt, wts_far.begin() + far_src_dsp, false);

          SCTL_ASSERT(K_near_cnt[elem_idx] == elem_nds_cnt[elem_idx]*trg_cnt);
          Matrix<Real> K_near_(elem_nds_cnt[elem_idx]*KDIM0, trg_cnt*KDIM1_, K_near.begin()+K_near_dsp[elem_idx]*KDIM0*KDIM1_, false);
          { // Set K_near_
            Matrix<Real> Mker(far_src_cnt*KDIM0, trg_cnt*KDIM1_);
            if (trg_normal_dot_prod_) {
              Matrix<Real> Mker_;
              constexpr Integer KDIM1_ = KDIM1/COORD_DIM;
              ker_.template KernelMatrix<Real,true>(Mker_, Xtrg_near_, X, Xn);
              #pragma omp parallel for schedule(static)
              for (Long s = 0; s < far_src_cnt; s++) {
                for (Long k0 = 0; k0 < KDIM0; k0++) {
                  for (Long t = 0; t < trg_cnt; t++) {
                    for (Long k1 = 0; k1 < KDIM1_; k1++) {
                      Mker[s*KDIM0+k0][t*KDIM1_+k1] = 0;
                      for (Long l = 0; l < COORD_DIM; l++) {
                        Mker[s*KDIM0+k0][t*KDIM1_+k1] += Mker_[s*KDIM0+k0][(t*KDIM1_+k1)*COORD_DIM+l] * wts[s] * Xn_trg_near_[t*COORD_DIM+l];
                      }
                    }
                  }
                }
              }
            } else {
              ker_.template KernelMatrix<Real,true>(Mker, Xtrg_near_, X, Xn);
              #pragma omp parallel for schedule(static)
              for (Long s = 0; s < far_src_cnt; s++) {
                for (Long k0 = 0; k0 < KDIM0; k0++) {
                  for (Long t = 0; t < trg_cnt*KDIM1; t++) {
                    Mker[s*KDIM0+k0][t] *= wts[s];
                  }
                }
              }
            }

            Matrix<Real> K_direct;
            elem_lst->FarFieldDensityOperatorTranspose(K_direct, Mker, j);
            const Long N = K_near_.Dim(0)*K_near_.Dim(1);
            if (K_direct.Dim(0) != 0 && K_direct.Dim(1) != 0) {
              #pragma omp parallel for schedule(static)
              for (Long k = 0; k < N; k++) K_near_[0][k] -= K_direct[0][k];
            } else {
              #pragma omp parallel for schedule(static)
              for (Long k = 0; k < N; k++) K_near_[0][k] -= Mker[0][k];
            }
          }
        }
      }
    }
    Profile::Toc();

    setup_near_flag = true;
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::ComputeFarField(Vector<Real>& U, const Vector<Real>& F) const {
    Profile::Tic("EvalFar", &comm_, true, 6);
    const Long Nsrc = X_far.Dim()/COORD_DIM;
    const Long Ntrg = Xtrg.Dim()/COORD_DIM;

    { // Set F_far
      const Long Nlst = elem_lst_map.size();
      if (F_far.Dim() != Nsrc*KDIM0) F_far.ReInit(Nsrc*KDIM0);
      for (Long i = 0; i < Nlst; i++) { // Init F_far
        Long elem_idx0 = elem_lst_dsp[i];
        Long elem_idx1 = elem_lst_dsp[i]+elem_lst_cnt[i];
        Long offset0 = (!elem_lst_cnt[i] ? 0 : elem_nds_dsp[elem_idx0]);
        Long offset1 = (!elem_lst_cnt[i] ? 0 : elem_nds_dsp[elem_idx1-1] + elem_nds_cnt[elem_idx1-1]);
        const Vector<Real> F_((offset1-offset0)*KDIM0, (Iterator<Real>)F.begin() + offset0*KDIM0, false);

        Long offset0_far = (!elem_lst_cnt[i] ? 0 : elem_nds_dsp_far[elem_idx0]);
        Long offset1_far = (!elem_lst_cnt[i] ? 0 : elem_nds_dsp_far[elem_idx1-1] + elem_nds_cnt_far[elem_idx1-1]);
        Vector<Real> F_far_((offset1_far-offset0_far)*KDIM0, F_far.begin() + offset0_far*KDIM0, false);

        elem_lst_map.at(elem_lst_name[i])->GetFarFieldDensity(F_far_, F_);
        if (F_far_.Dim()) { // F_far <-- F_far * wts_far
          #pragma omp parallel for schedule(static)
          for (Long i = offset0_far; i < offset1_far; i++) {
            for (Long j = 0; j < KDIM0; j++) {
              F_far[i*KDIM0+j] *= wts_far[i];
            }
          }
        } else { // F_far <-- F_ * wts_far
          #pragma omp parallel for schedule(static)
          for (Long i = offset0_far; i < offset1_far; i++) {
            const Long i_ = i - offset0_far;
            for (Long j = 0; j < KDIM0; j++) {
              F_far[i*KDIM0+j] = F_[i_*KDIM0+j] * wts_far[i];
            }
          }
        }
      }
    }
    fmm.SetSrcDensity("Src", F_far);

    const Integer KDIM1_ = (trg_normal_dot_prod_ ? KDIM1/COORD_DIM : KDIM1);
    if (U.Dim() != Ntrg*KDIM1_) U.ReInit(Ntrg*KDIM1_);
    U.SetZero();

    if (trg_normal_dot_prod_) {
      constexpr Integer KDIM1_ = KDIM1/COORD_DIM;
      Vector<Real> U_(Ntrg * KDIM1); U_.SetZero();
      fmm.Eval(U_, "Trg");
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < Ntrg; i++) {
        for (Long k = 0; k < KDIM1_; k++) {
          for (Long l = 0; l < COORD_DIM; l++) {
            U[i*KDIM1_+k] += U_[(i*KDIM1_+k)*COORD_DIM+l] * Xn_trg[i*COORD_DIM+l];
          }
        }
      }
    } else {
      fmm.Eval(U, "Trg");
    }

    Profile::Toc();
  }

  template <class Real, class Kernel> void BoundaryIntegralOp<Real,Kernel>::ComputeNearInterac(Vector<Real>& U, const Vector<Real>& F) const {
    const Integer KDIM1_ = (trg_normal_dot_prod_ ? KDIM1/COORD_DIM : KDIM1);

    Profile::Tic("EvalNear", &comm_, true, 6);
    const Long Ntrg = Xtrg.Dim()/COORD_DIM;
    const Long Nelem = near_elem_cnt.Dim();
    SCTL_ASSERT(Nelem == elem_nds_cnt.Dim());
    if (U.Dim() != Ntrg*KDIM1_) {
      U.ReInit(Ntrg*KDIM1_);
      U.SetZero();
    }

    Vector<Real> U_near(Nelem ? (near_elem_dsp[Nelem-1]+near_elem_cnt[Nelem-1])*KDIM1_ : 0);
    #pragma omp parallel for if(Nelem > omp_get_max_threads()) schedule(dynamic)
    for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) { // Compute near-interactions from precomputed operator matrix
      const Long src_dof = elem_nds_cnt[elem_idx]*KDIM0;
      const Long trg_dof = near_elem_cnt[elem_idx]*KDIM1_;
      if (src_dof==0 || trg_dof == 0 || K_near_cnt[elem_idx] == 0) continue;
      SCTL_ASSERT(src_dof * trg_dof == K_near_cnt[elem_idx]*KDIM0*KDIM1_);
      const Matrix<Real> K_near_(src_dof, trg_dof, K_near.begin() + K_near_dsp[elem_idx]*KDIM0*KDIM1_, false);
      const Matrix<Real> F_(1, src_dof, (Iterator<Real>)F.begin() + elem_nds_dsp[elem_idx]*KDIM0, false);
      Matrix<Real> U_(1, trg_dof, U_near.begin() + near_elem_dsp[elem_idx]*KDIM1_, false);
      Matrix<Real>::GEMM(U_, F_, K_near_);
    }

    for (Long i = 0; i < (Long)elem_lst_map.size(); i++) { // Compute near-interactions matrix-free (if EvalNearInterac is implemented)
      const auto& name = elem_lst_name[i];
      const auto& elem_lst = elem_lst_map.at(name);
      const auto& elem_data = elem_data_map.at(name);
      if (!elem_lst->MatrixFree()) continue;
      #pragma omp parallel for if(elem_lst_cnt[i]>4*omp_get_max_threads()) schedule(dynamic)
      for (Long j = 0; j < elem_lst_cnt[i]; j++) {
        const Long elem_idx = elem_lst_dsp[i]+j;
        const Long Ntrg = near_elem_cnt[elem_idx];
        const Vector<Real> Xt(Ntrg*COORD_DIM, Xtrg_near.begin() + near_elem_dsp[elem_idx]*COORD_DIM, false);
        const Vector<Real> Xn(Ntrg*(trg_normal_dot_prod_ ? COORD_DIM : 0), Xn_trg_near.begin() + near_elem_dsp[elem_idx]*COORD_DIM, false);

        {
          const Long src_dof = elem_nds_cnt[elem_idx]*KDIM0;
          const Long trg_dof = near_elem_cnt[elem_idx]*KDIM1_;
          if (src_dof==0 || trg_dof == 0) continue;
          const Vector<Real> F_(src_dof, (Iterator<Real>)F.begin() + elem_nds_dsp[elem_idx]*KDIM0, false);
          Vector<Real> U_(trg_dof, U_near.begin() + near_elem_dsp[elem_idx]*KDIM1_, false);
          elem_data.EvalNearInterac(U_, F_, Xt, Xn, ker_, tol_, elem_idx, elem_lst);
        }
      }
    }

    SCTL_ASSERT(near_trg_cnt.Dim() == Ntrg);
    Profile::Tic("Comm", &comm_, true, 7);
    comm_.ScatterForward(U_near, near_scatter_index);
    Profile::Toc();
    #pragma omp parallel for // schedule(static)
    for (Long i = 0; i < Ntrg; i++) { // Accumulate result to U
      Long near_cnt = near_trg_cnt[i];
      Long near_dsp = near_trg_dsp[i];
      for (Long j = 0; j < near_cnt; j++) {
        for (Long k = 0; k < KDIM1_; k++) {
          U[i*KDIM1_+k] += U_near[(near_dsp+j)*KDIM1_+k];
        }
      }
    }
    Profile::Toc();
  }

}

#endif // _SCTL_BOUNDARY_INTEGRAL_TXX_
