#ifndef _SCTL_TREE_TXX_
#define _SCTL_TREE_TXX_

#include <stdlib.h>               // for drand48
#include <algorithm>              // for lower_bound, max, min, sort
#include <cstdint>                // for int32_t, uint8_t
#include <cstdlib>                // for std::aligned_alloc, std::free
#include <map>                    // for map, operator!=, __map_iterator
#include <numeric>                // for exclusive_scan
#include <set>                    // for set, __tree_const_iterator
#include <string>                 // for basic_string, allocator, string
#include <vector>                 // for vector

#include "sctl/common.hpp"        // for Long, Integer, SCTL_ASSERT, SCTL_NA...
#include "sctl/tree.hpp"          // for Tree, PtTree, Morton
#include "sctl/comm.hpp"          // for Comm, CommOp
#include "sctl/comm.txx"          // for Comm::Alltoallv, Comm::Allreduce
#include "sctl/iterator.hpp"      // for Iterator, ConstIterator
#include "sctl/iterator.txx"      // for Iterator::Iterator<ValueType>, Iter...
#include "sctl/math_utils.txx"    // for pow
#include "sctl/morton.hpp"        // for Morton
#include "sctl/ompUtils.txx"      // for reduce, scan, sample_sort
#include "sctl/scratch_pool.hpp"  // for ScratchBuf
#include "sctl/scratch_pool.txx"
#include "sctl/static-array.hpp"  // for StaticArray
#include "sctl/static-array.txx"  // for StaticArray::operator[], StaticArra...
#include "sctl/vector.hpp"        // for Vector
#include "sctl/vector.txx"        // for Vector::operator[], Vector::begin
#include "sctl/vtudata.hpp"       // for VTUData
#include "sctl/vtudata.txx"       // for VTUData::WriteVTK
#include "sctl/matrix.hpp"
#include "sctl/matrix.txx"

namespace sctl {

  template <Integer DIM> template <class T> class Tree<DIM>::NodeArena {
   public:
    ~NodeArena() { for (Long b = 0; b < blocks_.Dim(); b++) std::free(&blocks_[b][0]); }

    T& Alloc() {
      if (cur_cnt_ == block_elems) {
        void* raw = std::aligned_alloc(SCTL_MEM_ALIGN, block_elems * sizeof(T));
        advise_huge_pages(raw, block_elems * (Long)sizeof(T));
        cur_ = Ptr2Itr<T>(raw, block_elems);
        cur_cnt_ = 0;
        blocks_.PushBack(cur_);
      }
      return cur_[cur_cnt_++];
    }

   private:
    static constexpr Long block_elems = ((8L<<20) / (Long)sizeof(T) / SCTL_MEM_ALIGN + 1) * SCTL_MEM_ALIGN;

    Vector<Iterator<T>> blocks_;
    Iterator<T> cur_;
    Long cur_cnt_{block_elems};
  };

  template <class Real, Integer DIM, class BaseTree> void PtTree<Real,DIM,BaseTree>::test() {
    Long N = 100000;
    Vector<Real> X(N*DIM), f(N);
    for (Long i = 0; i < N; i++) { // Set coordinates (X), and values (f)
      f[i] = 0;
      for (Integer k = 0; k < DIM; k++) {
        X[i*DIM+k] = pow<3>(drand48()*2-1.0)*0.5+0.5;
        f[i] += X[i*DIM+k]*k;
      }
    }

    PtTree<Real,DIM> tree;
    tree.AddParticles("pt", X);
    tree.AddParticleData("pt-value", "pt", f);
    tree.UpdateRefinement(X, 1000); // refine tree with max 1000 points per box.

    { // manipulate tree node data
      const auto& node_lst = tree.GetNodeLists(); // Get interaction lists
      //const auto& node_mid = tree.GetNodeMID();
      //const auto& node_attr = tree.GetNodeAttr();

      // get point values and count for each node
      Vector<Real> value;
      Vector<Long> cnt, dsp;
      tree.GetData(value, cnt, "pt-value");

      // compute the dsp (the point offset) for each node
      dsp.ReInit(cnt.Dim()); dsp = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());

      Long node_idx = 0;
      for (Long i = 0; i < cnt.Dim(); i++) { // find the tree node with maximum points
        if (cnt[node_idx] < cnt[i]) node_idx = i;
      }

      for (Long j = 0; j < cnt[node_idx]; j++) { // for this node, set all pt-value to -1
        value[dsp[node_idx]+j] = -1;
      }

      for (const Long nbr_idx : node_lst[node_idx].nbr) { // loop over the neighbors and set pt-value to 2
        if (nbr_idx >= 0 && nbr_idx != node_idx) {
          for (Long j = 0; j < cnt[nbr_idx]; j++) {
            value[dsp[nbr_idx]+j] = 2;
          }
        }
      }
    }

    // Generate visualization
    tree.WriteParticleVTK("pt", "pt-value");
    tree.WriteTreeVTK("tree");
  }

  template <Integer DIM> constexpr Integer Tree<DIM>::Dim() {
    return DIM;
  }

  template <Integer DIM> Tree<DIM>::Tree(const Comm& comm_) : comm(comm_) {
    Integer rank = comm.Rank();
    Integer np = comm.Size();

    Vector<double> coord;
    { // Set coord
      Long N0 = 1;
      while (sctl::pow<DIM,Long>(N0) < np) N0++;
      Long N = sctl::pow<DIM,Long>(N0);
      Long start = N * (rank+0) / np;
      Long end   = N * (rank+1) / np;
      coord.ReInit((end-start)*DIM);
      for (Long i = start; i < end; i++) {
        Long  idx = i;
        for (Integer k = 0; k < DIM; k++) {
          coord[(i-start)*DIM+k] = (idx % N0) / (double)N0;
          idx /= N0;
        }
      }
    }
    this->UpdateRefinement(coord);
  }

  template <Integer DIM> Tree<DIM>::~Tree() {
    #ifdef SCTL_MEMDEBUG
    for (auto& pair : node_data) {
      SCTL_ASSERT(node_cnt.find(pair.first) != node_cnt.end());
    }
    #endif
  }

  template <Integer DIM> const Vector<Morton<DIM>>& Tree<DIM>::GetPartitionMID() const {
    return mins;
  }
  template <Integer DIM> const Vector<Morton<DIM>>& Tree<DIM>::GetNodeMID() const {
    return node_mid;
  }
  template <Integer DIM> const Vector<typename Tree<DIM>::NodeAttr>& Tree<DIM>::GetNodeAttr() const {
    return node_attr;
  }
  template <Integer DIM> const Vector<typename Tree<DIM>::NodeLists>& Tree<DIM>::GetNodeLists() const {
    return node_lst;
  }
  template <Integer DIM> const Comm& Tree<DIM>::GetComm() const {
    return comm;
  }

  template <Integer DIM> template <class Real> void Tree<DIM>::UpdateRefinement(const Vector<Real>& coord, Long M, bool balance21, Periodicity periodicity, Integer halo_size) {
    const Integer np = comm.Size();
    const Integer rank = comm.Rank();

    static constexpr Integer MAX_CHILD = (1u << DIM);
    static constexpr Integer MAX_NBRS = sctl::pow<DIM,Integer>(3);
    static constexpr Integer MAX_DEPTH = Morton<DIM>::MAX_DEPTH;
    struct NbrPath {
      Integer p_nbr, p_nbr_child;
    };
    static const auto nbr_path = []() {
      const auto parent = (Morton<DIM>{}).Children()[0].Children()[MAX_CHILD-1];
      const auto parent_nbr_lst = parent.NbrList(parent.Depth(), Periodicity::NONE); // interior node, so periodicity doesn't matter

      Matrix<NbrPath> nbr_path(MAX_CHILD, MAX_NBRS);
      for (Integer p2n = 0; p2n < MAX_CHILD; p2n++) {
        const auto n0 = parent.Children()[p2n];
        const auto nlst = n0.NbrList(n0.Depth(), Periodicity::NONE);
        for (Integer nbr_idx = 0; nbr_idx < MAX_NBRS; nbr_idx++) {
          const auto& nbr = nlst[nbr_idx];

          for (Integer p_nbr = 0; p_nbr < (Integer)parent_nbr_lst.size(); p_nbr++) {
            if (parent_nbr_lst[p_nbr].isAncestor(nbr)) {
              const auto parent_nbr_child_lst = parent_nbr_lst[p_nbr].Children();
              nbr_path[p2n][nbr_idx].p_nbr = p_nbr;
              for (Integer p_nbr_child = 0; p_nbr_child < (Integer)parent_nbr_child_lst.size(); p_nbr_child++) {
                if (nbr == parent_nbr_child_lst[p_nbr_child]) {
                  nbr_path[p2n][nbr_idx].p_nbr_child = p_nbr_child;
                  break;
                }
              }
              break;
            }
          }

        }
      }
      return nbr_path;
    }();
    static const auto reverse_nbr_idx = []() {
      Matrix<Integer> reverse_nbr_idx(MAX_CHILD, MAX_NBRS);
      const auto parent = (Morton<DIM>{}).Children()[0].Children()[MAX_CHILD-1];
      for (Integer p2n = 0; p2n < MAX_CHILD; p2n++) {
        const auto n0 = parent.Children()[p2n];
        const auto nlst = n0.NbrList(n0.Depth(), Periodicity::NONE);
        for (Integer nbr_idx = 0; nbr_idx < MAX_NBRS; nbr_idx++) {
          const auto nbr_nlst = nlst[nbr_idx].NbrList(n0.Depth(), Periodicity::NONE);
          for (Integer i = 0; i < MAX_NBRS; i++) {
            if (nbr_nlst[i] == n0) {
              reverse_nbr_idx[p2n][nbr_idx] = i;
              break;
            }
          }

        }
      }
      return reverse_nbr_idx;
    }();

    Vector<Morton<DIM>> node_mid_orig;
    Long start_idx_orig, end_idx_orig;
    if (mins.Dim()) { // Set start_idx_orig, end_idx_orig
      start_idx_orig = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
      end_idx_orig = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
      node_mid_orig.ReInit(end_idx_orig - start_idx_orig, node_mid.begin() + start_idx_orig);
    } else {
      start_idx_orig = 0;
      end_idx_orig = 0;
    }

    const auto first_child = [](const Morton<DIM>& m) {
      return m.DFD(m.Depth()+1);
    };
    const auto complete_tree = [&first_child](Vector<Morton<DIM>>& mid_lst, const Morton<DIM>& mid_begin, const Morton<DIM>& mid_end) {
      // Fill in the nodes for a completed tree in the interval
      // [mid_begin, mid_end) and append to mid_lst.
      // Returns mid_end.
      SCTL_ASSERT(mid_begin <= mid_end);
      Morton<DIM> mid_iter = mid_begin;
      while (mid_iter != mid_end) {
        mid_lst.PushBack(mid_iter);
        if (mid_iter.isAncestor(mid_end)) mid_iter = first_child(mid_iter);
        else mid_iter = mid_iter.Next();
      }
      return mid_end;
    };

    const auto set_node_attr = [](Vector<NodeAttr>& node_attr, const Vector<Morton<DIM>>& node_mid, const Vector<Morton<DIM>>& mins, const Integer rank) { // Set node_attr
      const Integer np = (Integer)mins.Dim();
      Morton<DIM> m0 = (rank      ? mins[rank]   : Morton<DIM>()       );
      Morton<DIM> m1 = (rank+1<np ? mins[rank+1] : Morton<DIM>().Next());
      const Long Nnodes = node_mid.Dim();
      node_attr.ReInit(Nnodes);
      #pragma omp parallel for schedule(static)
      for (Long i = 0; i < Nnodes; i++) {
        node_attr[i].Leaf = !(i+1<Nnodes && node_mid[i].isAncestor(node_mid[i+1]));
        node_attr[i].Ghost = (node_mid[i] < m0 || node_mid[i] >= m1);
      }
    };
    const auto set_node_lst = [](Vector<NodeLists>& node_lst, const Vector<Morton<DIM>>& node_mid, const Periodicity periodicity) { // Set parent, child, nbr
      const Long Nnodes = node_mid.Dim();
      node_lst.ReInit(Nnodes);
      { // initialize root
        SCTL_ASSERT(Nnodes);
        node_lst[0].p2n = -1;
        node_lst[0].parent = -1;
        for (Integer j = 0; j < MAX_CHILD; j++) node_lst[0].child[j] = -1;
      }

      #pragma omp parallel
      { // Initialize node_lst (parent, child, p2n) for each thread's chunk
        const Integer tid = SCTL_GET_THREAD_NUM();
        const Integer nthreads = SCTL_GET_NUM_THREADS();
        const auto i_begin = (Nnodes *  tid     ) / nthreads;
        const auto i_end   = (Nnodes * (tid + 1)) / nthreads;
        const auto i_prev  = (Nnodes * (tid - 1)) / nthreads;
        const auto m_end = i_end < Nnodes ? node_mid[i_end] : Morton<DIM>().Next();

        ScratchBuf<Long> ancestors(Morton<DIM>::MAX_DEPTH+1);
        const auto& n0 = node_mid[i_begin];
        for (Long depth = 0; depth < n0.Depth(); depth++) {
          const auto ancestor = n0.Ancestor(depth);
          const Long ancestor_idx = std::lower_bound(node_mid.begin(), node_mid.end(), ancestor) - node_mid.begin();
          if (ancestor_idx >= i_prev) for (Integer j = 0; j < MAX_CHILD; j++) node_lst[ancestor_idx].child[j] = -1;
          ancestors[depth] = ancestor_idx;
        }
        #pragma omp barrier

        for (Long i = i_begin; i < i_end; i++) { // Set node_lst
          const Integer depth = node_mid[i].Depth();
          ancestors[depth] = i;
          if (depth) {
            const Integer p2n = node_mid[i].Path2Node();
            const Long p = ancestors[depth-1];
            node_lst[p].child[p2n] = i;
            node_lst[i].parent = p;
            node_lst[i].p2n = p2n;
            if (!node_mid[i].isAncestor(m_end)) for (Integer j = 0; j < MAX_CHILD; j++) node_lst[i].child[j] = -1;
            if (0) { // for debugging
              const auto parent_mid = node_mid[i].Ancestor(depth-1);
              const auto sibling_nds = parent_mid.Children();
              SCTL_ASSERT(sibling_nds[node_lst[i].p2n] == node_mid[i]);
              SCTL_ASSERT(node_lst[i].parent == std::lower_bound(node_mid.begin(), node_mid.end(), parent_mid) - node_mid.begin());
              SCTL_ASSERT(node_lst[p].child[p2n] == std::lower_bound(node_mid.begin(), node_mid.end(), node_mid[i]) - node_mid.begin());
              SCTL_ASSERT(node_lst[i].p2n == std::lower_bound(sibling_nds.begin(), sibling_nds.end(), node_mid[i]) - sibling_nds.begin());
            }
          }
        }
      }

      { // Set neighbor list
        { // Set root neighbors
          const auto& n0 = node_mid[0];
          const auto nlst = n0.NbrList(n0.Depth(), periodicity);
          static_assert(nlst.size() == MAX_NBRS);
          for (Long k = 0; k < MAX_NBRS; k++) {
            if (nlst[k].Depth() != Morton<DIM>::INVALID_DEPTH) node_lst[0].nbr[k] = 0;
            else node_lst[0].nbr[k] = -1;
          }
        }

        const auto set_nbrs = [&node_lst,&node_mid](Long idx) {
          const auto& n0_depth = node_mid[idx].Depth();
          SCTL_ASSERT(n0_depth != Morton<DIM>::INVALID_DEPTH);
          if (n0_depth == 0) return;

          auto& n0_lsts = node_lst[idx];
          SCTL_ASSERT(n0_lsts.parent >= 0);

          for (Integer nbd_idx = 0; nbd_idx < MAX_NBRS; nbd_idx++) {
            const NbrPath& path = nbr_path[n0_lsts.p2n][nbd_idx];
            const Long parent_nbr_idx = node_lst[n0_lsts.parent].nbr[path.p_nbr];
            if (parent_nbr_idx >= 0) n0_lsts.nbr[nbd_idx] = node_lst[parent_nbr_idx].child[path.p_nbr_child];
            else n0_lsts.nbr[nbd_idx] = -1;
          }
        };

        constexpr Integer stride = ((Morton<DIM>::MAX_DEPTH+1) + 7) & ~(Integer)7; // MAX_DEPTH+1 rounded up to a cache line (8 Longs) to prevent false-sharing
        const Integer nthreads = SCTL_GET_MAX_THREADS();
        ScratchBuf<Long> ancestors(nthreads * stride);
        #pragma omp parallel for schedule(static)
        for (Integer tid = 0; tid < nthreads; tid++) { // Set ancestors
          const Long idx0 = (Nnodes * tid) / nthreads;
          for (Long idx = idx0; node_mid[idx].Depth() > 0; idx = node_lst[idx].parent) {
            ancestors[tid*stride + node_mid[idx].Depth()] = idx;
          }
        }
        for (Integer tid = 0; tid < nthreads; tid++) { // Set neighbor list for ancestors shared by multiple threads
          const Long idx0 = (Nnodes *  tid     ) / nthreads;
          const Long idx1 = (Nnodes * (tid + 1)) / nthreads;
          const Long idx_ = (Nnodes * (tid - 1)) / nthreads;
          const Integer d0 = (tid > 0) ? node_mid[idx0].CommonAncestor(node_mid[idx_]).Depth() : 0;
          const Integer d1 = (tid + 1 < nthreads) ? node_mid[idx0].CommonAncestor(node_mid[idx1]).Depth() : 0;
          for (Integer d = std::max<Integer>(1, d0); d < d1; d++) set_nbrs(ancestors[tid*stride + d]);
        }
        #pragma omp parallel num_threads(nthreads)
        { // Set neighbor list for the rest of the nodes in each thread's chunk
          SCTL_ASSERT(SCTL_GET_NUM_THREADS() == nthreads);
          const Integer tid = SCTL_GET_THREAD_NUM();
          const Long idx0 = (Nnodes * tid) / nthreads;
          const Long idx1 = (Nnodes * (tid + 1)) / nthreads;
          { // Set neighbors for ancestors of each thread's first node
            const Long idx_ = (Nnodes * (tid - 1)) / nthreads;
            const Integer d0 = (tid > 0) ? node_mid[idx0].CommonAncestor(node_mid[idx_]).Depth() : 0;
            const Integer d1 = (tid + 1 < nthreads) ? node_mid[idx0].CommonAncestor(node_mid[idx1]).Depth() : 0;
            const Integer d2 = node_mid[idx0].Depth();
            for (Integer d = std::max<Integer>(1, std::max(d0,d1)); d < d2; d++) {
              set_nbrs(ancestors[tid*stride + d]);
            }
          }
          #pragma omp barrier
          for (Long i = idx0; i < idx1; i++) {
            if (tid + 1 < nthreads && node_mid[i].isAncestor(node_mid[idx1])) continue;
            set_nbrs(i);
          }
        }
      }
    };

    { // Build linear tree (node_mid) and set mins
      const auto split_anchor = [](const MortonCode<DIM>& m0, const MortonCode<DIM>& m1) {
        const uint8_t d = m0.CommonAncestor(m1).Depth();
        return m0.Ancestor(std::min<uint8_t>(Morton<DIM>::MAX_DEPTH,d+1)).Next();
      };

      Vector<MortonCode<DIM>> pt_mid;
      { // Construct sorted pt_mid
        Long Npt = coord.Dim() / DIM;
        ScratchBuf<MortonCode<DIM>> pt_mid_buf(Npt);
        Vector<MortonCode<DIM>> pt_mid_(pt_mid_buf);
        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < Npt; i++) {
          pt_mid_[i] = MortonCode<DIM>(&coord[i*DIM]);
        }
        comm.SampleSort(pt_mid_, pt_mid);
      }

      { // Update M = global_min(pt_mid.Dim(), M)
        StaticArray<Long,1> recv_buf, send_buf{std::min(pt_mid.Dim(), M)};
        comm.Allreduce<Long>(send_buf, recv_buf, 1, CommOp::MIN);
        M = recv_buf[0];
      }
      SCTL_ASSERT(M > 0);

      ScratchBuf<MortonCode<DIM>> pt_mid_(pt_mid.Dim() + 2*M);
      if (np > 1) { // Set mins, pt_mid <-- [M points from rank-1; pt_mid; M points from rank+1]
        Long send_size0 = (rank+1<np ? M : 0);
        Long send_size1 = (rank  > 0 ? M : 0);
        Long recv_size0 = (rank  > 0 ? M : 0);
        Long recv_size1 = (rank+1<np ? M : 0);
        SCTL_ASSERT(recv_size0 + pt_mid.Dim() + recv_size1 <= pt_mid_.Dim());
        omp_par::memcpy(pt_mid_.begin() + recv_size0, pt_mid.begin(), pt_mid.Dim());

        auto recv_req0 = comm.Irecv(pt_mid_.begin(), recv_size0, (rank+np-1)%np, 0);
        auto recv_req1 = comm.Irecv(pt_mid_.begin() + recv_size0 + pt_mid.Dim(), recv_size1, (rank+1)%np, 1);
        auto send_req0 = comm.Issend(pt_mid.begin() + pt_mid.Dim() - send_size0, send_size0, (rank+1)%np, 0);
        auto send_req1 = comm.Issend(pt_mid.begin(), send_size1, (rank+np-1)%np, 1);
        comm.Wait(std::move(recv_req0));
        comm.Wait(std::move(recv_req1));
        comm.Wait(std::move(send_req0));
        comm.Wait(std::move(send_req1));
        const Long Npts_buf = pt_mid.Dim() + recv_size0 + recv_size1;

        { // Set mins
          mins.ReInit(np);
          SCTL_ASSERT(Npts_buf > M);
          const Morton<DIM> m0 = (!rank ? Morton<DIM>{} :  split_anchor(pt_mid_[0], pt_mid_[M]) );
          comm.Allgather(Ptr2ConstItr<Morton<DIM>>(&m0,1), 1, mins.begin(), 1);
        }
        const Long idx0 = std::lower_bound(pt_mid_.begin(), pt_mid_.begin() + Npts_buf, mins[rank].mid) - pt_mid_.begin();
        const Long idx1 = std::lower_bound(pt_mid_.begin(), pt_mid_.begin() + Npts_buf, (rank==np-1) ? Morton<DIM>().Next().mid : mins[rank+1].mid) - pt_mid_.begin();
        pt_mid.ReInit(idx1-idx0, pt_mid_.begin()+idx0, false);
      } else {
        mins.ReInit(1);
        mins[0] = Morton<DIM>{};
      }

      { // Build linear MortonID tree from pt_mid (chunked parallel walk)
        const Long N = pt_mid.Dim();

        // Cap threads so each chunk has well over M particles (so begin + M stays in-bounds).
        const Integer max_threads = SCTL_GET_MAX_THREADS();
        const Long    min_chunk   = std::max<Long>(4 * M + 1, 1024);
        const Integer nthreads    = std::clamp<Integer>(static_cast<Integer>(N / min_chunk), 1, max_threads);

        // Upper bound: ~(MAX_DEPTH+1) nodes/leaf, chunk_size/M leaves/chunk, 4x slack.
        const Long chunk_size_max = (N + nthreads - 1) / nthreads;
        const Long max_emits      = 4 * chunk_size_max * (Morton<DIM>::MAX_DEPTH + 1) / std::max<Long>(1, M) + 4 * (Morton<DIM>::MAX_DEPTH + 1) * (Long(1) << DIM) + 16;

        struct alignas(64) PaddedLong { Long v; char pad[64 - sizeof(Long)]; };  // avoid false sharing
        ScratchBuf<PaddedLong> local_sizes(nthreads);
        ScratchBuf<Long>       offsets(nthreads);

        #pragma omp parallel num_threads(nthreads)
        {
          SCTL_ASSERT(SCTL_GET_NUM_THREADS() == nthreads);
          const Integer tid      = SCTL_GET_THREAD_NUM();
          const Long    begin_t  = (N *  tid     ) / nthreads;
          const Long    end_t    = (N * (tid + 1)) / nthreads;
          const bool    is_first = (tid == 0);
          const bool    is_last  = (tid == nthreads - 1);

          const Morton<DIM> start_anchor = is_first ? mins[rank] : split_anchor(pt_mid[begin_t], pt_mid[begin_t+M]);
          const Morton<DIM> end_anchor   = is_last  ? (rank+1<np ? mins[rank+1] : Morton<DIM>().Next()) : split_anchor(pt_mid[end_t], pt_mid[end_t+M]);
          const Long idx_start = is_first ? 0 : std::lower_bound(pt_mid.begin() + begin_t, pt_mid.begin() + begin_t + M, start_anchor.mid) - pt_mid.begin();
          const Long idx_end   = is_last  ? N : std::lower_bound(pt_mid.begin() + end_t,   pt_mid.begin() + end_t   + M, end_anchor.mid)   - pt_mid.begin();

          // NUMA-local per-thread scratch (first-touched on this thread's node).
          ScratchBuf<Morton<DIM>> buf(max_emits);
          Long count = 0;

          if (is_first) {
            Morton<DIM> m0{};
            while (m0 != start_anchor) {
              buf[count++] = m0;
              if (m0.isAncestor(start_anchor)) m0 = first_child(m0);
              else                             m0 = m0.Next();
            }
          }

          Morton<DIM> m0     = start_anchor;
          Long        pt_idx = idx_start;
          while (pt_idx < idx_end - M) {
            const Morton<DIM> m_ = split_anchor(pt_mid[pt_idx], pt_mid[pt_idx+M]);
            while (m0 != m_) {
              buf[count++] = m0;
              if (m0.isAncestor(m_)) m0 = first_child(m0);
              else                   m0 = m0.Next();
            }
            pt_idx = std::lower_bound(pt_mid.begin() + pt_idx, pt_mid.begin() + pt_idx + M, m0.mid) - pt_mid.begin();
            if (pt_idx < idx_end && pt_mid[pt_idx] < m0.mid) {
              pt_idx = std::lower_bound(pt_mid.begin() + pt_idx, pt_mid.begin() + idx_end, m0.mid) - pt_mid.begin();
            }
          }
          while (m0 != end_anchor) {  // tail to end_anchor / sentinel
            buf[count++] = m0;
            if (m0.isAncestor(end_anchor)) m0 = first_child(m0);
            else                           m0 = m0.Next();
          }

          if (is_last) {
            const Morton<DIM> end_anchor = Morton<DIM>().Next();
            while (m0 != end_anchor) {  // tail to end_anchor / sentinel
              buf[count++] = m0;
              if (m0.isAncestor(end_anchor)) m0 = first_child(m0);
              else                           m0 = m0.Next();
            }
          }
          local_sizes[tid].v = count;

          #pragma omp barrier
          #pragma omp single
          {
            Long total = 0;
            for (Integer s = 0; s < nthreads; ++s) { offsets[s] = total; total += local_sizes[s].v; }
            node_mid.ReInit(total);
          }

          std::copy(buf.begin(), buf.begin() + count, node_mid.begin() + offsets[tid]);
        }
      }
    }

    if (balance21) { // 2:1 balance refinement
      const Integer nthreads = SCTL_GET_MAX_THREADS();

      Vector<Morton<DIM>> parent_mid;
      { // add balancing Morton IDs
        static std::pair<Matrix<Integer>,Vector<Integer>> balance21_p_nbrs_precomp = []() { // for each p2n, list of parent's neighbors that must exist to be 2:1 balanced
          Matrix<Integer> p_nbr_lst(MAX_CHILD, MAX_NBRS);
          Vector<Integer> p_nbr_cnt(MAX_CHILD);
          p_nbr_cnt = 0;
          for (Integer i = 0; i < MAX_CHILD; i++) {
            std::set<Integer> p_nbr_set;
            for (Integer j = 0; j < MAX_NBRS; j++) {
              p_nbr_set.insert(nbr_path[i][j].p_nbr);
            }
            for (const auto &n : p_nbr_set) {
              p_nbr_lst[i][p_nbr_cnt[i]++] = n;
            }
          }
          return std::make_pair(p_nbr_lst, p_nbr_cnt);
        }();
        const Matrix<Integer> &p_nbr_lst = balance21_p_nbrs_precomp.first; // MAX_CHILD x MAX_NBRS
        const Vector<Integer> &p_nbr_cnt = balance21_p_nbrs_precomp.second; // MAX_CHILD

        ScratchBuf<Vector<Morton<DIM>>> parent_mid_t(nthreads);
        #pragma omp parallel num_threads(nthreads)
        { // build list of parent nodes parent_mid
          const Integer tid = SCTL_GET_THREAD_NUM();
          const Integer nt = SCTL_GET_NUM_THREADS();

          const Long Nnodes = node_mid.Dim();
          const Long idx_start = (Nnodes *  tid     ) / nt;
          const Long idx_end   = (Nnodes * (tid + 1)) / nt;

          Vector<Morton<DIM>>& parent_mid_ = parent_mid_t[tid];
          for (Long i = idx_start; i < idx_end; ++i) {
            if (i+1<Nnodes && node_mid[i+1].Depth() == node_mid[i].Depth()+1) parent_mid_.PushBack(node_mid[i]);
          }

          #pragma omp barrier
          Long dsp = 0;
          for (Long i = 0; i < tid; i++) dsp += parent_mid_t[i].Dim();
          if (tid == nt-1 && parent_mid.Dim() != dsp + parent_mid_.Dim()) parent_mid.ReInit(dsp + parent_mid_.Dim());
          #pragma omp barrier
          std::copy(parent_mid_.begin(), parent_mid_.end(), parent_mid.begin() + dsp);
        }
        const Long Nnodes = parent_mid.Dim();

        struct TreeNode {
          Morton<DIM> m;
          TreeNode* parent;
          TreeNode* child[MAX_CHILD];
          TreeNode* nbr[MAX_NBRS];
          Integer flags;
        };
        ScratchBuf<TreeNode> ptree(Nnodes);
        if (Nnodes) { // Init root
          ptree[0].m = Morton<DIM>{};
          ptree[0].parent = nullptr;
          for (Integer i = 0; i < MAX_CHILD; i++) ptree[0].child[i] = nullptr;
          const auto nbr_lst = Morton<DIM>().NbrList(0, periodicity);
          for (Integer i = 0; i < MAX_NBRS; i++) ptree[0].nbr[i] = (nbr_lst[i].Depth() == Morton<DIM>::INVALID_DEPTH ? nullptr : &ptree[0]);
          ptree[0].flags = 0;
        }

        const auto set_nbrs = [](TreeNode& node) {
          const Integer p2n = node.m.Path2Node();
          for (Integer k = 0; k < MAX_NBRS; k++) {
            const auto nbr_path_ = nbr_path[p2n][k];
            const TreeNode* const p_nbr_ptr = node.parent->nbr[nbr_path_.p_nbr];
            if (p_nbr_ptr) {
              TreeNode* const nbr_ptr = p_nbr_ptr->child[nbr_path_.p_nbr_child];
              if (nbr_ptr) node.nbr[k] = nbr_ptr;
            }
          }
        };
        const auto set_ancestor_nbrs = [](TreeNode& root, const Morton<DIM>& min_m, const Morton<DIM>& m0) {
          // excluding root and m0
          const Integer d0 = m0.Depth();
          TreeNode* ancestor_nbr[MAX_DEPTH+1][MAX_NBRS];
          if (d0) for (Integer i = 0; i < MAX_NBRS; i++) ancestor_nbr[0][i] = root.nbr[i];

          TreeNode* node = &root;
          for (Integer d = 1; d < d0; d++) { // Set the neighbor pointers for the partition ancestors
            const Integer p2n = m0.Ancestor(d).Path2Node();
            node = node->child[p2n];
            for (Integer k = 0; k < MAX_NBRS; k++) {
              ancestor_nbr[d][k] = nullptr;
              const auto nbr_path_ = nbr_path[p2n][k];
              const TreeNode* const p_nbr_ptr = ancestor_nbr[d-1][nbr_path_.p_nbr];
              if (p_nbr_ptr) {
                TreeNode* const nbr_ptr = p_nbr_ptr->child[nbr_path_.p_nbr_child];
                if (nbr_ptr) ancestor_nbr[d][k] = nbr_ptr;
              }
            }
            if (min_m <= node->m) {
              for (Integer k = 0; k < MAX_NBRS; k++) node->nbr[k] = ancestor_nbr[d][k];
              node->flags = 0;
            }
          }
        };

        ScratchBuf<Vector<Morton<DIM>>*> shared_new_pnodes(nthreads);
        ScratchBuf<Long> shared_pnode_cnt(nthreads), shared_pnode_dsp(nthreads);
        #pragma omp parallel num_threads(nthreads)
        if (Nnodes) { // Build ptree, balance it, and add nodes to parent_mid
          const Integer tid = SCTL_GET_THREAD_NUM();
          const Integer nt = SCTL_GET_NUM_THREADS();
          const Long idx0 = Nnodes * tid / nt;
          const Long idx1 = Nnodes * (tid+1) / nt;
          const Long idx_ = Nnodes * (tid-1) / nt;
          Long local_pnode_cnt = idx1 - idx0;

          Vector<TreeNode*> new_node_lst(idx1-idx0);
          { // Build parent tree ptree from parent_mid, and set parent, child pointers; add nodes to new_node_lst
            Long ancestors[MAX_DEPTH+1];
            ancestors[0] = 0;
            const Long d0 = (idx0 < Nnodes) ? parent_mid[idx0].Depth() : 0;
            for (Integer d = 1; d < d0; d++) { // Set ancestors, and initialize their child pointers to nullptr
              const Long i = std::lower_bound(parent_mid.begin(), parent_mid.end(), parent_mid[idx0].Ancestor(d)) - parent_mid.begin();
              ancestors[d] = i;

              if (i >= idx_) {
                auto& node = ptree[i];
                for (Integer k = 0; k < MAX_CHILD; k++) node.child[k] = nullptr;
              }
            }
            #pragma omp barrier
            if (idx0 == 0 && idx0 < idx1) new_node_lst[0] = &ptree[0];
            for (Long i = std::max<Long>(1,idx0); i < idx1; i++) { // Set parent and child pointers
              const auto m = parent_mid[i];
              const Integer d = m.Depth();
              ancestors[d] = i;

              auto& node = ptree[i];
              new_node_lst[i-idx0] = &node;
              node.m = m;
              node.parent = &ptree[ancestors[d-1]];
              node.parent->child[m.Path2Node()] = &node;
              if (idx1 == Nnodes || !m.isAncestor(parent_mid[idx1])) {
                for (Integer k = 0; k < MAX_CHILD; k++) node.child[k] = nullptr;
                for (Integer k = 0; k < MAX_NBRS; k++) node.nbr[k] = nullptr;
              }
            }
          }
          #pragma omp barrier

          { // Set nbr pointers
            if (idx0 < Nnodes) set_ancestor_nbrs(ptree[0], (idx_ >= 0 ? parent_mid[idx_] : Morton<DIM>()), parent_mid[idx0]);
            #pragma omp barrier
            for (Long i = std::max<Long>(1,idx0); i < idx1; i++) { // Set the neighbor pointers
              TreeNode& node = ptree[i];
              node.flags = 1;
              if (idx1 < Nnodes && node.m.isAncestor(parent_mid[idx1])) continue; // do not touch the ancestor nodes, other threads may be reading them
              set_nbrs(node);
            }
          }

          Vector<Morton<DIM>> new_mid;
          NodeArena<TreeNode> new_pnodes_;
          for (Integer iter = 0; iter <= MAX_DEPTH; iter++) {
            new_mid.ReInit(0);
            for (const auto node : new_node_lst) { // Collect missing parent-neighbors into new_mid
              if (node->m.Depth() <= 1) continue;
              const Integer p2n = node->m.Path2Node();
              const auto p_nbr_lst_ = p_nbr_lst[p2n];
              const Integer p_nbr_cnt_ = p_nbr_cnt[p2n];

              Integer tmp_buf[MAX_NBRS];
              Integer tmp_buf_cnt = 0;
              for (Integer k = 0; k < p_nbr_cnt_; k++) {
                if (node->parent->nbr[p_nbr_lst_[k]] == nullptr)
                  tmp_buf[tmp_buf_cnt++] = p_nbr_lst_[k];
              }
              if (tmp_buf_cnt) {
                const auto p_nbrs = node->m.NbrList(node->m.Depth()-1, periodicity);
                for (Integer k = 0; k < tmp_buf_cnt; k++)
                  if (p_nbrs[tmp_buf[k]].Depth() != Morton<DIM>::INVALID_DEPTH)
                    new_mid.PushBack(p_nbrs[tmp_buf[k]]);
              }
            }
            std::sort(new_mid.begin(), new_mid.end());
            shared_new_pnodes[tid] = &new_mid;
            #pragma omp barrier

            bool early_exit = true;
            for (Integer t = 0; t < nt; t++) {
              if (shared_new_pnodes[t]->Dim()) early_exit = false;
            }
            if (early_exit) break;

            { // Add new_mid to ptree, set their parent, child pointers, rebuild new_node_lst
              new_node_lst.ReInit(0);
              ScratchBuf<Long> src0(nt), src1(nt);
              { // bounds of this thread's owned range in each source list
                const Morton<DIM> r0 = parent_mid[Nnodes * tid / nt];
                const Morton<DIM> r1 = (tid+1 < nt ? parent_mid[Nnodes * (tid+1) / nt] : Morton<DIM>().Next());
                for (Integer t = 0; t < nt; t++) {
                  const Vector<Morton<DIM>>& lst = *shared_new_pnodes[t];
                  src0[t] = std::lower_bound(lst.begin(), lst.end(), r0) - lst.begin();
                  src1[t] = std::lower_bound(lst.begin(), lst.end(), r1) - lst.begin();
                }
              }
              for (Integer dt = 0; dt < nt; dt++) { // Add new nodes to ptree
                const Integer t = (tid + dt) % nt;
                const Vector<Morton<DIM>>& src = *shared_new_pnodes[t];
                for (Long i = src0[t]; i < src1[t]; i++) {
                  if (i > src0[t] && src[i] == src[i-1]) continue;

                  const auto m = src[i];
                  const Integer d0 = m.Depth();
                  TreeNode* parent = &ptree[0];
                  for (Integer d = 1; d <= d0; d++) {
                    const auto m_d = m.Ancestor(d);
                    const Integer p2n = m_d.Path2Node();
                    TreeNode* node = parent->child[p2n];
                    if (node == nullptr) {
                      node = &new_pnodes_.Alloc();
                      new_node_lst.PushBack(node);
                      node->m = m.Ancestor(d);
                      node->parent = parent;
                      for (Integer k = 0; k < MAX_CHILD; k++) node->child[k] = nullptr;
                      for (Integer k = 0; k < MAX_NBRS; k++) node->nbr[k] = nullptr;
                      parent->child[p2n] = node;
                      node->flags = iter+1;
                    }
                    parent = node;
                  }
                }
              }
            }
            local_pnode_cnt += new_node_lst.Dim();
            #pragma omp barrier

            { // Set nbr pointers
              if (idx0 < Nnodes) set_ancestor_nbrs(ptree[0], (idx_ >= 0 ? parent_mid[idx_] : Morton<DIM>()), parent_mid[idx0]);
              #pragma omp barrier
              for (auto& node : new_node_lst) {
                Integer cnt = 0;
                TreeNode *node_ptr = node, *ancestor[MAX_DEPTH+1];
                while (node_ptr && node_ptr->flags == iter+1) {
                  ancestor[cnt++] = node_ptr;
                  node_ptr = node_ptr->parent;
                }
                for (Integer i = cnt-1; i >= 0; i--) {
                  set_nbrs(*ancestor[i]);
                  ancestor[i]->flags = iter+2;
                }
              }
              #pragma omp barrier
              for (auto& node : new_node_lst) {
                for (Integer i = 0; i < MAX_NBRS; i++) {
                  TreeNode* nbr = node->nbr[i];
                  if (nbr && nbr->flags != iter+2) {
                    const Integer p2n = node->m.Path2Node();
                    nbr->nbr[reverse_nbr_idx[p2n][i]] = node;
                  }
                }
              }
            }
            #pragma omp barrier
          }

          static constexpr Integer FLAG_MINS_ANC = -1;  // ancestor of a min: exclude from parent_mid
          { // Set exclude flag for ancestors of mins
            Long local_excl = 0;
            const Morton<DIM> b0 = parent_mid[idx0];
            const Morton<DIM> b1 = (idx1 < Nnodes ? parent_mid[idx1] : Morton<DIM>().Next());
            const Long r0 = std::lower_bound(mins.begin(), mins.end(), b0) - mins.begin();
            const Long r1 = std::lower_bound(mins.begin(), mins.end(), b1) - mins.begin();
            for (Long r = r0; r < std::min<Long>(r1+1, np); r++) {
              TreeNode* node = &ptree[0];
              const Integer d0 = mins[r].Depth();
              for (Integer d = 0; d < d0 && node; d++) {
                if (node->m >= b0 && node->m < b1 && node->flags != FLAG_MINS_ANC) { // chains of consecutive mins share prefixes: count once
                  node->flags = FLAG_MINS_ANC;
                  local_excl++;
                }
                const Integer p2n = mins[r].Ancestor(d+1).Path2Node();
                node = node->child[p2n];
              }
            }
            shared_pnode_cnt[tid] = local_pnode_cnt - local_excl;
            #pragma omp barrier
          }

          #pragma omp single
          { // Resize parent_mid
            std::exclusive_scan(shared_pnode_cnt.begin(), shared_pnode_cnt.begin()+nt, shared_pnode_dsp.begin(), Long(0));
            parent_mid.ReInit(shared_pnode_dsp[nt-1] + shared_pnode_cnt[nt-1]);
          }

          if (idx0 < ptree.Dim()) { // preorder traversal to add local nodes to parent_mid
            TreeNode* node = &ptree[idx0];
            const Morton<DIM> m_end = (idx1 < ptree.Dim() ? ptree[idx1].m : Morton<DIM>().Next());
            Long out = shared_pnode_dsp[tid];
            while (node->m < m_end) {
              if (node->flags != FLAG_MINS_ANC) parent_mid[out++] = node->m;

              TreeNode* next = nullptr;
              for (Integer k = 0; k < MAX_CHILD; k++) { // descend to first child
                if (node->child[k]) { next = node->child[k]; break; }
              }
              while (next == nullptr && node->parent) { // no child, ascend to next sibling
                TreeNode* const parent = node->parent;
                const Integer p2n = node->m.Path2Node();
                for (Integer k = p2n+1; k < MAX_CHILD; k++) {
                  if (parent->child[k]) { next = parent->child[k]; break; }
                }
                node = parent;
              }
              if (next == nullptr) break; // ascended past root: end of tree
              node = next;
            }
            SCTL_ASSERT(out == shared_pnode_dsp[tid] + shared_pnode_cnt[tid]);
          }
        }
      }

      { // global_sort parent_mid and remove duplicates
        Vector<Morton<DIM>> parent_mid_sorted;
        comm.SampleSort(parent_mid, parent_mid_sorted, mins[comm.Rank()]);

        ScratchBuf<Long> cnt(nthreads), dsp(nthreads);
        if (parent_mid_sorted.Dim()) { // remove duplicates
          #pragma omp parallel num_threads(nthreads)
          {
            const Integer nt = SCTL_GET_NUM_THREADS();
            const Integer tid = SCTL_GET_THREAD_NUM();
            const Long start = 1+((parent_mid_sorted.Dim()-1) *  tid     ) / nt;
            const Long end   = 1+((parent_mid_sorted.Dim()-1) * (tid + 1)) / nt;

            Long loc_cnt = 0;
            for (Long j = start; j < end; j++) {
              if (parent_mid_sorted[j]!=parent_mid_sorted[j-1]) loc_cnt++;
            }
            cnt[tid] = loc_cnt;

            #pragma omp barrier
            #pragma omp single
            {
              std::exclusive_scan(cnt.begin(), cnt.begin()+nt, dsp.begin(), Long(1));
              parent_mid.ReInit(dsp[nt-1] + cnt[nt-1]);
              parent_mid[0] = parent_mid_sorted[0];
            } // implicit barrier at end of single

            Long loc_idx = dsp[tid];
            for (Long j = start; j < end; j++) {
              if (parent_mid_sorted[j]!=parent_mid_sorted[j-1]) parent_mid[loc_idx++] = parent_mid_sorted[j];
            }
          }
        } else {
          parent_mid.ReInit(0);
        }
      }

      if (parent_mid.Dim()) { // add children of parent_mid
        const Integer nthreads = SCTL_GET_MAX_THREADS();
        ScratchBuf<Long> local_cnt(nthreads), local_dsp(nthreads);
        #pragma omp parallel num_threads(nthreads)
        {
          const Integer nt = SCTL_GET_NUM_THREADS();
          const Integer tid = SCTL_GET_THREAD_NUM();
          const Long start = (parent_mid.Dim() *  tid     ) / nt;
          const Long end   = (parent_mid.Dim() * (tid + 1)) / nt;

          Vector<Morton<DIM>> node_mid_t; // local node_list TODO: replace by ScratchBuf
          if (tid == 0) {
            complete_tree(node_mid_t, Morton<DIM>(), mins[rank]);
            if (parent_mid.Dim()) complete_tree(node_mid_t, mins[rank], first_child(parent_mid[0]));
          }
          if (start < parent_mid.Dim()) {
            for (Long i = start; i < end-1; i++) complete_tree(node_mid_t, first_child(parent_mid[i]), first_child(parent_mid[i+1]));
            if (start <= end-1 && end < parent_mid.Dim()) complete_tree(node_mid_t, first_child(parent_mid[end-1]), first_child(parent_mid[end]));
          }
          if (tid == nt-1) {
            const auto m0 = parent_mid.Dim() ? first_child(parent_mid[parent_mid.Dim()-1]) : mins[rank];
            const auto m1 = (rank+1 < np) ? mins[rank+1] : Morton<DIM>().Next();
            complete_tree(node_mid_t, m0, m1);
            complete_tree(node_mid_t, m1, Morton<DIM>().Next());
          }
          local_cnt[tid] = node_mid_t.Dim();

          #pragma omp barrier
          #pragma omp single
          { // local_dsp <-- exclusive_scan(local_cnt), resize node_mid
            std::exclusive_scan(local_cnt.begin(), local_cnt.begin()+nt, local_dsp.begin(), Long(0));
            node_mid.ReInit(local_dsp[nt-1] + local_cnt[nt-1]);
          } // implicit barrier at end of single

          std::copy(node_mid_t.begin(), node_mid_t.end(), node_mid.begin() + local_dsp[tid]);
        }
      }
    }

    if (np > 1 && halo_size >= 0) { // Add place-holder for ghost nodes
      Long start_idx, end_idx;
      { // Set start_idx, end_idx
        start_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
        end_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
      }
      { // Set user_mid, user_cnt
        // user_mid: Morton IDs of nodes that need to be sent
        // user_cnt: number of nodes to be sent to each rank

        const Integer nthreads = SCTL_GET_MAX_THREADS();
        ScratchBuf<Vector<std::pair<Long,Morton<DIM>>>> user_node_lst_t_(nthreads);
        #pragma omp parallel num_threads(nthreads)
        { // Set user_node_lst_t_[tid]
          const Integer tid = SCTL_GET_THREAD_NUM();
          const Integer nthreads = SCTL_GET_NUM_THREADS();
          Vector<std::pair<Long,Morton<DIM>>>& user_node_lst_t = user_node_lst_t_[tid];

          Long user_procs_cnt = 0;
          ScratchBuf<Long> user_procs(np);
          ScratchBuf<bool> user_procs_flag(np);
          for (Long i = 0; i < np; i++) user_procs_flag[i] = false;
          user_procs_flag[rank] = true; // skip self
          const Long start_idx_t = start_idx + (end_idx - start_idx) *  tid      / nthreads;
          const Long end_idx_t   = start_idx + (end_idx - start_idx) * (tid + 1) / nthreads;
          for (Long i = start_idx_t; i < end_idx_t; i++) {
            Morton<DIM> m0 = node_mid[i];
            Integer d0 = m0.Depth();
            const auto nlst = m0.NbrList(std::max<Integer>(d0-halo_size,0), periodicity);
            for (const auto& m : nlst) {
              if (m.Depth() != Morton<DIM>::INVALID_DEPTH) {
                Morton<DIM> m_start = m.DFD();
                Morton<DIM> m_end = m.Next();
                Integer p_start = std::lower_bound(mins.begin(), mins.end(), m_start) - mins.begin() - 1;
                Integer p_end   = std::lower_bound(mins.begin(), mins.end(), m_end  ) - mins.begin();
                SCTL_ASSERT(0 <= p_start);
                SCTL_ASSERT(p_start < p_end);
                SCTL_ASSERT(p_end <= np);
                for (Long p = p_start; p < p_end; p++) {
                  if (!user_procs_flag[p]) {
                    user_procs_flag[p] = true;
                    user_procs[user_procs_cnt++] = p;
                  }
                }
              }
            }
            for (Long k = 0; k < user_procs_cnt; k++) {
              const Long p = user_procs[k];
              user_node_lst_t.PushBack(std::make_pair(p, m0));
              user_procs_flag[p] = false;
            }
            user_procs_cnt = 0;
          }
        }

        ScratchBuf<Long> cnt(nthreads), dsp(nthreads);
        { // Set cnt, dsp
          for (Integer tid = 0; tid < nthreads; tid++) cnt[tid] = user_node_lst_t_[tid].Dim();
          std::exclusive_scan(cnt.begin(), cnt.begin()+nthreads, dsp.begin(), Long(0));
        }

        const Long Nsend = dsp[nthreads-1] + cnt[nthreads-1];
        ScratchBuf<std::pair<Long,Morton<DIM>>> user_node_lst_buf(Nsend);
        Vector<std::pair<Long,Morton<DIM>>> user_node_lst(user_node_lst_buf);
        #pragma omp parallel num_threads(nthreads)
        { // user_node_lst <-- concatenate user_node_lst_t_[tid]
          const Integer tid = SCTL_GET_THREAD_NUM();
          Vector<std::pair<Long,Morton<DIM>>>& user_node_lst_t = user_node_lst_t_[tid];
          omp_par::copy(user_node_lst_t.begin(), user_node_lst_t.end(), user_node_lst.begin() + dsp[tid]);
        }
        omp_par::sample_sort(user_node_lst.begin(), user_node_lst.end());

        user_cnt.ReInit(np);
        user_mid.ReInit(Nsend);
        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < Nsend; i++) {
          user_mid[i] = user_node_lst[i].second;
        }
        #pragma omp parallel for schedule(static)
        for (Integer i = 0; i < np; i++) {
          const auto pair_start = std::make_pair(Long(i), Morton<DIM>());
          const auto pair_end = std::make_pair(Long(i+1), Morton<DIM>());
          Long cnt_start = std::lower_bound(user_node_lst.begin(), user_node_lst.end(), pair_start) - user_node_lst.begin();
          Long cnt_end   = std::lower_bound(user_node_lst.begin(), user_node_lst.end(), pair_end  ) - user_node_lst.begin();
          user_cnt[i] = cnt_end - cnt_start;
        }
      }

      Vector<Morton<DIM>> ghost_mid;
      { // SendRecv user_mid
        ScratchBuf<Long> send_dsp(np), recv_cnt(np), recv_dsp(np);
        send_dsp[0] = 0;
        recv_dsp[0] = 0;

        const Vector<Long>& send_cnt = user_cnt;
        omp_par::scan(send_cnt.begin(), send_dsp.begin(), np);

        comm.Alltoall(send_cnt.begin(), 1, recv_cnt.begin(), 1);
        omp_par::scan(recv_cnt.begin(), recv_dsp.begin(), np);

        const Vector<Morton<DIM>>& send_mid = user_mid;
        Long Nsend = send_dsp[np-1] + send_cnt[np-1];
        Long Nrecv = recv_dsp[np-1] + recv_cnt[np-1];
        SCTL_ASSERT(send_mid.Dim() == Nsend);

        ghost_mid.ReInit(Nrecv);
        comm.Alltoallv(send_mid.begin(), send_cnt.begin(), send_dsp.begin(), ghost_mid.begin(), recv_cnt.begin(), recv_dsp.begin());
      }

      { // Update node_mid <-- ghost_mid + node_mid
        const Long Nlocal = end_idx - start_idx;
        ScratchBuf<Morton<DIM>> local_mid_lst(Nlocal);
        omp_par::memcpy(local_mid_lst.begin(), node_mid.begin() + start_idx, Nlocal);

        const Integer nthreads = SCTL_GET_MAX_THREADS();
        ScratchBuf<Vector<Morton<DIM>>> node_mid_t0(nthreads);
        ScratchBuf<Vector<Morton<DIM>>> node_mid_t1(nthreads);
        const Long Nsplit = std::lower_bound(ghost_mid.begin(), ghost_mid.end(), mins[rank]) - ghost_mid.begin();
        #pragma omp parallel num_threads(nthreads)
        { // Set node_mid_t_[tid]
          const Integer tid = SCTL_GET_THREAD_NUM();
          const Integer nthreads = SCTL_GET_NUM_THREADS();
          if (rank != 0) { // Set node_mid_t0
            Vector<Morton<DIM>>& node_mid_t = node_mid_t0[tid];
            if (Nsplit) {
              const Long start_idx = ((Nsplit-1) *  tid     ) / nthreads;
              const Long end_idx   = ((Nsplit-1) * (tid + 1)) / nthreads;
              if (tid == 0)  complete_tree(node_mid_t, Morton<DIM>(), ghost_mid[start_idx]);
              for (Long i = start_idx; i < end_idx; i++) complete_tree(node_mid_t, ghost_mid[i], ghost_mid[i+1]);
              if (tid == nthreads-1) complete_tree(node_mid_t, ghost_mid[end_idx], mins[rank]);
            } else {
              if (tid == 0) complete_tree(node_mid_t, Morton<DIM>(), mins[rank]);
            }
          }
          if (rank != np-1) { // Set node_mid_t1
            Vector<Morton<DIM>>& node_mid_t = node_mid_t1[tid];
            const Long N = ghost_mid.Dim() - Nsplit;

            if (N) {
              const Long start_idx = Nsplit + ((N-1) *  tid     ) / nthreads;
              const Long end_idx   = Nsplit + ((N-1) * (tid + 1)) / nthreads;
              if (tid == 0)  complete_tree(node_mid_t, mins[rank+1], ghost_mid[start_idx]);
              for (Long i = start_idx; i < end_idx; i++) complete_tree(node_mid_t, ghost_mid[i], ghost_mid[i+1]);
              if (tid == nthreads-1) complete_tree(node_mid_t, ghost_mid[end_idx], Morton<DIM>().Next());
            } else {
              if (tid == 0) complete_tree(node_mid_t, mins[rank+1], Morton<DIM>().Next());
            }
          }
        }

        { // node_mid <-- concatenate: node_mid_t0, local_mid_lst, node_mid_t1
          ScratchBuf<Long> dsp(nthreads + 1 + nthreads);
          { // Set dsp, ReInit node_mid
            Long sum = 0;
            for (Integer t = 0; t < nthreads; t++) {
              dsp[t] = sum;
              sum += node_mid_t0[t].Dim();
            }
            dsp[nthreads] = sum;
            sum += Nlocal;
            for (Integer t = 0; t < nthreads; t++) {
              dsp[nthreads+1+t] = sum;
              sum += node_mid_t1[t].Dim();
            }
            node_mid.ReInit(sum);
          }
          #pragma omp parallel for schedule(static)
          for (Integer tid = 0; tid < nthreads; tid++) {
            omp_par::memcpy(node_mid.begin() + dsp[tid], node_mid_t0[tid].begin(), node_mid_t0[tid].Dim());
            omp_par::memcpy(node_mid.begin() + dsp[nthreads+1+tid], node_mid_t1[tid].begin(), node_mid_t1[tid].Dim());
          }
          omp_par::memcpy(node_mid.begin() + dsp[nthreads], local_mid_lst.begin(), Nlocal);
        }
      }
    } else {
      user_mid.ReInit(0);
      user_cnt.ReInit(np);
      user_cnt = 0;
    }

    set_node_attr(node_attr, node_mid, mins, rank);
    set_node_lst(node_lst, node_mid, periodicity);

    if (0) { // Check tree
      Morton<DIM> m0{};
      SCTL_ASSERT(node_mid.Dim() && m0 == node_mid[0]);
      for (Long i = 1; i < node_mid.Dim(); i++) {
        const auto& m = node_mid[i];
        if (m0.isAncestor(m)) m0 = m0.Ancestor(m0.Depth()+1);
        else m0 = m0.Next();
        SCTL_ASSERT(m0 == m);
      }
      SCTL_ASSERT(m0.Next() == Morton<DIM>().Next());
      const Long i0 = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
      const Long i1 = (rank == np-1 ? node_mid.Dim() : std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank+1]) - node_mid.begin() );
      SCTL_ASSERT(node_mid[i0] == mins[rank]);

      for (Long i = 0; i < node_mid.Dim()-1; i++) {
        SCTL_ASSERT(node_mid[i].Next() == node_mid[i+1] || first_child(node_mid[i]) == node_mid[i+1]);
      }
      SCTL_ASSERT(rank == np-1 || i1 == 0 || node_mid[i1-1].Next() == mins[rank+1] || first_child(node_mid[i1-1]) == mins[rank+1]);

      for (Long i = 0; i < i0; i++) SCTL_ASSERT(node_attr[i].Ghost == true);
      for (Long i = i0; i < i1; i++) SCTL_ASSERT(node_attr[i].Ghost == false);
      for (Long i = i1; i < node_mid.Dim(); i++) SCTL_ASSERT(node_attr[i].Ghost == true);
    }

    { // Update node_data, node_cnt
      comm.PartitionS(node_mid_orig, mins[comm.Rank()]);

      ScratchBuf<Long> new_cnt_range0(node_mid.Dim()+1);
      { // Set new_cnt_range0
        const Long start_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
        const Long end_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();

        #pragma omp parallel for schedule(static)
        for (Long i = 0; i < start_idx; i++) {
          new_cnt_range0[i] = 0;
        }
        #pragma omp parallel
        {
          const Integer tid = SCTL_GET_THREAD_NUM();
          const Integer nthreads = SCTL_GET_NUM_THREADS();
          const Long idx0 = start_idx + (end_idx - start_idx) *  tid      / nthreads;
          const Long idx1 = start_idx + (end_idx - start_idx) * (tid + 1) / nthreads;

          if (idx0 < end_idx) {
            Long j = std::lower_bound(node_mid_orig.begin(), node_mid_orig.end(), node_mid[idx0]) - node_mid_orig.begin();
            for (Long i = idx0; i < idx1; i++) {
              while (j < node_mid_orig.Dim() && node_mid_orig[j] < node_mid[i]) j++;
              new_cnt_range0[i] = j;
            }
          }
        }
        #pragma omp parallel for schedule(static)
        for (Long i = end_idx; i <= node_mid.Dim(); i++) {
          new_cnt_range0[i] = node_mid_orig.Dim();
        }
      }

      for (const auto& pair : node_data) {
        const std::string& data_name = pair.first;

        Iterator<Vector<char>> data_;
        Iterator<Vector<Long>> cnt_;
        GetData_(data_, cnt_, data_name);
        const Long dof = [this,&data_,&cnt_]() {
          StaticArray<Long,2> Nl, Ng;
          Nl[0] = data_->Dim();
          Nl[1] = omp_par::reduce(cnt_->begin(), cnt_->Dim());
          comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, CommOp::SUM);
          const Long dof = Ng[0] / std::max<Long>(Ng[1],1);
          SCTL_ASSERT(Nl[0] == Nl[1] * dof);
          SCTL_ASSERT(Ng[0] == Ng[1] * dof);
          return dof;
        }();

        const Long data_begin = omp_par::reduce(cnt_->begin(), start_idx_orig);
        const Long data_count = omp_par::reduce(cnt_->begin() + start_idx_orig, end_idx_orig - start_idx_orig);
        { // Recompute cnt_
          Vector<Long> cnt_tmp(end_idx_orig - start_idx_orig, cnt_->begin() + start_idx_orig);
          comm.PartitionN(cnt_tmp, node_mid_orig.Dim());

          cnt_->ReInit(node_mid.Dim());
          #pragma omp parallel for schedule(static)
          for (Long i = 0; i < node_mid.Dim(); i++) {
            Long sum = 0;
            Long j0 = new_cnt_range0[i];
            Long j1 = new_cnt_range0[i+1];
            for (Long j = j0; j < j1; j++) sum += cnt_tmp[j];
            cnt_[0][i] = sum;
          }
          SCTL_ASSERT(omp_par::reduce(cnt_->begin(), cnt_->Dim()) == omp_par::reduce(cnt_tmp.begin(), cnt_tmp.Dim()));
        }
        { // Repartition data_ according to new cnt_
          const Long Ndata = omp_par::reduce(cnt_->begin(), cnt_->Dim()) * dof;
          Vector<char> data_tmp(data_count * dof, data_->begin() + data_begin * dof, false);
          comm.PartitionN(data_tmp, Ndata);
          SCTL_ASSERT(data_tmp.Dim() == Ndata);

          if (data_tmp.OwnData()) data_->Swap(data_tmp);
          else if (data_begin != 0 || data_count * dof != data_->Dim()) { // make a new copy
            Vector<char> data_new = data_tmp;
            data_->Swap(data_new);
          } // else no change to data_
        }
      }
    }
  }

  template <Integer DIM> template <class ValueType> void Tree<DIM>::AddData(const std::string& name, const Vector<ValueType>& data, const Vector<Long>& cnt) {
    Long dof;
    { // Check dof
      StaticArray<Long,2> Nl, Ng;
      Nl[0] = data.Dim();
      Nl[1] = omp_par::reduce(cnt.begin(), cnt.Dim());
      comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, CommOp::SUM);
      dof = Ng[0] / std::max<Long>(Ng[1],1);
      SCTL_ASSERT(Nl[0] == Nl[1] * dof);
      SCTL_ASSERT(Ng[0] == Ng[1] * dof);
    }
    if (dof) SCTL_ASSERT(cnt.Dim() == node_mid.Dim());

    SCTL_ASSERT(node_data.find(name) == node_data.end());
    node_data[name].ReInit(data.Dim()*sizeof(ValueType), (Iterator<char>)data.begin(), true);
    node_cnt [name] = cnt;
  }

  template <Integer DIM> template <class ValueType> void Tree<DIM>::GetData(Vector<ValueType>& data, Vector<Long>& cnt, const std::string& name) const {
    const auto data_ = node_data.find(name);
    const auto cnt_ = node_cnt.find(name);
    SCTL_ASSERT(data_ != node_data.end());
    SCTL_ASSERT( cnt_ != node_cnt .end());
    data.ReInit(data_->second.Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->second.begin(), false);
    SCTL_ASSERT(data.Dim()*(Long)sizeof(ValueType) == data_->second.Dim());
    cnt .ReInit( cnt_->second.Dim(), (Iterator<Long>)cnt_->second.begin(), false);
  }

  template <Integer DIM> template <class ValueType> void Tree<DIM>::ReduceBroadcast(const std::string& name) {
    Integer np = comm.Size();
    Integer rank = comm.Rank();

    Vector<Long> dsp;
    Iterator<Vector<char>> data_;
    Iterator<Vector<Long>> cnt_;
    GetData_(data_, cnt_, name);
    Vector<ValueType> data(data_->Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->begin(), false);
    Vector<Long>& cnt = *cnt_;
    scan(dsp, cnt);

    Long dof;
    { // Set dof
      StaticArray<Long,2> Nl, Ng;
      Nl[0] = data.Dim();
      Nl[1] = omp_par::reduce(cnt.begin(), cnt.Dim());
      comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, CommOp::SUM);
      dof = Ng[0] / std::max<Long>(Ng[1],1);
      SCTL_ASSERT(Nl[0] == Nl[1] * dof);
      SCTL_ASSERT(Ng[0] == Ng[1] * dof);
    }

    { // Reduce
      Vector<Morton<DIM>> send_mid, recv_mid;
      Vector<Long> send_node_cnt(np), send_node_dsp(np);
      Vector<Long> recv_node_cnt(np), recv_node_dsp(np);
      { // Set send_mid, send_node_cnt, send_node_dsp, recv_mid, recv_node_cnt, recv_node_dsp
        { // Set send_mid
          Morton<DIM> m0 = mins[rank];
          for (Integer d = 0; d < m0.Depth(); d++) {
            send_mid.PushBack(m0.Ancestor(d));
          }
        }
        for (Integer p = 0; p < np; p++) {
          Long start_idx = std::lower_bound(send_mid.begin(), send_mid.end(), mins[p]) - send_mid.begin();
          Long end_idx = std::lower_bound(send_mid.begin(), send_mid.end(), (p+1==np ? Morton<DIM>().Next() : mins[p+1])) - send_mid.begin();
          send_node_cnt[p] = end_idx - start_idx;
        }
        scan(send_node_dsp, send_node_cnt);
        SCTL_ASSERT(send_node_dsp[np-1]+send_node_cnt[np-1] == send_mid.Dim());
        comm.Alltoall(send_node_cnt.begin(), 1, recv_node_cnt.begin(), 1);
        scan(recv_node_dsp, recv_node_cnt);

        recv_mid.ReInit(recv_node_dsp[np-1] + recv_node_cnt[np-1]);
        comm.Alltoallv(send_mid.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_mid.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
      }

      Vector<Long> send_data_cnt, send_data_dsp;
      Vector<Long> recv_data_cnt, recv_data_dsp;
      { // Set send_data_cnt, send_data_dsp
        send_data_cnt.ReInit(send_mid.Dim());
        recv_data_cnt.ReInit(recv_mid.Dim());
        for (Long i = 0; i < send_mid.Dim(); i++) {
          Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
          SCTL_ASSERT(send_mid[i] == node_mid[idx]);
          send_data_cnt[i] = cnt[idx];
        }
        scan(send_data_dsp, send_data_cnt);
        comm.Alltoallv(send_data_cnt.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_data_cnt.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
        scan(recv_data_dsp, recv_data_cnt);
      }

      Vector<ValueType> send_buff, recv_buff;
      Vector<Long> send_buff_cnt(np), send_buff_dsp(np);
      Vector<Long> recv_buff_cnt(np), recv_buff_dsp(np);
      { // Set send_buff, send_buff_cnt, send_buff_dsp, recv_buff, recv_buff_cnt, recv_buff_dsp
        Long N_send_nodes = send_mid.Dim();
        Long N_recv_nodes = recv_mid.Dim();
        if (N_send_nodes) send_buff.ReInit((send_data_dsp[N_send_nodes-1] + send_data_cnt[N_send_nodes-1]) * dof);
        if (N_recv_nodes) recv_buff.ReInit((recv_data_dsp[N_recv_nodes-1] + recv_data_cnt[N_recv_nodes-1]) * dof);
        for (Long i = 0; i < N_send_nodes; i++) {
          Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
          SCTL_ASSERT(send_mid[i] == node_mid[idx]);
          Long dsp_ = dsp[idx] * dof;
          Long cnt_ = cnt[idx] * dof;
          Long send_data_dsp_ = send_data_dsp[i] * dof;
          Long send_data_cnt_ = send_data_cnt[i] * dof;
          SCTL_ASSERT(send_data_cnt_ == cnt_);
          for (Long j = 0; j < cnt_; j++) {
            send_buff[send_data_dsp_+j] = data[dsp_+j];
          }
        }
        for (Integer p = 0; p < np; p++) {
          Long send_buff_cnt_ = 0;
          Long recv_buff_cnt_ = 0;
          for (Long i = 0; i < send_node_cnt[p]; i++) {
            send_buff_cnt_ += send_data_cnt[send_node_dsp[p]+i];
          }
          for (Long i = 0; i < recv_node_cnt[p]; i++) {
            recv_buff_cnt_ += recv_data_cnt[recv_node_dsp[p]+i];
          }
          send_buff_cnt[p] = send_buff_cnt_ * dof;
          recv_buff_cnt[p] = recv_buff_cnt_ * dof;
        }
        scan(send_buff_dsp, send_buff_cnt);
        scan(recv_buff_dsp, recv_buff_cnt);
        comm.Alltoallv(send_buff.begin(), send_buff_cnt.begin(), send_buff_dsp.begin(), recv_buff.begin(), recv_buff_cnt.begin(), recv_buff_dsp.begin());
      }

      { // Reduction
        Long N_recv_nodes = recv_mid.Dim();
        for (Long i = 0; i < N_recv_nodes; i++) {
          Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), recv_mid[i]) - node_mid.begin();
          Long dsp_ = dsp[idx] * dof;
          Long cnt_ = cnt[idx] * dof;
          Long recv_data_dsp_ = recv_data_dsp[i] * dof;
          Long recv_data_cnt_ = recv_data_cnt[i] * dof;
          SCTL_ASSERT(recv_data_cnt_ == cnt_ || recv_data_cnt_ == 0);
          if (recv_data_cnt_ == cnt_) {
            for (Long j = 0; j < cnt_; j++) {
              data[dsp_+j] += recv_buff[recv_data_dsp_+j];
            }
          }
        }
      }
    }

    Broadcast<ValueType>(name);
  }

  template <Integer DIM> template <class ValueType> void Tree<DIM>::Broadcast(const std::string& name) {
    Integer np = comm.Size();
    Integer rank = comm.Rank();

    Vector<Long> dsp;
    Iterator<Vector<char>> data_;
    Iterator<Vector<Long>> cnt_;
    GetData_(data_, cnt_, name);
    Vector<ValueType> data(data_->Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->begin(), false);
    Vector<Long>& cnt = *cnt_;
    scan(dsp, cnt);

    Long dof;
    { // Set dof
      StaticArray<Long,2> Nl, Ng;
      Nl[0] = data.Dim();
      Nl[1] = omp_par::reduce(cnt.begin(), cnt.Dim());
      comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, CommOp::SUM);
      dof = Ng[0] / std::max<Long>(Ng[1],1);
      SCTL_ASSERT(Nl[0] == Nl[1] * dof);
      SCTL_ASSERT(Ng[0] == Ng[1] * dof);
    }

    { // Broadcast
      const Vector<Morton<DIM>>& send_mid = user_mid;
      const Vector<Long>& send_node_cnt = user_cnt;
      Vector<Long> send_node_dsp(np);
      { // Set send_dsp
        SCTL_ASSERT(send_node_cnt.Dim() == np);
        scan(send_node_dsp, send_node_cnt);
        SCTL_ASSERT(send_node_dsp[np-1] + send_node_cnt[np-1] == send_mid.Dim());
      }

      Vector<Morton<DIM>> recv_mid;
      Vector<Long> recv_node_cnt(np), recv_node_dsp(np);
      { // Set recv_mid, recv_node_cnt, recv_node_dsp
        comm.Alltoall(send_node_cnt.begin(), 1, recv_node_cnt.begin(), 1);
        scan(recv_node_dsp, recv_node_cnt);

        recv_mid.ReInit(recv_node_dsp[np-1] + recv_node_cnt[np-1]);
        comm.Alltoallv(send_mid.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_mid.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
      }

      Vector<Long> send_data_cnt, send_data_dsp;
      Vector<Long> recv_data_cnt, recv_data_dsp;
      { // Set send_data_cnt, send_data_dsp
        send_data_cnt.ReInit(send_mid.Dim());
        recv_data_cnt.ReInit(recv_mid.Dim());
        for (Long i = 0; i < send_mid.Dim(); i++) {
          Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
          SCTL_ASSERT(send_mid[i] == node_mid[idx]);
          send_data_cnt[i] = cnt[idx];
        }
        scan(send_data_dsp, send_data_cnt);
        comm.Alltoallv(send_data_cnt.begin(), send_node_cnt.begin(), send_node_dsp.begin(), recv_data_cnt.begin(), recv_node_cnt.begin(), recv_node_dsp.begin());
        scan(recv_data_dsp, recv_data_cnt);
      }

      Vector<ValueType> send_buff, recv_buff;
      Vector<Long> send_buff_cnt(np), send_buff_dsp(np);
      Vector<Long> recv_buff_cnt(np), recv_buff_dsp(np);
      { // Set send_buff, send_buff_cnt, send_buff_dsp, recv_buff, recv_buff_cnt, recv_buff_dsp
        Long N_send_nodes = send_mid.Dim();
        Long N_recv_nodes = recv_mid.Dim();
        if (N_send_nodes) send_buff.ReInit((send_data_dsp[N_send_nodes-1] + send_data_cnt[N_send_nodes-1]) * dof);
        if (N_recv_nodes) recv_buff.ReInit((recv_data_dsp[N_recv_nodes-1] + recv_data_cnt[N_recv_nodes-1]) * dof);
        for (Long i = 0; i < N_send_nodes; i++) {
          Long idx = std::lower_bound(node_mid.begin(), node_mid.end(), send_mid[i]) - node_mid.begin();
          SCTL_ASSERT(send_mid[i] == node_mid[idx]);
          Long dsp_ = dsp[idx] * dof;
          Long cnt_ = cnt[idx] * dof;
          Long send_data_dsp_ = send_data_dsp[i] * dof;
          Long send_data_cnt_ = send_data_cnt[i] * dof;
          SCTL_ASSERT(send_data_cnt_ == cnt_);
          for (Long j = 0; j < cnt_; j++) {
            send_buff[send_data_dsp_+j] = data[dsp_+j];
          }
        }
        for (Integer p = 0; p < np; p++) {
          Long send_buff_cnt_ = 0;
          Long recv_buff_cnt_ = 0;
          for (Long i = 0; i < send_node_cnt[p]; i++) {
            send_buff_cnt_ += send_data_cnt[send_node_dsp[p]+i];
          }
          for (Long i = 0; i < recv_node_cnt[p]; i++) {
            recv_buff_cnt_ += recv_data_cnt[recv_node_dsp[p]+i];
          }
          send_buff_cnt[p] = send_buff_cnt_ * dof;
          recv_buff_cnt[p] = recv_buff_cnt_ * dof;
        }
        scan(send_buff_dsp, send_buff_cnt);
        scan(recv_buff_dsp, recv_buff_cnt);
        comm.Alltoallv(send_buff.begin(), send_buff_cnt.begin(), send_buff_dsp.begin(), recv_buff.begin(), recv_buff_cnt.begin(), recv_buff_dsp.begin());
      }

      Long start_idx, end_idx;
      { // Set start_idx, end_idx
        start_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
        end_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
        SCTL_ASSERT(0 <= start_idx);
        SCTL_ASSERT(start_idx < end_idx);
        SCTL_ASSERT(end_idx <= node_mid.Dim());
      }

      { // Update data <-- data + recv_buff
        Long Nsplit = std::lower_bound(recv_mid.begin(), recv_mid.end(), mins[rank]) - recv_mid.begin();

        Long N0 = (start_idx ? dsp[start_idx-1] + cnt[start_idx-1] : 0) * dof;
        Long N1 = (end_idx ? dsp[end_idx-1] + cnt[end_idx-1] : 0) * dof;
        Long Ns = (Nsplit ? recv_data_dsp[Nsplit-1] + recv_data_cnt[Nsplit-1] : 0) * dof;
        if (N0 != Ns || recv_buff.Dim() != N0+data.Dim()-N1) { // resize data and preserve non-ghost data
          Vector<char> data_new((recv_buff.Dim() + N1-N0) * sizeof(ValueType));
          omp_par::memcpy(data_new.begin() + Ns * sizeof(ValueType), data_->begin() + N0 * sizeof(ValueType), (N1-N0) * sizeof(ValueType));
          data_->Swap(data_new);
          data.ReInit(data_->Dim()/sizeof(ValueType), (Iterator<ValueType>)data_->begin(), false);
        }

        for (Long i = 0; i < start_idx; i++) cnt[i] = 0;
        for (Long i = end_idx; i < cnt.Dim(); i++) cnt[i] = 0;
        for (Long i = 0; i < recv_mid.Dim(); i++) {
          const auto idx = std::lower_bound(node_mid.begin(), node_mid.end(), recv_mid[i]) - node_mid.begin();
          SCTL_ASSERT(node_mid[idx] == recv_mid[i]);
          cnt[idx] = recv_data_cnt[i];
        }

        omp_par::memcpy(data.begin(), recv_buff.begin(), Ns);
        omp_par::memcpy(data.begin()+data.Dim()+Ns-recv_buff.Dim(), recv_buff.begin()+Ns, recv_buff.Dim()-Ns);
      }
    }
  }

  template <Integer DIM> void Tree<DIM>::DeleteData(const std::string& name) {
    SCTL_ASSERT(node_data.find(name) != node_data.end());
    SCTL_ASSERT(node_cnt .find(name) != node_cnt .end());
    node_data.erase(name);
    node_cnt .erase(name);
  }

  template <Integer DIM> void Tree<DIM>::WriteTreeVTK(std::string fname, bool show_ghost) const {
    typedef typename VTUData::VTKReal VTKReal;
    VTUData vtu_data;
    if (DIM <= 3) {  // Set vtu data
      static const Integer Ncorner = (1u << DIM);

      Vector<VTKReal> &coord = vtu_data.coord;
      //Vector<VTKReal> &value = vtu_data.value;

      Vector<int32_t> &connect = vtu_data.connect;
      Vector<int32_t> &offset = vtu_data.offset;
      Vector<uint8_t> &types = vtu_data.types;

      StaticArray<VTKReal, DIM> c;
      Long point_cnt = coord.Dim() / 3;
      Long connect_cnt = connect.Dim();
      for (Long nid = 0; nid < node_mid.Dim(); nid++) {
        const Morton<DIM> &mid = node_mid[nid];
        const NodeAttr &attr = node_attr[nid];
        if (!show_ghost && attr.Ghost) continue;
        if (!attr.Leaf) continue;

        mid.Coord((Iterator<VTKReal>)c);
        VTKReal s = sctl::pow<VTKReal>(0.5, mid.Depth());
        for (Integer j = 0; j < Ncorner; j++) {
          for (Integer i = 0; i < DIM; i++) coord.PushBack(c[i] + ((j & (1u << i)) ? 1 : 0) * s);
          for (Integer i = DIM; i < 3; i++) coord.PushBack(0);
          connect.PushBack(point_cnt);
          connect_cnt++;
          point_cnt++;
        }
        offset.PushBack(connect_cnt);
        if (DIM == 2)
          types.PushBack(8);
        else if (DIM == 3)
          types.PushBack(11);
        else
          types.PushBack(4);
      }
    }
    vtu_data.WriteVTK(fname, comm);
  }

  template <Integer DIM> void Tree<DIM>::GetData_(Iterator<Vector<char>>& data, Iterator<Vector<Long>>& cnt, const std::string& name) {
    auto data_ = node_data.find(name);
    const auto cnt_ = node_cnt.find(name);
    SCTL_ASSERT(data_ != node_data.end());
    SCTL_ASSERT( cnt_ != node_cnt .end());
    data = Ptr2Itr<Vector<char>>(&data_->second,1);
    cnt  = Ptr2Itr<Vector<Long>>(& cnt_->second,1);
  }

  template <Integer DIM> void Tree<DIM>::scan(Vector<Long>& dsp, const Vector<Long>& cnt) {
    dsp.ReInit(cnt.Dim());
    if (cnt.Dim()) dsp[0] = 0;
    omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());
  }



  template <class Real, Integer DIM, class BaseTree> PtTree<Real,DIM,BaseTree>::PtTree(const Comm& comm) : BaseTree(comm) {}

  template <class Real, Integer DIM, class BaseTree> PtTree<Real,DIM,BaseTree>::~PtTree() {
    #ifdef SCTL_MEMDEBUG
    for (auto& pair : data_pt_name) {
      Vector<Real> data;
      Vector<Long> cnt;
      this->GetData(data, cnt, pair.second);
      SCTL_ASSERT(scatter_idx.find(pair.second) != scatter_idx.end());
    }
    #endif
  }

  template <class Real, Integer DIM, class BaseTree> void PtTree<Real,DIM,BaseTree>::UpdateRefinement(const Vector<Real>& coord, Long M, bool balance21, Periodicity periodicity, Integer halo_size) {
    const auto& comm = this->GetComm();
    BaseTree::UpdateRefinement(coord, M, balance21, periodicity, halo_size);

    Long start_node_idx, end_node_idx;
    { // Set start_node_idx, end_node_idx
      const auto& mins = this->GetPartitionMID();
      const auto& node_mid = this->GetNodeMID();
      const Integer np = comm.Size();
      const Integer rank = comm.Rank();
      start_node_idx = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
      end_node_idx = std::lower_bound(node_mid.begin(), node_mid.end(), (rank+1==np ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
    }

    const auto& mins = this->GetPartitionMID();
    const auto& node_mid = this->GetNodeMID();
    for (const auto& pair : pt_mid) {
      const auto& pt_name = pair.first;

      auto& pt_mid_ = pt_mid[pt_name];
      auto& scatter_idx_ = scatter_idx[pt_name];
      comm.PartitionS(pt_mid_, mins[comm.Rank()]);
      comm.PartitionN(scatter_idx_, pt_mid_.Dim());

      ScratchBuf<Long> pt_cnt(node_mid.Dim());
      #pragma omp parallel
      { // Set pt_cnt
        const Integer tid = SCTL_GET_THREAD_NUM();
        const Integer nthreads = SCTL_GET_NUM_THREADS();
        const Long idx0 = (node_mid.Dim() *  tid     ) / nthreads;
        const Long idx1 = (node_mid.Dim() * (tid + 1)) / nthreads;

        if (idx0 < node_mid.Dim()) {
          Long j0 = std::lower_bound(pt_mid_.begin(), pt_mid_.end(), node_mid[idx0]) - pt_mid_.begin();
          if (idx0 == 0) SCTL_ASSERT(j0 == 0);
          for (Long i = idx0; i < idx1; i++) {
            const auto m1 = (i+1<node_mid.Dim() ? node_mid[i+1] : Morton<DIM>().Next());

            Long j = 1;
            while (j0+j < pt_mid_.Dim() && pt_mid_[j0+j] < m1) j *= 2;
            const Long j1 = std::lower_bound(pt_mid_.begin()+j0+(j>>1), pt_mid_.begin()+std::min<Long>(j0+j, pt_mid_.Dim()), m1) - pt_mid_.begin();
            pt_cnt[i] = j1 - j0;
            j0 = j1;
          }
          if (idx1 == node_mid.Dim()) SCTL_ASSERT(j0 == pt_mid_.Dim());
        }
      }

      Vector<char> data_tmp;
      for (const auto& pair : data_pt_name) {
        if (pair.second == pt_name) {
          const auto& data_name = pair.first;

          Iterator<Vector<char>> data;
          Iterator<Vector<Long>> cnt;
          this->GetData_(data, cnt, data_name);
          SCTL_ASSERT(cnt->Dim() == node_mid.Dim());

          { // Update data
            const Long dof = [&comm,&cnt,&data]() {
              StaticArray<Long,2> Nl, Ng;
              Nl[0] = data->Dim();
              Nl[1] = omp_par::reduce(cnt->begin(), cnt->Dim());
              comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, CommOp::SUM);
              const Long dof = Ng[0] / std::max<Long>(Ng[1],1);
              SCTL_ASSERT(Nl[0] == Nl[1] * dof);
              SCTL_ASSERT(Ng[0] == Ng[1] * dof);
              return dof;
            }();
            const Long data_begin = omp_par::reduce(cnt->begin(), start_node_idx);
            const Long data_count = omp_par::reduce(cnt->begin() + start_node_idx, end_node_idx - start_node_idx);

            data_tmp.ReInit(data_count * dof, data->begin() + data_begin * dof, false);
            comm.PartitionN(data_tmp, pt_mid_.Dim());

            if (data_tmp.OwnData()) data->Swap(data_tmp);
            else if (data_begin != 0 || data_count * dof != data->Dim()) { // make a new copy
              Vector<char> data_new = data_tmp;
              data->Swap(data_new);
            } // else no change to data
          }
          (*cnt) = Vector<Long>(pt_cnt);
        }
      }
    }
  }

  template <class Real, Integer DIM, class BaseTree> void PtTree<Real,DIM,BaseTree>::AddParticles(const std::string& name, const Vector<Real>& coord) {
    const auto& mins = this->GetPartitionMID();
    const auto& node_mid = this->GetNodeMID();
    const auto& comm = this->GetComm();

    SCTL_ASSERT(scatter_idx.find(name) == scatter_idx.end());
    Vector<Long>& scatter_idx_ = scatter_idx[name];

    Long N = coord.Dim() / DIM;
    SCTL_ASSERT(coord.Dim() == N * DIM);
    Nlocal[name] = N;

    Vector<Morton<DIM>>& pt_mid_ = pt_mid[name];
    if (pt_mid_.Dim() != N) pt_mid_.ReInit(N);
    for (Long i = 0; i < N; i++) {
      pt_mid_[i] = Morton<DIM>(coord.begin() + i*DIM);
    }
    comm.SortScatterIndex(pt_mid_, scatter_idx_, &mins[comm.Rank()]);
    comm.ScatterForward(pt_mid_, scatter_idx_);
    AddParticleData(name, name, coord);

    { // Set node_cnt
      Iterator<Vector<char>> data_;
      Iterator<Vector<Long>> cnt_;
      this->GetData_(data_,cnt_,name);
      cnt_[0].ReInit(node_mid.Dim());
      for (Long i = 0; i < node_mid.Dim(); i++) {
        Long start = std::lower_bound(pt_mid_.begin(), pt_mid_.end(), node_mid[i]) - pt_mid_.begin();
        Long end = std::lower_bound(pt_mid_.begin(), pt_mid_.end(), (i+1==node_mid.Dim() ? Morton<DIM>().Next() : node_mid[i+1])) - pt_mid_.begin();
        if (i == 0) SCTL_ASSERT(start == 0);
        if (i+1 == node_mid.Dim()) SCTL_ASSERT(end == pt_mid_.Dim());
        cnt_[0][i] = end - start;
      }
    }
  }

  template <class Real, Integer DIM, class BaseTree> void PtTree<Real,DIM,BaseTree>::AddParticleData(const std::string& data_name, const std::string& particle_name, const Vector<Real>& data) {
    SCTL_ASSERT(scatter_idx.find(particle_name) != scatter_idx.end());
    SCTL_ASSERT(data_pt_name.find(data_name) == data_pt_name.end());
    data_pt_name[data_name] = particle_name;

    Iterator<Vector<char>> data_;
    Iterator<Vector<Long>> cnt_;
    this->AddData(data_name, Vector<Real>(), Vector<Long>());
    this->GetData_(data_,cnt_,data_name);
    { // Set data_[0]
      data_[0].ReInit(data.Dim()*sizeof(Real), (Iterator<char>)data.begin(), true);
      this->GetComm().ScatterForward(data_[0], scatter_idx[particle_name]);
    }
    if (data_name != particle_name) { // Set cnt_[0]
      Vector<Real> pt_coord;
      Vector<Long> pt_cnt;
      this->GetData(pt_coord, pt_cnt, particle_name);
      cnt_[0] = pt_cnt;

      const auto& node_attr = this->GetNodeAttr();
      SCTL_ASSERT(node_attr.Dim() == cnt_[0].Dim());
      for (Long i = 0; i < node_attr.Dim(); i++) {
        if (node_attr[i].Ghost) cnt_[0][i] = 0;
        SCTL_ASSERT(node_attr[i].Leaf || !cnt_[0][i]);
      }
    }
  }

  template <class Real, Integer DIM, class BaseTree> void PtTree<Real,DIM,BaseTree>::GetParticleData(Vector<Real>& data, const std::string& data_name) const {
    SCTL_ASSERT(data_pt_name.find(data_name) != data_pt_name.end());
    const std::string& particle_name = data_pt_name.find(data_name)->second;
    SCTL_ASSERT(scatter_idx.find(particle_name) != scatter_idx.end());
    const auto& scatter_idx_ = scatter_idx.find(particle_name)->second;
    const Long Nlocal_ = Nlocal.find(particle_name)->second;

    const auto& mins = this->GetPartitionMID();
    const auto& node_mid = this->GetNodeMID();
    const auto& comm = this->GetComm();

    Long dof;
    Vector<Long> dsp;
    Vector<Long> cnt_;
    Vector<Real> data_;
    this->GetData(data_, cnt_, data_name);
    SCTL_ASSERT(cnt_.Dim() == node_mid.Dim());
    BaseTree::scan(dsp, cnt_);
    { // Set dof
      Long Nn = node_mid.Dim();
      StaticArray<Long,2> Ng, Nl{data_.Dim(), dsp[Nn-1]+cnt_[Nn-1]};
      comm.Allreduce((ConstIterator<Long>)Nl, (Iterator<Long>)Ng, 2, CommOp::SUM);
      dof = Ng[0] / std::max<Long>(Ng[1],1);
    }
    { // Set data
      Integer np = comm.Size();
      Integer rank = comm.Rank();
      Long N0 = std::lower_bound(node_mid.begin(), node_mid.end(), mins[rank]) - node_mid.begin();
      Long N1 = std::lower_bound(node_mid.begin(), node_mid.end(), (rank==np-1 ? Morton<DIM>().Next() : mins[rank+1])) - node_mid.begin();
      Long start = dsp[N0] * dof;
      Long end = (N1 ? (dsp[N1-1]+cnt_[N1-1])*dof : start);
      data.ReInit(end-start, data_.begin()+start, true);
      comm.ScatterReverse(data, scatter_idx_, Nlocal_ * dof);
      SCTL_ASSERT(data.Dim() == Nlocal_ * dof);
    }
  }

  template <class Real, Integer DIM, class BaseTree> void PtTree<Real,DIM,BaseTree>::DeleteParticleData(const std::string& data_name) {
    SCTL_ASSERT(data_pt_name.find(data_name) != data_pt_name.end());
    auto particle_name = data_pt_name[data_name];
    if (data_name == particle_name) {
      std::vector<std::string> data_name_lst;
      for (auto& pair : data_pt_name) {
        if (pair.second == particle_name) {
          data_name_lst.push_back(pair.first);
        }
      }
      for (auto x : data_name_lst) {
        if (x != particle_name) {
          DeleteParticleData(x);
        }
      }
      Nlocal.erase(particle_name);
    }
    this->DeleteData(data_name);
    data_pt_name.erase(data_name);
  }

  template <class Real, Integer DIM, class BaseTree> void PtTree<Real,DIM,BaseTree>::WriteParticleVTK(std::string fname, std::string data_name, bool show_ghost) const {
    typedef typename VTUData::VTKReal VTKReal;
    const auto& node_mid = this->GetNodeMID();
    const auto& node_attr = this->GetNodeAttr();

    VTUData vtu_data;
    if (DIM <= 3) {  // Set vtu data
      SCTL_ASSERT(data_pt_name.find(data_name) != data_pt_name.end());
      std::string particle_name = data_pt_name.find(data_name)->second;

      Vector<Real> pt_coord;
      Vector<Real> pt_value;
      Vector<Long> pt_cnt;
      Vector<Long> pt_dsp;
      Long value_dof = 0;
      { // Set pt_coord, pt_cnt, pt_dsp
        this->GetData(pt_coord, pt_cnt, particle_name);
        Tree<DIM>::scan(pt_dsp, pt_cnt);
      }
      if (particle_name != data_name) { // Set pt_value, value_dof
        Vector<Long> pt_cnt;
        this->GetData(pt_value, pt_cnt, data_name);
        Long Npt = omp_par::reduce(pt_cnt.begin(), pt_cnt.Dim());
        value_dof = pt_value.Dim() / std::max<Long>(Npt,1);
      }

      Vector<VTKReal> &coord = vtu_data.coord;
      Vector<VTKReal> &value = vtu_data.value;

      Vector<int32_t> &connect = vtu_data.connect;
      Vector<int32_t> &offset = vtu_data.offset;
      Vector<uint8_t> &types = vtu_data.types;

      Long point_cnt = coord.Dim() / DIM;
      Long connect_cnt = connect.Dim();
      value.ReInit(point_cnt * value_dof);
      value.SetZero();

      SCTL_ASSERT(node_mid.Dim() == node_attr.Dim());
      SCTL_ASSERT(node_mid.Dim() == pt_cnt.Dim());
      for (Long i = 0; i < node_mid.Dim(); i++) {
        if (!show_ghost && node_attr[i].Ghost) continue;
        if (!node_attr[i].Leaf) continue;

        for (Long j = 0; j < pt_cnt[i]; j++) {
          ConstIterator<Real> pt_coord_ = pt_coord.begin() + (pt_dsp[i] + j) * DIM;
          ConstIterator<Real> pt_value_ = (value_dof ? pt_value.begin() + (pt_dsp[i] + j) * value_dof : NullIterator<Real>());

          for (Integer k = 0; k < DIM; k++) coord.PushBack((VTKReal)pt_coord_[k]);
          for (Integer k = DIM; k < 3; k++) coord.PushBack(0);
          for (Integer k = 0; k < value_dof; k++) value.PushBack((VTKReal)pt_value_[k]);
          connect.PushBack(point_cnt);
          connect_cnt++;
          point_cnt++;

          offset.PushBack(connect_cnt);
          types.PushBack(1);
        }
      }
    }
    vtu_data.WriteVTK(fname, this->GetComm());
  }


}

#endif // _SCTL_TREE_TXX_
