#ifndef SCTL_MPI_COUNT_LIMIT
#define SCTL_MPI_COUNT_LIMIT 3
#endif

#include "sctl.hpp"

#include <iostream>

namespace {

using sctl::Comm;
using sctl::CommOp;
using sctl::Integer;
using sctl::Long;
using sctl::Vector;

constexpr Long kChunkLimit = static_cast<Long>(SCTL_MPI_COUNT_LIMIT);

template <class Type> void AssertEqual(const Type& a, const Type& b) {
  SCTL_ASSERT(a == b);
}

Long InterestingCount(Integer i) {
  switch (i) {
    case 0: return 0;
    case 1: return 1;
    case 2: return kChunkLimit;
    case 3: return kChunkLimit + 1;
    case 4: return 2 * kChunkLimit;
    default: return 2 * kChunkLimit + 1;
  }
}

constexpr Integer NInterestingCount = 6;

void FillSequence(Vector<Long>& v, Long base) {
  for (Long i = 0; i < v.Dim(); i++) v[i] = base + i;
}

void CheckSequence(const Vector<Long>& v, Long base) {
  for (Long i = 0; i < v.Dim(); i++) AssertEqual(v[i], base + i);
}

void TestIsendIrecv(const Comm& comm) {
  const Integer np = comm.Size();
  if (np < 2) return;
  const Integer rank = comm.Rank();
  const Integer dest = (rank + 1) % np;
  const Integer src = (rank + np - 1) % np;

  for (Integer test_id = 0; test_id < NInterestingCount; test_id++) {
    const Long count = InterestingCount(test_id);
    Vector<Long> send(count), recv(count);
    FillSequence(send, rank * 100000 + test_id * 1000);
    for (Long i = 0; i < count; i++) recv[i] = -1;

    auto recv_req = comm.Irecv(recv.begin(), recv.Dim(), src, 100 + test_id * 16);
    auto send_req = comm.Isend(send.begin(), send.Dim(), dest, 100 + test_id * 16);
    comm.Wait(std::move(send_req));
    comm.Wait(std::move(recv_req));

    CheckSequence(recv, src * 100000 + test_id * 1000);
  }

  // The counts must name the same number of bytes at both ends. A message split into chunks is one
  // receive per chunk of rcount, so a receive buffer larger than the message waits on chunks that
  // were never sent -- the counts below straddle the chunk boundary in both directions, and each
  // pair matches exactly. kChunkLimit is small here, so this is the same shape a message over the
  // implementation's count limit takes in a default build.
  for (Integer test_id = 0; test_id < NInterestingCount; test_id++) {
    const Long count = InterestingCount(test_id);
    Vector<Long> send(count), recv(count);
    FillSequence(send, rank * 100000 + 500 + test_id);
    for (Long i = 0; i < count; i++) recv[i] = -1;

    auto recv_req = comm.Irecv(recv.begin(), count, src, 400 + test_id);
    auto send_req = comm.Isend(send.begin(), count, dest, 400 + test_id);
    comm.Wait(std::move(send_req));
    comm.Wait(std::move(recv_req));

    CheckSequence(recv, src * 100000 + 500 + test_id);
  }
}

void TestIsendIrecvConsecutiveTags(const Comm& comm) {
  const Integer np = comm.Size();
  if (np < 2) return;
  const Integer rank = comm.Rank();
  const Integer dest = (rank + 1) % np;
  const Integer src = (rank + np - 1) % np;
  const Integer tag0 = 400;
  const Integer tag1 = tag0 + 1;
  const Long count0 = 2 * kChunkLimit + 1;
  const Long count1 = 2 * kChunkLimit + 2;

  Vector<Long> send0(count0), recv0(count0);
  Vector<Long> send1(count1), recv1(count1);
  FillSequence(send0, rank * 100000 + 10000);
  FillSequence(send1, rank * 100000 + 20000);
  for (Long i = 0; i < count0; i++) recv0[i] = -1;
  for (Long i = 0; i < count1; i++) recv1[i] = -1;

  auto recv_req1 = comm.Irecv(recv1.begin(), recv1.Dim(), src, tag1);
  auto recv_req0 = comm.Irecv(recv0.begin(), recv0.Dim(), src, tag0);
  auto send_req0 = comm.Isend(send0.begin(), send0.Dim(), dest, tag0);
  auto send_req1 = comm.Isend(send1.begin(), send1.Dim(), dest, tag1);

  comm.Wait(std::move(send_req1));
  comm.Wait(std::move(recv_req1));
  comm.Wait(std::move(send_req0));
  comm.Wait(std::move(recv_req0));

  CheckSequence(recv0, src * 100000 + 10000);
  CheckSequence(recv1, src * 100000 + 20000);
}

void TestBcast(const Comm& comm) {
  const Integer roots[2] = {0, comm.Size() - 1};
  for (Integer root_it = 0; root_it < 2; root_it++) {
    for (Integer test_id = 0; test_id < NInterestingCount; test_id++) {
      const Long count = InterestingCount(test_id);
      Vector<Long> buff(count);
      for (Long i = 0; i < count; i++) {
        buff[i] = (comm.Rank() == roots[root_it] ? 10000 * (root_it + 1) + test_id * 100 + i : -1);
      }
      comm.Bcast(buff.begin(), buff.Dim(), roots[root_it]);
      CheckSequence(buff, 10000 * (root_it + 1) + test_id * 100);
    }
  }
}

void TestAllreduce(const Comm& comm) {
  const Integer np = comm.Size();
  const Integer rank = comm.Rank();
  for (Integer test_id = 0; test_id < NInterestingCount; test_id++) {
    const Long count = InterestingCount(test_id);
    Vector<Long> send(count), recv(count);
    for (Long i = 0; i < count; i++) send[i] = rank + 10 * i;
    comm.Allreduce(send.begin(), recv.begin(), count, CommOp::SUM);
    for (Long i = 0; i < count; i++) {
      AssertEqual(recv[i], static_cast<Long>(np) * 10 * i + static_cast<Long>(np - 1) * np / 2);
    }

    for (Long i = 0; i < count; i++) send[i] = rank + i;
    comm.Allreduce(send.begin(), recv.begin(), count, CommOp::MIN);
    CheckSequence(recv, 0);

    comm.Allreduce(send.begin(), recv.begin(), count, CommOp::MAX);
    CheckSequence(recv, np - 1);
  }
}

void TestScan(const Comm& comm) {
  const Integer rank = comm.Rank();
  for (Integer test_id = 0; test_id < NInterestingCount; test_id++) {
    const Long count = InterestingCount(test_id);
    Vector<Long> send(count), recv(count);

    for (Long i = 0; i < count; i++) send[i] = rank + 10 * i;
    comm.Scan(send.begin(), recv.begin(), count, CommOp::SUM);
    for (Long i = 0; i < count; i++) {
      AssertEqual(recv[i], static_cast<Long>(rank + 1) * 10 * i + static_cast<Long>(rank) * (rank + 1) / 2);
    }

    for (Long i = 0; i < count; i++) send[i] = rank + i;
    comm.Scan(send.begin(), recv.begin(), count, CommOp::MIN);
    CheckSequence(recv, 0);

    comm.Scan(send.begin(), recv.begin(), count, CommOp::MAX);
    CheckSequence(recv, rank);
  }
}

void TestAllgather(const Comm& comm) {
  const Integer np = comm.Size();
  const Integer rank = comm.Rank();
  for (Integer test_id = 0; test_id < NInterestingCount; test_id++) {
    const Long count = InterestingCount(test_id);
    Vector<Long> send(count), recv(count * np);
    FillSequence(send, rank * 100000 + test_id * 1000);
    comm.Allgather(send.begin(), send.Dim(), recv.begin(), send.Dim());
    for (Integer p = 0; p < np; p++) {
      for (Long i = 0; i < count; i++) {
        AssertEqual(recv[p * count + i], static_cast<Long>(p) * 100000 + test_id * 1000 + i);
      }
    }
  }
}

void InitDispls(Vector<Long>& displs, const Vector<Long>& counts) {
  if (!displs.Dim()) return;
  displs[0] = 0;
  for (Integer i = 1; i < displs.Dim(); i++) displs[i] = displs[i - 1] + counts[i - 1];
}

void InitScatteredDispls(Vector<Long>& displs, const Vector<Long>& counts, Long gap) {
  if (!displs.Dim()) return;
  displs[0] = 0;
  for (Integer i = 1; i < displs.Dim(); i++) displs[i] = displs[i - 1] + counts[i - 1] + gap;
}

void TestAllgatherv(const Comm& comm) {
  const Integer np = comm.Size();
  const Integer rank = comm.Rank();
  for (Integer mode = 0; mode < 3; mode++) {
    Vector<Long> counts(np), displs(np);
    for (Integer p = 0; p < np; p++) {
      if (mode == 0) {
        counts[p] = kChunkLimit + 1 + p;
      } else if (mode == 1) {
        counts[p] = (p % 2 == 0 ? 0 : 2 * kChunkLimit + 1 + p);
      } else {
        counts[p] = kChunkLimit + 1 + (p % 2);
      }
    }
    if (mode == 2) {
      InitScatteredDispls(displs, counts, 3 * kChunkLimit + 2);
    } else {
      InitDispls(displs, counts);
    }

    Vector<Long> send(counts[rank]), recv(displs[np - 1] + counts[np - 1]);
    FillSequence(send, mode * 1000000 + rank * 10000);
    comm.Allgatherv(send.begin(), send.Dim(), recv.begin(), counts.begin(), displs.begin());

    for (Integer p = 0; p < np; p++) {
      for (Long i = 0; i < counts[p]; i++) {
        AssertEqual(recv[displs[p] + i], mode * 1000000 + static_cast<Long>(p) * 10000 + i);
      }
    }
  }
}

struct IntPair {
  int a;
  int b;
};

void TestAllgathervMixedTypes(const Comm& comm) {
  const Integer np = comm.Size();
  const Integer rank = comm.Rank();
  for (Integer mode = 0; mode < 2; mode++) {
    Vector<Long> counts(np), displs(np);
    for (Integer p = 0; p < np; p++) counts[p] = 2 * (kChunkLimit + 1 + (p % 2));
    if (mode == 0) {
      InitDispls(displs, counts);
    } else {
      InitScatteredDispls(displs, counts, 2 * kChunkLimit + 3);
    }

    const Long send_count = counts[rank] / 2;
    Vector<IntPair> send(send_count);
    Vector<int> recv(displs[np - 1] + counts[np - 1]);
    for (Long i = 0; i < send_count; i++) {
      send[i].a = static_cast<int>(mode * 100000 + rank * 10000 + 2 * i + 0);
      send[i].b = static_cast<int>(mode * 100000 + rank * 10000 + 2 * i + 1);
    }

    comm.Allgatherv(send.begin(), send.Dim(), recv.begin(), counts.begin(), displs.begin());

    for (Integer p = 0; p < np; p++) {
      for (Long i = 0; i < counts[p]; i++) {
        AssertEqual(recv[displs[p] + i], static_cast<int>(mode * 100000 + p * 10000 + i));
      }
    }
  }
}

void TestAlltoall(const Comm& comm) {
  const Integer np = comm.Size();
  const Integer rank = comm.Rank();
  for (Integer test_id = 1; test_id < NInterestingCount; test_id++) {
    const Long count = InterestingCount(test_id);
    Vector<Long> send(np * count), recv(np * count);
    for (Integer p = 0; p < np; p++) {
      for (Long i = 0; i < count; i++) send[p * count + i] = rank * 100000 + p * 1000 + i;
    }
    comm.Alltoall(send.begin(), count, recv.begin(), count);
    for (Integer p = 0; p < np; p++) {
      for (Long i = 0; i < count; i++) {
        AssertEqual(recv[p * count + i], static_cast<Long>(p) * 100000 + rank * 1000 + i);
      }
    }
  }
}

void CheckAlltoallvPayload(const Vector<Long>& recv, const Vector<Long>& recv_cnt, const Vector<Long>& recv_dsp, Integer rank) {
  for (Integer p = 0; p < recv_cnt.Dim(); p++) {
    for (Long i = 0; i < recv_cnt[p]; i++) {
      AssertEqual(recv[recv_dsp[p] + i], static_cast<Long>(p) * 100000 + rank * 1000 + i);
    }
  }
}

void TestAlltoallv(const Comm& comm) {
  const Integer np = comm.Size();
  const Integer rank = comm.Rank();
  for (Integer mode = 0; mode < 3; mode++) {
    Vector<Long> send_cnt(np), send_dsp(np), recv_cnt(np), recv_dsp(np);
    for (Integer p = 0; p < np; p++) {
      if (mode == 0) {
        send_cnt[p] = kChunkLimit + 1 + ((rank + p) % 2);
      } else if (mode == 1) {
        send_cnt[p] = ((rank + p) % 2 == 0 ? 0 : 2 * kChunkLimit + 1);
      } else {
        send_cnt[p] = (p == rank ? 2 * kChunkLimit + 1 : (p == (rank + 1) % np ? kChunkLimit + 1 : 0));
      }
    }
    InitDispls(send_dsp, send_cnt);

    comm.Alltoall(send_cnt.begin(), 1, recv_cnt.begin(), 1);
    InitDispls(recv_dsp, recv_cnt);

    Vector<Long> send(send_dsp[np - 1] + send_cnt[np - 1]);
    Vector<Long> recv(recv_dsp[np - 1] + recv_cnt[np - 1]);
    for (Integer p = 0; p < np; p++) {
      for (Long i = 0; i < send_cnt[p]; i++) send[send_dsp[p] + i] = rank * 100000 + p * 1000 + i;
    }

    comm.Alltoallv(send.begin(), send_cnt.begin(), send_dsp.begin(), recv.begin(), recv_cnt.begin(), recv_dsp.begin());
    CheckAlltoallvPayload(recv, recv_cnt, recv_dsp, rank);

    // Same payload via the recursive-bitonic implementation; result must match.
    Vector<Long> recv_dense(recv_dsp[np - 1] + recv_cnt[np - 1]);
    comm.Alltoallv_dense(send.begin(), send_cnt.begin(), send_dsp.begin(), recv_dense.begin(), recv_cnt.begin(), recv_dsp.begin());
    CheckAlltoallvPayload(recv_dense, recv_cnt, recv_dsp, rank);
  }
}

void TestIalltoallvSparse(const Comm& comm) {
  const Integer np = comm.Size();
  const Integer rank = comm.Rank();
  Vector<Long> send_cnt(np), send_dsp(np), recv_cnt(np), recv_dsp(np);
  for (Integer p = 0; p < np; p++) {
    send_cnt[p] = (p == rank || p == (rank + 1) % np ? 2 * kChunkLimit + 1 + p : 0);
  }
  InitDispls(send_dsp, send_cnt);

  comm.Alltoall(send_cnt.begin(), 1, recv_cnt.begin(), 1);
  InitDispls(recv_dsp, recv_cnt);

  Vector<Long> send(send_dsp[np - 1] + send_cnt[np - 1]);
  Vector<Long> recv(recv_dsp[np - 1] + recv_cnt[np - 1]);
  for (Integer p = 0; p < np; p++) {
    for (Long i = 0; i < send_cnt[p]; i++) send[send_dsp[p] + i] = rank * 100000 + p * 1000 + i;
  }

  auto req = comm.Ialltoallv_sparse(send.begin(), send_cnt.begin(), send_dsp.begin(), recv.begin(), recv_cnt.begin(), recv_dsp.begin(), 23);
  comm.Wait(std::move(req));
  CheckAlltoallvPayload(recv, recv_cnt, recv_dsp, rank);
}

// Send/Recv pair up with a rank on this node wherever the kernel permits it, so the payload is read
// out of the sender's buffer rather than sent. Both ends must agree on that without exchanging the
// decision, and the fallback must match message for message, so the pairs below run whichever
// transport the run gets: neighbour ranks are on the same node under one mpirun, off it otherwise.
void TestSendRecv(const Comm& comm) {
  const Integer np = comm.Size();
  if (np < 2) return;
  const Integer rank = comm.Rank();
  const Integer peer = (rank % 2 == 0 ? rank + 1 : rank - 1);
  if (peer >= np) return;  // odd rank count leaves the last one out
  const bool first = (rank < peer);

  // The pair sends in one order and receives in the other, so neither waits on a send it must
  // itself receive.
  const auto exchange = [&comm, peer, first](Vector<Long>& send, Vector<Long>& recv, Integer tag) {
    if (first) {
      comm.Send(send.begin(), send.Dim(), peer, tag);
      comm.Recv(recv.begin(), recv.Dim(), peer, tag);
    } else {
      comm.Recv(recv.begin(), recv.Dim(), peer, tag);
      comm.Send(send.begin(), send.Dim(), peer, tag);
    }
  };

  for (Integer test_id = 0; test_id < NInterestingCount; test_id++) {  // 0 through several chunks
    const Long count = InterestingCount(test_id);
    Vector<Long> send(count), recv(count);
    FillSequence(send, rank * 100000 + test_id * 1000);
    for (Long i = 0; i < count; i++) recv[i] = -1;
    exchange(send, recv, 200 + test_id);
    CheckSequence(recv, peer * 100000 + test_id * 1000);
  }

  { // the same payload through Isend/Irecv: whichever transport Send/Recv takes, it agrees
    const Long count = 2 * kChunkLimit + 1;
    Vector<Long> send(count), blocking(count), nonblocking(count);
    FillSequence(send, rank * 100000 + 700);
    for (Long i = 0; i < count; i++) blocking[i] = nonblocking[i] = -7;
    exchange(send, blocking, 260);
    auto rq = comm.Irecv(nonblocking.begin(), count, peer, 261);
    auto sq = comm.Isend(send.begin(), count, peer, 261);
    comm.Wait(std::move(sq));
    comm.Wait(std::move(rq));
    for (Long i = 0; i < count; i++) AssertEqual(blocking[i], nonblocking[i]);
  }

  { // two messages on one tag arrive in the order they were sent
    Vector<Long> a(kChunkLimit + 1), b(kChunkLimit + 1), ra(a.Dim()), rb(b.Dim());
    FillSequence(a, rank * 100000 + 800);
    FillSequence(b, rank * 100000 + 900);
    if (first) {
      comm.Send(a.begin(), a.Dim(), peer, 270);
      comm.Send(b.begin(), b.Dim(), peer, 270);
      comm.Recv(ra.begin(), ra.Dim(), peer, 270);
      comm.Recv(rb.begin(), rb.Dim(), peer, 270);
    } else {
      comm.Recv(ra.begin(), ra.Dim(), peer, 270);
      comm.Recv(rb.begin(), rb.Dim(), peer, 270);
      comm.Send(a.begin(), a.Dim(), peer, 270);
      comm.Send(b.begin(), b.Dim(), peer, 270);
    }
    CheckSequence(ra, peer * 100000 + 800);
    CheckSequence(rb, peer * 100000 + 900);
  }
}

// SameNode reports the topology, not what the direct-read flags allow, so it holds for this rank
// whatever the build says.
void TestSameNode(const Comm& comm) {
  const Integer np = comm.Size(), rank = comm.Rank();
  SCTL_ASSERT(comm.SameNode(rank));  // this rank shares its own node

  Long local = 0;
  for (Integer i = 0; i < np; i++) local += (comm.SameNode(i) ? 1 : 0);
  SCTL_ASSERT(local >= 1 && local <= np);

  { // every rank this one calls a node peer says the same of it
    Vector<Long> mine(np), theirs(np);
    for (Integer i = 0; i < np; i++) mine[i] = (comm.SameNode(i) ? 1 : 0);
    comm.Alltoall(mine.begin(), 1, theirs.begin(), 1);
    for (Integer i = 0; i < np; i++) AssertEqual(mine[i], theirs[i]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  const Comm& comm = Comm::World();

  TestIsendIrecv(comm);
  TestIsendIrecvConsecutiveTags(comm);
  TestSendRecv(comm);
  TestSameNode(comm);
  TestBcast(comm);
  TestAllreduce(comm);
  TestScan(comm);
  TestAllgather(comm);
  TestAllgatherv(comm);
  TestAllgathervMixedTypes(comm);
  TestAlltoall(comm);
  TestAlltoallv(comm);
  TestIalltoallvSparse(comm);

  comm.Barrier();
  if (!comm.Rank()) {
    std::cout << "Comm large-count tests passed with forced chunk limit " << kChunkLimit << '\n';
  }

  Comm::MPI_Finalize();
  return 0;
}
