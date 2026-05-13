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

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  const Comm comm = Comm::World();

  TestIsendIrecv(comm);
  TestIsendIrecvConsecutiveTags(comm);
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
