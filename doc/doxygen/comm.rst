.. _comm_hpp:

comm.hpp
========

This header file provides an object-oriented wrapper to MPI (Message Passing Interface) operations; utilizing MPI when available, or defaulting to a self communicator otherwise.
It encapsulates common communication operations such as point-to-point communication, collective communication, data partitioning, sorting, and scatter operations.

Classes and Types
-----------------

.. doxygenclass:: SCTL_NAMESPACE::Comm
..   :members:
..

    **Methods**:

    - ``MPI_Init(argc, argv)``: Initialize MPI.

    - ``MPI_Finalize()``: Finalize MPI.

    - ``Self()``: Get the *self* communicator.

    - ``World()``: Get the *world* communicator.

    - ``Rank()``: Get the rank of the current process.

    - ``Size()``: Get the size of the communicator.

    - ``Barrier()``: Synchronize all processes.

    - ``Isend(sbuf, scount, dest, tag)``: Non-blocking send.

    - ``Irecv(rbuf, rcount, source, tag)``: Non-blocking receive.

    - ``Wait(req_ptr)``: Wait for a non-blocking send or receive.

    - ``Bcast(buf, count, root)``: Broadcast to all processes in the communicator.

    - ``Allgather(sbuf, scount, rbuf, rcount)``: Gather and concatenate equal-size messages from all processes.

    - ``Allgatherv(sbuf, scount, rbuf, rcounts, rdispls)``: Gather and concatenate messages of different lengths from all processes.

    - ``Alltoall(sbuf, scount, rbuf, rcount)``: Perform all-to-all operation for equal-size messages.

    - ``Ialltoallv_sparse(sbuf, scounts, sdispls, rbuf, rcounts, rdispls, tag)``: Sparse all-to-all communication.

    - ``Alltoallv(sbuf, scounts, sdispls, rbuf, rcounts, rdispls)``: All-to-all communication with varying send and receive counts and displacements.

    - ``Allreduce(sbuf, rbuf, count, op)``: Perform an all-reduce operation.

    - ``Scan(sbuf, rbuf, count, op)``: Perform a scan operation.

    - ``PartitionW(nodeList, wts_)``: Perform a weighted partitioning of a vector.

    - ``PartitionN(v, N)``: Partition a vector given the number of local elements after partitioning.

    - ``PartitionS(nodeList, splitter, comp)``: Perform a partitioning of a vector with a splitter element using a custom comparison function.

    - ``HyperQuickSort(arr, SortedElem, comp)``: Sort the elements of an array using HyperQuickSort algorithm with a custom comparison function.

    - ``SortScatterIndex(key, scatter_index, split_key)``: Generate scatter indices corresponding to a sorted array.

    - ``ScatterForward(data_, scatter_index)``: Scatter data elements forward using the provided scatter index.

    - ``ScatterReverse(data_, scatter_index_, loc_size_)``: Scatter data elements in reverse using the provided scatter index.

    **Usage guide**: :ref:`Using the Comm class <tutorial-comm>`

.. doxygenenum:: SCTL_NAMESPACE::CommOp
..

..    - ``SUM``: Sum operation.
..
..    - ``MIN``: Minimum operation.
..
..    - ``MAX``: Maximum operation.

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/comm.hpp
   :language: c++
