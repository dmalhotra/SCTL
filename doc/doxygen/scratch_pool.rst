.. _scratch_pool_hpp:

scratch_pool.hpp
================

This header file provides ``ScratchBuf<T>`` and ``ScratchPool`` — a per-thread
stack allocator for short-lived buffers. Allocation is a pointer bump within a
thread-local chunk (lock-free, NUMA-local first-touch), making it the
recommended replacement for owning ``Vector<T>`` temporaries in hot loops.

Classes and Types
-----------------

.. doxygenclass:: sctl::ScratchBuf
..   :members:
..

    **Construction**:

    - ``ScratchBuf(Long count)``: Allocate ``count`` elements from the calling
      thread's pool (``ScratchPool::Instance()``).

    - ``ScratchBuf(Long count, ScratchPool& pool)``: Allocate from a
      user-supplied pool (for tests / isolation; not thread-safe).

    Copy, move, ``new``, and ``delete`` are all deleted — ``ScratchBuf`` is
    stack-only and bound to LIFO destruction order within its pool.

    **Methods**:

    - ``begin()``, ``end()``: Iterators over the buffer.

    - ``Dim()``: Number of elements.

    - ``operator[](Long i)``: Element access.

|

.. doxygenclass:: sctl::ScratchPool
..   :members:
..

    **Static methods**:

    - ``Instance()``: Returns the calling thread's pool (``thread_local``
      storage; persists across OpenMP parallel regions).

    **Diagnostics**:

    - ``DebugChunkCount()``: Number of chunks currently held.

    - ``DebugLiveCount()``: Number of live allocations (exact under
      ``SCTL_MEMDEBUG``; ``0`` if known-empty, ``-1`` otherwise in release).

|

Usage
-----

Replace short-lived owning ``Vector<T>``:

.. code-block:: cpp

   {
     ScratchBuf<double> buf(N);
     Vector<double> v(buf);                 // non-owning, fixed-size view
     // ... use v[i], v.begin(), v.end() ...
   }                                        // buf is freed here (LIFO pop)

Pass ``disable_reinit=false`` to the ``Vector`` constructor only when a
callback may legitimately need to resize/rebind the view.

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/scratch_pool.hpp
   :language: c++
