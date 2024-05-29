.. _profile_hpp:

profile.hpp
============

This header file provides functionality for manual instrumentation of code, allowing users to define profiling blocks and report changes in various counters (e.g., time, flops, memory allocations) between the start and end of each profiling block.

Classes and Types
-----------------

.. doxygenclass:: SCTL_NAMESPACE::Profile
..   :members:
..

    **Methods**:

    - ``Enable(bool state)``: Enable or disable the profiler.

    - ``Tic(name, comm_ptr = None, sync = False, verbose = 1)``: Marks the start of a profiling block.

    - ``Toc()``: Marks the end of a profiling block.

    - ``IncrementCounter(prof_field, x)``: Increment a profiling counter.

    - ``GetProfField(name)``: Returns a profiling expression identified by a string name.

    - ``SetProfField(name, expr)``: Create a named profiling field from a given expr object.

    - ``UnaryExpr(e1, op)``: Construct a profiling expression from a given unary operator op acting on the output of e1.

    - ``BinaryExpr(e1, e2, op)``: Construct a profiling expression from a given binary operator op acting on the output of e1 and e2.

    - ``CommReduceExpr(e1, comm_op)``: Construct a profiling expression by applying a distributed reduction operation on the output of e1.

    - ``print(comm_ptr = None, fields = [], format = [])``: Display the profiling output.

    - ``reset()``: Clear all profiling data.

    **Types**:

    - ``ProfileCounter``: Enumerates the available counters for profiling.

    - ``Scoped``: Defines a profiling block through the lifetime/scope of its instance.

    - ``ProfExpr``: Represents a profiling expression defined in terms of one or more counters and is printed in the profiling output.

    **Usage guide**: :ref:`Using Profile class <tutorial-profile>`

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/profile.hpp
   :language: c++
