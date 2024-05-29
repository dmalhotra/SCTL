.. _tutorial-profile:

Using the Profile class
======================

The `Profile` class allows you to instrument your code for profiling, enabling you to measure various metrics like time, FLOPs, memory allocations, and more. It provides functions to mark the beginning and end of profiling blocks and to increment profiling counters. This tutorial will guide you through using the `Profile` class in your code.
For more advanced usage and additional features, please refer to the Profile class API in :ref:`profile.hpp <profile_hpp>`.

.. :ref:`Profile class documentation <profile-dox>`.

Enabling and Disabling Profiling
--------------------------------

You can enable or disable profiling using the `Enable` function. By default, profiling is disabled.

.. code-block:: cpp

    sctl::Profile::Enable(true)  // Enable profiling

Marking the Start and End of Profiling Blocks
----------------------------------------------

Use `Tic` to mark the start of a profiling block and `Toc` to mark its end.

.. code-block:: cpp

    sctl::Profile::Tic("BlockName")  // Start of profiling block
    // Your code to be profiled
    sctl::Profile::Toc()  // End of profiling block

Incrementing Profiling Counters
--------------------------------

You can manually increment profiling counters using `IncrementCounter`.

.. code-block:: cpp

    sctl::Profile::IncrementCounter(ProfileCounter::FLOP, numFlops)

Displaying Profiling Results
-----------------------------

You can print the profiling results using the `print` function. It displays various profiling metrics.

.. code-block:: cpp

    sctl::Profile::print()

Example Usage
-------------

Let's take a look at how you can use the `Profile` class in your code:

.. code-block:: cpp

    #include "sctl.hpp"

    int main(int argc, char** argv) {
      sctl::Comm::MPI_Init(&argc, &argv);

      // Enable profiling
      sctl::Profile::Enable(true);

      // Start profiling block
      sctl::Profile::Tic("TestBlock");

      // Your code to be profiled
      // ...

      // End profiling block
      sctl::Profile::Toc();

      // Print profiling results
      sctl::Profile::print();

      sctl::Comm::MPI_Finalize();
      return 0;
    }

Additional Notes
----------------

- You can create a scoped profiling block using the `Scoped` struct, which automatically marks the beginning and end of a block within its scope.
- There are various predefined profiling counters like time, FLOPs, heap allocations, etc., that you can use.

