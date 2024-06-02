.. _tutorial-comm:

Using the Comm Class
====================

The `Comm` class provides an object-oriented wrapper to communication operations, primarily designed to work with MPI. When MPI is available (indicated by the macro ``SCTL_HAVE_MPI``), it uses MPI functionalities. Otherwise, it defaults to the *self* communicator.
The following tutorial provides a brief introduction to using the `Comm` class. For more advanced usage and additional features, please refer to the Comm class API in :ref:`comm.hpp <comm_hpp>`.

Initialization
---------------

To begin using the `Comm` class, you should initialize MPI if it's available in your environment.

.. code-block:: cpp

   Comm::MPI_Init(&argc, &argv);

Creating Communicators
-----------------------

You can create instances of the `Comm` class to represent different communication contexts.

1. **Default Constructor**: This initializes a communicator to represent the *self* communicator.

2. **Copy Constructor**: You can duplicate an existing communicator.

3. **Static Methods**: The class provides static methods `Self()` and `World()` to get the *self* and *world* communicators, respectively.

Communication Methods
----------------------

Once you have a communicator, you can perform various communication operations using it.

1. **Rank and Size**: You can obtain the rank and size of processes within the communicator.

2. **Barrier**: Synchronize all processes within the communicator.

3. **Send and Receive**: Perform non-blocking send and receive operations.

4. **Broadcast (Bcast)**: Broadcast data from one process to all others.

5. **Gather and Scatter**: Gather and scatter data among processes.

6. **Allreduce and Scan**: Perform all-reduce and scan operations.

7. **Partitioning and Sorting**: Perform partitioning and sorting operations on data vectors.

MPI Conversion
----------------

If you are working with MPI directly and need to convert between `MPI_Comm` and `Comm`, the class provides methods for conversion.

Cleanup
---------

When you're done with MPI communication, make sure to finalize MPI.

.. code-block:: cpp

   Comm::MPI_Finalize()


Example Usage
---------------

Below is a simplified example demonstrating the use of `Comm`:

.. code-block:: cpp

   #include "sctl.hpp"

   int main(int argc, char** argv) {
     // Initialize MPI if available
     Comm::MPI_Init(&argc, &argv);

     // Create a communicator representing the *self* communicator
     Comm comm = Comm::Self();

     // Get rank and size
     Integer rank = comm.Rank();
     Integer size = comm.Size();

     // Perform communication operations...

     // Finalize MPI
     Comm::MPI_Finalize();
   }


Compiling Code
--------------

To compile code that utilizes the `Comm` class, follow these steps:

**Without MPI**: If you're compiling code without MPI support, you can use a standard C++ compiler. Here's a basic example using g++:

.. code-block:: bash

    g++ -std=c++11 your_code.cpp -o your_executable

**With MPI**: If your code uses MPI functionality, you need to compile it with an MPI compiler and link against the MPI library. Here's an example using `mpicxx`:

.. code-block:: bash

    mpicxx -std=c++11 -DSCTL_HAVE_MPI your_code.cpp -o your_executable

Ensure to define the macro ``SCTL_HAVE_MPI`` during compilation.

