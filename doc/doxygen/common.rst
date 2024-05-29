.. _common_hpp:

common.hpp
==========

This header file provides common definitions and utilities. It defines macros, types, and functions that are used across various components of the library.

Macros
------

``SCTL_DATA_PATH``:
    This macro defines the path to the data directory used by the library. If not defined, it defaults to "./data/".

.. ``SCTL_NAMESPACE``:
..     This macro defines the namespace used by the SCTL framework. If not defined, it defaults to "sctl".

``SCTL_PROFILE``:
    This macro defines the granularity level for profiling. If not defined, it defaults to -1.

``SCTL_ALIGN_BYTES``:
    This macro defines the alignment requirement for memory allocation based on the processor architecture.

``SCTL_MEM_ALIGN``:
    This macro defines the alignment requirement for memory management. It defaults to the maximum of 64 and ``SCTL_ALIGN_BYTES``.

``SCTL_GLOBAL_MEM_BUFF``:
    This macro defines the size of the global memory buffer in MB used for memory management. It defaults to 0.

Functions
---------

``SCTL_WARN(msg)``:
    Prints a warning message to the standard error stream.

``SCTL_ERROR(msg)``:
    Prints an error message to the standard error stream and aborts the program.

``SCTL_ASSERT_MSG(cond, msg)``:
    Asserts a condition and prints an error message if the condition is not met.

``SCTL_ASSERT(cond)``:
    Asserts a condition and prints an error message if the condition is not met, including file name, line number, and function name.

``SCTL_UNUSED(x)``:
    Macro to silence unused variable warnings.

Namespaces
----------

``SCTL_NAMESPACE``:
    The namespace under which all SCTL components reside.

Classes and Types
-----------------

``SCTL_NAMESPACE::ConstIterator``:
    Template class representing a constant iterator.

``SCTL_NAMESPACE::Iterator``:
    Template class representing a mutable iterator.

``SCTL_NAMESPACE::StaticArray``:
    Template class representing a static array.

    - Parameters:
        - ``ValueType``: Type of elements in the array.
        - ``DIM``: Dimension of the array.

    If ``SCTL_MEMDEBUG`` is defined, these classes are defined with additional debugging functionality. Otherwise, they are defined as aliases for pointer types.

``SCTL_NAMESPACE::Integer``:
    Alias for a long integer type representing bounded numbers less than 32k.

``SCTL_NAMESPACE::Long``:
    Alias for a 64-bit integer type representing problem size.

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/common.hpp
   :language: c++

