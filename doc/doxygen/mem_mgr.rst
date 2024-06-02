.. _mem_mgr_hpp:

mem_mgr.hpp
===========

This header file defines memory management functions for aligned allocation and deallocation using placement new and object destructor calls. These functions provide an alternative to using `new` and `delete` for memory management.

Functions
---------

.. doxygenfunction:: sctl::aligned_new

..

.. doxygenfunction:: sctl::aligned_delete


.. **Aligned Allocation**:
.. 
.. - ``Iterator<ValueType> aligned_new<ValueType>(n_elem, mem_mgr = &MemoryManager::glbMemMgr())``:
..   Aligned allocation as an alternative to `new`. Uses placement new to construct objects. Returns an iterator to the allocated memory.
.. 
.. **Aligned Deallocation**:
.. 
.. - ``aligned_delete<ValueType>(Iterator<ValueType> A, mem_mgr = &MemoryManager::glbMemMgr())``:
..   Aligned deallocation as an alternative to `delete`. Calls the object destructor for deallocated memory.
.. 
.. |
.. 
.. .. raw:: html
.. 
..    <div style="border-top: 3px solid"></div>
..    <br>
.. 
.. .. literalinclude:: ../../include/sctl/mem_mgr.hpp
..    :language: c++
.. 
