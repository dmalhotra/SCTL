.. _ompUtils_hpp:

ompUtils.hpp
============

This header file defines utility functions for OpenMP parallel operations such as merge, merge sort, reduce, and scan. These functions are part of the `sctl::omp_par` namespace.

Functions
---------


- ``merge(ConstIter A_begin, ConstIter A_end, ConstIter B_begin, ConstIter B_end, Iter C_begin, Int count, comp)``:
  Merges two sorted ranges into a single sorted range.

..

- ``merge_sort(ConstIter begin, ConstIter end, comp)``:
  Performs merge sort on a range using a custom comparison function.

..

- ``merge_sort(ConstIter begin, ConstIter end)``:
  Performs merge sort on a range.

..

- ``value_type reduce(ConstIter input, Int cnt)``:
  Reduces the elements in a range to a single value.

..

- ``scan(ConstIter input, Iter output, Int cnt)``:
  Performs a parallel prefix sum (scan) operation on a range.


|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/ompUtils.hpp
   :language: c++
