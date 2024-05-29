.. _vtudata_hpp:

vtudata.hpp
===========

This header file provides a data structure for storing data in the VTK (Visualization Toolkit) unstructured grid format. It includes facilities for managing point and cell data, as well as methods for writing the data to a VTK file.

Classes and Types
-----------------

.. doxygenclass:: SCTL_NAMESPACE::VTUData
..   :members:
..

    **Attributes**:

    - ``coord``: Vector storing 3D coordinates of points.

    - ``value``: Vector storing values associated with points.

    - ``connect``: Vector storing connectivity information for cells.

    - ``offset``: Vector storing offset information for cells.

    - ``types``: Vector storing cell types.

    **Methods**:

    - ``WriteVTK(fname, comm=Comm::Self())``: Writes the VTU data to a VTK file.

    **Usage guide**: :ref:`Using VTUData class <tutorial-vtudata>`

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/vtudata.hpp
   :language: c++

