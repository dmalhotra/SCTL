.. _tree_hpp:

tree.hpp
========

This header file provides functionality for creating and manipulating tree data structures, particularly designed for spatial partitioning and distributed computing.

Classes and Types
-----------------

.. doxygenclass:: sctl::Tree
..   :members:
..

    **Structs**:

    - ``NodeAttr``: Struct defining attributes of tree nodes.

    - ``NodeLists``: Struct defining node-lists of tree nodes.

    **Constructors**:

    - ``Tree(comm)``: Construct a distributed memory tree.

    **Methods**:

    - ``Dim()``: Returns the number of spatial dimensions.

    - ``GetPartitionMID()``: Returns the vector of Morton IDs partitioning the processor domains.

    - ``GetNodeMID()``: Returns the vector of Morton IDs of tree nodes.

    - ``GetNodeAttr()``: Returns the vector of attributes of tree nodes.

    - ``GetNodeLists()``: Returns the vector of node-lists of tree nodes.

    - ``GetComm()``: Retrieves the communicator associated with the tree.

    - ``UpdateRefinement(coord, M, balance21, periodic)``: Update tree refinement and repartition node data among the new tree nodes.

    - ``AddData(name, data, cnt)``: Add named data to the tree nodes.

    - ``GetData(data, cnt, name)``: Get node data.

    - ``ReduceBroadcast(name)``: Perform reduction operation and broadcast.

    - ``Broadcast(name)``: Broadcast operation.

    - ``DeleteData(name)``: Delete data from the tree nodes.

    - ``WriteTreeVTK(fname, show_ghost)``: Write VTK visualization.

    **Usage guide**: :ref:`Using Tree and PtTree classes <tutorial-tree>`

|

.. doxygenclass:: sctl::PtTree
..   :members:
..

    **Constructors**:

    - ``PtTree(comm)``: Construct a distributed memory particle tree.

    **Methods**:

    - ``AddParticles(name, coord)``: Add particles to the point tree.

    - ``AddParticleData(data_name, particle_name, data)``: Add particle data to the point tree.

    - ``GetParticleData(data, data_name)``: Get particle data from the point tree.

    - ``UpdateRefinement(coord, M, balance21, periodic)``: Update refinement of the point tree based on given coordinates.

    - ``DeleteParticleData(data_name)``: Delete particle data from the point tree.

    - ``WriteParticleVTK(fname, data_name, show_ghost)``: Write particle data to a VTK file.

    **Usage guide**: :ref:`Using Tree and PtTree classes <tutorial-tree>`

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/tree.hpp
   :language: c++

