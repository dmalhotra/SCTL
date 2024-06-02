.. _morton_hpp:

morton.hpp
==========

This header file provides a template class for representing a Morton index in a space-filling curve.

Classes and Types
-----------------

.. doxygenclass:: sctl::Morton
..   :members:
..

    **Constructor**:

    - ``Morton()``: Default constructor for Morton.
    - ``template <class T> explicit Morton(ConstIterator<T> coord, uint8_t depth_ = MAX_DEPTH)``: Constructor for Morton using coordinate iterators.

    **Methods**:

    - ``Depth() const``: Get the depth of the Morton index.
    - ``Coord() const``: Get the coordinates of the origin of a Morton box.
    - ``Next() const``: Get the Morton index of the next box.
    - ``Ancestor(ancestor_level) const``: Get the Morton index of the ancestor box at a given level.
    - ``DFD(level = MAX_DEPTH) const``: Get the Morton index of the deepest first descendant box.
    - ``NbrList(Vector<Morton>& nbrs, uint8_t level, bool periodic) const``: Get a list of the 3^DIM neighbor Morton IDs.
    - ``Children(Vector<Morton> &nlst) const``: Get the Morton indices of the children boxes.
    - ``operator<``, ``operator>``, ``operator!=``, ``operator==``, ``operator<=``, ``operator>=``: comparison operators.
    - ``isAncestor(Morton const &descendant) const``: Check if this Morton index is an ancestor of another Morton index.
    - ``Long operator-(const Morton<DIM> &I) const``: Compute the difference in Morton indices.

    **Friend Functions**:

    - ``operator<<``: Overloaded stream insertion operator.

|

.. raw:: html

   <div style="border-top: 3px solid"></div>
   <br>

.. literalinclude:: ../../include/sctl/morton.hpp
   :language: c++
