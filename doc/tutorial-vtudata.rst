.. _tutorial-vtudata:


Using VTUDatal
================

This tutorial provided an overview of how to use the `VTUData` class to store and write data in the VTK unstructured mesh format.
This format is commonly used in scientific computing for visualizing complex data sets.

For more information, refer to the :ref:`VTUData class documentation <vtudata-dox>`.

Usage
-----

To use the `VTUData` class, follow these steps:

1. **Create a `VTUData` object:** Instantiate a `VTUData` object to store your data.

.. code-block:: cpp

    VTUData vtu_data;

2. **Add point data:** Populate the `coord` and `value` vectors with the coordinates and values of the data points, respectively.

.. code-block:: cpp

    for (long i = 0; i < num_points; i++) {
        // Add coordinates
        for (long k = 0; k < 3; k++) {
            vtu_data.coord.PushBack((VTUData::VTKReal)drand48());
        }
        // Add values
        vtu_data.value.PushBack((VTUData::VTKReal)drand48());
    }

3. **Add cell data:** Add cell data by specifying the connectivity, offset, and cell types.

.. code-block:: cpp

    // Add tetrahedron
    vtu_data.types.PushBack(10); // VTK_TETRA
    for (long i = 0; i < 4; i++) {
        vtu_data.connect.PushBack(i);
    }
    vtu_data.offset.PushBack(vtu_data.connect.Dim());

    // Add triangle
    vtu_data.types.PushBack(5); // VTK_TRIANGLE
    for (long i = 4; i < 7; i++) {
        vtu_data.connect.PushBack(i);
    }
    vtu_data.offset.PushBack(vtu_data.connect.Dim());

4. **Write to VTK file:** Finally, write the `VTUData` object to a VTK file using the `WriteVTK` method.

.. code-block:: cpp

    vtu_data.WriteVTK("output.vtu");

