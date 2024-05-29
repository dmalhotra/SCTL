.. _tutorial-tree:

Using the Tree and PtTree classes
=================================


For more advanced usage and additional features, please refer to the class API in :ref:`tree.hpp <tree_hpp>`.


1. Overview of Tree and PtTree Classes
---------------------------------------

- **Tree Class**:

  - Represents a generic distributed/parallel tree data structure.

  - Provides functionalities for tree manipulation, refinement, data addition, retrieval, and visualization.

  - Designed to handle tree structures in arbitrary dimensions.

- **PtTree Class**:

  - Inherits from the Tree class and specializes in handling point data within the tree structure.

  - Specifically designed for scenarios where points are distributed across multiple processors.

  - Provides additional functionalities for adding, retrieving, and visualizing point data.

2. Usage of Tree Class
----------------------

To use the `Tree` class, follow these steps:

1. **Initialization**:

   .. code-block:: cpp

      Tree<Real, DIM> tree;

   Create an instance of the `Tree` class. Optionally, you can specify the datatype `Real` and the dimensionality `DIM`.

2. **Adding Data**:

   .. code-block:: cpp

      Vector<Real> data;
      Vector<Long> cnt;
      tree.AddData("name", data, cnt);

   Add data to the tree nodes. Provide a name for the data, along with the corresponding data vector and count vector.

3. **Data Retrieval**:

   .. code-block:: cpp

      Vector<Real> retrievedData;
      Vector<Long> counts;
      tree.GetData(retrievedData, counts, "name");

   Retrieve data from the tree nodes using the specified data name.

4. **Visualization**:

   .. code-block:: cpp

      tree.WriteTreeVTK("tree");

   Generate a VTK visualization of the tree structure.

3. Usage of PtTree Class
------------------------

To utilize the `PtTree` class for point data management, follow these steps:

1. **Initialization**:

   .. code-block:: cpp

      PtTree<Real, DIM> ptTree;

   Create an instance of the `PtTree` class, which inherits from the `Tree` class.

2. **Adding Particles**:

   .. code-block:: cpp

      Vector<Real> coordinates;
      ptTree.AddParticles("pt", coordinates);

   Add particle coordinates to the point tree, specifying a name for the particle group.

3. **Adding Particle Data**:

   .. code-block:: cpp

      Vector<Real> particleData;
      ptTree.AddParticleData("data_name", "pt", particleData);

   Add data associated with the particles. Provide a name for the data, along with the corresponding particle group name.

4. **Update Refinement**:

   .. code-block:: cpp

      ptTree.UpdateRefinement(coordinates, 1000);

   Update the refinement of the point tree based on the given coordinates, with a maximum number of points per box.

4. Example: Tree and PtTree in Action
--------------------------------------

Here's an example demonstrating the usage of the `PtTree` class:

.. code-block:: cpp

      template <class Real, Integer DIM> void ExamplePtTree() {
          Long N = 100000;
          Vector<Real> X(N*DIM), f(N);
          for (Long i = 0; i < N; i++) { // Set coordinates (X), and values (f)
            f[i] = 0;
            for (Integer k = 0; k < DIM; k++) {
              X[i*DIM+k] = pow<3>(drand48()*2-1.0)*0.5+0.5;
              f[i] += X[i*DIM+k]*k;
            }
          }

          PtTree<Real,DIM> tree;
          tree.AddParticles("pt", X);
          tree.AddParticleData("pt-value", "pt", f);
          tree.UpdateRefinement(X, 1000); // refine tree with max 1000 points per box.

          { // manipulate tree node data
            const auto& node_lst = tree.GetNodeLists(); // Get interaction lists
            //const auto& node_mid = tree.GetNodeMID();
            //const auto& node_attr = tree.GetNodeAttr();

            // get point values and count for each node
            Vector<Real> value;
            Vector<Long> cnt, dsp;
            tree.GetData(value, cnt, "pt-value");

            // compute the dsp (the point offset) for each node
            dsp.ReInit(cnt.Dim()); dsp = 0;
            omp_par::scan(cnt.begin(), dsp.begin(), cnt.Dim());

            Long node_idx = 0;
            for (Long i = 0; i < cnt.Dim(); i++) { // find the tree node with maximum points
              if (cnt[node_idx] < cnt[i]) node_idx = i;
            }

            for (Long j = 0; j < cnt[node_idx]; j++) { // for this node, set all pt-value to -1
              value[dsp[node_idx]+j] = -1;
            }

            for (const Long nbr_idx : node_lst[node_idx].nbr) { // loop over the neighbors and set pt-value to 2
              if (nbr_idx >= 0 && nbr_idx != node_idx) {
                for (Long j = 0; j < cnt[nbr_idx]; j++) {
                  value[dsp[nbr_idx]+j] = 2;
                }
              }
            }
          }

          // Generate visualization
          tree.WriteParticleVTK("pt", "pt-value");
          tree.WriteTreeVTK("tree");
      }


This example initializes a point tree, adds particles with associated values, updates the refinement of the tree, manipulates node data, and generates visualization outputs.

