.. scikit-sparse documentation master file, created by
   sphinx-quickstart on Sat Dec 12 22:10:41 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================================
Scikit-Sparse -- Sparse Matrix Extensions for SciPy
===================================================

Scikit-Sparse is a collection of sparse matrix extensions for SciPy,
with a focus on factorization routines and reordering methods.

The package is a thin wrapper around the `SuiteSparse <suitesparse_website_>`_
library, to make it compatible with :mod:`scipy.sparse` arrays.

.. _suitesparse_website: https://people.engr.tamu.edu/davis/suitesparse.html


Features
--------

- Fill-reducing orderings: AMD and COLAMD
- Cholesky factorization via CHOLMOD
- Integration with SciPy sparse arrays


Installation
------------

.. code-block:: bash

    conda install -c conda-forge scikit-sparse-dev
    # or
    pip install scikit-sparse-dev


Quick Example
-------------

.. code-block:: python

    import numpy as np
    from scipy.sparse import csc_array
    from sksparse.cholmod import cho_factor

    # Create a sparse positive definite matrix
    A = csc_array([[4, 1, 0],
                    [1, 3, 0],
                    [0, 0, 2]])

    # Perform Cholesky factorization
    f = cho_factor(A)

    # Solve Ax = b
    b = np.array([1, 2, 3])
    x = f.solve(b)


Learn More
----------

* :doc:`Overview <overview>` - Introduction and installation instructions
* :doc:`User Guide <tutorial/index>` - Tutorials and examples
* :doc:`API Reference <reference/index>` - Detailed API documentation
* :doc:`Change Log <changes>` - List of changes by version

.. _github_repo: https://github.com/scikit-sparse/scikit-sparse
.. _github_issues: https://github.com/scikit-sparse/scikit-sparse/issues


.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   Overview <overview>

   User Guide <tutorial/index>

   API Reference <reference/index>

   Change Log <changes>
