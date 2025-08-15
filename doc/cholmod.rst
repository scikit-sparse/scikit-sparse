.. Copyright (C) 2025, The scikit-sparse developers. All rights reserved.
   Part of the scikit-sparse project.
   See pyproject.toml for full author list and LICENSE.txt for license details.
   SPDX-License-Identifier: BSD-2-Clause

Cholesky Decomposition (:mod:`sksparse.cholmod`)
================================================

.. module:: sksparse.cholmod
   :synopsis: Cholesky decomposition using CHOLMOD

.. versionadded:: 0.1

.. versionchanged:: 0.5
   Major API updates to more closely resemble the :func:`scipy.linalg.cholesky`
   dense interface, and incorporate more functions from the CHOLMOD MATLAB
   interface.

The :mod:`sksparse.cholmod` module provides an interface to the SuiteSparse
`CHOLMOD <https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/CHOLMOD>`_
package, which computes basic linear algebra operations for sparse, symmetric,
positive-definite matrices.

The main function of this module is to compute the `Cholesky factor
<http://en.wikipedia.org/wiki/Cholesky_decomposition>`_ :math:`L` of a sparse,
symmetric (Hermitian if complex), positive-definite matrix :math:`A` with
a fill-reducing permutation :math:`P`, such that:

.. math::
    LL^{\top} = PAP^{\top}.

For symmetric, indefinite matrices, compute the LDL factorization:

.. math::
   LDL^{\top} = PAP^{\top}.

Either of these factors can then be used to solve linear systems of the form
:math:`Ax = b`.

The :mod:`.cholmod` module exposes most of the capabilities of the CHOLMOD
package including:

* Computation of the Cholesky factor with fill-reducing
  permutation for both real and complex sparse matrices :math:`A`, in any
  format supported by :mod:`scipy.sparse`.
* An interface for using this decomposition to solve problems of the form
  :math:`Ax = b`.
* Functions to compute a rank-:math:`k` "update" or "downdate" of the Cholesky
  factor.
* The ability to perform the fill-reduction analysis once, and then
  re-use it to efficiently decompose many matrices with the same pattern of
  non-zero entries.


Quickstart
----------

If :math:`A` is a sparse, symmetric, positive-definite matrix, then the
following code computes the Cholesky decomposition of :math:`A`:

.. code:: python

  from sksparse.cholmod import cholesky
  L = cholesky(A)

or, with a fill-reducing permutation:

.. code:: python

  L, p = cholesky(A, order="default")

See the :ref:`example <cholesky-example>` below for a demonstration of the
effect of the fill-reducing permutation.

Once the factorization has been computed, it can be used to solve a linear
system:

.. code:: python

  from sksparse.cholmod import ldl, ldlsolve
  A = ...                # a symmetric, positive-definite sparse matrix
  b = ...                # right-hand side
  L, D = ldl(A)          # compute LDL^T factorization
  x = ldlsolve(L, D, b)  # solve Ax = b


Top-level functions
-------------------

The main function this module provides is :func:`cholesky`, for computing the
Cholesy factor and optionally a fill-reducing permutation of a sparse matrix.

.. autofunction:: cholesky

For matrices that are symmetric but not positive-definite, the LDL factorization
can be computed using the :func:`ldl` function.

.. autofunction:: ldl

Once the factorization has been computed, the resulting matrices can be used to
solve linear systems using :func:`ldlsolve`:

.. autofunction:: ldlsolve


Error handling
--------------

Warnings issued by CHOLMOD are converted into Python warnings of
type :exc:`CholmodWarning`. The module will also issue
a :exc:`~scipy.sparse.SparseEfficiencyWarning` if the input matrix is not
a :class:`~scipy.sparse.csc_array` (note that the
:class:`~scipy.sparse.csc_matrix` class is not supported, as it will be
deprecated and is not recommended for use in new code).

.. autoexception:: CholmodWarning
  :show-inheritance:

.. autoexception:: CholmodSmallDiagonalWarning
  :show-inheritance:


Errors detected by CHOLMOD or by our wrapper code are converted into exceptions
of type :exc:`CholmodError` or an appropriate subclass.

.. autoexception:: CholmodError
  :show-inheritance:

.. autoexception:: CholmodNotPositiveDefiniteError
  :show-inheritance:

.. autoexception:: CholmodNotInstalledError
  :show-inheritance:

.. autoexception:: CholmodOutOfMemoryError
  :show-inheritance:

.. autoexception:: CholmodOverflowError
  :show-inheritance:

.. autoexception:: CholmodInvalidInputError
  :show-inheritance:

.. autoexception:: CholmodGpuProblemError
  :show-inheritance:


.. _cholesky-example:

Example
-------

This figure shows the effect of AMD ordering that reduces the fill-in of the
Cholesky factorization of a sparse matrix.

.. figure:: examples/cholesky_example.svg
   :alt: Cholesky Example with AMD Ordering
   :align: center
   :width: 90%

   The number of non-zeros in the Cholesky factorization of the original matrix
   (left) and the permuted matrix (right) using AMD ordering.

The source code for this example is:

.. literalinclude:: examples/cholmod_example.py
   :language: python
