================================================
Cholesky Decomposition (:mod:`sksparse.cholmod`)
================================================

.. currentmodule:: sksparse.cholmod


The :mod:`sksparse.cholmod` module provides an interface to the SuiteSparse
`CHOLMOD <cholmod_github_>`_ package, which computes basic linear algebra
operations for sparse, symmetric, positive-definite matrices.

The main function of this module is to compute the `Cholesky factor
<cholesky_wiki_>`_ :math:`L` of a sparse,
symmetric (Hermitian if complex), positive-definite matrix :math:`A` with
a fill-reducing permutation :math:`P`, such that:

.. math::

    LL^{\top} = PAP^{\top}.

For matrices that are symmetric but may be numerically close to semi-definite,
the module can compute the LDL factorization:

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

This wrapper handles both 32-bit and 64-bit integer types, depending on the
input matrix format.

.. _cholmod_github: https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/CHOLMOD
.. _cholesky_wiki: http://en.wikipedia.org/wiki/Cholesky_decomposition


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

  from sksparse.cholmod import ldl_factor
  A = ...            # a symmetric, positive-definite sparse matrix
  b = ...            # right-hand side
  f = ldl_factor(A)  # compute LDL^T factorization
  x = f.solve(b)     # solve Ax = b


Examples
--------

.. _cholesky-example:

Cholesky Example
++++++++++++++++

To see how to use the Cholesky factorization, we can load a sparse matrix
from the `SuiteSparse Matrix Collection <SSMC_>`_ and compute its ordering.

.. _SSMC: https://sparse.tamu.edu

.. literalinclude:: examples/cholmod_example.py
   :language: python

This figure shows the effect of AMD ordering that reduces the fill-in of the
Cholesky factorization of a sparse matrix.

.. figure:: examples/cholesky_example.svg
   :alt: Cholesky Example with AMD Ordering
   :align: center
   :width: 90%

   The number of non-zeros in the Cholesky factorization of the original matrix
   (left) and the permuted matrix (right) using AMD ordering.


Nested Dissection Example
+++++++++++++++++++++++++

To see the effects of nested dissection ordering, we can load a sparse matrix
and compute its ordering.

.. literalinclude:: examples/nesdis_example.py
   :language: python

This figure shows the effect of nested dissection ordering that reduces the
fill-in of the LU factorization of a sparse matrix in a case where the AMD
order *does not* help.

.. figure:: examples/nesdis_example.svg
   :alt: LU Example with Nesdis Ordering
   :align: center
   :width: 90%

   The number of non-zeros in the LU factorization of the original matrix
   and the permuted matrix using AMD and nested dissection ordering.


Function Interface
------------------

For users who want to directly compute the factorization without needing to
manipulate the :class:`CholeskyFactor` object, the :mod:`.cholmod` module
provides the :func:`cholesky` and :func:`ldl` functions that perform both the
symbolic analysis and the numerical factorization in one step, and return the
matrices directly.


Object Interface
----------------

For more advanced usage, users can instantiate the :class:`CholeskyFactor`
class. This class can be instantiated directly using its constructor, or more
conveniently using the :func:`cho_factor` or :func:`ldl_factor` functions.

When instantiated directly, the constructor performs a symbolic analysis of the
matrix, but does not compute the numerical factorization. The numerical
factorization is then performed by calling the :meth:`.CholeskyFactor.factorize` method.

The :func:`cho_factor` and :func:`ldl_factor` functions perform both the
symbolic analysis and the numerical factorization in one step, and return an
instance of the :class:`CholeskyFactor` class.

The resulting :class:`CholeskyFactor` object can then be used to solve linear
systems using its :meth:`CholeskyFactor.solve` method, or to update the
factorization in-place using the :meth:`.update`, :meth:`.rowadd`,
:meth:`.rowdel`, and :meth:`.resymbol` methods.

The :meth:`CholeskyFactor.factorize` method can be called again to factor a new
matrix with the same sparsity pattern.


Symbolic Analysis
-----------------

In addition to numerical factorization, :mod:`.cholmod` provides symbolic
operations :func:`symbfact`, and :func:`etree` that can be used to analyze the
structure of the Cholesky factor and to compute fill-reducing permutations.


Graph Partitioning
------------------

The :mod:`.cholmod` module also includes functions for graph partitioning and
node reordering, which can be used like the :mod:`~sksparse.amd` and
:mod:`~sksparse.colamd` (and their constrained counterparts) modules to reduce
fill-in during factorization.

These functions provide a direct interface to the corresponding CHOLMOD
functions that are used internally by :func:`cholesky` and :func:`ldl` when the
``order`` argument is specified. The functions :func:`bisect`, :func:`metis`,
and :func:`nesdis` can be used to compute fill-reducing orderings, and the
:class:`SeparatorTree` class represents the resulting separator tree.


Exceptions and Warnings
-----------------------

Warnings issued by CHOLMOD are converted into Python warnings of
type :exc:`CholmodWarning`. The module will also issue
a :exc:`~scipy.sparse.SparseEfficiencyWarning` if the input matrix is not
a :class:`~scipy.sparse.csc_array` (note that the
:class:`~scipy.sparse.csc_matrix` class is not supported, as it will be
deprecated and is not recommended for use in new code).

Errors detected by CHOLMOD or by our wrapper code are converted into exceptions
of type :exc:`CholmodError` or an appropriate subclass. See the
:ref:`cholmod-exceptions` for details.
