==============================================
Sparse QR Decomposition (:mod:`sksparse.spqr`)
==============================================

.. currentmodule:: sksparse.spqr


The :mod:`sksparse.spqr` module provides an interface to the SuiteSparse
`SPQR <spqr_github_>`_ package, which computes basic linear algebra
operations for sparse matrices.

The main function of this module is to compute the `QR factorization
<QR_wiki_>`_ of a sparse matrix :math:`A` with a fill-reducing permutation
:math:`E` such that:

.. math::

    QR = AE

These factors can then be used to solve linear systems of the form
:math:`Ax = b`.

The :mod:`.spqr` module exposes most of the capabilities of the SPQR
package including:

* Computation of the QR factors with fill-reducing
  permutation for both real and complex sparse matrices :math:`A`, in any
  format supported by :mod:`scipy.sparse`.
* An interface for using this decomposition to solve problems of the form
  :math:`Ax = b`.
* The ability to perform the fill-reduction analysis once, and then
  re-use it to efficiently decompose many matrices with the same pattern of
  non-zero entries.

.. _spqr_github: https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/SPQR
.. _QR_wiki: https://en.wikipedia.org/wiki/QR_decomposition


Quickstart
----------

If :math:`A` is a sparse matrix then the
following code computes the QR decomposition of :math:`A`:

.. code:: python

  from sksparse.spqr import spqr_factor
  Q, R, p = spqr(A)
  # Q @ R == A[:, p]

See the :ref:`example <spqr-example>` below for a demonstration of the
effect of the fill-reducing permutation.

Once the factorization has been computed, it can be used to solve a square
linear system:

.. code:: python

  from sksparse.spqr import spqr_factor
  A = ...  # a sparse matrix
  b = ...  # right-hand side
  x = spqr_solve(A, b)


.. _spqr-example:

Example
-------

To see how to use the QR factorization, we can load a sparse matrix
from the `SuiteSparse Matrix Collection <SSMC_>`_ and compute its ordering.

.. _SSMC: https://sparse.tamu.edu

.. literalinclude:: examples/spqr_example.py
   :language: python

This figure shows the effect of COLAMD ordering that reduces the fill-in of the
QR factorization of a sparse matrix.

.. figure:: examples/spqr_example.svg
   :alt: SPQR Example with COLAMD Ordering
   :align: center
   :width: 90%

   The number of non-zeros in the QR factorization of the original matrix
   (left) and the permuted matrix (right) using COLAMD ordering.


Function Interface
------------------

For users who just need to compute the QR factorization and return the factors
as sparse matrices, the :func:`spqr` function can be used. This function takes
a sparse matrix as input and returns the :math:`Q` and :math:`R` factors as
sparse matrices, along with the fill-reducing column permutation vector. There
is also a ``mode`` argument that is similar to the one in
:func:`scipy.linalg.qr`, which allows users to specify whether they want the
full or reduced factors, just the :math:`R` factor, or the Householder form of
the :math:`Q` factor.

Typically, the Householder form is substantially less memory-intensive than the
explicit :math:`Q` factor, especially for large sparse matrices. The function
:func:`spqr_qmult` can then be used to efficiently apply the :math:`Q` factor
to a dense matrix or vector without explicitly forming :math:`Q`.

To directly solve a linear system without explicitly computing the
factorization, the :func:`spqr_solve` function can be used. This function
performs the factorization internally and returns the solution to the linear
system. It can be used on non-square matrices as well. If :math:`A` is a matrix
of shape :math:`M` by :math:`N`, solving an overdetermined system
(:math:`M > N`) will return the least-squares solution, while solving an
underdetermined system (:math:`M < N`) will return the minimum-norm solution.


Object Interface
----------------

For more advanced usage, users can instantiate the :class:`SPQRFactor`
class. This class can be instantiated directly using its constructor, or more
conveniently using the :func:`spqr_factor` function.

When instantiated directly, the constructor performs a symbolic analysis of the
matrix, but does not compute the numeric factorization. The numeric
factorization is then performed by calling the :meth:`.SPQRFactor.factorize`
method.

The :func:`spqr_factor` function performs both the symbolic analysis and the
numeric factorization in one step, and returns an instance of the
:class:`SPQRFactor` class. The resulting :class:`SPQRFactor` object can then be
used to solve linear systems using its :meth:`SPQRFactor.solve` method. The
:meth:`SPQRFactor.factorize` method can be called again to factor a new matrix
with the same sparsity pattern. The :class:`SPQRFactor` class also provides
a method to apply the :math:`Q` factor to a dense matrix or vector, using the
:meth:`SPQRFactor.qmult` method.

Currently, it is not possible to extract the :math:`Q` and :math:`R` factors as
sparse matrices from the :class:`SPQRFactor` object. Users who need the
explicit factors should use the :func:`spqr` function instead.

After a factorization is computed, statistics about the factorization can be
accessed via the :attr:`SPQRFactor.info` attribute, which is an instance of
the :class:`SPQRInfo` class. This object contains various performance metrics
and statistics about the factorization process.


Exceptions and Warnings
-----------------------

Warnings issued by SPQR are converted into Python warnings of
type :exc:`SPQRWarning`. The module will also issue
a :exc:`~scipy.sparse.SparseEfficiencyWarning` if the input matrix is not
a :class:`~scipy.sparse.csc_array` (note that the
:class:`~scipy.sparse.csc_matrix` class is not supported, as it will be
deprecated and is not recommended for use in new code).

Errors detected by SPQR or by our wrapper code are converted into exceptions
of type :exc:`SPQRError` or an appropriate subclass. See the
:ref:`spqr-exceptions` for details.
