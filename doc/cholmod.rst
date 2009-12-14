Sparse Cholesky decomposition (:mod:`scikits.sparse.cholmod`)
=============================================================

.. module:: scikits.sparse.cholmod
   :synopsis: Sparse Cholesky decomposition using CHOLMOD

.. versionadded:: 0.1

Overview
--------

This module exposes most of the capabilities of the `CHOLMOD
<http://www.cise.ufl.edu/research/sparse/cholmod/>`_
package. Specifically, it provides:

* Computation of the `Cholesky decomposition
  <http://en.wikipedia.org/wiki/Cholesky_decomposition>`_ :math:`LL' =
  A` or :math:`LDL' = A` (with fill-reducing permutation) for both
  real and complex sparse matrices :math:`A`, in any format supported
  by :mod:`scipy.sparse`. (However, CSC matrices will be most
  efficient.)
* A convenient and efficient interface for using this decomposition to
  solve problems of the form :math:`Ax = b`.
* The ability to perform the costly fill-reduction analysis once, and
  then re-use it to efficiently decompose many matrices with the same
  pattern of non-zero entries.
* In-place 'update' and 'downdate' operations, for computing the
  Cholesky decomposition of a product :math:`AA'` when the columns of
  :math:`A` become available incrementally (e.g., due to memory
  constraints), or when many matrices with similar but non-identical
  columns must be factored.

The most common use is probably for solving least squares problems.

Quickstart
----------

If :math:`A` is a sparse, symmetric, positive-definite matrix, and
:math:`b` is a matrix or vector (either sparse or dense), then the
following code solves the equation :math:`Ax = b`::

  from scikits.sparse.cholmod import cholesky
  factor = cholesky(A)
  x = factor(b)

If you have a least-squares problem to solve, minimizing :math:`||Mx -
b||^2`, and :math:`M` is a sparse matrix, the `solution
<http://en.wikipedia.org/wiki/Linear_least_squares#Derivation_of_the_normal_equations>`_
is :math:`x = (M'M)^{-1} M'b`, which can be efficiently calculated
as::

  from scikits.sparse.cholmod import cholesky_AAt
  # Notice that CHOLMOD computes AA' and we want M'M, so we must set A = M'!
  factor = cholesky_AAt(M.T)
  x = factor(M.T * b)

Top-level functions
-------------------

All usage of this module starts by calling one of four functions, all
of which return a :class:`Factor` object, documented below.

Most users will want one of the ``cholesky`` functions, which perform
a fill-reduction analysis and decomposition together:

.. autofunction:: cholesky(A, beta=0, mode="auto")

.. autofunction:: cholesky_AAt(A, beta=0, mode="auto")

However, some users may want to break the fill-reduction analysis and
actual decomposition into separate steps, and instead begin with one
of the ``analyze`` functions, which perform only fill-reduction:

.. autofunction:: analyze(A, mode="auto")

.. autofunction:: analyze_AAt(A, mode="auto")

.. note:: Even if you used :func:`cholesky` or :func:`cholesky_AAt`,
  you can still call :meth:`cholesky_inplace()
  <Factor.cholesky_inplace>` or :meth:`cholesky_AAt_inplace()
  <Factor.cholesky_AAt_inplace>` on the resulting :class:`Factor` to
  quickly factor another matrix with the same non-zero pattern as your
  original matrix.

:class:`Factor` objects
-----------------------

.. class:: Factor

  A :class:`Factor` object represents the Cholesky decomposition of some
  matrix :math:`A` (or :math:`AA'`). Each :class:`Factor` fixes:

  * A specific fill-reducing permutation
  * A choice of which Cholesky algorithm to use (see :func:`analyze`)
  * Whether we are currently working with real numbers or complex

  Given a :class:`Factor` object, you can:

  * Compute new Cholesky decompositions of matrices that have the same
    pattern of non-zeros
  * Perform 'updates' or 'downdates'
  * Access the various Cholesky factors
  * Solve equations involving those factors

Factoring new matrices
++++++++++++++++++++++

.. automethod:: Factor.cholesky_inplace(A, beta=0)

.. automethod:: Factor.cholesky_AAt_inplace(A, beta=0)

.. automethod:: Factor.cholesky(A, beta=0)

.. automethod:: Factor.cholesky_AAt(A, beta=0)

Updating/Downdating
+++++++++++++++++++

.. automethod:: Factor.update_inplace(C, subtract=False)

Accessing Cholesky factors explicitly
+++++++++++++++++++++++++++++++++++++

.. note:: When possible, it is generally more efficient to use the
  ``solve_...`` functions documented below rather than extracting the
  Cholesky factors explicitly.

.. automethod:: Factor.P

.. automethod:: Factor.D

.. automethod:: Factor.L

.. automethod:: Factor.LD

.. automethod:: Factor.L_D

Solving equations
+++++++++++++++++

All methods in this section accept both sparse and dense matrices (or
vectors) ``b``, and return either a sparse or dense ``x``
accordingly.

All methods in this section act on :math:`LDL'` factorizations; `L`
always refers to the matrix returned by :meth:`L_D`, not that
returned by :meth:`L` (though conversion is not performed unless
necessary).

.. note:: If you need an efficient implementation of :meth:`solve_L`
   or :meth:`solve_Lt` that works with the :math:`LL'` factorization,
   then drop us a line, it'd be easy to add.

.. automethod:: Factor.solve_A(b)

.. automethod:: Factor.__call__(b)

.. automethod:: Factor.solve_LDLt(b)

.. automethod:: Factor.solve_LD(b)

.. automethod:: Factor.solve_DLt(b)

.. automethod:: Factor.solve_L(b)

.. automethod:: Factor.solve_Lt(b)

.. automethod:: Factor.solve_D(b)

.. automethod:: Factor.solve_P(b)

.. automethod:: Factor.solve_Pt(b)

Error handling
--------------

.. class:: CholmodError

  Errors detected by CHOLMOD or by our wrapper code are converted into
  exceptions of type :class:`CholmodError`.

.. class:: CholmodWarning

  Warnings issued by CHOLMOD are converted into Python warnings of
  type :class:`CholmodWarning`.

.. class:: CholmodTypeConversionWarning

  CHOLMOD itself supports matrices in CSC form with 32-bit integer
  indices and 'double' precision floats (64-bits, or 128-bits total
  for complex numbers). If you pass some other sort of matrix, then
  the wrapper code will convert it for you before passing it to
  CHOLMOD, and issue a warning of type
  :class:`CholmodTypeConversionWarning` to let you know that your
  efficiency is not as high as it might be.

  .. warning:: Not all conversions currently produce warnings. This is
    a bug.

  Child of :class:`CholmodWarning`.


Citing CHOLMOD
--------------

Tim Davies, the author of CHOLMOD, `asks
<http://www.cise.ufl.edu/research/sparse/cholmod/>`_ that if you use
CHOLMOD in the production of published scientific research, you cite
one or more of the following papers:

* Dynamic supernodes in sparse Cholesky update/downdate and triangular
  solves, T. A. Davis and W. W. Hager, ACM Trans. Math. Software, Vol
  35, No. 4, 2009.
* Algorithm 887: CHOLMOD, supernodal sparse Cholesky factorization and
  update/downdate , Y. Chen, T. A. Davis, W. W. Hager, and
  S. Rajamanickam, ACM Trans. Math. Software, Vol 35, No. 3, 2009.
* Row modifications of a sparse Cholesky factorization, T. A. Davis
  and W. W. Hager, SIAM Journal on Matrix Analysis and Applications,
  vol 26, no 3, pp. 621-639, 2005.
* Multiple-rank modifications of a sparse Cholesky factorization,
  T. A. Davis and W. W. Hager, SIAM Journal on Matrix Analysis and
  Applications, vol. 22, no. 4, pp. 997-1013, 2001.
* Modifying a sparse Cholesky factorization, T. A. Davis and
  W. W. Hager, SIAM Journal on Matrix Analysis and Applications,
  vol. 20, no. 3, pp. 606-627, 1999.
