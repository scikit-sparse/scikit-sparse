Changes
=======

.. module:: sksparse.cholmod

v0.3
----
  * Dropped deprecated :meth:`Factor.solve_P` and :meth:`Factor.solve_P`.
  * Fixed a memory leak upon garbage collection of :class:`Factor`.

v0.2
----
  * :class:`Factor` solve methods now return 1d output for 1d input
    (just like ``np.dot`` does).
  * :meth:`Factor.solve_P` and :meth:`Factor.solve_P` deprecated; use
    :meth:`Factor.apply_P` and :meth:`Factor.apply_Pt` instead.
  * New methods for computing determinants of positive-definite
    matrices: :meth:`Factor.det`, :meth:`Factor.logdet`,
    :meth:`Factor.slogdet`.
  * New method for explicitly computing inverse of a positive-definite
    matrix: :meth:`Factor.inv`.
  * :meth:`Factor.D` has much better implementation.
  * Build system improvements.
  * Wrapper code re-licensed under BSD terms.

v0.1
----
  First public release.
