Changes
=======

.. module:: scikits.sparse.cholmod

v0.2
-------
  * :class:`Factor` solve methods now return 1d output for 1d input
    (just like ``np.dot`` does).
  * :meth:`Factor.solve_P` and :meth:`Factor.solve_P` deprecated; use
    :meth:`Factor.apply_P` and :meth:`Factor.apply_Pt` instead.
  * New methods for computing determinants of positive-definite
    matrices: :meth:`Factor.det`, :meth:`Factor.logdet`,
    :meth:`Factor.slogdet`.
  * :meth:`Factor.D` has much-improved implementation.
  * Build system improvements.
  * Wrapper code re-licensed under BSD terms.

v0.1
------
  First public release.
