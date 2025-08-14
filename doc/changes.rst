Changes
=======

v0.5.0
------
* Major API updates to the :mod:`sksparse.cholmod` module. The module has been
  updated to resemble the existing :func:`scipy.linalg.cholesky` interface, as
  well as provide additional functions present in the SuiteSparse CHOLMOD
  MATLAB interface, and MATLAB built-in Cholesky functions. The underlying
  Cython code has been refactored to provide greater type safety and
  performance.

  - The :func:`~.cholmod.cholesky` and :code:`cholesky_AAt` functions are
    now combined into single functions with the ``symmetric`` kwarg.

v0.4.4
------
* Bug in solve with dense array, where base of result is not set correctly, fixed.
* Travis tests are using conda now.
* Supported versions updated to:

  - Python: 3.7, 3.6
  - NumPy: 1.15, 1.14, 1.13
  - SciPy: 1.1, 1.0, 0.19
  - SuiteSparse: 5.2

v0.4.3
------
* The method :code:`Factor.solve_L` can now also use the `L` matrix of the LL' decomposition.
* Supported versions updated to:

  - Python: 3.6, 3.5
  - NumPy: 1.14, 1.13
  - SciPy: 1.0, 0.19

v0.4.2
------
* Bug where the ordering method is not taken into account is fixed.
* The Factor class has now a (public) copy method.

v0.4.1
------
* Bug with relaxed stride checking in NumPy 1.12 fixed.
* Supported versions updated to:

  - Python: 3.6, 3.5, 3.4, 2.7
  - NumPy: 1.8 to 1.12

v0.4
------
* 64-bit indices (type long) are now supported.
* The ordering method for Cholesky decomposition is now choosable.
* Specific exceptions subclasses are now thrown for each error condition.
* Setup does not rely on an installed Cython anymore.

v0.3.1
------
* Ensure that arrays returned by the :code:`Factor.solve_...` methods are
  writeable.

v0.3
----
* Dropped deprecated :code:`Factor.solve_P` and :code:`Factor.solve_P`.
* Fixed a memory leak upon garbage collection of :code:`Factor`.

v0.2
----
* :code:`Factor` solve methods now return 1d output for 1d input
  (just like :func:`numpy.dot` does).
* :code:`Factor.solve_P` and :code:`Factor.solve_P` deprecated; use
  :code:`Factor.apply_P` and :code:`Factor.apply_Pt` instead.
* New methods for computing determinants of positive-definite
  matrices: :code:`Factor.det`, :code:`Factor.logdet`,
  :code:`Factor.slogdet`.
* New method for explicitly computing inverse of a positive-definite
  matrix: :code:`Factor.inv`.
* :code:`Factor.D` has much better implementation.
* Build system improvements.
* Wrapper code re-licensed under BSD terms.

v0.1
----
First public release.
