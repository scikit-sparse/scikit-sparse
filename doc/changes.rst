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

  - The :code:`cholmod.Factor` class has been renamed to
    :obj:`~sksparse.cholmod.CholeskyFactor`.

  - The :code:`cholmod.Common` class has been removed. Its attributes have been
    subsumed into the :code:`CholeskyFactor` class.

  - The :func:`~sksparse.cholmod.cholesky` function now returns
    a :obj:`~scipy.sparse.csc_array` instead of a :code:`Factor` object, and an
    optional :obj:`~numpy.ndarray` containing the permutation vector.

  - The :func:`~sksparse.cholmod.ldl` function has been added. It returns
    a tuple (:obj:`~scipy.sparse.csc_array`, :obj:`~scipy.sparse.csc_array`),
    and an optional :obj:`~numpy.ndarray` containing the permutation vector.

  - A :func:`~sksparse.cholmod.cho_factor` function has been added to perform
    the numeric Cholesky factorization and return
    a :obj:`~sksparse.cholmod.CholeskyFactor` object.

  - Similarly, a :func:`~sksparse.cholmod.ldl_factor` function has been added
    to perform the numeric LDL factorization and return
    a :obj:`~sksparse.cholmod.CholeskyFactor` object.

  - The :code:`cholmod.analyze` function has been removed. The analysis step is
    now performed when calling the constructor of
    :obj:`~sksparse.cholmod.CholeskyFactor`.

  - The :code:`use_long` parameter has been removed from the
    :func:`~sksparse.cholmod.cholesky` and :func:`~sksparse.cholmod.ldl`
    functions. The type of indices is now inferred from the input matrix.

  - The :code:`mode` parameter has been renamed to :code:`supernodal_mode`.

  - The :code:`symmetric` parameter has been removed. It has been replaced by
    two new parameters: :code:`lower`, and :code:`sym_kind`.

    * Parameter :code:`lower` controls whether to use the lower or upper
      triangular part of the input matrix, or whether to return a lower or
      upper triangular factor.

    * Parameter :code:`sym_kind` has been added. It accepts a string argument
      in :code:`{"sym", "row", "col"}`, which controls the symmetry structure
      of the matrix to analyze.

  - The functions :code:`cholmod.analyze_AAt` and :code:`cholmod.cholesky_AAt`
    have been removed. Use :func:`~sksparse.cholmod.cho_factor` or
    :func:`~sksparse.cholmod.cholesky` with :code:`sym_kind="row"` instead.

  - The :code:`ordering_method` parameter has been renamed to :code:`order`.

  - The :code:`Factor` methods :code:`L`, :code:`D`, :code:`LD`, :code:`L_D`,
    and :code:`P`, have been removed in favor of the methods
    :meth:`~sksparse.cholmod.CholeskyFactor.get_factor` and
    :meth:`~sksparse.cholmod.CholeskyFactor.get_perm`.

  - The properties :attr:`~sksparse.cholmod.CholeskyFactor.perm` and
    :attr:`~sksparse.cholmod.CholeskyFactor.factor` have been added
    to return read-only views of the permutation vector and factor matrix,
    respectively.

  - The :code:`Factor.solve_A` method has been replaced by the
    :meth:`~sksparse.cholmod.CholeskyFactor.solve` method.
    The :code:`Factor` methods :code:`solve_LDLt`, :code:`solve_LD`,
    :code:`solve_DLt`, :code:`solve_L`, :code:`solve_Lt`, and :code:`solve_D`
    have been removed. The :obj:`~sksparse.cholmod.CholeskyFactor` is not
    callable.

  - The new :meth:`~sksparse.cholmod.CholeskyFactor.solve` method checks the
    condition number and raises a :exc:`~sksparse.cholmod.CholmodNotPositiveDefiniteError` if the
    matrix is exactly singular, or a :exc:`~sksparse.cholmod.CholmodWarning` if the matrix is
    ill-conditioned. Previously, no warning would be issued. See the
    :attr:`~sksparse.cholmod.CholeskyFactor.rcond` property for more details.

  - Add multiple properties to the :obj:`~sksparse.cholmod.CholeskyFactor`
    class for convenient access to :code:`cholmod_factor` attributes. See the
    full documentation for details.

  - Fix a bug in the previous version where sparse inputs with inconsistent
    ``has_sorted_indices`` or ``has_canonical_format`` flags would silently
    lead to incorrect results. The input matrix is now modified into
    a canonical CSC format, regardless of the input format.

  - Add support for single-precision (float32/complex64) input matrices. The
    output factor and solve results will match the input precision. Previously,
    all inputs were converted to double-precision.

* Create the :mod:`~sksparse.amd` module, which provides the AMD ordering method.
* Create the :mod:`~sksparse.btf` module, which provides the BTF ordering method.
* Create the :mod:`~sksparse.camd` module, which provides the constrained AMD
  ordering method.
* Create the :mod:`~sksparse.colamd` module, which provides the COLAMD
  ordering method.
* Create the :mod:`~sksparse.ccolamd` module, which provides the constrained
  COLAMD ordering method.
* Create the :mod:`~sksparse.klu` submodule, which provides an interface to
  the KLU sparse LU solver.
* Create the :mod:`~sksparse.spqr` submodule, which provides an interface to
  the SPQR sparse QR solver.
* Create the :mod:`~sksparse.umfpack` submodule, which provides an interface to
  the UMFPACK sparse LU solver.
* Remove support for the following versions:

  - Python < 3.10
  - NumPy < 2.0
  - SciPy < 1.14
  - SuiteSparse < 7.4.0

  Python 3.9 will reach its end of life in October 2025, so remove support for
  it now. Numpy will end support for all 1.x versions by September 2025. SciPy
  v1.14 (released June 2024) will be supported until the end of 2026.
  SuiteSparse 7.4.0 introduces single precision support in CHOLMOD 5.1.0.


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
