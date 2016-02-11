Overview
========

Introduction
------------

The :mod:`scikit-sparse` package (previously known as :mod:`scikits.sparse`)
is a companion to the :mod:`scipy.sparse` library for sparse matrix
manipulation in Python. All :mod:`scikit-sparse` routines expect and
return :mod:`scipy.sparse` matrices (usually in CSC format). The intent
of :mod:`scikit-sparse` is to wrap GPL'ed code such as `SuiteSparse
<http://www.cise.ufl.edu/research/sparse/SuiteSparse/>`_, which cannot be
included in SciPy proper.

Currently our coverage is rather... sparse, with only a wrapper for
the CHOLMOD routines for sparse Cholesky decomposition, but we hope
that this will expand over time. Contributions of new wrappers are
very welcome, especially if you can follow the style of the existing
interfaces.

Download
--------

The current release may be downloaded from the Python Package index at

  http://pypi.python.org/pypi/scikit-sparse/

Or from the `homepage <https://github.com/scikit-sparse/scikit-sparse>`_
at

  https://github.com/scikit-sparse/scikit-sparse/downloads

Or the latest *development version* may be found in our `Git
repository <https://github.com/scikit-sparse/scikit-sparse>`_::

  git clone git://github.com/scikit-sparse/scikit-sparse.git

Requirements
------------

Installing :mod:`scikit-sparse` requires:

* `Python <http://python.org/>`_
* `NumPy <http://numpy.scipy.org/>`_
* `SciPy <http://www.scipy.org/>`_
* `Cython <http://www.cython.org/>`_
* `CHOLMOD <http://www.cise.ufl.edu/research/sparse/cholmod/>`_

On Debian/Ubuntu systems, the following command should suffice::

  apt-get install python-scipy libsuitesparse-dev

On Arch Linux, the `python-scikit-sparse` AUR package declares the
required dependencies.

Installation
------------

As usual, ::

  pip install --user scikit-sparse

Contact
-------

Post your suggestions and questions directly to our `bug tracker
<https://github.com/scikit-sparse/scikit-sparse/issues>`_.
