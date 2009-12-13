Overview
========

Introduction
------------

The :mod:`scikits.sparse` package is a companion to the
:mod:`scipy.sparse` library for sparse matrix manipulation in
Python. All :mod:`scikits.sparse` routines expect and return
:mod:`scipy.sparse` matrices (usually in CSC format). Unlike SciPy
proper, :mod:`scikits.sparse` is covered by the GPLv2+ (see the file
COPYING for details), and thus can take advantage of GPL'ed code like
`SuiteSparse <http://www.cise.ufl.edu/research/sparse/SuiteSparse/>`_.

Currently our coverage is rather... sparse, with only a wrapper for
the CHOLMOD routines for sparse Cholesky decomposition, but we hope
that this will expand over time. Contributions of new wrappers are
very welcome, especially if you can follow the style of the existing
interfaces.

Download
--------

The current release may be downloaded from the Python Package index at

  http://pypi.python.org/pypi/scikits.sparse/

Or from the `homepage <http://code.google.com/p/scikits-sparse>`_ at

  http://code.google.com/p/scikits-sparse/downloads/list

Or the latest *development version* may be found in our `Mercurial
repository <http://code.google.com/p/scikits-sparse/source/list>`_::

  hg clone https://scikits-sparse.googlecode.com/hg/ scikits.sparse

Requirements
------------

Installing :mod:`scikits.sparse` requires:

* `Python <http://python.org/>`_
* `NumPy <http://numpy.scipy.org/>`_
* `SciPy <http://www.scipy.org/>`_
* `Cython <http://www.cython.org/>`_
* `CHOLMOD <http://www.cise.ufl.edu/research/sparse/cholmod/>`_

On Debian/Ubuntu systems, the following command should suffice::

  apt-get install python-scipy libsuitesparse-dev

.. note:: If you work out more detailed instructions for some other
  system, then please `drop us a note
  <scikits-sparse-discuss@lists.vorpus.org>`_ so that they may be
  included here.

Installation
------------

If you have ``easy_install`` installed, then a simple ::

  easy_install -U scikits.sparse

should get you the latest version. Otherwise, download and unpack the
source distribution, and then run ::

  python setup.py install

Contact
-------

Post your suggestions and questions directly to the `mailing list
<http://lists.vorpus.org/cgi-bin/mailman/listinfo/scikits-sparse-discuss>`_
(scikits-sparse-discuss@lists.vorpus.org), or to our `bug tracker
<http://code.google.com/p/scikits-sparse/issues/list>`_. You may also
contact `Nathaniel Smith <mailto:njs@pobox.com>`_ directly.
