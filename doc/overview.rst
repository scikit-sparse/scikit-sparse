Overview
========

Introduction
------------

The :mod:`scikit-sparse` package (previously known as :mod:`scikits.sparse`)
is a companion to the :mod:`scipy.sparse` library for sparse matrix
manipulation in Python. All :mod:`scikit-sparse` routines expect and
return :mod:`scipy.sparse` matrices (usually in CSC format). The intent
of :mod:`scikit-sparse` is to wrap GPL'ed code such as `SuiteSparse
<http://www.suitesparse.com>`_, which cannot be
included in SciPy proper.

Currently our coverage is rather... sparse, with only a wrapper for
the CHOLMOD routines for sparse Cholesky decomposition, but we hope
that this will expand over time. Contributions of new wrappers are
very welcome, especially if you can follow the style of the existing
interfaces.

Download
--------

The current release may be downloaded from the Python Package index at

  https://pypi.python.org/pypi/scikit-sparse/

Or from the `homepage <https://github.com/scikit-sparse/scikit-sparse>`_
at

  https://github.com/scikit-sparse/scikit-sparse/releases

Or the latest *development version* may be found in our `Git
repository <https://github.com/scikit-sparse/scikit-sparse>`_::

  $ git clone git://github.com/scikit-sparse/scikit-sparse.git

Requirements
------------

Installing :mod:`scikit-sparse` requires:

* `Python <http://python.org/>`_
* `NumPy <http://numpy.scipy.org/>`_
* `SciPy <http://www.scipy.org/>`_
* `Cython <http://www.cython.org/>`_
* CHOLMOD (included in `SuiteSparse <http://www.suitesparse.com>`_)

Test versions are:
* Python: 3.7, 3.6
* NumPy: 1.15, 1.14, 1.13
* SciPy: 1.1, 1.0, 0.19
* SuiteSparse: 5.2
(Other versions may work but are untested.)

On Debian/Ubuntu systems, the following command should suffice::

  $ sudo apt-get install python-scipy libsuitesparse-dev

On Arch Linux, run::

  $ sudo pacman -S suitesparse

Installation
------------

As usual, ::

  $ pip install --user scikit-sparse

or with conda ::

  $ conda install -c conda-forge scikit-sparse

Contact
-------

Post your suggestions and questions directly to our `bug tracker
<https://github.com/scikit-sparse/scikit-sparse/issues>`_.

Developers
----------

* 2008        `David Cournapeau        <cournape@gmail.com>`_
* 2009-2015   `Nathaniel Smith         <njs@pobox.com>`_
* 2010        `Dag Sverre Seljebotn    <dagss@student.matnat.uio.no>`_
* 2014        `Leon Barrett            <lbarrett@climate.com>`_
* 2015        `Yuri                    <yuri@tsoft.com>`_
* 2016-2017   `Antony Lee              <anntzer.lee@gmail.com>`_
* 2016        `Alex Grigorievskiy      <alex.grigorievskiy@gmail.com>`_
* 2016-2018   `Joscha Reimer           <jor@informatik.uni-kiel.de>`_
