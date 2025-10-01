.. start-badges

.. image:: https://img.shields.io/github/v/release/broesler/scikit-sparse
   :target: https://github.com/broesler/scikit-sparse/releases/latest
   :alt: Latest GitHub release

.. image:: https://img.shields.io/pypi/v/scikit-sparse-dev
   :target: https://pypi.org/project/scikit-sparse-dev/
   :alt: Latest PyPI release

.. image:: https://img.shields.io/conda/vn/conda-forge/scikit-sparse-dev
   :target: https://anaconda.org/conda-forge/scikit-sparse-dev
   :alt: Latest conda-forge release

.. image:: https://github.com/broesler/scikit-sparse/actions/workflows/ci-dev.yml/badge.svg?branch=dev
   :target: https://github.com/broesler/scikit-sparse/actions/workflows/ci-dev.yml
   :alt: CI Status

.. image:: https://readthedocs.org/projects/scikit-sparse-dev/badge/?version=latest
   :target: https://scikit-sparse-dev.readthedocs.io/en/latest/

.. end-badges

========================
Scikit-Sparse (sksparse)
========================

**NOTE**:

    This is the README for the development version of scikit-sparse.
    For the stable version, see `the GitHub repository <upstream_repo_>`_, and
    `the stable docs <upstream_docs_>`_.


The ``scikit-sparse`` package is a companion to the `scipy.sparse
<scipy_sparse_>`_ package for sparse matrix manipulation in Python. It provides
routines that are not suitable for inclusion in `scipy.sparse <scipy_sparse_>`_
proper, typically because they depend on external libraries with
GPL licenses, such as `SuiteSparse <suitesparse_website_>`_.

For more details on usage see `the docs <sksparse_docs_>`_.

.. _upstream_repo: https://github.com/scikit-sparse/scikit-sparse
.. _upstream_docs: https://scikit-sparse.readthedocs.io
.. _scipy_sparse: https://docs.scipy.org/doc/scipy/reference/sparse.html
.. _suitesparse_website: https://people.engr.tamu.edu/davis/suitesparse.html
.. _sksparse_docs: https://scikit-sparse-dev.readthedocs.org
   
.. start-installation

Requirements
------------

Installing ``scikit-sparse`` requires:

* `Python <http://python.org/>`_ >= 3.10
* `NumPy <http://numpy.org/>`_ >= 2.0
* `SciPy <http://scipy.org/>`_ >= 1.14
* `Cython <http://www.cython.org/>`_ >= 3.0
* `SuiteSparse <suitesparse_website_>`_ >= 7.4.0

Older versions may work but are untested.


Installation
------------

Installing SuiteSparse
++++++++++++++++++++++

To install ``scikit-sparse``, you need to have the `SuiteSparse
<suitesparse_website_>`_ library installed on your system.

It is recommended that you install SuiteSparse and the scikit-sparse
dependencies in a virtual environment, to avoid conflicts with other packages.
We recommend using Anaconda::

    $ conda create -n scikit-sparse python>=3.10 suitesparse
    $ conda activate scikit-sparse

If you are not using Anaconda, you can install SuiteSparse using your preferred
package manager.

On MacOS, you can use `Homebrew <http://brew.sh>`_::

    $ brew install suite-sparse

On Debian/Ubuntu systems, use the following command::

    $ sudo apt-get install python-scipy libsuitesparse-dev

On Arch Linux, run::

    $ sudo pacman -S suitesparse


Installing Scikit-Sparse
++++++++++++++++++++++++

Once you have SuiteSparse installed, you can install ``scikit-sparse`` with::

    $ conda install -c conda-forge scikit-sparse-dev

or if you prefer to use pip, you can install it with::

    $ pip install scikit-sparse-dev

Check if the installation was successful by running the following command::

    $ python -c "import sksparse; print(sksparse.__version__)"


.. end-installation

See `Troubleshooting <docs_trouble_>`_ for more information on determining
which SuiteSparse library is being used.

.. _docs_trouble: https://scikit-sparse-dev.readthedocs.io/en/latest/overview.html#troubleshooting


----

Copyright © 2009–2025, the `scikit-sparse developers <docs_dev_>`_.

.. _docs_dev: https://scikit-sparse-dev.readthedocs.io/en/latest/overview.html#developers
