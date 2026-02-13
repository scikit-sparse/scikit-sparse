=============================
Contributing to scikit-sparse
=============================

Thank you for your interest in contributing! This project uses a hybrid
development environment consisting of `Mamba <mamba_website_>`_ 
(or `Conda <conda_website_>`_, for C-libraries and the Python interpreter)
and `uv <uv_website_>`_ (for Python package management).


Prerequisites
=============

Ensure you have the following installed:

* `Mamba <mamba_website_>`_ (or Miniforge)

.. _mamba_website: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
.. _conda_website: https://www.anaconda.com/docs/getting-started/anaconda/install
.. _uv_website: https://docs.astral.sh/uv/#installation


Setting Up the Development Environment
======================================

We use Mamba to provide the SuiteSparse C-libraries and the C++ compiler.

Create the Environment
----------------------

From the project root, run:

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate sksparse-dev

Install scikit-sparse (Editable)
--------------------------------

We use ``uv`` to install the Python dependencies and compile the Cython
extensions. This command installs the project in "editable" mode along with all
development tools (pytest, ruff, sphinx).

.. code-block:: bash

   uv pip install -e ".[dev]"

.. note::

   If the install fails to find SuiteSparse headers, ensure your
   ``CONDA_PREFIX`` is set correctly (``echo $CONDA_PREFIX``). Our ``setup.py``
   is configured to prioritize this path.


Development Workflow
====================

New features, bug fixes, and documentation improvements should be developed in
a separate branch off of the `dev <sksparse_dev_url_>`_ branch (*e.g.*
``feature/my-new-feature``). Once your changes are ready, submit a pull request
for review. Follow the NumPy/SciPy contribution guidelines for best practices
on `documentation <numpy_docstyle_>`_ and `commit messages <numpy_commitmsg_>`_.

.. _sksparse_dev_url: https://github.com/scikit-sparse/scikit-sparse/tree/dev
.. _numpy_docstyle: https://numpy.org/doc/stable/dev/howto-docs.html#howto-document
.. _numpy_commitmsg: https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message

Code Linting and Formatting
---------------------------

We use ``ruff`` for linting. It is installed as part of the ``[dev]``
dependencies. It runs automatically on pre-commit, but you can also run it
manually:

.. code-block:: bash

   ruff check .

Running Tests
-------------

We use ``pytest`` for all unit tests.

.. code-block:: bash

   pytest

Rebuilding Extensions
---------------------

If you modify ``.pyx`` or ``.pxd`` files, you must re-run the ``uv`` install
command to trigger a re-compilation of the Cython extensions:

.. code-block:: bash

   uv pip install -e ".[dev]"


Building Documentation
======================

The documentation is built using Sphinx and the Furo theme.

.. code-block:: bash

   cd doc
   make html

The results will be available in ``doc/_build/html/index.html``.

Benchmarks
==========

We use `asv <asv_website_>`_ for benchmarking. See the ``./benchmarks/``
directory. To run the benchmarks, run the following command from the project
root:

.. code-block:: bash

   asv run

See also the `SciPy benchmark guidelines <scipy_benchmarks_website_>`_ for best
practices on writing benchmarks. We do not use ``spin``, so refer to the
``asv`` commands therein.

.. _asv_website: https://asv.readthedocs.io/en/stable/
.. _scipy_benchmarks_website: https://scipy.github.io/devdocs/dev/contributor/benchmarking.html#benchmarking-with-asv

