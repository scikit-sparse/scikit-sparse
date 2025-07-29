Overview
========

Introduction
------------

The :mod:`scikit-sparse` package (previously known as :mod:`scikits.sparse`)
is a companion to the :mod:`scipy.sparse` library for sparse matrix
manipulation in Python. All :mod:`scikit-sparse` routines expect and
return :mod:`scipy.sparse` matrices (usually in CSC format). The intent
of :mod:`scikit-sparse` is to wrap GPL'ed code such as `SuiteSparse
<suitesparse_website_>`_, which cannot be
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
* `SuiteSparse <suitesparse_website_>`_

Test versions are:

* Python: 3.10, 3.11, 3.12, 3.13
* NumPy: 2.0
* SciPy: 1.14
* SuiteSparse CHOLMOD: 5.3

(Other versions may work but are untested.)


Installation
------------

Installing SuiteSparse
++++++++++++++++++++++

To install :mod:`scikit-sparse`, you need to have the `SuiteSparse
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

On Debian/Ubuntu systems, the following command should suffice::

  $ sudo apt-get install python-scipy libsuitesparse-dev

On Arch Linux, run::

  $ sudo pacman -S suitesparse


Installing Scikit-Sparse
++++++++++++++++++++++++

Once you have SuiteSparse installed, you can install :mod:`scikit-sparse` with::

  $ conda install -c conda-forge scikit-sparse

or if you prefer to use pip, you can install it with::

  $ pip install scikit-sparse

Check if the installation was successful by running the following command::

  $ python -c "import sksparse; print(sksparse.__version__)"


Troubleshooting
+++++++++++++++

The installation will automatically detect the SuiteSparse library and compile
the necessary Cython code. It will check for the SuiteSparse library in the
following order:

    1. The environment variables ``SUITESPARSE_INCLUDE_DIR`` and
       ``SUITESPARSE_LIB_DIR`` (if set, these will override the default search
       paths)
    2. Your active conda environment path
    3. Your homebrew paths (*e.g.* ``/opt/homebrew/include/suitesparse``)
    4. Typical system paths (*e.g.* ``/usr/include/suitesparse`` on Linux, or
       ``/usr/local/include/suitesparse`` on macOS)

The first path that contains the SuiteSparse headers and libraries will be used.

To see which SuiteSparse library was found, you can run the following command::

    $ CHECK_SKSPARSE_INSTALL=$(python -c 'import sksparse.cholmod; print(sksparse.cholmod.__file__)')

then, on MacOS::

    $ otool -L $CHECK_SKSPARSE_INSTALL | grep cholmod

Look for a line that contains ``cholmod.*\.dylib`` or ``cholmod.*\.a``. The
output might be something like::

    $ otool -L $CHECK_SKSPARSE_INSTALL | grep cholmod
    /Users/username/src/scikit-sparse/sksparse/cholmod.cpython-313-darwin.so:
            @rpath/libcholmod.5.dylib (compatibility version 5.0.0, current version 5.3.1)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1351.0.0)

The ``@rpath/libcholmod.5.dylib`` indicates that the library was found on the
relative path. To resolve this path, run::

    $ otool -l @rpath/libcholmod.5.dylib | grep -A2 LC_RPATH
          cmd LC_RPATH
      cmdsize 72
         path /Users/username/anaconda3/envs/scikit-sparse/lib (offset 12)

which indicates that the library was found on the conda path.

On Linux, use the following commands instead::

    $ ldd $CHECK_SKSPARSE_INSTALL | grep cholmod
    $ readelf -d $CHECK_SKSPARSE_INSTALL | grep -E '(RPATH|RUNPATH)'
    0x000000000000001d (RUNPATH)            Library runpath: [/home/user/anaconda3/envs/scikit-sparse/lib]

also confirming installation on the conda path.

Contact
-------

Post your suggestions and questions directly to our `GitHub Issues page
<https://github.com/scikit-sparse/scikit-sparse/issues>`_.


Developers
----------

* 2008        `David Cournapeau        <cournape@gmail.com>`_
* 2009–2015   `Nathaniel Smith         <njs@pobox.com>`_
* 2010        `Dag Sverre Seljebotn    <dagss@student.matnat.uio.no>`_
* 2014        `Leon Barrett            <lbarrett@climate.com>`_
* 2015        `Yuri                    <yuri@tsoft.com>`_
* 2016–2017   `Antony Lee              <anntzer.lee@gmail.com>`_
* 2016        `Alex Grigorievskiy      <alex.grigorievskiy@gmail.com>`_
* 2016–2018   `Joscha Reimer           <jor@informatik.uni-kiel.de>`_
* 2025–       `Bernard Roesler         <bernard.roesler@gmail.com>`_

.. _suitesparse_website: https://people.engr.tamu.edu/davis/suitesparse.html
