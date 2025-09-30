========
Overview
========

Introduction
------------

The ``scikit-sparse`` package (previously known as ``scikits.sparse``)
is a companion to the :mod:`scipy.sparse` library for sparse matrix manipulation
in Python. All :mod:`sksparse` routines expect and return :mod:`scipy.sparse`
matrices (usually in CSC format). The intent of :mod:`sksparse` is to wrap code
with a GPL license, such as `SuiteSparse <suitesparse_website_>`_, which cannot
be included in SciPy proper.

.. _suitesparse_website: https://people.engr.tamu.edu/davis/suitesparse.html

.. include:: ../README.rst
   :start-after: .. start-installation
   :end-before:  .. end-installation


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


To see which SuiteSparse library was found, you can run the following command on
MacOS or Linux::

    $ CHECK_SKSPARSE_INSTALL=$(python -c 'import sksparse.cholmod; print(sksparse.cholmod.__file__)')

Then, use one of the following commands depending on your operating system.


MacOS
^^^^^

On MacOS, use the following command to check where the SuiteSparse
installation was found::

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


Linux
^^^^^

On Linux, use the following commands instead::

    $ ldd $CHECK_SKSPARSE_INSTALL | grep cholmod
    $ readelf -d $CHECK_SKSPARSE_INSTALL | grep -E '(RPATH|RUNPATH)'
    0x000000000000001d (RUNPATH)            Library runpath: [/home/user/anaconda3/envs/scikit-sparse/lib]

also confirming installation on the conda path.


Contact
-------

Post your suggestions and questions directly to our `GitHub Issues page
<github_issues_>`_.

.. _github_issues: https://github.com/scikit-sparse/scikit-sparse/issues

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
* 2021-       `Justin Ellis            <justin.ellis18@gmail.com>`_
* 2022-       `Aaron Johnson           <aaron9035@gmail.com>`_
* 2025–       `Bernard Roesler         <bernard.roesler@gmail.com>`_
