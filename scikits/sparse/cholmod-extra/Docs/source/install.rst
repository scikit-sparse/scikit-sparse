Installation
============

The module depends on `SuiteSparse
<http://www.cise.ufl.edu/research/sparse/SuiteSparse/>`_, thus install
it if you do not have it installed.  Instructions can be found at
http://www.cise.ufl.edu/research/sparse/SuiteSparse/.  After
installing the dependencies, clone the Git repository:

::

    git clone https://github.com/jluttine/cholmod-extra.git
    
Compile and install the module:

::
    
    cd cholmod-extra
    make
    make install

TODO: Check the installation directory in Makefile!

This documentation can be found in Docs/ folder.  The documentation
source files are readable as such in reStructuredText format.  If you
have `Sphinx <http://sphinx.pocoo.org/>`_ installed, the documentation
can be compiled to, for instance, HTML or PDF using

::

    cd Docs
    make html
    make latexpdf

The documentation can be found also at
http://cholmod-extra.readthedocs.org/ in various formats.
