.. Part of the scikit-sparse project.
.. Copyright (C) 2008-2025 The scikit-sparse developers. All rights reserved.
.. See pyproject.toml for full author list and LICENSE.txt for license details.
.. SPDX-License-Identifier: BSD-2-Clause

.. _user_guide:

************************
Scikit-Sparse User Guide
************************

.. currentmodule:: sksparse

.. sectionauthor:: Bernard T. Roesler

Scikit-sparse is a collection of sparse matrix algorithms and convenience
functions built to work with SciPySparse_ arrays. It is largely an interface to
the parts of the SuiteSparse_ library by Timothy A. Davis that have a GPL
license and are not suitable for inclusion in SciPy proper.

.. _SciPySparse: https://docs.scipy.org/doc/scipy/tutorial/sparse.html
.. _SuiteSparse: http://faculty.cse.tamu.edu/davis/suitesparse.html


Subpackages and User Guides
---------------------------

Scikit-sparse is organized into submodules corresponding to the submodules of
SuiteSparse. These are summarized in the following table:

==================    ========================================
Subpackage            Description and User Guide
==================    ========================================
``amd``                 :doc:`./amd`
``btf``                 :doc:`./btf`
``camd``                :doc:`./camd`
``ccolamd``             :doc:`./ccolamd`
``cholmod``             :doc:`./cholmod`
``colamd``              :doc:`./colamd`
``klu``                 :doc:`./klu`
``umfpack``             :doc:`./umfpack`
==================    ========================================

.. toctree::
   :caption: User Guide
   :maxdepth: 1
   :hidden:

   amd
   btf
   camd
   ccolamd
   cholmod
   colamd
   klu
   umfpack
