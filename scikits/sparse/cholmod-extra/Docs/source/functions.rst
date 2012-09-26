Functions
=========

Sparse inverse
--------------

.. cpp:function:: cholmod_sparse* cholmod_spinv(cholmod_factor *L, cholmod_common *Common)

   Return the sparse inverse given the Cholesky factor.  The sparse
   inverse contains elements from the inverse matrix but has the same
   sparsity structure as the Cholesky factor (symbolically).

Although the inverse of a sparse matrix is dense in general, it is
sometimes sufficient to compute only some elements of the inverse.
For instance, in order to compute
:math:`\operatorname{tr}(\mathbf{K}^{-1}\mathbf{A})`, it is sufficient
to find those elements of :math:`\mathbf{K}^{-1}` that are non-zero in
:math:`\mathbf{A}^{\mathrm{T}}`.  If :math:`\mathbf{A}^{\mathrm{T}}`
has the same sparsity structure as :math:`\mathbf{K}` (e.g.,
:math:`\mathbf{A}^{\mathrm{T}}=\partial\mathbf{K}/\partial\theta`),
one only needs to compute those elements of the inverse
:math:`\mathbf{K}^{-1}` that are non-zero in :math:`\mathbf{K}`.
These elements can be computed using an efficient algorithm if
:math:`\mathbf{K}` is symmetric positive-definite [Takahashi:1973]_.
The resulting sparse matrix is called the sparse inverse.

The algorithm for computing the sparse inverse can be derived as
follows [Vanhatalo:2008]_.  Denote the inverse as
:math:`\mathbf{Z}=\mathbf{K}^{-1}` and the Cholesky decomposition as
:math:`\mathbf{LL}^{\mathrm{T}} = \mathbf{K}`, where
:math:`\mathbf{L}` is a lower triangular matrix.  We have the identity

.. math::
   :label: ZL

   \mathbf{ZL} = \mathbf{L}^{-\mathrm{T}}.

Taking the diagonal elements of the Cholesky factor,
:math:`\mathbf{\Lambda} = \operatorname{mask}(\mathbf{L},\mathbf{I})`,
the equation :eq:`ZL` can be written as

.. math::
   
   \mathbf{Z\Lambda} + \mathbf{Z} (\mathbf{L} - \mathbf{\Lambda}) =
   \mathbf{L}^{-\mathrm{T}}.

Subtracting the second term on the left and multiplying by
:math:`\mathbf{\Lambda}^{-1}` from the right yields

.. math::
   :label: recursion

   \mathbf{Z} = \mathbf{L}^{-\mathrm{T}} \mathbf{\Lambda}^{-1} -
   \mathbf{Z} (\mathbf{L} - \mathbf{\Lambda}) \mathbf{\Lambda}^{-1}.

One can also use Cholesky decomposition of the form
:math:`\tilde{\mathbf{L}} \mathbf{D} \tilde{\mathbf{L}}^{\mathrm{T}} =
\mathbf{K}`, where :math:`\tilde{\mathbf{L}}` has unit diagonal and
:math:`\mathbf{D}` is a diagonal matrix.  In that case, equation
:eq:`recursion` transforms to

.. math::

   \mathbf{Z} = \tilde{\mathbf{L}}^{-\mathrm{T}} \mathbf{D}^{-1} -
   \mathbf{Z} (\tilde{\mathbf{L}} - \mathbf{I}).

These formulae can be used to solve the inverse recursively.  The
recursive update formulae are shown for the supernodal factorization,
because the update formulae for the simplicial factorization can be
seen as a special case, and possible permutations are ignored.


The inverse is computed for each supernodal block at a time starting
from the lower right corner. Now, consider one iteration step.  Let
:math:`\mathbf{Z}_C` denote the lower right part of the inverse which
has already been computed.  The supernodal block that is updated
consists of :math:`\mathbf{Z}_A` and :math:`\mathbf{Z}_B` as

.. math::

   \mathbf{Z} = 
   \left[ \begin{matrix}
     \ddots & \vdots       & \vdots \\
     \cdots & \mathbf{Z}_A & \mathbf{Z}^{\mathrm{T}}_B \\
     \cdots & \mathbf{Z}_B & \mathbf{Z}_C
   \end{matrix} \right],

where :math:`\mathbf{Z}_A` and :math:`\mathbf{Z}_C` are square
matrices on the diagonal.  Using the same division to blocks for
:math:`\mathbf{L}`, from :eq:`recursion` follows that

.. math::
   
   \mathbf{Z}_B &= - \mathbf{Z}_C \mathbf{L}_B \mathbf{L}^{-1}_A,
   \\
   \mathbf{Z}_A &= \mathbf{L}^{-\mathrm{T}}_{A} \mathbf{L}^{-1}_A -
   \mathbf{Z}^{\mathrm{T}}_B \mathbf{L}_B \mathbf{L}^{-1}_A.

For the first iteration step, the update equation is
:math:`\mathbf{Z}_A = \mathbf{L}^{-\mathrm{T}}_{A} \mathbf{L}^{-1}_A`.

Instead of computing the full inverse using this recursion, it is
possible to gain significant speed-up if one computes the sparse
inverse, because then it is sufficient to compute only those elements
that are symbolically non-zero in :math:`\mathbf{L}`.  It follows that
one can discard those rows from the block :math:`B` that are
symbolically zero in :math:`\mathbf{L}_B`.  Also, the same rows and
the corresponding columns can be discarded from the block :math:`C`.
Thus, the blocks :math:`B` and :math:`C` are effectively very small
for the numerical matrix product computations.  For the simplicial
factorization, each block is one column wide, that is,
:math:`\mathbf{Z}_A` is a scalar and :math:`\mathbf{Z}_B` is a vector.

For :math:`\mathbf{K} = \tilde{\mathbf{L}} \mathbf{D}
\tilde{\mathbf{L}}^{\mathrm{T}}` factorization, the update equations
are

.. math::
   
   \mathbf{Z}_B &= - \mathbf{Z}_C \tilde{\mathbf{L}}_B
   \tilde{\mathbf{L}}^{-1}_A, 
   \\ 
   \mathbf{Z}_A &=
   \tilde{\mathbf{L}}^{-\mathrm{T}}_{A} \mathbf{D}^{-1}_A 
   \tilde{\mathbf{L}}^{-1}_A -
   \mathbf{Z}^{\mathrm{T}}_B \tilde{\mathbf{L}}_B 
   \tilde{\mathbf{L}}^{-1}_A,

and for the first iteration step, :math:`\mathbf{Z}_A =
\tilde{\mathbf{L}}^{-\mathrm{T}}_{A} \mathbf{D}_A
\tilde{\mathbf{L}}^{-1}_A`.


The following methods have been implemented in cholmod-extra.

..
   ========== ==== ======= ======= ==== ======= =======
   a                   LL                   LDL
   ---------- -------------------- --------------------
   b          Real Complex Zomplex Real Complex Zomplex
   ========== ==== ======= ======= ==== ======= =======
   Simplicial no   no      no      yes  no      no
   Supernodal yes  no      no      no   no      no
   ========== ==== ======= ======= ==== ======= =======

.. daksjl
   tabularcolumns:: |r|r|r|r|r|r|r|

.. table:: Implemented sparse inverse methods. 

   +------------+------+---------+---------+------+---------+---------+
   |            | :math:`\mathbf{LL}       | :math:`\tilde{\mathbf{L}}|
   |            | ^{\mathrm{T}}`           | \mathbf{D}               |
   |            |                          | \tilde{\mathbf{L}}       |
   |            |                          | ^{\mathrm{T}}`           |
   +------------+------+---------+---------+------+---------+---------+
   |            | Real | Complex | Zomplex | Real | Complex | Zomplex |
   +============+======+=========+=========+======+=========+=========+
   | Simplicial | no   | no      | no      | yes  | no      | no      |
   +------------+------+---------+---------+------+---------+---------+
   | Supernodal | yes  | no      | no      | no   | no      | no      |
   +------------+------+---------+---------+------+---------+---------+


.. [Takahashi:1973] Takahashi K, Fagan J, and Chen M-S
                    (1973). Formation of a sparse bus impedance matrix
                    and its application to short circuit study. In
                    *Power Industry Computer Application Conference
                    Proceedings*. IEEE Power Engineering Society.

.. [Vanhatalo:2008] Vanhatalo J and Vehtari A (2008). Modelling local
                    and global phenomena with sparse Gaussian
                    processes. In *Proceedings of the 24th Conference
                    in Uncertainty in Artificial Intelligence*. AU AI
                    Press.
