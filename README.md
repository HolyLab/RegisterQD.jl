# RegisterQD

This package exports 3 functions: `qd_translate`, `qd_rigid`, and `qd_affine`. `qd_translate` only supports translations and therefore offers the fewest degrees of freedom; `qd_affine` offers the most. In general, using more degrees of freedom allows you to solve harder optimization problems but also makes it harder to finding the global optimum. Your best strategy is to permit no more degrees of freedom than needed to solve the problem.

See the help on these functions for details about how to call them.
