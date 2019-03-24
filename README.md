# RegisterQD

[![Build Status](https://travis-ci.com/HolyLab/RegisterQD.jl.svg?branch=master)](https://travis-ci.com/HolyLab/RegisterQD.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/o82s9bc4m8lgmgf5?svg=true)](https://ci.appveyor.com/project/Cody-G/registerqd-jl)
[![codecov](https://codecov.io/gh/HolyLab/RegisterQD.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterQD.jl)

This package exports 3 functions: `qd_translate`, `qd_rigid`, and `qd_affine`. `qd_translate` only supports translations and therefore offers the fewest degrees of freedom; `qd_affine` offers the most. In general, using more degrees of freedom allows you to solve harder optimization problems but also makes it harder to find the global optimum. Your best strategy is to permit no more degrees of freedom than needed to solve the problem.

See the help on these functions for details about how to call them.
