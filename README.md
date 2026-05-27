# RegisterQD

[![CI](https://github.com/HolyLab/RegisterQD.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterQD.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/RegisterQD.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterQD.jl)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/RegisterQD.jl/stable/)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/RegisterQD.jl/dev/)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

RegisterQD performs image registration using the global optimization routine [QuadDIRECT](https://github.com/timholy/QuadDIRECT.jl).
Unlike many other registration packages, this is not "greedy" descent based on an initial guess — it attempts to find the globally-optimal alignment of your images.

## Installation

RegisterQD and its dependencies live in the [HolyLab registry](https://github.com/HolyLab/HolyLabRegistry).
Add the registry once, then install:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterQD")
```

You also need a mismatch backend. For CPU processing, load [RegisterMismatch](https://github.com/HolyLab/RegisterMismatch.jl):

```julia
Pkg.add("RegisterMismatch")
```

For GPU processing, use [RegisterMismatchCuda](https://github.com/HolyLab/RegisterMismatchCuda.jl) instead.
*Do not load both in the same session — they conflict.*

## Quick start

```julia
using RegisterMismatch, RegisterQD

fixed  = Float64.(reshape(1:25, 5, 5))
moving = circshift(fixed, (2, 1))   # known shift: 2 rows, 1 column

tform, mm = qd_translate(fixed, moving, (3, 3))
# tform.translation == [2.0, 1.0]
# mm == 0.0
```

## Registration functions

- `qd_translate`: register images by shifting one with respect to another (translations only)
- `qd_rigid`: register images using rotations and translations
- `qd_affine`: register images using arbitrary affine transformations

In general, using more degrees of freedom allows you to solve harder optimization problems, but also makes it harder to find the global optimum.
Use no more degrees of freedom than your problem requires.

## Anisotropic sampling

This package supports images sampled anisotropically, which is common in 3-D biomedical imaging (e.g. MRI, optical sections where the axial resolution differs from the in-plane resolution).
Pass `SD = diagm(voxelspacing)` to the registration functions to account for non-uniform spacing.
See [`arrayscale`](https://HolyLab.github.io/RegisterQD.jl/stable/api/#RegisterQD.arrayscale) and [`getSD`](https://HolyLab.github.io/RegisterQD.jl/stable/api/#RegisterQD.getSD) for details, and the [User Guide](https://HolyLab.github.io/RegisterQD.jl/stable/guide/) for a full explanation.

**NOTE**: see NEWS.md for information about recent breaking changes.
