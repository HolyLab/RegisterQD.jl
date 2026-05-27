# RegisterQD.jl

```@docs
RegisterQD.RegisterQD
```

RegisterQD performs image registration using the global optimization routine
[QuadDIRECT](https://github.com/timholy/QuadDIRECT.jl).
Unlike greedy descent methods, it searches for the globally-optimal alignment,
making it robust to poor initial guesses.

## Installation

RegisterQD and its dependencies live in the
[HolyLab registry](https://github.com/HolyLab/HolyLabRegistry).
Add the registry once, then install the package:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterQD")
```

RegisterQD also requires a *mismatch backend* to be loaded before calling any
registration function. For CPU processing, install and load
[RegisterMismatch](https://github.com/HolyLab/RegisterMismatch.jl):

```julia
Pkg.add("RegisterMismatch")
```

For GPU processing, use
[RegisterMismatchCuda](https://github.com/HolyLab/RegisterMismatchCuda.jl) instead.
Do **not** load both in the same session — they conflict.

## Quick start

```julia
using RegisterMismatch, RegisterQD

fixed  = Float64.(reshape(1:25, 5, 5))
moving = circshift(fixed, (2, 1))   # known shift: 2 rows, 1 column

tform, mm = qd_translate(fixed, moving, (3, 3))
# tform.translation ≈ [2.0, 1.0]
# mm ≈ 0.0
```

For a rotation search:

```julia
using RegisterMismatch, RegisterQD, CoordinateTransformations, Rotations, ImageTransformations

fixed  = Float64.(reshape(1:100, 10, 10))
moving = warp(centered(fixed), LinearMap(RotMatrix(0.1)))

tform, mm = qd_rigid(collect(centered(fixed)), collect(float(moving)), (2,2), (0.3,))
# mm ≈ 0.0  (rotation recovered)
```

See the [User Guide](@ref) for a full explanation of concepts, and the
[API Reference](@ref) for all exported functions.
