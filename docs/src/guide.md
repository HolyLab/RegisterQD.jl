# User Guide

## How registration works

Image registration is the problem of finding a geometric transformation that
aligns one image (`moving`) to another (`fixed`).  RegisterQD measures how well
two images are aligned using a *mismatch* value — roughly, the sum-of-squared
differences after normalising for intensity — and minimises it with the
[QuadDIRECT](https://github.com/timholy/QuadDIRECT.jl) global optimiser.

Because QuadDIRECT searches the entire parameter space rather than following a
local gradient, it can find the correct alignment even without a good starting
guess.  The trade-off is that it is slower than greedy methods; providing a good
`initial_tfm` when one is available can speed things up significantly.

## Mismatch backend

The mismatch computation is implemented in a separate *backend* package that must
be loaded before calling any registration function:

| Use case | Package to load |
|----------|----------------|
| CPU      | [RegisterMismatch](https://github.comlyHolyLab/RegisterMismatch.jl) |
| GPU (CUDA) | [RegisterMismatchCuda](https://github.com/HolyLab/RegisterMismatchCuda.jl) |

Load exactly one backend — loading both in the same session causes method
conflicts.

## Choosing a registration function

| Function | Degrees of freedom | When to use |
|---|---|---|
| [`qd_translate`](@ref) | N translations | images are related by a pure shift |
| [`qd_rigid`](@ref) | N translations + rotation(s) | images share the same shape but may be rotated |
| [`qd_affine`](@ref) | N translations + N² linear | images may have scaling or shear in addition to rotation |

Start with the fewest degrees of freedom that your problem requires.
More degrees of freedom make the global search harder and increase the risk of
finding a degenerate (low-overlap) solution.

## Anisotropic sampling and the `SD` parameter

Three-dimensional biomedical images (MRI, optical sections) are often sampled at
different resolutions along different axes — for example, 0.5 mm in-plane but
2 mm axially.  A physical-space rotation of such a volume does not look like a
rotation in array-index space.

The `SD` ("spatial displacements") parameter encodes the physical spacing.  Its
columns are the physical displacements corresponding to one array-element step
along each axis:

```julia
# 2-D example: dim 2 sampled 4× coarser than dim 1
SD = [1.0 0.0;
      0.0 4.0]

tform, mm = qd_rigid(fixed, moving, mxshift, mxrot; SD)
```

If your image carries axis metadata (e.g. from
[AxisArrays.jl](https://github.com/JuliaArrays/AxisArrays.jl)), use
[`getSD`](@ref) to extract `SD` automatically:

```julia
SD = getSD(fixed)
tform, mm = qd_rigid(fixed, moving, mxshift, mxrot; SD)
```

The returned transformation is always in *physical* coordinates.  Before warping
the array, convert it with [`arrayscale`](@ref):

```julia
itfm = arrayscale(tform, SD)
warped = warp(moving, itfm)
```

## Centre of rotation

Rotations are defined around the *origin of coordinates*, which for a plain
Julia array is index `(1,1,…)`.  For most images, rotation around the image
centre is more natural.  Call `centered` (from ImageTransformations) on both
images to shift the origin to the centre before registering:

```julia
using ImageTransformations: centered

cfixed  = centered(fixed)
cmoving = centered(moving)

tform, mm = qd_rigid(collect(cfixed), collect(float(cmoving)), mxshift, mxrot)
```

Remember to call `centered` again when applying the result to a new image, or
re-encode the transformation with a different origin using
`recenter(tform, newctr)`.

## Initial guess

If you already have an approximate transformation, pass it as `initial_tfm` to
warm-start the search:

```julia
guess = Translation(1.0, 0.0)
tform, mm = qd_translate(fixed, moving, mxshift; initial_tfm=guess)
```

## Pre-smoothing with `qsmooth`

[`qsmooth`](@ref) smooths an image with a quadratic B-spline kernel and returns
an interpolant that can be warped cheaply.  Pre-smoothing `fixed` and passing
`presmoothed=true` reduces the mismatch evaluation cost:

```julia
fixed_s = qsmooth(fixed)
tform, mm = qd_translate(fixed_s, moving, mxshift; presmoothed=true)
```

Do **not** smooth `moving`.

## Overlap threshold (`thresh`)

`thresh` prevents the optimiser from "aligning" images by sliding one entirely
out of view.  The default is 10 % of the sum-of-squared intensity of `fixed`
for translations and rigid transforms, and 50 % for affine transforms (because
the extra degrees of freedom make degenerate solutions more tempting).  Increase
`thresh` if you see unexpected results.

## Coarse rotation search

For large rotations where QuadDIRECT may not converge, use
[`rotation_gridsearch`](@ref) to identify a good starting rotation, then refine
with [`qd_rigid`](@ref):

```julia
best_rot, _ = rotation_gridsearch(fixed, moving, mxshift, maxradians, gridsz)
tform, mm   = qd_rigid(fixed, moving, mxshift, mxrot; initial_tfm=best_rot)
```

[`grid_rotations`](@ref) generates the grid of candidate rotations used
internally by `rotation_gridsearch` and can also be used standalone.
