## Transformations

# Rotation only
"""
    R2 = rot(θ)

Calculate a two-dimensional rotation transformation `R2`.
`θ` is the angle of rotation around the `z` axis applied to the `xy` plane.
"""
rot(θ::Number) = LinearMap(RotMatrix(θ))

"""
    R3 = rot(qx, qy, qz)

    Calculate a three-dimensional rotation transformation from the supplied parameters.
    `qx`, `qy`, and `qz` specify the rotation via the components of the corresponding
    [unit quaternion](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).
    Briefly, `R3` is a rotation around the axis specified by the unit vector `q/|q|`
    (where `q` is the 3-component vector), with an angle `θ` given by `sin(θ/2) = |q|`.
    If `|q| > 1`, `rot` returns `nothing` instead of erroring.
"""
function rot(qx::Number, qy::Number, qz::Number)
    # Unit-quaternion parametrization
    r2 = qx^2 + qy^2 + qz^2
    r2 > 1 && return nothing
    qr = sqrt(1 - r2)
    R = @SMatrix([1 - 2*(qy^2 + qz^2)  2*(qx*qy - qz*qr)    2*(qx*qz + qy*qr);
                  2*(qx*qy + qz*qr)    1 - 2*(qx^2 + qz^2)  2*(qy*qz - qx*qr);
                  2*(qx*qz - qy*qr)    2*(qy*qz + qx*qr)    1 - 2*(qx^2 + qy^2)])
    return LinearMap(RotMatrix(R))
end

#rotation + translation
tfmrigid(dx::Number, dy::Number, θ::Number) = Translation(dx, dy) ∘ rot(θ)

tfmrigid(dx::Number, dy::Number, dz::Number, qx::Number, qy::Number, qz::Number) =
    Translation(dx, dy, dz) ∘ rot(qx, qy, qz)

## Transformations supplied as vectors, for which we pass the array so we can infer the dimensionality

@noinline dimcheck(v, lv) = length(v) == lv || throw(DimensionMismatch("expected $lv parameters, got $(length(v))"))

function rot(theta, img::AbstractArray{T,2}) where {T}
    dimcheck(theta, 1)
    return rot(theta[1])
end

function rot(qs, img::AbstractArray{T,3}) where {T}
    dimcheck(qs, 3)
    qx, qy, qz = qs
    return rot(qx, qy, qz)
end

function tfmrigid(params, img::AbstractArray{T,2}) where {T}
    dimcheck(params, 3)
    dx, dy, θ = params
    return tfmrigid(dx, dy, θ)
end

function tfmrigid(params, img::AbstractArray{T,3}) where {T}
    dimcheck(params, 6)
    dx, dy, dz, qx, qy, qz = params
    return tfmrigid(dx, dy, dz, qx, qy, qz)
end

## Computing mismatch

#rotation + shift, fast because it uses fourier method for shift
function rigid_mm_fast(theta, mxshift, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    # The reason this is `initial_tfm ∘ rot` rather than `rot ∘ initial_tfm`
    # is explained in `linmap`
    tfm = arrayscale(initial_tfm ∘ rot(theta, moving), SD)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    bshft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity)
    return mm
end

#rotation + shift, slow because it warps for every rotation and shift
function rigid_mm_slow(params, fixed, moving, thresh, SD; initial_tfm=IdentityTransformation())
    tfm = arrayscale(initial_tfm ∘ tfmrigid(params, moving), SD)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

function qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot;
                         SD=I,
                         initial_tfm=IdentityTransformation(),
                         thresh=0.1*sum(_abs2.(fixed[.!(isnan.(fixed))])),
                         kwargs...)
    #note: if a trial rotation results in image overlap < thresh for all possible shifts then QuadDIRECT throws an error
    f(x) = rigid_mm_fast(x, mxshift, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper = [mxrot...]
    lower = -upper
    root_coarse, x0coarse = _analyze(f, lower, upper;
                                     minwidth=minwidth_rot, print_interval=100, maxevals=5e4, kwargs..., atol=0, rtol=1e-3)
    box_coarse = minimum(root_coarse)
    tfmcoarse0 = initial_tfm ∘ rot(position(box_coarse, x0coarse), moving)
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization=:intensity, initial_tfm=arrayscale(tfmcoarse0, SD))
    tfmcoarse = tfmcoarse0 ∘ pscale(Translation(best_shft), SD)
    return tfmcoarse, mm
end

function qd_rigid_fine(fixed, moving, mxrot, minwidth_rot;
                       SD=I,
                       initial_tfm=IdentityTransformation(),
                       thresh=0.1*sum(_abs2.(fixed[.!(isnan.(fixed))])),
                       kwargs...)
    f(x) = rigid_mm_slow(x, fixed, moving, thresh, SD; initial_tfm=initial_tfm)
    upper_shft = fill(2.0, ndims(fixed))
    upper_rot = mxrot
    upper = vcat(upper_shft, upper_rot)
    lower = -upper
    minwidth_shfts = fill(0.005, ndims(fixed))
    minwidth = vcat(minwidth_shfts, minwidth_rot)
    root, x0 = _analyze(f, lower, upper; minwidth=minwidth, print_interval=100, maxevals=5e4, kwargs...)
    box = minimum(root)
    tfmfine = initial_tfm ∘ tfmrigid(position(box, x0), moving)
    return tfmfine, value(box)
end

"""
    tform, mm = qd_rigid(fixed, moving, mxshift, mxrot;
                         presmoothed=false,
                         SD=I, minwidth_rot=default_minwidth_rot(fixed, SD),
                         thresh=thresh, initial_tfm=IdentityTransformation(), kwargs...)

Optimize a rigid transformation (rotation + shift) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm.  The algorithm is run twice: the first step finds the optimal rotation,
using a fourier method to speed up the search for the best whole-pixel shift.
The second step performs the rotation + shift in combination to obtain sub-pixel accuracy.
Any `NaN`-valued pixels are not included in the mismatch; you can use this to mask out
any regions of `fixed` that you don't want to align against.

`mxshift` is the maximum-allowed translation, in units of array indices. It can be passed
as a vector or tuple.
`mxrot` is the maximum-allowed rotation, in radians for 2d or
[quaternion-units](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) for 3d.
See [`RegisterQD.rot`](@ref) for more information.
`minwidth_rot` optionally specifies the lower limit of resolution for the rotation;
the default is a rotation that moves corner elements by 0.1 pixel.

`kwargs...` can include any keyword argument that can be passed to `QuadDIRECT.analyze`.
It's recommended that you pass your own stopping criteria when possible
(i.e. `rtol`, `atol`, and/or `fvalue`).
If you provide `rtol` and/or `atol` they will apply only to the second (fine) step of the registration;
there is currently no way to adjust these criteria for the coarse step.

The rotation returned will be centered on the origin-of-coordinates, i.e. (0,0) for a 2D image.
Usually it is more natural to consider rotations around the center of the image.
If you would like `mxrot` and the returned rotation to act relative to the center of the image,
then you must move the origin to the center of the image by calling `centered(img)` from the
`ImageTransformations` package.
Call `centered` on both the fixed and moving image to generate the `fixed` and `moving` that you provide as arguments.
If you later want to apply the returned transform to an image you must remember to call `centered`
on that image as well.  Alternatively you can re-encode the transformation in terms of a
different origin by calling `recenter(tform, newctr)` where `newctr` is the displacement of the new center from the old center.

Use `SD` ("spatial displacements") if your axes are not uniformly sampled, for example
`SD = diagm(voxelspacing)` where `voxelspacing` is a vector encoding the spacing along all axes
of the image.
See [`arrayscale`](@ref) for more information about `SD`.

`thresh` enforces a certain amount of sum-of-squared-intensity overlap between
the two images; with non-zero `thresh`, it is not permissible to "align" the images by
shifting one entirely out of the way of the other. The default value for `thresh` is 10%
of the sum-of-squared-intensity of `fixed`.

If you have a good initial guess at the solution, pass it with the `initial_tfm` kwarg to jump-start the search.

Use `presmoothed=true` if you have called [`qsmooth`](@ref) on `fixed` before calling `qd_rigid`.
Do not smooth `moving`.

Both the output `tfm` and any `initial_tfm` are represented in *physical* coordinates;
as long as `initial_tfm` is a rigid transformation, `tfm` will be a pure rotation+translation.
If `SD` is not the identity, use `arrayscale` before applying the result to `moving`.
"""
function qd_rigid(fixed, moving, mxshift::VecLike, mxrot::Union{Number,VecLike};
                  presmoothed=false,
                  SD=I,
                  minwidth_rot=default_minwidth_rot(CartesianIndices(fixed), SD),
                  thresh=0.1*sum(_abs2.(fixed[.!(isnan.(fixed))])),
                  initial_tfm=IdentityTransformation(),
                  print_interval=100,
                  kwargs...)
    #TODO make sure isrotation(initial_tfm.linear) returns true, else throw a warning and an option to terminate? do I still allow the main block to run?
    if initial_tfm == IdentityTransformation() || isrotation(initial_tfm.linear)
    else
        @warn "initial_tfm is not a rigid transformation"
    end
    fixed, moving = float(fixed), float(moving)
    if presmoothed
        moving = qinterp(eltype(fixed), moving)
    end
    mxrot = [mxrot...]
    print_interval < typemax(Int) && print("Running coarse step\n")
    tfm_coarse, mm_coarse = qd_rigid_coarse(fixed, moving, mxshift, mxrot, minwidth_rot; SD=SD, initial_tfm=initial_tfm, thresh=thresh, print_interval=print_interval, kwargs...)
    print_interval < typemax(Int) && print("Obtained mismatch $mm_coarse from coarse step, running fine step\n")
    final_tfm, mm_fine = qd_rigid_fine(fixed, moving, mxrot./2, minwidth_rot; SD=SD, initial_tfm=tfm_coarse, thresh=thresh, print_interval=print_interval, kwargs...)
    return final_tfm, mm_fine
end
