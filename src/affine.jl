"""
    tfm = linmap(mat, img, initial_tfm=IdentityTransformation())

Construct a linear transformation from the values in `mat`.
`mat` can be any list of `N^2` values, where `N` is the dimensionality.
`img` is the array (or array of the same dimensionality) that `tfm` will be applied to.

If `initial_tfm` is supplied, it will be composed with the one from `mat`.

See also [`RegisterQD.aff`](@ref), which includes a translational component.
"""
function linmap(mat, img::AbstractArray{T, N}, initial_tfm = IdentityTransformation()) where {T, N}
    # img isn't used *except* to get the dimensionality in an inferrable way, so that
    # the (static) size of the array in `tfm` is known.
    mat = [mat...]
    mat = reshape(mat, N, N)
    lm = LinearMap(SMatrix{N, N}(mat))
    # The order below is subtle: you might think that `initial_tfm` should be applied
    # to `x` first, so that the order of composition should be `lm ∘ initial_tfm`.
    # But what we're going to be doing is calculating an `lm` that aligns
    #    imgw(x) = moving(initial_tfm(x))
    # to `fixed`. The final aligned image is
    #    imgw2(x) = imgw(lm(x))
    # Putting these two together, we see that
    #    imgw2(x) = moving(initial_tfm(lm(x)))
    # and thus the resulting transformation is
    #    initial_tfm ∘ lm
    return initial_tfm ∘ lm
end

"""
    tfm = aff(params, img, initial_tfm=IdentityTransformation())

Construct an affine transformation from the values in `params`.
`params` can be any list of `N + N^2` values, where `N` is the dimensionality;
the first `N` are for the translation,
and the last `N^2` are for the linear portion (see [`RegisterQD.linmap`](@ref)).

`img` is the array (or array of the same dimensionality) that `tfm` will be applied to.

If `initial_tfm` is supplied, it will be composed with the one from `params`.
"""
function aff(params, img::AbstractArray{T, N}, initial_tfm = IdentityTransformation()) where {T, N}
    # img isn't used *except* to get the dimensionality in an inferrable way, so that
    # the (static) sizes of the arrays in `tfm` are known.
    params = [params...]
    length(params) == (N + N^2) || throw(DimensionMismatch("expected $(N + N^2) parameters, got $(length(params))"))
    offs = Float64.(params[1:N])
    mat = Float64.(params[(N + 1):end])
    mat = reshape(mat, N, N)
    # The order below is explained in `linmap`
    return initial_tfm ∘ AffineMap(SMatrix{N, N}(mat), SVector{N}(offs))
end

# This acts as the objective function during optimization, see qd_affine_coarse
# which defines `f(x)` in terms of this function.
# here params contains parameters of a linear map
# The "fast" part means that it handles translation using `best_shift`, so this is
# concerned only with the linear-map portion of the affine transformation.
function affine_mm_fast(params, mxshift, fixed, moving, thresh, SD; initial_tfm = IdentityTransformation())
    tfm = arrayscale(linmap(params, moving, initial_tfm), SD)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    bshft, mm = best_shift(fixed, moving, mxshift, thresh; normalization = :intensity)
    return mm
end

# Similar to the above but includes the translation in `params`
# This is used for optimization of the complete affine transformation.
function affine_mm_slow(params, fixed, moving, thresh, SD; initial_tfm = IdentityTransformation())
    tfm = arrayscale(aff(params, moving, initial_tfm), SD)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch_zeroshift(fixed, moving; normalization = :intensity)
    return ratio(mm, thresh; fillval = Inf)
end

"""
    tfm, mm = qd_affine_coarse(fixed, moving, mxshift, linmins, linmaxs; kwargs...)

Compute an affine transformation `tfm` that coarsely minimizes the mismatch between
`fixed(x)` and `moving(tfm(x))`. `mm` is the total mismatch.
The translational component is handled at the level of integer-pixel shifts,
which is why this is "coarse" optimization.
"""
function qd_affine_coarse(
        fixed, moving, mxshift, linmins, linmaxs;
        SD = I,
        initial_tfm = IdentityTransformation(),
        thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])),
        minwidth = default_lin_minwidths(moving),
        maxevals = 5.0e4,
        kwargs...
    )
    # Define the objective function for the coarse stage
    f(x) = affine_mm_fast(x, mxshift, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    # QuadDIRECT benefits from bounds on the search region, build those bounds
    upper = linmaxs
    lower = linmins
    # Try to find the global minimum (within the needed precision)
    root, x0 = _analyze(
        f, lower, upper;
        minwidth = minwidth, print_interval = 100, maxevals = maxevals, kwargs..., atol = 0, rtol = 1.0e-3
    )
    box = minimum(root)
    # Convert the coarse-minimum into a complete affine transformation
    params = position(box, x0)
    tfmcoarse0 = linmap(params, moving, initial_tfm)
    best_shft, mm = best_shift(fixed, moving, mxshift, thresh; normalization = :intensity, initial_tfm = arrayscale(tfmcoarse0, SD))
    tfmcoarse = tfmcoarse0 ∘ pscale(Translation(best_shft), SD)
    return tfmcoarse, mm
end

"""
    tfm, mm = qd_affine_fine(fixed, moving, linmins, linmaxs; initial_tfm=IdentityTransformation(), kwargs...)

Compute an affine transformation `tfm` that minimizes the mismatch between
`fixed(x)` and `moving(tfm(x))`. `mm` is the total mismatch.
This only allows small translations (just a couple of pixels).
As a consequence, any large translations must be supplied with reasonable accuracy in
`initial_tfm` (see [`RegisterQD.qd_affine_coarse`](@ref)).

`minwidth_mat` optionally specifies the lower limit of resolution for the *linear-map*
parameters only; the translation resolution is fixed internally. The `_mat` suffix
distinguishes this from the full-search-space `minwidth` accepted by
[`RegisterQD.qd_affine_coarse`](@ref): here it names just the linear-map subspace, which
is combined internally with the (fixed) translation resolution.
"""
function qd_affine_fine(
        fixed, moving, linmins, linmaxs;
        SD = I,
        initial_tfm = IdentityTransformation(),
        thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])),
        minwidth_mat = default_lin_minwidths(fixed) ./ 10,
        maxevals = 5.0e4,
        kwargs...
    )
    # Define the objective function
    f(x) = affine_mm_slow(x, fixed, moving, thresh, SD; initial_tfm = initial_tfm)
    # Limit translations to only a couple of pixels
    upper_shft = fill(2.0, ndims(fixed))
    # Combine translation limits with the affine-transform limits
    upper = vcat(upper_shft, linmaxs)
    lower = vcat(-upper_shft, linmins)
    # Don't worry about translations that less than 1% of a pixel
    minwidth_shfts = fill(0.01, ndims(fixed))
    minwidth = vcat(minwidth_shfts, minwidth_mat)
    # Try to find the global minimum
    root, x0 = _analyze(
        f, lower, upper;
        minwidth = minwidth, print_interval = 100, maxevals = maxevals, kwargs...
    )
    box = minimum(root)
    # Convert the result to an affine transformation
    params = position(box, x0)
    tfmfine = aff(params, moving, initial_tfm)
    return tfmfine, value(box)
end

#TODO oops maybe dmax and ndax is too closely related to linmax and linmin?
# Response: dmax and ndmax are designed to allow you to control linmax and linmin.
# You supply one or the other, so I don't see a problem.
#TODO I think that this is a tad loquatious, and not enough examples. Permission to rework it?
"""
    tform, mm = qd_affine(fixed, moving, mxshift, linmins, linmaxs;
                          presmoothed=false, SD=I,
                          thresh=0.5*sum(abs2,fixed), initial_tfm=IdentityTransformation(),
                          kwargs...)
    tform, mm = qd_affine(fixed, moving, mxshift;
                          dmax=0.05, ndmax=0.05,
                          presmoothed=false, SD=I,
                          thresh=0.5*sum(abs2,fixed), initial_tfm=IdentityTransformation(),
                          kwargs...)

Optimize an affine transformation (linear map + translation) to minimize the mismatch
between `fixed` and `moving` using the QuadDIRECT algorithm.

Returns `(tform, mm)` where `tform` is an `AffineMap` and `mm` is the residual mismatch
value (lower is better).

The algorithm runs in two steps: the first step samples the search space at a coarser
resolution than the second. `kwargs...` may contain any keyword argument accepted by
`QuadDIRECT.analyze`. Supplying your own stopping criteria (`rtol`, `atol`, and/or
`fvalue`) is recommended. Any `rtol`/`atol` you supply will apply only to the second
(fine) step; the coarse step uses fixed internal criteria.

`tform` is centered on the origin of coordinates, i.e. `(0, 0)` for 2D images. To
rotate around the image center instead, call `centered(img)` (from `ImageFiltering` or
`ImageTransformations`) on both `fixed` and `moving` before calling `qd_affine`.
To re-encode the result relative to a different center, use
`recenter(tform, newctr)`.

`linmins` and `linmaxs` bound the allowable values of the linear-map matrix entries.
They can be `N×N` matrices or flattened vectors. If omitted, a modest default search
space is used, controllable via the `dmax` (diagonal) and `ndmax` (off-diagonal) keyword
arguments; e.g. `dmax=0.05` permits diagonal entries in `[0.95, 1.05]`.

`mxshift` sets the maximum allowable translation in each dimension.

Use `presmoothed=true` if you have called [`qsmooth`](@ref) on `fixed` before calling
`qd_affine`. Do not smooth `moving`.

If you have a good initial guess, pass it with `initial_tfm` to jump-start the search.

Use `SD` if your axes are not uniformly sampled, for example
`SD = diagm(voxelspacing)` where `voxelspacing` encodes the physical spacing along each
axis. See [`arrayscale`](@ref) for details.

`thresh` enforces a minimum sum-of-squared-intensity overlap; with non-zero `thresh`,
it is not permissible to "align" the images by shifting one entirely out of view.
The default is 50% of the sum-of-squared-intensity of `fixed` — higher than the 10%
used by [`qd_translate`](@ref) and [`qd_rigid`](@ref) because affine transformations
have extra degrees of freedom (scaling, shear) that make degenerate low-overlap
solutions more likely.

!!! note
    A mismatch backend such as
    [RegisterMismatch.jl](https://github.com/HolyLab/RegisterMismatch.jl) must be
    loaded before calling this function.

# Examples

```julia-repl
julia> using RegisterMismatch

julia> fixed = Float64.(reshape(1:25, 5, 5));

julia> moving = circshift(fixed, (1, 0));

julia> tform, mm = qd_affine(fixed, moving, (3, 3); print_interval=typemax(Int));

julia> mm < 1e-6
true
```
"""
function qd_affine(
        fixed, moving, mxshift, linmins, linmaxs;
        presmoothed = false,
        SD = I,
        # Affine's extra degrees of freedom (scaling/shear) make degenerate
        # low-overlap solutions more likely, so the default requires more overlap
        # (50%) than the 10% used by qd_translate/qd_rigid.
        thresh = 0.5 * sum(abs2.(fixed[.!(isnan.(fixed))])),
        initial_tfm = IdentityTransformation(),
        print_interval = 100,
        kwargs...
    )
    fixed, moving = float(fixed), float(moving)
    if presmoothed
        moving = qinterp(eltype(fixed), moving)
    end
    linmins = [linmins...]
    linmaxs = [linmaxs...]
    print_interval < typemax(Int) && print("Running coarse step\n")
    mw = default_lin_minwidths(moving)
    tfm_coarse, mm_coarse = qd_affine_coarse(
        fixed, moving, mxshift, linmins, linmaxs;
        SD = SD, minwidth = mw, initial_tfm = initial_tfm, thresh = thresh, print_interval = print_interval, kwargs...
    )
    print_interval < typemax(Int) && print("Running fine step\n")
    mw = mw ./ 100
    linmins, linmaxs = scalebounds(linmins, linmaxs, 0.5) # should these be narrowed further?
    final_tfm, final_mm = qd_affine_fine(
        fixed, moving, linmins, linmaxs;
        SD = SD, minwidth_mat = mw, initial_tfm = tfm_coarse, thresh = thresh, print_interval = print_interval, kwargs...
    )
    return final_tfm, final_mm
end

function qd_affine(
        fixed, moving, mxshift;
        dmax = 0.05, ndmax = 0.05,
        kwargs...
    )
    minb, maxb = default_linmap_bounds(fixed; dmax = dmax, ndmax = ndmax)
    return qd_affine(fixed, moving, mxshift, minb, maxb; kwargs...)
end
