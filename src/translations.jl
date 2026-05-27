####################  Translation Search ##########################

function tfmshift(params, img::AbstractArray{T, N}) where {T, N}
    length(params) == N || throw(DimensionMismatch("expected $N parameters, got $(length(params))"))
    return Translation(params...)
end

#slow because it warps for every shift instead of using fourier method
function translate_mm_slow(params, fixed, moving, thresh; initial_tfm = IdentityTransformation())
    tfm = initial_tfm ∘ tfmshift(params, moving)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch_zeroshift(fixed, moving; normalization = :intensity)
    return ratio(mm, thresh; fillval = Inf)
end

function qd_translate_fine(
        fixed, moving;
        initial_tfm = IdentityTransformation(),
        minwidth = fill(0.01, ndims(fixed)),
        thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])),
        kwargs...
    )
    f(x) = translate_mm_slow(x, fixed, moving, thresh; initial_tfm = initial_tfm)
    upper = fill(1.0, ndims(fixed))
    lower = -upper
    root, x0 = _analyze(f, lower, upper; minwidth = minwidth, print_interval = 100, kwargs...)
    box = minimum(root)
    tfmfine = initial_tfm ∘ tfmshift(position(box, x0), moving)
    return tfmfine, value(box)
end

"""
    tform, mm = qd_translate(fixed, moving, mxshift;
                             presmoothed=false, thresh=0.1*sum(abs2,fixed), kwargs...)

Optimize a translation to minimize the mismatch between `fixed` and `moving` using the
QuadDIRECT algorithm. No shift larger than `mxshift` (after an optional `initial_tfm`)
will be considered.

Returns `(tform, mm)` where `tform` is a `Translation` and `mm` is the residual
mismatch value (lower is better).

Both `mxshift` and the returned translation are in pixel units, so the algorithm does
not need to know the physical sampling.

The algorithm runs in two steps: the first uses a Fourier method to find the best
whole-pixel shift; the second refines for sub-pixel accuracy with default precision of
1% of one pixel (`minwidth=fill(0.01, ndims(fixed))`). Override with the `minwidth`
keyword argument. `kwargs...` can include any keyword argument accepted by
`QuadDIRECT.analyze`. Supplying your own stopping criteria (`rtol`, `atol`, and/or
`fvalue`) is recommended.

Use `presmoothed=true` if you have called [`qsmooth`](@ref) on `fixed` before calling
`qd_translate`. Do not smooth `moving`.

If you have a good initial guess, pass it with `initial_tfm` to jump-start the search.

`thresh` enforces a minimum sum-of-squared-intensity overlap between the two images;
with non-zero `thresh`, shifting one image entirely out of view is not a valid solution.
The default is 10% of the sum-of-squared-intensity of `fixed`.

If `crop=true`, `fixed` is cropped by `mxshift` on all sides so that there is complete
overlap between `fixed` and `moving` for every evaluated shift. This avoids edge-effect
normalization artifacts when the transformed `moving` does not fully overlap `fixed`.

!!! note
    A mismatch backend such as
    [RegisterMismatch.jl](https://github.com/HolyLab/RegisterMismatch.jl) must be
    loaded before calling this function.

# Examples

```jldoctest
julia> using RegisterMismatch

julia> fixed = Float64.(reshape(1:25, 5, 5));

julia> moving = circshift(fixed, (2, 1));  # known shift: 2 rows, 1 column

julia> tform, mm = qd_translate(fixed, moving, (3, 3); print_interval=typemax(Int));

julia> println(tform.translation)
[2.0, 1.0]

julia> mm
0.0
```
"""
function qd_translate(
        fixed, moving, mxshift;
        presmoothed = false,
        thresh = 0.1 * sum(abs2.(fixed[.!(isnan.(fixed))])),
        initial_tfm = IdentityTransformation(),
        minwidth = fill(0.01, ndims(fixed)), print_interval = 100, crop = false, kwargs...
    )
    fixed, moving = float(fixed), float(moving)
    if presmoothed
        moving = qinterp(eltype(fixed), moving)
    end
    print_interval < typemax(Int) && print("Running coarse step\n")
    if crop
        #we enforce that moving is always bigger than fixed by amount 2*(maxshift+1) (the +1 is for the fine step)
        sz = size(fixed) .- (2 .* mxshift)
        if any(size(moving) .< (2 .* (mxshift .+ 1)))
            error("Moving image size must be at least 2 * (mxshift+1) when crop_edges is set to true")
        end
        cropped_inds_f = crop_rng.(axes(fixed), mxshift)
        cropped_inds_m = crop_rng.(axes(moving), mxshift)
        moving_inner = view(moving, cropped_inds_m...)
        fixed_inner = view(fixed, cropped_inds_f...)
        moving_fine = OffsetArray(moving, (-1 .* mxshift)...)
    else
        fixed_inner = fixed
        moving_inner = moving_fine = moving
    end
    best_shft, mm = best_shift(fixed_inner, moving_inner, mxshift, thresh; normalization = :intensity, initial_tfm = initial_tfm)
    tfm_coarse = initial_tfm ∘ Translation(best_shft)
    print_interval < typemax(Int) && print("Running fine step\n")
    return qd_translate_fine(fixed_inner, moving_fine; initial_tfm = tfm_coarse, thresh = thresh, minwidth = minwidth, print_interval = print_interval, kwargs...)
end

crop_rng(rng::AbstractUnitRange{Int}, amt::Int) = (first(rng) + amt):(last(rng) - amt)
