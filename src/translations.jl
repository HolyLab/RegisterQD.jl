####################  Translation Search ##########################

function tfmshift(params, img::AbstractArray{T,N}) where {T,N}
    length(params) == N || throw(DimensionMismatch("expected $N parameters, got $(length(params))"))
    return Translation(params...)
end

#slow because it warps for every shift instead of using fourier method
function translate_mm_slow(params, fixed, moving, thresh; initial_tfm=IdentityTransformation())
    tfm = initial_tfm ∘ tfmshift(params, moving)
    moving, fixed = warp_and_intersect(moving, fixed, tfm)
    mm = mismatch0(fixed, moving; normalization=:intensity)
    return ratio(mm, thresh, Inf)
end

function qd_translate_fine(fixed, moving;
                           initial_tfm=IdentityTransformation(),
                           minwidth=fill(0.01, ndims(fixed)),
                           thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                           kwargs...)
    f(x) = translate_mm_slow(x, fixed, moving, thresh; initial_tfm=initial_tfm)
    upper = fill(1.0, ndims(fixed))
    lower = -upper
    root, x0 = _analyze(f, lower, upper; minwidth=minwidth, print_interval=100, maxevals=5e4, kwargs...)
    box = minimum(root)
    tfmfine = initial_tfm ∘ tfmshift(position(box, x0), moving)
    return tfmfine, value(box)
end

"""
`tform, mm = qd_translate(fixed, moving, mxshift; thresh=thresh, kwargs...)`
optimizes a simple shift (translation) to minimize the mismatch between `fixed` and
`moving` using the QuadDIRECT algorithm with the constraint that no shifts larger than
``mxshift` will be considered.

Both `mxshift` and the returned translation are specified in terms of pixel units, so the
algorithm need not be aware of anisotropic sampling.

The algorithm involves two steps: the first step uses a fourier method to speed up
the search for the best whole-pixel shift.  The second step refines the search for sub-pixel accuracy.
The default precision of this step is 1% of one pixel (0.01) for each dimension of the image.
You can override the default with the `minwidth` argument.  `kwargs...` can also include 
any other keyword argument that can be passed to `QuadDIRECT.analyze`.
It's recommended that you pass your own stopping criteria when possible (i.e. `rtol`, `atol`, and/or `fvalue`).

If you have a good initial guess at the solution, pass it with the `initial_tfm` kwarg to jump-start the search.
`thresh` enforces a certain amount of sum-of-squared-intensity overlap between the two images;
with non-zero `thresh`, it is not permissible to "align" the images by shifting one entirely out of the way of the other.

If the `crop` keyword arg is `true` then `fixed` is cropped by `mxshift` on all sides so that there will be 
complete overlap between `fixed` and `moving` for any evaluated shift. This avoids edge effects that can
occur due to normalization when the transformed `moving` doesn't fully overlap with `fixed`.
"""
function qd_translate(fixed, moving, mxshift;
                      thresh=0.1*sum(abs2.(fixed[.!(isnan.(fixed))])),
                      initial_tfm=IdentityTransformation(),
                      minwidth=fill(0.01, ndims(fixed)), print_interval=100, crop=false, kwargs...)
    fixed, moving = float(fixed), float(moving)
    print_interval < typemax(Int) && print("Running coarse step\n")
    if crop
        #we enforce that moving is always bigger than fixed by amount 2*(maxshift+1) (the +1 is for the fine step)
        sz = size(fixed) .- (2 .* mxshift)
        if any(size(moving) .< (2 .* (mxshift.+1)))
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
    best_shft, mm = best_shift(fixed_inner, moving_inner, mxshift, thresh; normalization=:intensity, initial_tfm=initial_tfm)
    tfm_coarse = initial_tfm ∘ Translation(best_shft)
    print_interval < typemax(Int) && print("Running fine step\n")
    return qd_translate_fine(fixed_inner, moving_fine; initial_tfm=tfm_coarse, thresh=thresh, minwidth=minwidth, print_interval=print_interval, kwargs...)
end

crop_rng(rng::AbstractUnitRange{Int}, amt::Int) = (first(rng)+amt):(last(rng)-amt)
